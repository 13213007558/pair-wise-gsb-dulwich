# pathspec.py -- Git pathspec compilation and matching
# Copyright (C) 2026 Jelmer Vernooij <jelmer@jelmer.uk>
#
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# Dulwich is dual-licensed under the Apache License, Version 2.0 and the GNU
# General Public License as published by the Free Software Foundation; version 2.0
# or (at your option) any later version. You can redistribute it and/or
# modify it under the terms of either of these two licenses.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# You should have received a copy of the licenses; if not, see
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/LICENSE-2.0> for a copy of the Apache
# License, Version 2.0.
#

"""Git pathspec compilation and matching.

A pathspec (see gitglossary(7)) selects a subset of the paths in a tree.
This module compiles a list of pathspecs into a :class:`Pathspec` object
that can be reused to match paths from any tree, without touching the
local filesystem.

Supported magic:

* `top` (`:/`): interpret the pattern from the repository root,
  regardless of the current working directory.
* `literal`: treat wildcard characters literally.
* `glob`: wildmatch(3) with `WM_PATHNAME`; `*` and `?` do not
  match `/`, while `**` matches across directories.
* `icase`: match case-insensitively.
* `exclude` (`:!` or `:^`): subtract matches from the result.

Without magic, a pattern is matched with shell-style wildcards in which
`*` and `?` also match `/`; a pattern without any wildcard matches
as a leading directory prefix, so `src` matches `src/a` but not
`src2` and `src/a` does not match `src/abc`.

Attribute matching (`:(attr:...)`) is not supported and raises
:exc:`PathspecError`, as do unknown or conflicting magic signatures.

All matching is done on `bytes` paths; non-UTF-8 path names are never
decoded lossily for comparison.
"""

__all__ = [
    "Pathspec",
    "PathspecError",
    "compile_pathspecs",
    "iter_tree_matches",
]

import os
import posixpath
import re
from collections.abc import Iterable, Iterator, Sequence

from .object_store import iter_tree_contents
from .objects import ObjectID, TreeEntry
from .pack import ObjectContainer
from .wildmatch import MalformedPattern, translate_bracket_expression
from .wildmatch import translate as translate_glob


class PathspecError(ValueError):
    """Raised when a pathspec cannot be parsed or compiled."""


_SHORT_MAGIC = {
    b"/": "top",
    b"!": "exclude",
    b"^": "exclude",
}

_SUPPORTED_MAGIC = frozenset({"top", "literal", "icase", "glob", "exclude"})

_BACKSLASH = 0x5C


def _parse_magic(spec: bytes) -> tuple[frozenset[str], bytes]:
    """Split a raw pathspec into its magic signatures and pattern."""
    if not spec.startswith(b":"):
        return frozenset(), spec
    if spec.startswith(b"::"):
        # "::" explicitly resets to no magic
        return frozenset(), spec[2:]
    if spec.startswith(b":("):
        end = spec.find(b")")
        if end == -1:
            raise PathspecError(f"unterminated magic signature in {spec!r}")
        magic = set()
        for name in spec[2:end].split(b","):
            if not name:
                continue
            if name == b"attr" or name.startswith(b"attr:"):
                raise PathspecError("attribute pathspec magic (attr:) is not supported")
            try:
                decoded = name.decode("ascii")
            except UnicodeDecodeError:
                raise PathspecError(f"invalid pathspec magic in {spec!r}") from None
            if decoded not in _SUPPORTED_MAGIC:
                raise PathspecError(f"unsupported pathspec magic: {decoded!r}")
            magic.add(decoded)
        return frozenset(magic), spec[end + 1 :]
    magic = set()
    i = 1
    while i < len(spec) and spec[i : i + 1] in _SHORT_MAGIC:
        magic.add(_SHORT_MAGIC[spec[i : i + 1]])
        i += 1
    if i == 1:
        raise PathspecError(f"invalid pathspec magic in {spec!r}")
    return frozenset(magic), spec[i:]


def _has_wildcard(pattern: bytes) -> bool:
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == _BACKSLASH:
            i += 2
            continue
        if c in b"*?[":
            return True
        i += 1
    return False


def _unescape(pattern: bytes) -> bytes:
    r"""Remove backslash escapes, turning e.g. `\*` into a literal `*`."""
    if _BACKSLASH not in pattern:
        return pattern
    out = bytearray()
    i = 0
    while i < len(pattern):
        if pattern[i] == _BACKSLASH and i + 1 < len(pattern):
            i += 1
        out.append(pattern[i])
        i += 1
    return bytes(out)


def _translate_default(pattern: bytes) -> bytes:
    """Translate a default-magic pathspec pattern to a regex.

    Unlike :func:`dulwich.wildmatch.translate` (`WM_PATHNAME`), `*`
    and `?` match `/` as well.
    """
    out = bytearray()
    i = 0
    while i < len(pattern):
        c = pattern[i : i + 1]
        if c == b"*":
            out += b".*"
        elif c == b"?":
            out += b"."
        elif c == b"[":
            i, rx = translate_bracket_expression(pattern, i)
            out += rx
            continue
        elif c == b"\\":
            i += 1
            if i >= len(pattern):
                out += rb"\\"
                break
            out += re.escape(pattern[i : i + 1])
        else:
            out += re.escape(c)
        i += 1
    return bytes(out)


class _LiteralItem:
    """A pathspec item without wildcards, matched as a directory prefix."""

    __slots__ = ("_icase", "_literal")

    def __init__(self, literal: bytes, icase: bool) -> None:
        self._literal = literal.lower() if icase else literal
        self._icase = icase

    def matches(self, path: bytes) -> bool:
        if self._literal == b"":
            return True
        candidate = path.lower() if self._icase else path
        return candidate == self._literal or candidate.startswith(self._literal + b"/")


class _RegexItem:
    """A pathspec item with wildcards, matched against the whole path."""

    __slots__ = ("_re",)

    def __init__(self, regex: bytes, icase: bool) -> None:
        flags = re.IGNORECASE if icase else 0
        self._re = re.compile(b"\\A(?:" + regex + b")\\Z", flags)

    def matches(self, path: bytes) -> bool:
        return self._re.match(path) is not None


def _build_item(pattern: bytes, magic: frozenset[str]) -> "_LiteralItem | _RegexItem":
    icase = "icase" in magic
    if "literal" in magic:
        return _LiteralItem(pattern, icase)
    if not _has_wildcard(pattern):
        return _LiteralItem(_unescape(pattern), icase)
    try:
        if "glob" in magic:
            regex = translate_glob(pattern)
        else:
            regex = _translate_default(pattern)
    except MalformedPattern as e:
        raise PathspecError(f"malformed pathspec pattern {pattern!r}: {e}") from e
    return _RegexItem(regex, icase)


def _normalize_prefix(cwd: str | bytes | None, root: str | bytes | None) -> bytes:
    """Resolve `cwd` to a repository-root-relative prefix.

    Raises:
      PathspecError: if `cwd` escapes the repository root.
    """
    if cwd is None:
        return b""
    cwd_b = os.fsencode(cwd) if isinstance(cwd, str) else bytes(cwd)
    if os.path.isabs(cwd_b) or posixpath.isabs(cwd_b):
        if root is None:
            raise PathspecError(
                "an absolute cwd requires the repository root to be given"
            )
        root_b = os.fsencode(root) if isinstance(root, str) else bytes(root)
        root_n = posixpath.normpath(root_b)
        cwd_n = posixpath.normpath(cwd_b)
        if cwd_n != root_n and not cwd_n.startswith(root_n.rstrip(b"/") + b"/"):
            raise PathspecError(
                f"cwd {cwd_b!r} is outside the repository root {root_b!r}"
            )
        rel = posixpath.relpath(cwd_n, root_n)
    else:
        rel = posixpath.normpath(cwd_b)
    if rel == b".":
        return b""
    if rel == b".." or rel.startswith(b"../"):
        raise PathspecError(f"cwd {cwd_b!r} is outside the repository root")
    return rel


class Pathspec:
    """A compiled set of pathspecs, reusable across trees.

    Positive items form the set of candidate paths; negative (`exclude`)
    items are then subtracted. When there are only negative items, the
    candidate set is everything.
    """

    def __init__(
        self,
        positives: "Sequence[_LiteralItem | _RegexItem]",
        negatives: "Sequence[_LiteralItem | _RegexItem]",
    ) -> None:
        """Create a pathspec from compiled positive and negative items."""
        self._positives = tuple(positives)
        self._negatives = tuple(negatives)

    @property
    def is_null(self) -> bool:
        """Whether this pathspec places no restriction on paths."""
        return not self._positives and not self._negatives

    def match(self, path: bytes) -> bool:
        """Check whether a repository-relative path is selected."""
        if self._positives:
            if not any(item.matches(path) for item in self._positives):
                return False
        # With only exclude items, everything is a candidate
        return not any(item.matches(path) for item in self._negatives)

    def match_paths(self, paths: Iterable[bytes]) -> Iterator[bytes]:
        """Yield the paths from `paths` that are selected."""
        for path in paths:
            if self.match(path):
                yield path


def compile_pathspecs(
    pathspecs: Sequence[str | bytes],
    *,
    cwd: str | bytes | None = None,
    root: str | bytes | None = None,
) -> Pathspec:
    """Compile pathspecs into a reusable :class:`Pathspec`.

    Args:
      pathspecs: Pathspecs, with optional magic signatures. `str`
        pathspecs are encoded with the filesystem encoding; `bytes`
        pathspecs are used as-is.
      cwd: Directory relative to which patterns without `top` magic are
        interpreted. May be relative to the repository root, or absolute
        (in which case `root` is required). Defaults to the repository
        root.
      root: Repository root, used to validate an absolute `cwd`.

    Returns:
      A compiled :class:`Pathspec` that matches repository-relative
      `bytes` paths and never touches the filesystem.

    Raises:
      PathspecError: on unknown, unsupported (`attr`) or conflicting
        magic, malformed patterns, or a `cwd` outside the repository
        root.
    """
    prefix = _normalize_prefix(cwd, root)
    positives: list[_LiteralItem | _RegexItem] = []
    negatives: list[_LiteralItem | _RegexItem] = []
    for spec in pathspecs:
        raw = os.fsencode(spec) if isinstance(spec, str) else bytes(spec)
        magic, pattern = _parse_magic(raw)
        if "literal" in magic and "glob" in magic:
            raise PathspecError(f"literal and glob magic are incompatible in {raw!r}")
        if "top" not in magic and prefix:
            pattern = prefix + b"/" + pattern if pattern else prefix
        pattern = pattern.rstrip(b"/")
        item = _build_item(pattern, magic)
        if "exclude" in magic:
            negatives.append(item)
        else:
            positives.append(item)
    if prefix and not positives:
        # With only exclude pathspecs, the current directory is the
        # starting set (as when running git from a subdirectory)
        positives.append(_LiteralItem(prefix, False))
    return Pathspec(positives, negatives)


def iter_tree_matches(
    store: ObjectContainer, tree_id: ObjectID | None, pathspec: Pathspec
) -> Iterator[TreeEntry]:
    """Iterate the entries of a tree selected by a compiled pathspec.

    Args:
      store: Object store to read trees from.
      tree_id: SHA1 of the tree to enumerate.
      pathspec: A compiled :class:`Pathspec`.

    Yields:
      :class:`TreeEntry` namedtuples for the non-tree entries of the tree
      whose paths are selected by `pathspec`.
    """
    for entry in iter_tree_contents(store, tree_id):
        assert entry.path is not None
        if pathspec.match(entry.path):
            yield entry

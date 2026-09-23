# pathspec.py -- Git pathspec compilation
# Copyright (C) 2026 Jelmer Vernooij <jelmer@jelmer.uk>
#
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# Dulwich is dual-licensed under the Apache License, Version 2.0 and the GNU
# General Public License as published by the Free Software Foundation; version 2.0
# or (at your option) any later version. You can redistribute it and/or
# modify it under the terms of either of these two licenses.
#
# You should have received a copy of the licenses; if not, see
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/LICENSE-2.0> for a copy of the Apache
# License, Version 2.0.
#

"""Git pathspec compilation.

This module turns Git pathspecs (see gitglossary(7)) into a reusable
matcher that can be applied to any tree or list of paths without touching
the filesystem.  Paths are kept as bytes throughout; no lossy UTF-8
decoding is involved in matching.

Supported magic signatures:

* top (':/' or ':(top)'): match from the repository root instead of
  relative to the current directory.
* literal ('::' or ':(literal)'): wildcards are literal characters.
* glob (':(glob)'): shell glob with FNM_PATHNAME semantics, i.e. '*' and
  '?' do not cross '/' and '**' spans directories.
* icase (':(icase)'): case-insensitive matching (ASCII-only, like Git).
* exclude (':!<pat>', ':^<pat>' or ':(exclude)'): drop matches.

Without a magic signature a pattern is matched literally as a directory
prefix, and wildcards (if any) use fnmatch semantics *without*
FNM_PATHNAME -- '*' crosses directory boundaries.

The 'attr' magic is not supported and raises :exc:'PathspecError', as do
unknown or conflicting ('literal' + 'glob') magic signatures.  All errors
are raised at compile time, before any tree enumeration happens.

Matching follows Git's set semantics: the positive pathspecs form a union
of candidate paths, then every exclude pathspec is subtracted.  When only
exclude pathspecs are given, the starting set is everything.
"""

from __future__ import annotations

__all__ = [
    "Pathspec",
    "PathspecError",
    "PathspecMatcher",
    "compile_pathspecs",
    "iter_tree_matches",
]

import posixpath
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from .wildmatch import MalformedPattern, translate_bracket_expression
from .wildmatch import translate as translate_wildmatch

if TYPE_CHECKING:
    from .object_store import BaseObjectStore
    from .objects import ObjectID, TreeEntry

_KNOWN_MAGICS = (b"top", b"literal", b"glob", b"icase", b"exclude")

_WILDCARD_RE = re.compile(rb"[*?[\\]")


class PathspecError(ValueError):
    """Raised for invalid or unsupported pathspecs."""


def _to_bytes(value: str | bytes, what: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError(f"{what} must be str or bytes, not {type(value).__name__}")


def _parse_long_magic(spec: bytes) -> tuple[set[bytes], bytes]:
    """Parse a '(...)' magic signature at the start of spec."""
    end = spec.find(b")")
    if end == -1:
        raise PathspecError(f"unterminated magic signature in {spec!r}")
    magics: set[bytes] = set()
    for word in spec[1:end].split(b","):
        if word in _KNOWN_MAGICS:
            magics.add(word)
        elif word == b"attr" or word.startswith(b"attr:"):
            raise PathspecError(
                f"unsupported magic 'attr' in {spec!r}:"
                " attribute matching is not implemented"
            )
        else:
            raise PathspecError(f"invalid pathspec magic {word!r} in {spec!r}")
    return magics, spec[end + 1 :]


def _validate_magics(magics: set[bytes], spec: bytes) -> frozenset[bytes]:
    if b"literal" in magics and b"glob" in magics:
        raise PathspecError(f"{spec!r}: 'literal' and 'glob' are incompatible")
    return frozenset(magics)


def _split_magic(spec: bytes) -> tuple[frozenset[bytes], bytes]:
    """Split a pathspec into its magic signatures and pattern."""
    if not spec.startswith(b":"):
        return frozenset(), spec
    magics: set[bytes] = set()
    i = 1
    while i < len(spec):
        c = spec[i : i + 1]
        if c == b"/":
            magics.add(b"top")
        elif c in (b"!", b"^"):
            magics.add(b"exclude")
        elif c == b":":
            magics.add(b"literal")
        elif c == b"(":
            long_magics, pattern = _parse_long_magic(spec[i:])
            magics |= long_magics
            return _validate_magics(magics, spec), pattern
        else:
            # The first character that is not a magic signature ends the
            # short form; the remainder is the pattern.
            break
        i += 1
    return _validate_magics(magics, spec), spec[i:]


def _normalize_cwd(cwd: bytes, root: bytes | None) -> bytes:
    """Normalize cwd to a repository-relative path, rejecting escapes."""
    if not cwd or cwd == b".":
        return b""
    if cwd.startswith(b"/"):
        if root is None:
            raise PathspecError(f"absolute cwd {cwd!r} requires the repository root")
        root_n = posixpath.normpath(root)
        cwd_n = posixpath.normpath(cwd)
        if cwd_n != root_n and not cwd_n.startswith(root_n + b"/"):
            raise PathspecError(f"{cwd!r} is outside repository at {root!r}")
        cwd = cwd_n[len(root_n) :].lstrip(b"/")
    rel = posixpath.normpath(cwd)
    if rel in (b"", b"."):
        return b""
    if rel == b".." or rel.startswith(b"../"):
        raise PathspecError(f"{cwd!r} is outside repository")
    return rel


def _join(prefix: bytes, pattern: bytes, raw: bytes) -> bytes:
    """Join a cwd prefix and a pattern, normalizing like Git does."""
    if pattern.startswith(b"/"):
        raise PathspecError(f"{raw!r} is outside repository")
    full = prefix + b"/" + pattern if prefix else pattern
    full = posixpath.normpath(full)
    if full in (b"", b"."):
        return b""
    if full == b".." or full.startswith((b"../", b"/")):
        raise PathspecError(f"{raw!r} is outside repository")
    return full


def _translate_nopn(pattern: bytes) -> bytes:
    """Translate a wildmatch() pattern without WM_PATHNAME.

    In this mode '*' and '?' also match '/' and '**' is just two ordinary
    stars.  Bracket expressions reuse the WM_PATHNAME translator; a negated
    class therefore still refuses '/', a negligible deviation for pathspec
    use.
    """
    res = b""
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i : i + 1]
        i += 1
        if c == b"*":
            # Collapse runs of '*' (including '**') into one quantifier.
            while i < n and pattern[i : i + 1] == b"*":
                i += 1
            res += b".*"
        elif c == b"?":
            res += b"."
        elif c == b"\\":
            if i < n:
                res += re.escape(pattern[i : i + 1])
                i += 1
            else:
                res += re.escape(c)
        elif c == b"[":
            i, bracket = translate_bracket_expression(pattern, i)
            res += bracket
        else:
            res += re.escape(c)
    return res


class Pathspec:
    """A single compiled pathspec item.

    The pattern is stored repository-relative, as bytes.  Instances are
    immutable and may be reused against any number of trees.
    """

    def __init__(self, pattern: bytes, magic: frozenset[bytes]) -> None:
        """Compile a single pattern with the given magic signatures."""
        self.pattern = pattern
        self.magic = magic
        self.exclude = b"exclude" in magic
        self._icase = b"icase" in magic
        self._literal: bytes | None = None
        self._re: re.Pattern[bytes] | None = None
        if b"literal" in magic or not _WILDCARD_RE.search(pattern):
            self._literal = pattern
        else:
            try:
                if b"glob" in magic:
                    body = translate_wildmatch(pattern)
                else:
                    body = _translate_nopn(pattern)
            except MalformedPattern as e:
                raise PathspecError(f"malformed pathspec pattern {pattern!r}") from e
            flags = re.IGNORECASE if self._icase else 0
            self._re = re.compile(b"(?ms)" + body + b"\\Z", flags)

    def match(self, path: bytes) -> bool:
        """Check whether path (repository-relative bytes) matches."""
        if self._literal is not None:
            pat = self._literal
            if not pat:
                return True
            candidate, wanted = path, pat
            if self._icase:
                candidate, wanted = candidate.lower(), wanted.lower()
            return candidate == wanted or candidate.startswith(wanted + b"/")
        assert self._re is not None
        return self._re.match(path) is not None

    def __repr__(self) -> str:
        """Return a debug representation of this pathspec."""
        return f"Pathspec({self.pattern!r}, {sorted(self.magic)!r})"


class PathspecMatcher:
    """An ordered set of compiled pathspecs with Git's include/exclude rules.

    A path matches when it matches at least one positive (non-exclude)
    pathspec -- or there are no positive pathspecs at all, in which case
    everything is a candidate -- and does not match any exclude pathspec.
    """

    def __init__(self, items: Sequence[Pathspec]) -> None:
        """Create a matcher from compiled :class:'Pathspec' items."""
        self._items = list(items)
        self._include = [item for item in self._items if not item.exclude]
        self._exclude = [item for item in self._items if item.exclude]

    @property
    def items(self) -> list[Pathspec]:
        """The compiled pathspec items, in order."""
        return list(self._items)

    def __len__(self) -> int:
        """Return the number of compiled pathspec items."""
        return len(self._items)

    def match(self, path: str | bytes) -> bool:
        """Check whether path is selected by this pathspec set."""
        path_b = _to_bytes(path, "path")
        if self._include and not any(item.match(path_b) for item in self._include):
            return False
        return not any(item.match(path_b) for item in self._exclude)

    def match_paths(self, paths: Iterable[str | bytes]) -> Iterator[bytes]:
        """Yield the paths from paths selected by this pathspec set.

        The positive set is formed first and exclusions are applied to it,
        mirroring Git.  Yields paths as bytes.
        """
        for path in paths:
            path_b = _to_bytes(path, "path")
            if self.match(path_b):
                yield path_b


def compile_pathspecs(
    pathspecs: Iterable[str | bytes],
    cwd: str | bytes | None = None,
    root: str | bytes | None = None,
) -> PathspecMatcher:
    """Compile pathspecs into a reusable :class:'PathspecMatcher'.

    Args:
      pathspecs: The pathspecs to compile, with optional magic signatures.
      cwd: Directory that relative (non-top) patterns are resolved against.
        May be repository-relative or absolute; an absolute cwd requires
        root so it can be validated.  Defaults to the repository root.
      root: Repository worktree root, used to validate an absolute cwd.

    Returns:
      A :class:'PathspecMatcher' that can be applied to any tree; it never
      touches the filesystem.

    Raises:
      PathspecError: For unknown, unsupported ('attr') or conflicting magic
        signatures, malformed patterns, or a cwd/pattern that escapes the
        repository.
    """
    cwd_b = _to_bytes(cwd, "cwd") if cwd is not None else b""
    root_b = _to_bytes(root, "root") if root is not None else None
    prefix = _normalize_cwd(cwd_b, root_b)
    items = []
    for spec in pathspecs:
        spec_b = _to_bytes(spec, "pathspec")
        if not spec_b:
            raise PathspecError(
                "empty string is not a valid pathspec;"
                " please use . instead if you meant to match all paths"
            )
        magic, pattern = _split_magic(spec_b)
        if b"top" in magic:
            full = _join(b"", pattern, spec_b)
        else:
            full = _join(prefix, pattern, spec_b)
        items.append(Pathspec(full, magic))
    return PathspecMatcher(items)


def iter_tree_matches(
    store: BaseObjectStore,
    tree_id: ObjectID | None,
    matcher: PathspecMatcher,
    *,
    include_trees: bool = False,
) -> Iterator[TreeEntry]:
    """Iterate the entries of a tree selected by a compiled pathspec set.

    Args:
      store: Object store to read trees from.
      tree_id: SHA1 of the tree to enumerate.
      matcher: A :class:'PathspecMatcher' from :func:'compile_pathspecs'.
      include_trees: If True, include tree (directory) entries as well.

    Yields:
      TreeEntry namedtuples whose path matches the pathspec set.
    """
    from .object_store import iter_tree_contents

    for entry in iter_tree_contents(store, tree_id, include_trees=include_trees):
        assert entry.path is not None
        if matcher.match(entry.path):
            yield entry

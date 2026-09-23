# archive.py -- Creating an archive from a tarball
# Copyright (C) 2015 Jonas Haag <jonas@lophus.org>
# Copyright (C) 2015 Jelmer Vernooij <jelmer@jelmer.uk>
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

"""Generates tarballs for Git trees.

The archive writer honours the "export-ignore" and "export-subst"
gitattributes, read from the tree being archived (never from the local
working tree):

* "export-ignore" excludes the matching path from the archive. When it
  is set on a directory, the whole subtree below it is omitted.
* "export-subst" enables "$Format:...$" placeholder expansion in the
  matching files. The supported placeholders are %H (full commit hash),
  %h (abbreviated commit hash), %aI (author time, strict ISO 8601), %s
  (commit subject) and %% (a literal percent sign). A "$Format:...$"
  string containing any other placeholder is copied into the archive
  verbatim, unexpanded.

Attribute resolution follows git's rules: ".gitattributes" files are
read from the archived tree itself, files deeper in the directory
hierarchy override shallower ones, and within one file later lines
override earlier ones. "-attr" unsets an attribute and "!attr" resets it
to the unspecified state, dropping any value inherited from a shallower
file.

Placeholder expansion needs a commit to take values from. When
"tar_stream" is called with only a tree (no "commit" argument),
"export-subst" files are archived with their "$Format:...$" strings
left untouched.
"""

__all__ = ["ChunkedBytesIO", "UnsafeArchivePathError", "tar_stream"]

import datetime
import posixpath
import re
import stat
import struct
import tarfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from io import BytesIO
from os import SEEK_END
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .object_store import BaseObjectStore
    from .objects import Commit, TreeEntry

from .attrs import (
    AttributeValue,
    Pattern,
    compile_gitattributes_patterns,
    parse_git_attributes,
)
from .index import cleanup_mode
from .objects import Blob, Tree


class UnsafeArchivePathError(ValueError):
    """Raised when a tree contains a path that is unsafe to include in an archive.

    Mirrors git's ``verify_path`` rejection (``error: invalid path``) for tree
    entries whose names would escape the archive root or alias a special
    directory on extraction. Examples include absolute paths, ``..`` or
    ``.git`` components, embedded backslashes, and NTFS alternate-data-stream
    separators (``:``).
    """

    def __init__(self, path: bytes) -> None:
        """Initialize the exception with the offending path."""
        self.path = path
        super().__init__(
            f"unsafe path in tree: {path!r}",
        )


# Characters that are never safe in a tree path destined for an archive:
# NUL terminates C strings, ``\`` is the Windows path separator (and would
# let a hostile name like ``..\evil.txt`` slip past a POSIX-only split), and
# ``:`` is the NTFS alternate-data-stream / drive-letter separator.
_UNSAFE_PATH_CHARS = (b"\x00", b"\\", b":")

_INVALID_PATH_COMPONENTS = (b".", b"..", b".git", b"")


def _is_unsafe_archive_path(path: bytes) -> bool:
    """Return True if ``path`` should not be emitted into an archive.

    Matches the spirit of git's ``verify_path``: absolute paths, ``.``/``..``/
    ``.git`` components (case-insensitive), embedded NUL bytes, backslashes,
    and colons are all rejected, regardless of host platform, because the
    resulting archive may be extracted anywhere.
    """
    if not path:
        return True
    for ch in _UNSAFE_PATH_CHARS:
        if ch in path:
            return True
    for component in path.split(b"/"):
        # Match git's protectNTFS-style normalization: trailing dots and
        # spaces are stripped before comparison so ``.git.`` / ``.GIT `` are
        # still recognised as ``.git`` on Windows.
        normalized = component.lower().rstrip(b". ")
        if normalized in _INVALID_PATH_COMPONENTS:
            return True
    return False


def _tree_gitattributes(
    store: "BaseObjectStore", tree: Tree
) -> list[tuple[Pattern, Mapping[bytes, AttributeValue]]]:
    """Read and compile the .gitattributes patterns stored in a tree.

    The attributes are taken from the tree being archived, never from the
    local working tree, so archiving a commit does not depend on whatever
    happens to be checked out.
    """
    for name, mode, sha in tree.iteritems():
        if name == b".gitattributes" and mode is not None and stat.S_ISREG(mode):
            blob = store[sha]
            if isinstance(blob, Blob):
                return compile_gitattributes_patterns(
                    parse_git_attributes(BytesIO(blob.data)),
                    source=b".gitattributes",
                )
    return []


# A stack of (directory, patterns) pairs, from the archive root downwards.
# Patterns in each entry match paths relative to their directory.
_AttrStack = Sequence[
    tuple[bytes, Sequence[tuple[Pattern, Mapping[bytes, AttributeValue]]]]
]


def _match_attrs(
    attr_stack: _AttrStack, path: bytes, is_dir: bool = False
) -> dict[bytes, AttributeValue]:
    """Resolve the attributes for a repo-relative path.

    Patterns from shallower ".gitattributes" files are applied first and
    deeper ones override them; within one file, later lines override
    earlier ones. A "-attr" line unsets the attribute and "!attr" resets
    it to the unspecified state, discarding whatever a shallower file
    said. Directories are additionally matched with a trailing slash so
    that "dir/" patterns apply to them, as in git.
    """
    attrs: dict[bytes, AttributeValue] = {}
    for base, patterns in attr_stack:
        if base:
            if path != base and not path.startswith(base + b"/"):
                continue
            rel = path[len(base) :].lstrip(b"/")
        else:
            rel = path
        candidates = [rel, rel + b"/"] if is_dir else [rel]
        for pattern, pat_attrs in patterns:
            if any(pattern.match(candidate) for candidate in candidates):
                for name, value in pat_attrs.items():
                    if value is None:
                        # Unspecified: forget any inherited state.
                        attrs.pop(name, None)
                    else:
                        attrs[name] = value
    return attrs


def _attr_is_set(attrs: Mapping[bytes, AttributeValue], name: bytes) -> bool:
    """Return True if an attribute is set (true or a string value)."""
    value = attrs.get(name)
    return value is not None and value is not False


_FORMAT_RE = re.compile(rb"\$Format:([^$]*)\$")


def _format_author_time(timestamp: int, timezone: int) -> bytes:
    """Format a commit's author time as strict ISO 8601 (git's %aI)."""
    tz = datetime.timezone(datetime.timedelta(seconds=timezone))
    dt = datetime.datetime.fromtimestamp(timestamp, tz)
    return dt.isoformat().encode("ascii")


def _commit_subject(message: bytes) -> bytes:
    """Return the commit subject (git's %s): first paragraph, lines joined."""
    paragraph = message.split(b"\n\n", 1)[0]
    return b" ".join(paragraph.split())


def _expand_format(data: bytes, commit: "Commit") -> bytes:
    """Expand the supported "$Format:...$" placeholders in blob contents.

    Supported placeholders: %H (full commit hash), %h (abbreviated commit
    hash), %aI (author time, strict ISO 8601), %s (commit subject) and %%
    (a literal percent sign). A "$Format:...$" string containing any other
    placeholder is left verbatim, as documented in the module docstring.
    """
    full_hash = commit.id
    placeholders = {
        b"%H": full_hash,
        b"%h": full_hash[:7],
        b"%aI": _format_author_time(commit.author_time, commit.author_timezone),
        b"%s": _commit_subject(commit.message),
        b"%%": b"%",
    }

    def replace(match: re.Match[bytes]) -> bytes:
        fmt = match.group(1)
        out = []
        i = 0
        while i < len(fmt):
            if fmt[i : i + 1] == b"%":
                for token in (b"%aI", fmt[i : i + 2]):
                    if token == fmt[i : i + len(token)] and token in placeholders:
                        out.append(placeholders[token])
                        i += len(token)
                        break
                else:
                    # Unknown placeholder: keep the whole string verbatim.
                    return match.group(0)
            else:
                out.append(fmt[i : i + 1])
                i += 1
        return b"".join(out)

    return _FORMAT_RE.sub(replace, data)


class ChunkedBytesIO:
    """Turn a list of bytestrings into a file-like object.

    This is similar to creating a `BytesIO` from a concatenation of the
    bytestring list, but saves memory by NOT creating one giant bytestring
    first::

        BytesIO(b''.join(list_of_bytestrings)) =~= ChunkedBytesIO(
            list_of_bytestrings)
    """

    def __init__(self, contents: list[bytes]) -> None:
        """Initialize ChunkedBytesIO.

        Args:
            contents: List of byte chunks
        """
        self.contents = contents
        self.pos = (0, 0)

    def read(self, maxbytes: int | None = None) -> bytes:
        """Read bytes from the chunked stream.

        Args:
            maxbytes: Maximum number of bytes to read (None for all)

        Returns:
            Bytes read
        """
        if maxbytes is None or maxbytes < 0:
            remaining = None
        else:
            remaining = maxbytes

        buf = []
        chunk, cursor = self.pos

        while chunk < len(self.contents):
            chunk_remainder = len(self.contents[chunk]) - cursor
            if remaining is not None and remaining < chunk_remainder:
                buf.append(self.contents[chunk][cursor : cursor + remaining])
                cursor += remaining
                self.pos = (chunk, cursor)
                break
            else:
                buf.append(self.contents[chunk][cursor:])
                if remaining is not None:
                    remaining -= chunk_remainder
                chunk += 1
                cursor = 0
                self.pos = (chunk, cursor)
        return b"".join(buf)


def tar_stream(
    store: "BaseObjectStore",
    tree: "Tree",
    mtime: int,
    prefix: bytes = b"",
    format: str = "",
    commit: "Commit | None" = None,
) -> Generator[bytes, None, None]:
    """Generate a tar stream for the contents of a Git tree.

    Returns a generator that lazily assembles a .tar.gz archive, yielding it in
    pieces (bytestrings). To obtain the complete .tar.gz binary file, simply
    concatenate these chunks.

    The "export-ignore" and "export-subst" gitattributes are honoured, read
    from ".gitattributes" files stored in "tree" itself (never from the
    working tree). Paths carrying "export-ignore" are omitted, directories
    included; files carrying "export-subst" get the supported
    "$Format:...$" placeholders (%H, %h, %aI, %s, %%) expanded, while
    unknown placeholders are left verbatim. Symbolic links are emitted as
    tar symlink entries: their targets are neither followed nor
    substituted.

    Args:
      store: Object store to retrieve objects from
      tree: Tree object for the tree root
      mtime: UNIX timestamp that is assigned as the modification time for
        all files, and the gzip header modification time if format='gz'
      prefix: Optional prefix to prepend to all paths in the archive. It
        is not part of the repository and does not affect attribute
        matching.
      format: Optional compression format for tarball
      commit: Optional commit to take "$Format:...$" substitution values
        from. When omitted (tree-only archiving), files marked
        "export-subst" are archived with their placeholders unexpanded.

    Returns:
      Bytestrings
    """
    buf = BytesIO()
    mode = "w:" + format if format else "w"

    # The tarfile.open overloads are complex; cast to Any to avoid issues
    with closing(cast(Any, tarfile.open)(name=None, mode=mode, fileobj=buf)) as tar:
        if format == "gz":
            # Manually correct the gzip header file modification time so that
            # archives created from the same Git tree are always identical.
            # The gzip header file modification time is not currently
            # accessible from the tarfile API, see:
            # https://bugs.python.org/issue31526
            buf.seek(0)
            assert buf.read(2) == b"\x1f\x8b", "Invalid gzip header"
            buf.seek(4)
            buf.write(struct.pack("<L", mtime))
            buf.seek(0, SEEK_END)

        for entry_abspath, entry, attrs in _walk_tree(store, tree):
            assert entry.sha is not None
            try:
                blob = store[entry.sha]
            except KeyError:
                # Entry probably refers to a submodule, which we don't yet
                # support.
                continue

            # The prefix is purely an archive-side name decoration; it must
            # not influence gitattributes matching, which ran above on the
            # repo-relative path.
            if prefix:
                archive_path = posixpath.join(prefix, entry_abspath)
            else:
                archive_path = entry_abspath
            info = tarfile.TarInfo()
            # tarfile only works with ascii.
            info.name = archive_path.decode("utf-8", "surrogateescape")
            info.mtime = mtime
            assert entry.mode is not None

            if stat.S_ISLNK(entry.mode):
                # Symbolic link: keep the tar symlink type, store the blob
                # contents as the link target, and never dereference or
                # substitute them.
                info.type = tarfile.SYMTYPE
                info.linkname = blob.as_raw_string().decode("utf-8", "surrogateescape")
                info.mode = 0o777
                info.size = 0
                tar.addfile(info)
                yield buf.getvalue()
                buf.truncate(0)
                buf.seek(0)
                continue

            if isinstance(blob, Blob):
                chunks: list[bytes] = list(blob.chunked)
            else:
                # Fallback for objects without chunked attribute
                chunks = [blob.as_raw_string()]
            if commit is not None and _attr_is_set(attrs, b"export-subst"):
                chunks = [_expand_format(b"".join(chunks), commit)]
            data = ChunkedBytesIO(chunks)

            info.size = sum(len(chunk) for chunk in chunks)
            # Canonicalize the tree-supplied mode the same way git's archive
            # writer does before emitting a permission field. A crafted tree
            # entry can carry setuid/setgid/sticky or other non-canonical bits
            # (e.g. 0o104755), which tarfile would otherwise copy verbatim into
            # the archive so an extracted file lands setuid. cleanup_mode maps a
            # regular file to 0o644/0o755 (matching checkout), and masking to the
            # permission bits drops the special bits.
            info.mode = cleanup_mode(entry.mode) & 0o777

            tar.addfile(info, data)
            yield buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
    yield buf.getvalue()


def _walk_tree(
    store: "BaseObjectStore",
    tree: "Tree",
    root: bytes = b"",
    attr_stack: _AttrStack = (),
) -> Generator[tuple[bytes, "TreeEntry", dict[bytes, AttributeValue]], None, None]:
    """Recursively walk a dulwich Tree, yielding (path, TreeEntry, attributes).

    Paths are repo-relative (the archive prefix is not involved), and the
    attributes are resolved from the ".gitattributes" files stored in the
    walked trees. Entries whose attributes set "export-ignore" are skipped;
    for a directory that drops the whole subtree.

    Raises:
      UnsafeArchivePathError: if a tree entry's name is unsafe to emit into
        an archive (matches the spirit of git's ``verify_path`` /
        ``error: invalid path``).
    """
    patterns = _tree_gitattributes(store, tree)
    if patterns:
        attr_stack = [*attr_stack, (root, patterns)]
    for entry in tree.iteritems():
        assert entry.path is not None
        if _is_unsafe_archive_path(entry.path):
            raise UnsafeArchivePathError(entry.path)
        entry_abspath = posixpath.join(root, entry.path)
        assert entry.mode is not None
        attrs = _match_attrs(attr_stack, entry_abspath, stat.S_ISDIR(entry.mode))
        if _attr_is_set(attrs, b"export-ignore"):
            # Ignored files are dropped; an ignored directory takes its
            # whole subtree with it.
            continue
        if stat.S_ISDIR(entry.mode):
            assert entry.sha is not None
            subtree = store[entry.sha]
            if isinstance(subtree, Tree):
                yield from _walk_tree(store, subtree, entry_abspath, attr_stack)
        else:
            yield (entry_abspath, entry, attrs)

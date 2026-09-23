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

Git attributes
==============

The tar stream honours the ``export-ignore`` and ``export-subst`` git
attributes, read from ``.gitattributes`` files *contained in the tree
being archived* -- never from the local working tree, the index, or
``$GIT_DIR/info/attributes``. An archive generated for a commit therefore
reflects the attributes as committed, even if the worktree has been modified
since.

Attribute lookup follows git's directory hierarchy rules: the root
``.gitattributes`` applies repository-wide, a ``.gitattributes`` in a
subdirectory applies to the paths beneath it (matched relative to that
directory) and overrides the root file, and within a single file later lines
override earlier ones. ``-attr`` explicitly unsets an attribute and
``!attr`` resets it to the unspecified state; see
:func:`dulwich.attrs.match_path_hierarchical`.

``export-ignore``
    Files -- and directories -- with this attribute set are omitted from the
    archive. An ignored directory is omitted wholesale: none of the entries
    beneath it are emitted.

``export-subst``
    For regular files with this attribute set, ``$Format:...$``
    placeholders are expanded using commit metadata. The supported
    placeholders are:

    * ``%H``  -- full commit hash
    * ``%h``  -- abbreviated commit hash
    * ``%aI`` -- author date, strict ISO 8601 format
    * ``%at`` -- author date, seconds since the epoch
    * ``%ad`` -- author date, git's default format
    * ``%s``  -- commit message subject

    Any other ``$Format:...$`` placeholder is left untouched (emitted
    verbatim) rather than being expanded into something misleading.

    Expansion requires commit context: when :func:`tar_stream` is called
    with only a tree (``commit=None``), no substitution is performed and
    ``$Format:...$`` placeholders are left as-is, because there is no
    commit to source the values from.

Other notes:

* The archive ``prefix`` only affects member names; attribute matching
  always happens against repo-relative paths, so a prefix never changes
  what is ignored or substituted.
* Symbolic links are emitted as tar symlink entries: their content (the
  link target) is never substituted and never dereferenced. Regular files
  and executable files keep their usual tar types and permission bits.
"""

__all__ = ["ChunkedBytesIO", "UnsafeArchivePathError", "tar_stream"]

import posixpath
import re
import stat
import struct
import tarfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from datetime import datetime, timedelta, timezone
from io import BytesIO
from os import SEEK_END
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .object_store import BaseObjectStore
    from .objects import TreeEntry

from .attrs import (
    AttributeValue,
    Pattern,
    compile_gitattributes_patterns,
    match_path_hierarchical,
    parse_git_attributes,
)
from .index import cleanup_mode
from .objects import Blob, Commit, Tree, format_timezone


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


# Placeholders of the form $Format:...$ in files carrying the export-subst
# attribute. Anything between the markers that is not a supported directive
# is preserved verbatim.
_FORMAT_PLACEHOLDER_RE = re.compile(rb"\$Format:([^$]*)\$")

# Length of the abbreviated commit hash emitted for %h, matching git's
# default abbreviation.
_ABBREV_HASH_LENGTH = 7

# Locale-independent names for git's default date format (%ad); strftime's
# %a/%b depend on the process locale, which must not leak into archives.
_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _format_iso8601(timestamp: int, tz_offset: int) -> bytes:
    """Format a timestamp as strict ISO 8601 (git's %aI)."""
    tz = timezone(timedelta(seconds=tz_offset))
    dt = datetime.fromtimestamp(timestamp, tz)
    offset = format_timezone(tz_offset).decode("ascii")
    return (dt.strftime("%Y-%m-%dT%H:%M:%S") + offset[:3] + ":" + offset[3:]).encode(
        "ascii"
    )


def _format_git_date(timestamp: int, tz_offset: int) -> bytes:
    """Format a timestamp in git's default date format (%ad)."""
    tz = timezone(timedelta(seconds=tz_offset))
    dt = datetime.fromtimestamp(timestamp, tz)
    return (
        f"{_WEEKDAY_NAMES[dt.weekday()]} {_MONTH_NAMES[dt.month - 1]} {dt.day} "
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} {dt.year} "
        f"{format_timezone(tz_offset).decode('ascii')}"
    ).encode("ascii")


def _commit_subject(commit: Commit) -> bytes:
    """Return the subject of a commit message (git's %s)."""
    message = commit.message or b""
    paragraph = message.split(b"\n\n", 1)[0]
    return b" ".join(paragraph.splitlines()).strip()


def _expand_format_placeholders(data: bytes, commit: Commit) -> bytes:
    """Expand supported $Format:...$ placeholders in blob content.

    Unknown directives are preserved verbatim, so content that merely looks
    like a placeholder is never corrupted.
    """

    def replace(match: "re.Match[bytes]") -> bytes:
        directive = match.group(1)
        if directive == b"%H":
            return commit.id
        if directive == b"%h":
            return commit.id[:_ABBREV_HASH_LENGTH]
        if directive == b"%aI":
            return _format_iso8601(commit.author_time, commit.author_timezone)
        if directive == b"%at":
            return str(commit.author_time).encode("ascii")
        if directive == b"%ad":
            return _format_git_date(commit.author_time, commit.author_timezone)
        if directive == b"%s":
            return _commit_subject(commit)
        # Unknown placeholder: keep the original text untouched.
        return match.group(0)

    return _FORMAT_PLACEHOLDER_RE.sub(replace, data)


def _tree_gitattributes(
    store: "BaseObjectStore", tree: "Tree", dirpath: bytes
) -> list[tuple[Pattern, Mapping[bytes, AttributeValue]]]:
    """Load the compiled patterns of a tree's own .gitattributes, if any."""
    if b".gitattributes" not in tree:
        return []
    _mode, sha = tree[b".gitattributes"]
    try:
        blob = store[sha]
    except KeyError:
        return []
    if not isinstance(blob, Blob):
        return []
    source = dirpath + b"/.gitattributes" if dirpath else b".gitattributes"
    return compile_gitattributes_patterns(
        parse_git_attributes(BytesIO(blob.data)), source
    )


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

    The ``export-ignore`` and ``export-subst`` attributes are honoured
    using ``.gitattributes`` files read from ``tree`` itself; see the
    module docstring for the exact semantics.

    Args:
      store: Object store to retrieve objects from
      tree: Tree object for the tree root
      mtime: UNIX timestamp that is assigned as the modification time for
        all files, and the gzip header modification time if format='gz'
      prefix: Optional prefix to prepend to all paths in the archive. The
        prefix is not part of the repository path and does not affect
        attribute matching.
      format: Optional compression format for tarball
      commit: Optional commit providing context for export-subst
        placeholder expansion. When omitted, $Format:...$ placeholders are
        left unexpanded.

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

            info = tarfile.TarInfo()
            # The prefix only affects the member name; attribute matching
            # already happened against the repo-relative path.
            name = posixpath.join(prefix, entry_abspath) if prefix else entry_abspath
            # tarfile only works with ascii.
            info.name = name.decode("utf-8", "surrogateescape")
            info.mtime = mtime
            assert entry.mode is not None

            if stat.S_ISLNK(entry.mode):
                # Symbolic link: the blob content is the link target. Emit a
                # symlink entry; never substitute or dereference the target.
                info.type = tarfile.SYMTYPE
                info.linkname = blob.as_raw_string().decode("utf-8", "surrogateescape")
                info.mode = 0o777
                info.size = 0
                tar.addfile(info)
            else:
                if isinstance(blob, Blob):
                    chunks = blob.chunked
                else:
                    # Fallback for objects without chunked attribute
                    chunks = [blob.as_raw_string()]
                if commit is not None and attrs.get(b"export-subst") is True:
                    chunks = [_expand_format_placeholders(b"".join(chunks), commit)]
                info.size = sum(len(chunk) for chunk in chunks)
                # Canonicalize the tree-supplied mode the same way git's archive
                # writer does before emitting a permission field. A crafted tree
                # entry can carry setuid/setgid/sticky or other non-canonical bits
                # (e.g. 0o104755), which tarfile would otherwise copy verbatim into
                # the archive so an extracted file lands setuid. cleanup_mode maps a
                # regular file to 0o644/0o755 (matching checkout), and masking to the
                # permission bits drops the special bits.
                info.mode = cleanup_mode(entry.mode) & 0o777
                tar.addfile(info, ChunkedBytesIO(list(chunks)))

            yield buf.getvalue()
            buf.truncate(0)
            buf.seek(0)
    yield buf.getvalue()


def _walk_tree(
    store: "BaseObjectStore",
    tree: "Tree",
    root: bytes = b"",
    sources: Sequence[
        tuple[bytes, Sequence[tuple[Pattern, Mapping[bytes, AttributeValue]]]]
    ] = (),
) -> Generator[tuple[bytes, "TreeEntry", dict[bytes, AttributeValue]], None, None]:
    """Recursively walk a dulwich Tree, yielding (path, TreeEntry, attributes).

    Paths are repo-relative; any archive prefix is applied by the caller.
    Files with the export-ignore attribute set are skipped, and directories
    with it set are pruned wholesale along with everything beneath them.

    Raises:
      UnsafeArchivePathError: if a tree entry's name is unsafe to emit into
        an archive (matches the spirit of git's ``verify_path`` /
        ``error: invalid path``).
    """
    patterns = _tree_gitattributes(store, tree, root)
    if patterns:
        sources = [*sources, (root, patterns)]
    for entry in tree.iteritems():
        assert entry.path is not None
        if _is_unsafe_archive_path(entry.path):
            raise UnsafeArchivePathError(entry.path)
        entry_abspath = posixpath.join(root, entry.path)
        assert entry.mode is not None
        if stat.S_ISDIR(entry.mode):
            assert entry.sha is not None
            subtree = store[entry.sha]
            if isinstance(subtree, Tree):
                attrs = match_path_hierarchical(sources, entry_abspath)
                if attrs.get(b"export-ignore") is True:
                    # Ignored directory: prune the whole subtree.
                    continue
                yield from _walk_tree(store, subtree, entry_abspath, sources)
        else:
            attrs = match_path_hierarchical(sources, entry_abspath)
            if attrs.get(b"export-ignore") is True:
                continue
            yield (entry_abspath, entry, attrs)

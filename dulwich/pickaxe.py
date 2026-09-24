# pickaxe.py -- Pickaxe-style content filtering for commit walks.
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

"""Pickaxe-style content filtering for commit history.

This module implements the two classic "pickaxe" filters known from
git log:

* OccurrenceFilter (like git log -S<string>) matches commits in which
  the number of occurrences of a literal byte string in a file changes.
  Merely moving text to a different line does *not* match, since the
  occurrence count stays the same.
* DiffRegexFilter (like git log -G<regex>) matches commits in which an
  added or removed line matches a regular expression. Moving text to
  another line *does* match, since the line is removed and re-added.

Binary file content is handled according to an explicit BinaryPolicy:
it is either skipped entirely or matched as raw bytes. Content is never
decoded with error-replacing codecs.
"""

__all__ = [
    "BinaryPolicy",
    "DiffRegexFilter",
    "OccurrenceFilter",
    "PickaxeFilter",
    "PickaxeMatch",
]

import difflib
import re
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from .diff_tree import TreeChange
from .objects import S_ISGITLINK, Blob, ObjectID, TreeEntry

if TYPE_CHECKING:
    from .object_store import BaseObjectStore

# Number of leading bytes to inspect when guessing whether content is
# binary, mirroring git's FIRST_FEW_BYTES.
_BINARY_CHECK_BYTES = 8000


class BinaryPolicy(Enum):
    """How pickaxe filters treat binary file content."""

    SKIP = "skip"
    """Never match binary files."""

    MATCH = "match"
    """Match against the raw bytes of binary files."""


class PickaxeMatch(NamedTuple):
    """Record of a pickaxe hit for a single commit.

    Attributes:
      parent: SHA of the parent commit the comparison was made against,
        or None for root commits (which are compared against the empty
        tree).
      changes: The tree changes (relative to parent) that matched the
        filter.
    """

    parent: ObjectID | None
    changes: list[TreeChange]


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:_BINARY_CHECK_BYTES]


def _blob_data(store: "BaseObjectStore", entry: TreeEntry | None) -> bytes | None:
    """Return the content of the blob for a tree entry.

    Returns None if there is no entry, the entry is a gitlink, or the
    object it points to is not a blob.
    """
    if entry is None:
        return None
    if S_ISGITLINK(entry.mode):
        return None
    obj = store[entry.sha]
    if not isinstance(obj, Blob):
        return None
    return obj.data


def _changed_lines(old: bytes, new: bytes) -> Iterator[bytes]:
    """Yield the lines removed from old or added in new."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            yield from old_lines[i1:i2]
        if tag in ("replace", "insert"):
            yield from new_lines[j1:j2]


class PickaxeFilter:
    """Base class for pickaxe-style commit filters.

    Subclasses decide whether a single tree change matches. All
    comparisons are done on raw bytes; file content is never decoded
    with error-replacing codecs.
    """

    def __init__(self, binary_policy: BinaryPolicy = BinaryPolicy.SKIP) -> None:
        """Constructor.

        Args:
          binary_policy: How to treat binary file content; either
            BinaryPolicy.SKIP (never match binary files) or
            BinaryPolicy.MATCH (match against raw bytes).
        """
        self.binary_policy = binary_policy

    def _contents(
        self, store: "BaseObjectStore", change: TreeChange
    ) -> tuple[bytes, bytes] | None:
        """Return the (old, new) content for a change, or None to skip it.

        A missing side (added or deleted files) is treated as empty
        content. Binary content causes the change to be skipped unless
        the binary policy is BinaryPolicy.MATCH.
        """
        old = _blob_data(store, change.old)
        new = _blob_data(store, change.new)
        if self.binary_policy is BinaryPolicy.SKIP:
            if (old is not None and _is_binary(old)) or (
                new is not None and _is_binary(new)
            ):
                return None
        return (old or b"", new or b"")

    def matches(self, store: "BaseObjectStore", change: TreeChange) -> bool:
        """Check whether a single tree change matches this filter.

        Args:
          store: Object store to retrieve blob contents from.
          change: The tree change to consider.
        Returns: True if the change matches.
        """
        raise NotImplementedError


class OccurrenceFilter(PickaxeFilter):
    """Filter matching when the occurrence count of a literal changes.

    Equivalent to git log -S<string>: a commit matches if the number of
    times needle occurs in a file differs from the parent version of
    that file.
    """

    def __init__(
        self,
        needle: bytes | str,
        binary_policy: BinaryPolicy = BinaryPolicy.SKIP,
    ) -> None:
        """Constructor.

        Args:
          needle: Literal byte string to count occurrences of. Text
            strings are encoded as UTF-8.
          binary_policy: How to treat binary file content.
        """
        super().__init__(binary_policy)
        if isinstance(needle, str):
            needle = needle.encode("utf-8")
        if not needle:
            raise ValueError("needle must not be empty")
        self.needle = needle

    def matches(self, store: "BaseObjectStore", change: TreeChange) -> bool:
        """Check whether the occurrence count of the needle changed."""
        contents = self._contents(store, change)
        if contents is None:
            return False
        old, new = contents
        return old.count(self.needle) != new.count(self.needle)


class DiffRegexFilter(PickaxeFilter):
    """Filter matching when an added or removed line matches a regex.

    Equivalent to git log -G<regex>: a commit matches if any line added
    or removed by it matches the pattern. Unlike OccurrenceFilter,
    moving text to another line matches, since the line is removed in
    one place and re-added in another.
    """

    def __init__(
        self,
        pattern: "bytes | str | re.Pattern[bytes]",
        binary_policy: BinaryPolicy = BinaryPolicy.SKIP,
    ) -> None:
        """Constructor.

        Args:
          pattern: Regular expression matched against added and removed
            lines. May be a compiled bytes pattern, a bytes pattern, or
            a text pattern (encoded as UTF-8).
          binary_policy: How to treat binary file content.
        """
        super().__init__(binary_policy)
        if isinstance(pattern, str):
            pattern = pattern.encode("utf-8")
        if isinstance(pattern, bytes):
            pattern = re.compile(pattern)
        self.pattern = pattern

    def matches(self, store: "BaseObjectStore", change: TreeChange) -> bool:
        """Check whether any added or removed line matches the pattern."""
        contents = self._contents(store, change)
        if contents is None:
            return False
        old, new = contents
        if old == new:
            return False
        search = self.pattern.search
        for line in _changed_lines(old, new):
            if search(line):
                return True
        return False

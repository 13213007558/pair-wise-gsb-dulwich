# pickaxe.py -- Pickaxe conditions for filtering history by content changes.
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

"""Pickaxe conditions for selecting commits by the content they change.

This provides the building blocks for history queries like ``git log -S``
(when did the number of occurrences of a literal string change?) and
``git log -G`` (when did an added or removed line match a pattern?).
"""

__all__ = [
    "BINARY_BYTES",
    "BINARY_POLICIES",
    "BINARY_SKIP",
    "Pickaxe",
]

import difflib
import re
import stat
from typing import TYPE_CHECKING

from .diff_tree import TreeChange
from .objects import S_ISGITLINK, Blob, TreeEntry
from .patch import is_binary

if TYPE_CHECKING:
    from .object_store import BaseObjectStore

# Binary files are never considered for pickaxe matches.
BINARY_SKIP = "skip"
# Binary files are matched as raw bytes, without any decoding.
BINARY_BYTES = "bytes"

BINARY_POLICIES = (BINARY_SKIP, BINARY_BYTES)


class Pickaxe:
    """A condition matching tree changes by the content they add or remove.

    Two mutually exclusive modes are supported:

    * literal: matches when the number of occurrences of a literal byte
      string in a file changes (like ``git log -S``).
    * regex: matches when a regular expression matches a line added or
      removed by the change (like ``git log -G``).

    These are deliberately different filters: moving a line of text from
    one place in a file to another does not change the number of
    occurrences, so the literal mode does not match, while the regex mode
    does, since the line shows up as removed and added in the diff.

    Content is always matched as raw bytes; it is never decoded with a
    lossy encoding. Binary content (as determined by
    ``dulwich.patch.is_binary``) is handled according to the binary
    policy: either skipped entirely (BINARY_SKIP, the default) or matched
    as raw bytes (BINARY_BYTES).
    """

    def __init__(
        self,
        needle: "bytes | None" = None,
        regex: "bytes | re.Pattern[bytes] | None" = None,
        binary: str = BINARY_SKIP,
    ) -> None:
        """Constructor.

        Args:
          needle: Literal byte string to count occurrences of. Mutually
            exclusive with regex.
          regex: Compiled bytes pattern (or bytes pattern to compile)
            matched against added/removed lines. Mutually exclusive with
            needle.
          binary: Policy for binary content, one of BINARY_POLICIES.

        Raises:
          ValueError: If both or neither of needle and regex are given,
            the needle is empty, the regex is not a bytes pattern, or the
            binary policy is unknown.
        """
        if (needle is None) == (regex is None):
            raise ValueError("Exactly one of needle and regex must be provided")
        if needle is not None and not needle:
            raise ValueError("Empty needle")
        if binary not in BINARY_POLICIES:
            raise ValueError(f"Unknown binary policy: {binary!r}")
        if isinstance(regex, bytes):
            regex = re.compile(regex)
        if regex is not None and isinstance(regex.pattern, str):
            raise ValueError("Regex pattern must be bytes, not str")
        self._needle = needle
        self._regex = regex
        self._binary = binary

    @classmethod
    def literal(cls, needle: bytes, *, binary: str = BINARY_SKIP) -> "Pickaxe":
        """Create a condition matching changes in occurrence counts.

        Args:
          needle: Literal byte string to count occurrences of.
          binary: Policy for binary content, one of BINARY_POLICIES.
        Returns: A Pickaxe that matches when the number of occurrences of
          needle in a file changes.
        """
        return cls(needle=needle, binary=binary)

    @classmethod
    def regex(
        cls,
        pattern: "bytes | re.Pattern[bytes]",
        *,
        binary: str = BINARY_SKIP,
    ) -> "Pickaxe":
        """Create a condition matching added/removed lines by regex.

        Args:
          pattern: Compiled bytes pattern, or bytes to compile.
          binary: Policy for binary content, one of BINARY_POLICIES.
        Returns: A Pickaxe that matches when a line added or removed by
          the change matches the pattern.
        """
        return cls(regex=pattern, binary=binary)

    @property
    def binary(self) -> str:
        """The binary content policy in effect."""
        return self._binary

    def __repr__(self) -> str:
        """Return string representation of Pickaxe."""
        if self._needle is not None:
            return f"Pickaxe.literal({self._needle!r}, binary={self._binary!r})"
        return f"Pickaxe.regex({self._regex.pattern!r}, binary={self._binary!r})"

    @staticmethod
    def _blob_data(
        store: "BaseObjectStore", entry: "TreeEntry | None"
    ) -> "bytes | None":
        if entry is None or entry.mode is None:
            return None
        if stat.S_ISDIR(entry.mode) or S_ISGITLINK(entry.mode):
            return None
        obj = store[entry.sha]
        if not isinstance(obj, Blob):
            return None
        return obj.data

    def matches_change(
        self, store: "BaseObjectStore", change: "TreeChange | None"
    ) -> bool:
        """Check whether a single tree change satisfies this condition.

        Args:
          store: ObjectStore for looking up blob contents.
          change: The TreeChange to check, or None.
        Returns: True if the change satisfies this condition.
        """
        if change is None:
            return False
        old = self._blob_data(store, change.old)
        new = self._blob_data(store, change.new)
        if old is None and new is None:
            return False
        old = old or b""
        new = new or b""
        if self._binary == BINARY_SKIP and (is_binary(old) or is_binary(new)):
            return False
        if self._needle is not None:
            return old.count(self._needle) != new.count(self._needle)
        assert self._regex is not None
        return self._matches_changed_lines(old, new)

    def _matches_changed_lines(self, old: bytes, new: bytes) -> bool:
        assert self._regex is not None
        regex = self._regex
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "delete"):
                for line in old_lines[i1:i2]:
                    if regex.search(line):
                        return True
            if tag in ("replace", "insert"):
                for line in new_lines[j1:j2]:
                    if regex.search(line):
                        return True
        return False

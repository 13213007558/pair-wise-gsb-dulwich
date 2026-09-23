# pickaxe.py -- Pickaxe-style search of commit history.
# Copyright (C) 2026 Dulwich contributors
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
# <http://www.gnu.org/licenses/> for a copy of the Apache
# License, Version 2.0.
#

"""Pickaxe-style search of commit history.

This module implements the two flavours of history search known from
'git log -S<string>' and 'git log -G<regex>':

* Occurrence counting (string): the number of non-overlapping
  occurrences of a literal string in the full blob contents on either
  side of a change is compared.  Simply moving text around (within a
  file, or to another file by renaming) does not change the count and
  therefore does not match.
* Diff line matching (regex): the regular expression is matched
  against the lines actually added or removed according to the line
  diff of each changed file.  Unchanged context is never searched.

Merge commits are compared either against their first parent only
(MERGE_FIRST_PARENT) or against each parent in turn (MERGE_ALL); the
resulting matches record which parent commit and which path matched.
"""

__all__ = [
    "MERGE_ALL",
    "MERGE_FIRST_PARENT",
    "PickaxeFileMatch",
    "PickaxeMatch",
    "pickaxe",
]

import re
from collections.abc import Callable, Iterator, Sequence
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, NamedTuple

from .diff_tree import (
    CHANGE_ADD,
    CHANGE_DELETE,
    CHANGE_RENAME,
    TreeChange,
    tree_changes,
)
from .objects import S_ISGITLINK, Blob, Commit, ObjectID, TreeEntry
from .walk import ORDER_DATE, Walker

if TYPE_CHECKING:
    from .object_store import BaseObjectStore

MERGE_FIRST_PARENT = "first-parent"
MERGE_ALL = "all"

_MERGE_MODES = (MERGE_FIRST_PARENT, MERGE_ALL)


class PickaxeFileMatch(NamedTuple):
    """A single file that matched a pickaxe query.

    Attributes:
      parent: SHA of the parent commit the comparison was made against,
        or None for root commits (which are compared against the empty
        tree).
      path: Path of the file that matched.  For renames this is the new
        path; for deletions it is the old path.
    """

    parent: ObjectID | None
    path: bytes


class PickaxeMatch(NamedTuple):
    """A commit that matched a pickaxe query.

    A commit is reported at most once per walk, no matter how many
    files matched; the individual file matches are listed in
    'matches'.
    """

    commit: Commit
    matches: list[PickaxeFileMatch]


def _split_lines(data: bytes) -> list[bytes]:
    """Split blob data into lines, without line terminators.

    A final line without a trailing newline is still returned as a
    line, matching how git treats files that lack a trailing newline.
    """
    if not data:
        return []
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


def _count_matcher(needle: bytes) -> Callable[[bytes | None, bytes | None], bool]:
    """Build a matcher comparing non-overlapping occurrence counts."""

    def matches(old: bytes | None, new: bytes | None) -> bool:
        old_count = old.count(needle) if old else 0
        new_count = new.count(needle) if new else 0
        return old_count != new_count

    return matches


def _diff_line_matcher(
    pattern: "re.Pattern[bytes]",
) -> Callable[[bytes | None, bytes | None], bool]:
    """Build a matcher checking added/removed diff lines against a regex."""

    def matches(old: bytes | None, new: bytes | None) -> bool:
        old_lines = _split_lines(old or b"")
        new_lines = _split_lines(new or b"")
        if old_lines == new_lines:
            return False
        sm = SequenceMatcher(a=old_lines, b=new_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "delete"):
                for line in old_lines[i1:i2]:
                    if pattern.search(line):
                        return True
            if tag in ("replace", "insert"):
                for line in new_lines[j1:j2]:
                    if pattern.search(line):
                        return True
        return False

    return matches


def _path_matches(paths: set[bytes] | None, changed_path: bytes | None) -> bool:
    if changed_path is None:
        return False
    if paths is None:
        return True
    for path in paths:
        if changed_path == path:
            return True
        if changed_path.startswith(path + b"/"):
            return True
    return False


def _pair_exact_renames(changes: list[TreeChange]) -> list[TreeChange]:
    """Pair add/delete changes with identical SHAs into renames.

    This only pairs changes with identical content (which can never
    match a pickaxe query by themselves), so unlike full rename
    detection it never needs to load blob contents.
    """
    adds: dict[ObjectID, list[int]] = {}
    deletes: dict[ObjectID, list[int]] = {}
    for i, change in enumerate(changes):
        if change.type == CHANGE_ADD:
            assert change.new is not None and change.new.sha is not None
            adds.setdefault(change.new.sha, []).append(i)
        elif change.type == CHANGE_DELETE:
            assert change.old is not None and change.old.sha is not None
            deletes.setdefault(change.old.sha, []).append(i)
    renames: dict[int, TreeChange] = {}
    paired_adds: set[int] = set()
    for sha, delete_indexes in deletes.items():
        add_indexes = adds.get(sha, [])
        for delete_index, add_index in zip(delete_indexes, add_indexes):
            delete = changes[delete_index]
            add = changes[add_index]
            renames[delete_index] = TreeChange(CHANGE_RENAME, delete.old, add.new)
            paired_adds.add(add_index)
    result = []
    for i, change in enumerate(changes):
        if i in paired_adds:
            continue
        result.append(renames.get(i, change))
    return result


def _entry_data(store: "BaseObjectStore", entry: "TreeEntry | None") -> bytes | None:
    """Return the blob data for a tree entry, or None if not a blob."""
    if entry is None:
        return None
    if entry.mode is not None and S_ISGITLINK(entry.mode):
        return None
    if entry.sha is None:
        return None
    obj = store[entry.sha]
    if not isinstance(obj, Blob):
        return None
    return obj.data


def _compile_regex(regex: bytes | str) -> "re.Pattern[bytes]":
    if isinstance(regex, str):
        regex = regex.encode("utf-8")
    return re.compile(regex)


def pickaxe(
    store: "BaseObjectStore",
    include: ObjectID | Sequence[ObjectID],
    exclude: Sequence[ObjectID] | None = None,
    *,
    string: bytes | str | None = None,
    regex: bytes | str | None = None,
    paths: Sequence[bytes] | None = None,
    merge_mode: str = MERGE_FIRST_PARENT,
    order: str = ORDER_DATE,
    reverse: bool = False,
    max_entries: int | None = None,
    since: int | None = None,
    until: int | None = None,
) -> Iterator[PickaxeMatch]:
    """Search commit history for changes to a string or pattern.

    Exactly one of 'string' and 'regex' must be given:

    * 'string' (like 'git log -S'): match commits where the number of
      non-overlapping occurrences of the literal string in a file
      changes.  Moving text without changing the number of occurrences
      does not match.
    * 'regex' (like 'git log -G'): match commits where a line matching
      the regular expression is added or removed, according to the
      actual line diff.  The rest of the file is not searched.

    Args:
      store: ObjectStore instance for looking up objects.
      include: SHA(s) of commits to include along with their ancestors
        (the 'stop' revision of the search).
      exclude: SHAs of commits to exclude along with their ancestors
        (the 'start' revision of the search), overriding includes.
      string: Literal string to count occurrences of.
      regex: Regular expression to match against added/removed lines.
        Invalid patterns raise 're.error' before any history is
        traversed.
      paths: Optional file or directory paths to restrict the search
        to.  Blobs outside these paths are never loaded.
      merge_mode: How to handle merge commits: MERGE_FIRST_PARENT
        compares against the first parent only; MERGE_ALL compares
        against each parent and reports every matching parent.
      order: ORDER_* constant specifying the order of results.
      reverse: If True, reverse the order of output.
      max_entries: Maximum number of matching commits to yield.
      since: Timestamp to list commits after.
      until: Timestamp to list commits before.

    Returns: An iterator of PickaxeMatch objects, in the order the
      underlying walker yields commits.  Each matching commit appears
      at most once, with all of its file matches attached.

    Raises:
      ValueError: If the arguments are inconsistent (neither or both
        of string/regex given, empty string, unknown merge mode).
      re.error: If the regex is invalid; raised before traversal.
    """
    if (string is None) == (regex is None):
        raise ValueError("Exactly one of string and regex must be given")
    if string is not None:
        if isinstance(string, str):
            string = string.encode("utf-8")
        if not string:
            raise ValueError("string must not be empty")
        matcher = _count_matcher(string)
    else:
        assert regex is not None
        # Compile eagerly so invalid patterns raise before any
        # history is traversed.
        matcher = _diff_line_matcher(_compile_regex(regex))
    if merge_mode not in _MERGE_MODES:
        raise ValueError(f"Unknown merge mode {merge_mode!r}")

    walker = Walker(
        store,
        include,
        exclude=exclude,
        order=order,
        reverse=reverse,
        since=since,
        until=until,
    )
    path_set = set(paths) if paths else None
    return _pickaxe_entries(store, walker, matcher, path_set, merge_mode, max_entries)


def _pickaxe_entries(
    store: "BaseObjectStore",
    walker: Walker,
    matcher: Callable[[bytes | None, bytes | None], bool],
    paths: set[bytes] | None,
    merge_mode: str,
    max_entries: int | None,
) -> Iterator[PickaxeMatch]:
    num_entries = 0
    for entry in walker:
        commit = entry.commit
        parents = commit.parents
        if not parents:
            compare_parents: list[ObjectID | None] = [None]
        elif len(parents) == 1 or merge_mode == MERGE_FIRST_PARENT:
            compare_parents = [parents[0]]
        else:
            compare_parents = list(parents)
        matches = []
        for parent_id in compare_parents:
            if parent_id is None:
                old_tree_id = None
            else:
                parent_commit = store[parent_id]
                assert isinstance(parent_commit, Commit)
                old_tree_id = parent_commit.tree
            changes = [
                change
                for change in tree_changes(store, old_tree_id, commit.tree)
                if _path_matches(
                    paths, change.new.path if change.new is not None else None
                )
                or _path_matches(
                    paths, change.old.path if change.old is not None else None
                )
            ]
            for change in _pair_exact_renames(changes):
                old_data = _entry_data(store, change.old)
                new_data = _entry_data(store, change.new)
                if matcher(old_data, new_data):
                    if change.new is not None:
                        assert change.new.path is not None
                        path = change.new.path
                    else:
                        assert change.old is not None and change.old.path is not None
                        path = change.old.path
                    matches.append(PickaxeFileMatch(parent_id, path))
        if matches:
            yield PickaxeMatch(commit, matches)
            num_entries += 1
            if max_entries is not None and num_entries >= max_entries:
                return

# pickaxe.py -- Find commits that introduce or remove occurrences of text.
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
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/> for a copy of the Apache License,
# version 2.0.
#

"""Pickaxe queries over commit history.

This module implements the two kinds of "pickaxe" search known from
"git log -S" and "git log -G":

* Passing 'string' counts the non-overlapping occurrences of a literal
  string in the full contents of each changed blob and reports commits
  where that count changes.  Merely moving text around inside a file does
  not change the count and therefore does not match.
* Passing 'pattern' matches a regular expression against the lines a
  commit actually adds or removes (as determined by a line-based diff),
  rather than searching the file contents as a whole.

Both modes report added, deleted, modified and renamed files, and handle
files without a trailing newline.
"""

__all__ = [
    "MERGE_ALL",
    "MERGE_FIRST",
    "PickaxeMatch",
    "PickaxeResult",
    "pickaxe",
]

import re
from collections.abc import Callable, Iterator, Sequence
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, NamedTuple

from .diff_tree import RenameDetector, TreeChange, tree_changes
from .objects import S_IFGITLINK, Blob, Commit, ObjectID
from .walk import ORDER_DATE, Walker

if TYPE_CHECKING:
    from .object_store import BaseObjectStore

# Compare merge commits against their first parent only.
MERGE_FIRST = "first"

# Compare merge commits against each of their parents in turn.
MERGE_ALL = "all"

ALL_MERGE_MODES = (MERGE_FIRST, MERGE_ALL)


class PickaxeMatch(NamedTuple):
    """A single pickaxe hit within a commit.

    Attributes:
      parent: SHA of the parent commit the comparison was made against,
        or None for a root commit (compared against the empty tree).
      path: Path of the file that matched, as it appears in the commit
        (the old path for deletions).
    """

    parent: ObjectID | None
    path: bytes


class PickaxeResult(NamedTuple):
    """Pickaxe result for a single commit.

    A commit is reported at most once, no matter how many files matched;
    every hit is listed in 'matches'.
    """

    commit: Commit
    matches: list[PickaxeMatch]


def _split_lines(content: bytes) -> list[bytes]:
    """Split content into lines, keeping terminators, git-style.

    Lines are split on a newline byte only and keep their trailing
    newline, so a line whose newline termination changed shows up as
    removed and re-added, like in git diffs.  A final line without a
    trailing newline is still returned as a line.
    """
    if not content:
        return []
    parts = content.split(b"\n")
    lines = [part + b"\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _string_matcher(needle: bytes) -> Callable[[bytes, bytes], bool]:
    def matches(old: bytes, new: bytes) -> bool:
        # bytes.count() counts non-overlapping occurrences, so moving text
        # around inside a file leaves the count unchanged.
        return old.count(needle) != new.count(needle)

    return matches


def _regex_matcher(regex: "re.Pattern[bytes]") -> Callable[[bytes, bytes], bool]:
    def matches(old: bytes, new: bytes) -> bool:
        old_lines = _split_lines(old)
        new_lines = _split_lines(new)
        opcodes = SequenceMatcher(a=old_lines, b=new_lines).get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag in ("replace", "delete"):
                for line in old_lines[i1:i2]:
                    if regex.search(line):
                        return True
            if tag in ("replace", "insert"):
                for line in new_lines[j1:j2]:
                    if regex.search(line):
                        return True
        return False

    return matches


def _path_matches(paths: set[bytes] | None, path: bytes | None) -> bool:
    if path is None:
        return False
    if paths is None:
        return True
    for followed_path in paths:
        if path == followed_path:
            return True
        if path.startswith(followed_path + b"/"):
            return True
    return False


def _is_gitlink(change: TreeChange) -> bool:
    for entry in (change.old, change.new):
        if entry is not None and entry.mode is not None:
            if (entry.mode & 0o170000) == S_IFGITLINK:
                return True
    return False


def _blob_data(store: "BaseObjectStore", change: TreeChange, old: bool) -> bytes:
    entry = change.old if old else change.new
    if entry is None or entry.sha is None:
        return b""
    obj = store[entry.sha]
    assert isinstance(obj, Blob)
    return obj.data


def pickaxe(
    store: "BaseObjectStore",
    include: ObjectID | Sequence[ObjectID],
    exclude: Sequence[ObjectID] | None = None,
    *,
    string: bytes | str | None = None,
    pattern: bytes | str | None = None,
    paths: Sequence[bytes] | None = None,
    merge: str = MERGE_FIRST,
    order: str = ORDER_DATE,
    reverse: bool = False,
    max_entries: int | None = None,
    since: int | None = None,
    until: int | None = None,
    rename_detector: RenameDetector | None = None,
) -> Iterator[PickaxeResult]:
    """Find commits that introduce or remove occurrences of text.

    Exactly one of 'string' and 'pattern' must be given:

    * 'string' (like "git log -S"): report commits where the number of
      non-overlapping occurrences of the literal string in a changed file
      differs from its parent.  Moving text within a file does not match.
    * 'pattern' (like "git log -G"): report commits whose diff adds or
      removes a line matching the regular expression.  The regex is only
      matched against added/removed lines, not against the whole file.

    Args:
      store: ObjectStore instance for looking up objects.
      include: SHA(s) of commits to start walking from (inclusive).
      exclude: SHA(s) of commits to stop walking at, along with their
        ancestors.
      string: Literal string to count occurrences of.
      pattern: Regular expression to match against added/removed lines.
        Compiled before any history is traversed, so invalid patterns
        raise re.error immediately.
      paths: Optional file or directory paths to restrict the search to.
        Blobs of files outside these paths are never loaded.
      merge: How to handle merge commits: MERGE_FIRST compares against the
        first parent only; MERGE_ALL compares against every parent.
      order: ORDER_* constant specifying the order of results.
      reverse: If True, reverse the order of output.
      max_entries: Maximum number of matching commits to yield.
      since: Timestamp to list commits after.
      until: Timestamp to list commits before.
      rename_detector: RenameDetector to use for pairing renames.  By
        default a detector is used that only pairs exact renames (by
        blob SHA) and never reads blob contents, so blobs of files
        outside 'paths' are not loaded as a side effect.  Renames with
        content changes then show up as a delete plus an add.  Pass a
        custom detector to enable content-based rename detection.
    Returns: Iterator over PickaxeResult objects, in walker order.  Each
      matching commit is yielded at most once; its matches list the parent
      commit SHAs and paths that matched.

    Raises:
      ValueError: If neither or both of string/pattern are given, the
        string is empty, or the merge mode is unknown.
      re.error: If the pattern is not a valid regular expression.
    """
    if (string is None) == (pattern is None):
        raise ValueError("Exactly one of string and pattern must be given")
    matcher: Callable[[bytes, bytes], bool]
    if string is not None:
        if isinstance(string, str):
            string = string.encode("utf-8")
        if not string:
            raise ValueError("Cannot search for the empty string")
        matcher = _string_matcher(string)
    else:
        assert pattern is not None
        if isinstance(pattern, str):
            pattern = pattern.encode("utf-8")
        # Compile up front so invalid patterns raise before any history
        # is traversed.
        matcher = _regex_matcher(re.compile(pattern))
    if merge not in ALL_MERGE_MODES:
        raise ValueError(f"Unknown merge mode {merge!r}")
    if rename_detector is None:
        # Pair exact renames (by blob SHA) without ever reading blob
        # contents, so blobs of files outside the path filter are not
        # loaded as a side effect of rename detection.
        rename_detector = RenameDetector(store, max_files=0)
    path_set = set(paths) if paths else None
    return _pickaxe(
        store,
        include,
        exclude,
        matcher=matcher,
        paths=path_set,
        merge=merge,
        order=order,
        reverse=reverse,
        max_entries=max_entries,
        since=since,
        until=until,
        rename_detector=rename_detector,
    )


def _pickaxe(
    store: "BaseObjectStore",
    include: ObjectID | Sequence[ObjectID],
    exclude: Sequence[ObjectID] | None,
    *,
    matcher: Callable[[bytes, bytes], bool],
    paths: set[bytes] | None,
    merge: str,
    order: str,
    reverse: bool,
    max_entries: int | None,
    since: int | None,
    until: int | None,
    rename_detector: RenameDetector,
) -> Iterator[PickaxeResult]:
    walker = Walker(
        store,
        include,
        exclude=exclude,
        order=order,
        reverse=reverse,
        since=since,
        until=until,
    )
    num_entries = 0
    for entry in walker:
        commit = entry.commit
        parents = commit.parents
        if not parents:
            compare_to: list[ObjectID | None] = [None]
        elif merge == MERGE_FIRST:
            compare_to = list(parents[:1])
        else:
            compare_to = list(parents)
        matches: list[PickaxeMatch] = []
        for parent_id in compare_to:
            if parent_id is None:
                parent_tree = None
            else:
                parent_commit = store[parent_id]
                assert isinstance(parent_commit, Commit)
                parent_tree = parent_commit.tree
            changes = tree_changes(
                store, parent_tree, commit.tree, rename_detector=rename_detector
            )
            for change in changes:
                if _is_gitlink(change):
                    continue
                if (
                    change.old is not None
                    and change.new is not None
                    and change.old.sha == change.new.sha
                ):
                    # Mode-only change or exact rename: contents are
                    # identical, so it cannot match.
                    continue
                old_path = change.old.path if change.old is not None else None
                new_path = change.new.path if change.new is not None else None
                if not (
                    _path_matches(paths, new_path) or _path_matches(paths, old_path)
                ):
                    # Not a requested path; do not load its blobs.
                    continue
                old_data = _blob_data(store, change, old=True)
                new_data = _blob_data(store, change, old=False)
                if matcher(old_data, new_data):
                    path = new_path if new_path is not None else old_path
                    assert path is not None
                    matches.append(PickaxeMatch(parent_id, path))
        if matches:
            yield PickaxeResult(commit, matches)
            num_entries += 1
            if max_entries is not None and num_entries >= max_entries:
                return

# annotate.py -- Annotate files with last changed revision
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

"""Annotate file contents indicating when they were last changed.

Annotated lines are represented as tuples with last modified revision SHA1
and contents.

Please note that this is a very naive annotate implementation. It works,
but its speed could be improved - in particular because it uses
Python's difflib.
"""

__all__ = [
    "AnnotatedLine",
    "BinaryFileError",
    "MissingBlobError",
    "annotate_lines",
    "annotate_lines_detailed",
    "update_lines",
]

import difflib
import itertools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from dulwich.errors import NotBlobError
from dulwich.objects import Blob, Commit
from dulwich.patch import is_binary
from dulwich.walk import (
    ORDER_DATE,
    Walker,
)

if TYPE_CHECKING:
    from dulwich.diff_tree import TreeChange
    from dulwich.object_store import BaseObjectStore
    from dulwich.objects import ObjectID, TreeEntry

# Walk over ancestry graph breadth-first
# When checking each revision, find lines that according to difflib.Differ()
# are common between versions.
# Any lines that are not in common were introduced by the newer revision.
# If there were no lines kept from the older version, stop going deeper in the
# graph.


def update_lines(
    annotated_lines: Sequence[tuple[tuple["Commit | None", "TreeEntry | None"], bytes]],
    new_history_data: tuple["Commit | None", "TreeEntry | None"],
    new_blob: "Blob",
) -> list[tuple[tuple["Commit | None", "TreeEntry | None"], bytes]]:
    """Update annotation lines with old blob lines."""
    ret: list[tuple[tuple[Commit | None, TreeEntry | None], bytes]] = []
    new_lines = new_blob.splitlines()
    matcher = difflib.SequenceMatcher(
        a=[line for (h, line) in annotated_lines], b=new_lines
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ret.extend(annotated_lines[i1:i2])
        elif tag in ("insert", "replace"):
            ret.extend([(new_history_data, line) for line in new_lines[j1:j2]])
        elif tag == "delete":
            pass  # don't care
        else:
            raise RuntimeError(f"Unknown tag {tag} returned in diff")
    return ret


class BinaryFileError(Exception):
    """Raised when a blob in the annotated history is binary."""


class MissingBlobError(Exception):
    """Raised when a blob in the annotated history is missing from the store."""


class AnnotatedLine(NamedTuple):
    """A single line of a file with detailed annotation information.

    Attributes:
      commit: Commit credited with introducing the line, or None when the
        line could not be reliably attributed (see unblamable).
      entry: Tree entry for the blob in commit, or None when unblamable.
      line: Contents of the line as it appears in the queried commit.
      lineno: 1-based line number in the queried commit.
      source_lineno: 1-based line number of the line in the version of the
        file at commit, or None when unblamable.
      unblamable: True when no reliable source commit could be determined,
        e.g. for lines that were introduced by an ignored revision and have
        no counterpart in its predecessor. No author is fabricated for such
        lines.
    """

    commit: "Commit | None"
    entry: "TreeEntry | None"
    line: bytes
    lineno: int
    source_lineno: "int | None"
    unblamable: bool


def _find_path_revs(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str,
    follow: bool,
) -> list[tuple["Commit", "TreeEntry"]]:
    """Find the revisions that touched a path, newest first.

    Renames are followed when follow is true. For merge commits the
    per-parent changes are considered in parent order and the first change
    touching path is followed.
    """
    walker = Walker(
        store, include=[commit_id], paths=[path], order=order, follow=follow
    )
    revs: list[tuple[Commit, TreeEntry]] = []
    for log_entry in walker:
        for tree_change in log_entry.changes():
            changes: list[TreeChange | None]
            if isinstance(tree_change, list):
                changes = list(tree_change)
            else:
                changes = [tree_change]
            for change in changes:
                if change is None:
                    continue
                if change.new is not None and change.new.path == path:
                    if change.old is not None and change.old.path is not None:
                        path = change.old.path
                    revs.append((log_entry.commit, change.new))
                    break
    return revs


def _blob_lines(
    store: "BaseObjectStore", entry: "TreeEntry", path: bytes
) -> list[bytes]:
    """Return the lines of a text blob, reporting binary/missing objects."""
    assert entry.sha is not None
    try:
        obj = store[entry.sha]
    except KeyError as exc:
        raise MissingBlobError(
            f"Blob {entry.sha.decode('ascii')} for path {path!r} is missing"
        ) from exc
    if not isinstance(obj, Blob):
        raise NotBlobError(entry.sha)
    if is_binary(obj.data):
        raise BinaryFileError(
            f"Path {path!r} is binary in blob {entry.sha.decode('ascii')}"
        )
    return obj.splitlines()


def _map_lines(
    old_lines: list[bytes], new_lines: list[bytes]
) -> list[tuple["int | None", bool]]:
    """Map each line of new_lines back to a line of old_lines.

    Returns a list with one (old_lineno, exact) tuple per line in new_lines.
    old_lineno is None for lines without any counterpart in old_lines (pure
    insertions). exact is True only for lines identical to their
    counterpart. Lines in replaced blocks are mapped to old lines pairwise
    by offset; surplus new lines are mapped to the last old line of the
    block.
    """
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    mapping: list[tuple[int | None, bool]] = [(None, False)] * len(new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(j2 - j1):
                mapping[j1 + off] = (i1 + off, True)
        elif tag == "replace":
            for off in range(j2 - j1):
                mapping[j1 + off] = (min(i1 + off, i2 - 1), False)
        elif tag in ("insert", "delete"):
            pass  # insertions have no old-side counterpart; deletions add none
        else:
            raise RuntimeError(f"Unknown tag {tag} returned in diff")
    return mapping


def _normalize_revisions(
    revisions: "Iterable[ObjectID | Commit]",
) -> "set[ObjectID]":
    """Normalize a collection of commit ids and/or Commit objects to ids."""
    return {rev.id if isinstance(rev, Commit) else rev for rev in revisions}


def annotate_lines_detailed(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str = ORDER_DATE,
    follow: bool = True,
    ignore_revisions: "Iterable[ObjectID | Commit] | None" = None,
) -> list[AnnotatedLine]:
    """Annotate the lines of a blob, returning detailed per-line information.

    Lines are processed backwards from commit_id; the line contents and
    their order always reflect the version of the file at commit_id.

    Revisions listed in ignore_revisions are not credited with any line.
    They are kept in the history chain (rather than dropped) so that lines
    they changed are mapped back through them to their predecessor: lines
    that are identical on both sides keep their line mapping, and lines in
    replaced blocks (e.g. split or joined by a reformat) are mapped to the
    predecessor lines they replaced. Lines introduced by an ignored
    revision without any counterpart in the predecessor (pure insertions)
    cannot be reliably attributed and are marked unblamable with commit
    set to None - no author is fabricated for them.

    Tie-breaking rules (fixed and deterministic):

    * For merge commits, per-parent changes are considered in parent order
      and the first change touching path is followed.
    * When an ignored revision replaces a block of lines, new lines are
      mapped to old lines pairwise by offset; surplus new lines are mapped
      to the last old line of the block.
    * Duplicate lines are matched by difflib.SequenceMatcher with autojunk
      disabled, which is deterministic for a given pair of blobs.

    Only text blobs are supported.

    :param store: Object store to retrieve objects from
    :param commit_id: Commit id in which to annotate path
    :param path: Path to annotate
    :param order: Order in which to process history (defaults to ORDER_DATE)
    :param follow: Whether to follow changes across renames/copies
    :param ignore_revisions: Revisions (commit ids and/or Commit objects)
        that should not be credited with any line
    :return: List of AnnotatedLine entries, one per line of the blob at
        commit_id
    :raises BinaryFileError: if a blob in the history of path is binary
    :raises MissingBlobError: if a blob in the history of path is missing
    """
    ignored = _normalize_revisions(ignore_revisions or ())
    revs = _find_path_revs(store, commit_id, path, order, follow)
    if not revs:
        return []
    versions = [
        (commit, entry, _blob_lines(store, entry, path)) for commit, entry in revs
    ]
    final_lines = versions[0][2]
    results: list[AnnotatedLine | None] = [None] * len(final_lines)
    # Suspects are (final 0-based lineno, 0-based lineno in current version)
    # pairs for lines that have not been attributed yet.
    suspects = [
        (final_lineno, final_lineno) for final_lineno in range(len(final_lines))
    ]
    for (commit, entry, current_lines), (
        _pcommit,
        _pentry,
        parent_lines,
    ) in itertools.pairwise(versions):
        if not suspects:
            break
        mapping = _map_lines(parent_lines, current_lines)
        next_suspects: list[tuple[int, int]] = []
        if commit.id in ignored:
            for final_lineno, current_lineno in suspects:
                old_lineno, _exact = mapping[current_lineno]
                if old_lineno is None:
                    results[final_lineno] = AnnotatedLine(
                        None,
                        None,
                        final_lines[final_lineno],
                        final_lineno + 1,
                        None,
                        True,
                    )
                else:
                    next_suspects.append((final_lineno, old_lineno))
        else:
            for final_lineno, current_lineno in suspects:
                old_lineno, exact = mapping[current_lineno]
                if old_lineno is not None and exact:
                    next_suspects.append((final_lineno, old_lineno))
                else:
                    results[final_lineno] = AnnotatedLine(
                        commit,
                        entry,
                        final_lines[final_lineno],
                        final_lineno + 1,
                        current_lineno + 1,
                        False,
                    )
        suspects = next_suspects
    # Any remaining suspects were introduced by the oldest revision found.
    commit, entry, _lines = versions[-1]
    for final_lineno, current_lineno in suspects:
        if commit.id in ignored:
            results[final_lineno] = AnnotatedLine(
                None,
                None,
                final_lines[final_lineno],
                final_lineno + 1,
                None,
                True,
            )
        else:
            results[final_lineno] = AnnotatedLine(
                commit,
                entry,
                final_lines[final_lineno],
                final_lineno + 1,
                current_lineno + 1,
                False,
            )
    return [result for result in results if result is not None]


def annotate_lines(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str = ORDER_DATE,
    lines: Sequence[tuple[tuple["Commit", "TreeEntry"], bytes]] | None = None,
    follow: bool = True,
    ignore_revisions: "Iterable[ObjectID | Commit] | None" = None,
) -> list[tuple[tuple["Commit | None", "TreeEntry | None"], bytes]]:
    """Annotate the lines of a blob.

    :param store: Object store to retrieve objects from
    :param commit_id: Commit id in which to annotate path
    :param path: Path to annotate
    :param order: Order in which to process history (defaults to ORDER_DATE)
    :param lines: Initial lines to compare to (defaults to specified)
    :param follow: Whether to follow changes across renames/copies
    :param ignore_revisions: Revisions (commit ids and/or Commit objects)
        that should not be credited with any line; lines they changed are
        mapped back to earlier revisions where possible. Lines that cannot
        be reliably attributed are returned with a (None, None) source.
        See annotate_lines_detailed for the exact mapping rules.
    :return: List of (commit, line) entries where
        commit is the oldest commit that changed a line
    """
    if ignore_revisions:
        return [
            ((annotated.commit, annotated.entry), annotated.line)
            for annotated in annotate_lines_detailed(
                store,
                commit_id,
                path,
                order=order,
                follow=follow,
                ignore_revisions=ignore_revisions,
            )
        ]
    revs = _find_path_revs(store, commit_id, path, order, follow)

    lines_annotated: list[tuple[tuple[Commit | None, TreeEntry | None], bytes]] = []
    for commit, entry in reversed(revs):
        assert entry.sha is not None
        blob_obj = store[entry.sha]
        assert isinstance(blob_obj, Blob)
        lines_annotated = update_lines(lines_annotated, (commit, entry), blob_obj)
    return lines_annotated

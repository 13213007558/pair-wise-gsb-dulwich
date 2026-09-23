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

Ignoring revisions
==================

Both ``annotate_lines`` and ``annotate_lines_detailed`` accept an
``ignore_revisions`` argument: an iterable of commit ids whose own changes
should not be blamed. This is useful to e.g. skip commits that only
reformatted the file.

Ignored commits are *not* simply dropped from the history: a reformatting
commit may split or join lines and move context around, so the lines it
introduced are mapped back onto the parent version instead. A line
introduced by an ignored commit is attributed to a line with identical
contents in the parent version that is not already matched to another
line. When several parent lines qualify, the one whose line number is
closest to the position at which the new line was introduced is used;
remaining ties are broken in favour of the lowest parent line number.
Each parent line can only be used as the source of one new line per
commit. Lines for which no parent line qualifies are marked as
*unblamable* and are not attributed to any commit, rather than being
blamed on a fabricated author.

Only text blobs are supported. Binary blobs and missing objects are
reported by raising ``BinaryBlobError`` and ``MissingBlobError``.
"""

__all__ = [
    "AnnotateError",
    "AnnotatedLine",
    "BinaryBlobError",
    "MissingBlobError",
    "annotate_lines",
    "annotate_lines_detailed",
    "update_lines",
]

import difflib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NamedTuple

from dulwich.objects import Blob
from dulwich.walk import (
    ORDER_DATE,
    Walker,
)

if TYPE_CHECKING:
    from dulwich.diff_tree import TreeChange
    from dulwich.object_store import BaseObjectStore
    from dulwich.objects import Commit, ObjectID, TreeEntry

# Number of leading bytes to check for NUL when detecting binary blobs,
# mirroring git's FIRST_FEW_BYTES.
_BINARY_CHECK_BYTES = 8000


class AnnotateError(Exception):
    """Base class for errors raised while annotating."""


class MissingBlobError(AnnotateError):
    """An object referenced while annotating is missing."""

    def __init__(self, sha: "ObjectID | None", path: bytes) -> None:
        """Create a MissingBlobError."""
        self.sha = sha
        self.path = path
        super().__init__(
            f"Missing blob {sha.decode('ascii') if sha else None} for path {path!r}"
        )


class BinaryBlobError(AnnotateError):
    """A blob encountered while annotating is binary."""

    def __init__(self, sha: "ObjectID", path: bytes) -> None:
        """Create a BinaryBlobError."""
        self.sha = sha
        self.path = path
        super().__init__(
            f"Blob {sha.decode('ascii')} for path {path!r} is binary; "
            f"only text blobs can be annotated"
        )


class AnnotatedLine(NamedTuple):
    """A single annotated line.

    :ivar commit: Commit that last changed the line, or None when the
        line is unblamable
    :ivar entry: Tree entry for the file at the commit, or None when the
        line is unblamable
    :ivar line: Contents of the line (as of the queried commit)
    :ivar line_no: 1-based line number in the queried commit's version
    :ivar source_line_no: 1-based line number in the version of the file
        at the source commit, or None when the line is unblamable
    :ivar unblamable: Whether the line could not be attributed to any
        commit
    """

    commit: "Commit | None"
    entry: "TreeEntry | None"
    line: bytes
    line_no: int
    source_line_no: "int | None"
    unblamable: bool


class _LineSource(NamedTuple):
    """Attribution of a line to a commit and a line number within it."""

    commit: "Commit | None"
    entry: "TreeEntry | None"
    line_no: "int | None"  # 1-based line number in the commit version


_UNBLAMABLE = _LineSource(None, None, None)

# Walk over ancestry graph breadth-first
# When checking each revision, find lines that according to difflib.Differ()
# are common between versions.
# Any lines that are not in common were introduced by the newer revision.
# If there were no lines kept from the older version, stop going deeper in the
# graph.


def update_lines(
    annotated_lines: Sequence[tuple[tuple["Commit", "TreeEntry"], bytes]],
    new_history_data: tuple["Commit", "TreeEntry"],
    new_blob: "Blob",
) -> list[tuple[tuple["Commit", "TreeEntry"], bytes]]:
    """Update annotation lines with old blob lines."""
    ret: list[tuple[tuple[Commit, TreeEntry], bytes]] = []
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


def _get_blob(store: "BaseObjectStore", entry: "TreeEntry") -> "Blob":
    """Retrieve the blob for a tree entry, checking it can be annotated."""
    path = entry.path
    sha = entry.sha
    if sha is None:
        raise MissingBlobError(None, path)
    try:
        obj = store[sha]
    except KeyError as exc:
        raise MissingBlobError(sha, path) from exc
    if not isinstance(obj, Blob):
        raise MissingBlobError(sha, path)
    if b"\0" in obj.data[:_BINARY_CHECK_BYTES]:
        raise BinaryBlobError(sha, path)
    return obj


def _find_source_line(
    annotated: "list[tuple[_LineSource, bytes]]",
    candidates: "list[int]",
    line: bytes,
    anchor: int,
) -> "int | None":
    """Find the parent line to attribute a new line to.

    :param annotated: Annotated lines of the parent version
    :param candidates: Indices into ``annotated`` of parent lines that
        are not already matched to a line in the new version
    :param line: Contents of the new line
    :param anchor: Position in the parent version at which the new line
        (or the block containing it) was introduced
    :return: Index into ``annotated`` of the source line, or None when
        no parent line qualifies
    """
    best: int | None = None
    for index in candidates:
        if annotated[index][1] != line:
            continue
        if best is None or (abs(index - anchor), index) < (
            abs(best - anchor),
            best,
        ):
            best = index
    return best


def _annotate_lines(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str,
    follow: bool,
    ignore_revisions: "Iterable[ObjectID] | None",
) -> "list[tuple[_LineSource, bytes]]":
    """Annotate the lines of a blob, returning (source, line) pairs."""
    ignored = set(ignore_revisions) if ignore_revisions else set()
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

    annotated: list[tuple[_LineSource, bytes]] = []
    for commit, entry in reversed(revs):
        new_lines = _get_blob(store, entry).splitlines()
        if not annotated:
            annotated = [
                (_LineSource(commit, entry, index + 1), line)
                for index, line in enumerate(new_lines)
            ]
            continue
        matcher = difflib.SequenceMatcher(
            a=[line for (_source, line) in annotated], b=new_lines
        )
        opcodes = matcher.get_opcodes()
        candidates: list[int] = []
        if commit.id in ignored:
            consumed = [False] * len(annotated)
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == "equal":
                    for index in range(i1, i2):
                        consumed[index] = True
            candidates = [index for index, used in enumerate(consumed) if not used]
        result: list[tuple[_LineSource, bytes]] = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                result.extend(annotated[i1:i2])
            elif tag in ("insert", "replace"):
                if commit.id in ignored:
                    for j in range(j1, j2):
                        line = new_lines[j]
                        match = _find_source_line(annotated, candidates, line, i1)
                        if match is None:
                            result.append((_UNBLAMABLE, line))
                        else:
                            candidates.remove(match)
                            result.append((annotated[match][0], line))
                else:
                    result.extend(
                        (_LineSource(commit, entry, j + 1), new_lines[j])
                        for j in range(j1, j2)
                    )
            elif tag == "delete":
                pass  # don't care
            else:
                raise RuntimeError(f"Unknown tag {tag} returned in diff")
        annotated = result
    return annotated


def annotate_lines_detailed(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str = ORDER_DATE,
    follow: bool = True,
    ignore_revisions: "Iterable[ObjectID] | None" = None,
) -> "list[AnnotatedLine]":
    """Annotate the lines of a blob, with per-line details.

    The contents and order of the lines are those of the version of the
    file at ``commit_id``.

    :param store: Object store to retrieve objects from
    :param commit_id: Commit id in which to annotate path
    :param path: Path to annotate
    :param order: Order in which to process history (defaults to ORDER_DATE)
    :param follow: Whether to follow changes across renames/copies
    :param ignore_revisions: Commit ids whose own changes should not be
        blamed; see the module documentation for the rules used to
        attribute their lines to earlier commits
    :return: List of AnnotatedLine tuples, one per line of the file at
        the queried commit
    """
    annotated = _annotate_lines(store, commit_id, path, order, follow, ignore_revisions)
    return [
        AnnotatedLine(
            commit=source.commit,
            entry=source.entry,
            line=line,
            line_no=index + 1,
            source_line_no=source.line_no,
            unblamable=source.commit is None,
        )
        for index, (source, line) in enumerate(annotated)
    ]


def annotate_lines(
    store: "BaseObjectStore",
    commit_id: "ObjectID",
    path: bytes,
    order: str = ORDER_DATE,
    lines: "Sequence[tuple[tuple[Commit, TreeEntry], bytes]] | None" = None,
    follow: bool = True,
    ignore_revisions: "Iterable[ObjectID] | None" = None,
) -> "list[tuple[tuple[Commit | None, TreeEntry | None], bytes]]":
    """Annotate the lines of a blob.

    :param store: Object store to retrieve objects from
    :param commit_id: Commit id in which to annotate path
    :param path: Path to annotate
    :param order: Order in which to process history (defaults to ORDER_DATE)
    :param lines: Initial lines to compare to (defaults to specified)
    :param follow: Whether to follow changes across renames/copies
    :param ignore_revisions: Commit ids whose own changes should not be
        blamed; see the module documentation for the rules used to
        attribute their lines to earlier commits
    :return: List of ((commit, entry), line) entries where
        commit is the oldest commit that changed a line. Lines that
        could not be attributed to any commit (only possible when
        ignore_revisions is used) have a (None, None) history.
    """
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

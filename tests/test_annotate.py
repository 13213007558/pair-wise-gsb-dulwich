# test_annotate.py -- tests for annotate
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

"""Tests for annotate support."""

import shutil
import tempfile
from typing import Any
from unittest import TestCase

from dulwich.annotate import (
    BinaryFileError,
    MissingBlobError,
    annotate_lines,
    annotate_lines_detailed,
    update_lines,
)
from dulwich.objects import Blob, Commit, Tree
from dulwich.repo import Repo


class UpdateLinesTestCase(TestCase):
    """Tests for update_lines function."""

    def test_update_lines_equal(self) -> None:
        """Test update_lines when all lines are equal."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = [
            (("commit1", "entry1"), b"line1"),
            (("commit2", "entry2"), b"line2"),
        ]
        new_blob = b"line1\nline2"
        new_history_data = ("commit3", "entry3")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        self.assertEqual(old_lines, result)

    def test_update_lines_insert(self) -> None:
        """Test update_lines when new lines are inserted."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = [
            (("commit1", "entry1"), b"line1"),
            (("commit2", "entry2"), b"line3"),
        ]
        new_blob = b"line1\nline2\nline3"
        new_history_data = ("commit3", "entry3")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        expected = [
            (("commit1", "entry1"), b"line1"),
            (("commit3", "entry3"), b"line2"),
            (("commit2", "entry2"), b"line3"),
        ]
        self.assertEqual(expected, result)

    def test_update_lines_delete(self) -> None:
        """Test update_lines when lines are deleted."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = [
            (("commit1", "entry1"), b"line1"),
            (("commit2", "entry2"), b"line2"),
            (("commit3", "entry3"), b"line3"),
        ]
        new_blob = b"line1\nline3"
        new_history_data = ("commit4", "entry4")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        expected = [
            (("commit1", "entry1"), b"line1"),
            (("commit3", "entry3"), b"line3"),
        ]
        self.assertEqual(expected, result)

    def test_update_lines_replace(self) -> None:
        """Test update_lines when lines are replaced."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = [
            (("commit1", "entry1"), b"line1"),
            (("commit2", "entry2"), b"line2"),
        ]
        new_blob = b"line1\nline2_modified"
        new_history_data = ("commit3", "entry3")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        expected = [
            (("commit1", "entry1"), b"line1"),
            (("commit3", "entry3"), b"line2_modified"),
        ]
        self.assertEqual(expected, result)

    def test_update_lines_empty_old(self) -> None:
        """Test update_lines with empty old lines."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = []
        new_blob = b"line1\nline2"
        new_history_data = ("commit1", "entry1")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        expected = [
            (("commit1", "entry1"), b"line1"),
            (("commit1", "entry1"), b"line2"),
        ]
        self.assertEqual(expected, result)

    def test_update_lines_empty_new(self) -> None:
        """Test update_lines with empty new blob."""
        old_lines: list[tuple[tuple[Any, Any], bytes]] = [
            (("commit1", "entry1"), b"line1")
        ]
        new_blob = b""
        new_history_data = ("commit2", "entry2")

        result = update_lines(old_lines, new_history_data, new_blob)  # type: ignore[arg-type]
        self.assertEqual([], result)


class AnnotateLinesTestCase(TestCase):
    """Tests for annotate_lines function."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Repo.init(self.temp_dir)

    def tearDown(self) -> None:
        self.repo.close()
        shutil.rmtree(self.temp_dir)

    def _make_commit(
        self, blob_content: bytes, message: str, parent: bytes | None = None
    ) -> bytes:
        """Helper to create a commit with a single file."""
        # Create blob
        blob = Blob()
        blob.data = blob_content
        self.repo.object_store.add_object(blob)

        # Create tree
        tree = Tree()
        tree.add(b"test.txt", 0o100644, blob.id)
        self.repo.object_store.add_object(tree)

        # Create commit
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test Author <test@example.com>"
        commit.author_time = commit.commit_time = 1000000000
        commit.author_timezone = commit.commit_timezone = 0
        commit.encoding = b"UTF-8"
        commit.message = message.encode("utf-8")

        if parent:
            commit.parents = [parent]
        else:
            commit.parents = []

        self.repo.object_store.add_object(commit)
        return commit.id

    def test_annotate_lines_single_commit(self) -> None:
        """Test annotating a file with a single commit."""
        commit_id = self._make_commit(b"line1\nline2\nline3\n", "Initial commit")

        result = annotate_lines(self.repo.object_store, commit_id, b"test.txt")

        self.assertEqual(3, len(result))
        for (commit, entry), line in result:
            self.assertEqual(commit_id, commit.id)
            self.assertIn(line, [b"line1\n", b"line2\n", b"line3\n"])

    def test_annotate_lines_multiple_commits(self) -> None:
        """Test annotating a file with multiple commits."""
        # First commit
        commit1_id = self._make_commit(b"line1\nline2\n", "Initial commit")

        # Second commit - add a line
        commit2_id = self._make_commit(
            b"line1\nline1.5\nline2\n", "Add line between", parent=commit1_id
        )

        # Third commit - modify a line
        commit3_id = self._make_commit(
            b"line1_modified\nline1.5\nline2\n", "Modify first line", parent=commit2_id
        )

        result = annotate_lines(self.repo.object_store, commit3_id, b"test.txt")

        self.assertEqual(3, len(result))
        # First line should be from commit3
        self.assertEqual(commit3_id, result[0][0][0].id)
        self.assertEqual(b"line1_modified\n", result[0][1])

        # Second line should be from commit2
        self.assertEqual(commit2_id, result[1][0][0].id)
        self.assertEqual(b"line1.5\n", result[1][1])

        # Third line should be from commit1
        self.assertEqual(commit1_id, result[2][0][0].id)
        self.assertEqual(b"line2\n", result[2][1])

    def test_annotate_lines_nonexistent_path(self) -> None:
        """Test annotating a nonexistent file."""
        commit_id = self._make_commit(b"content\n", "Initial commit")

        result = annotate_lines(self.repo.object_store, commit_id, b"nonexistent.txt")
        self.assertEqual([], result)


class AnnotateIgnoreRevisionsTestCase(TestCase):
    """Tests for annotate_lines with ignore_revisions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Repo.init(self.temp_dir)

    def tearDown(self) -> None:
        self.repo.close()
        shutil.rmtree(self.temp_dir)

    def _make_commit(
        self,
        files: dict[bytes, bytes],
        message: str,
        parents: tuple[bytes, ...] = (),
        time: int = 1000000000,
        add_blobs: bool = True,
    ) -> bytes:
        """Helper to create a commit with the given files."""
        tree = Tree()
        for path, content in files.items():
            blob = Blob()
            blob.data = content
            if add_blobs:
                self.repo.object_store.add_object(blob)
            tree.add(path, 0o100644, blob.id)
        self.repo.object_store.add_object(tree)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test Author <test@example.com>"
        commit.author_time = commit.commit_time = time
        commit.author_timezone = commit.commit_timezone = 0
        commit.message = message.encode("utf-8")
        commit.parents = list(parents)
        self.repo.object_store.add_object(commit)
        return commit.id

    def _build_format_history(self) -> tuple[bytes, bytes, bytes, bytes]:
        """Build: business change, reformat, business change, rename.

        The reformat commit splits a line and adds a line of its own; the
        file also contains duplicate lines. Commit times are deliberately
        not monotonically increasing.
        """
        c1 = self._make_commit(
            {
                b"app.txt": b"header = 1\ntotal = compute(a, b)\ndone\ndone\nfooter = 2\n"
            },
            "business: add app",
            time=1000000000,
        )
        c2 = self._make_commit(
            {
                b"app.txt": b"# formatted\nheader = 1\ntotal = compute(\n"
                b"    a, b)\ndone\ndone\nfooter = 2\n"
            },
            "style: reformat",
            parents=(c1,),
            time=1000000000,
        )
        c3 = self._make_commit(
            {
                b"app.txt": b"# formatted\nheader = 1\ntotal = compute(\n"
                b"    a, b)\ndone\ndone\nfooter = 3\n"
            },
            "business: bump footer",
            parents=(c2,),
            time=999999999,
        )
        c4 = self._make_commit(
            {
                b"app2.txt": b"# formatted\nheader = 1\ntotal = compute(\n"
                b"    a, b)\ndone\ndone\nfooter = 3\n"
            },
            "rename app.txt to app2.txt",
            parents=(c3,),
            time=1000000000,
        )
        return c1, c2, c3, c4

    def test_without_ignore_attributes_format_commit(self) -> None:
        """Without ignore_revisions the format commit is blamed."""
        c1, c2, c3, c4 = self._build_format_history()

        result = annotate_lines_detailed(self.repo.object_store, c4, b"app2.txt")

        self.assertEqual(
            [
                (c2, 1, b"# formatted\n"),
                (c1, 1, b"header = 1\n"),
                (c2, 3, b"total = compute(\n"),
                (c2, 4, b"    a, b)\n"),
                (c1, 3, b"done\n"),
                (c1, 4, b"done\n"),
                (c3, 7, b"footer = 3\n"),
            ],
            [
                (
                    annotated.commit.id if annotated.commit else None,
                    annotated.source_lineno,
                    annotated.line,
                )
                for annotated in result
            ],
        )
        self.assertEqual(list(range(1, 8)), [a.lineno for a in result])
        self.assertFalse(any(a.unblamable for a in result))

    def test_ignore_revisions_maps_lines_back(self) -> None:
        """Ignored format commit lines map back to earlier authors."""
        c1, c2, c3, c4 = self._build_format_history()

        result = annotate_lines_detailed(
            self.repo.object_store, c4, b"app2.txt", ignore_revisions={c2}
        )

        self.assertEqual(
            [
                (None, None, b"# formatted\n"),
                (c1, 1, b"header = 1\n"),
                (c1, 2, b"total = compute(\n"),
                (c1, 2, b"    a, b)\n"),
                (c1, 3, b"done\n"),
                (c1, 4, b"done\n"),
                (c3, 7, b"footer = 3\n"),
            ],
            [
                (
                    annotated.commit.id if annotated.commit else None,
                    annotated.source_lineno,
                    annotated.line,
                )
                for annotated in result
            ],
        )
        # Line contents and order still reflect the queried commit.
        self.assertEqual(list(range(1, 8)), [a.lineno for a in result])
        # Only the line added by the format commit alone is unblamable.
        self.assertEqual([True] + [False] * 6, [a.unblamable for a in result])
        self.assertIsNone(result[0].commit)
        self.assertIsNone(result[0].entry)

    def test_ignore_revisions_accepts_commit_objects(self) -> None:
        """ignore_revisions accepts Commit objects as well as ids."""
        _c1, c2, _c3, c4 = self._build_format_history()
        c2_commit = self.repo.object_store[c2]
        assert isinstance(c2_commit, Commit)

        by_id = annotate_lines_detailed(
            self.repo.object_store, c4, b"app2.txt", ignore_revisions={c2}
        )
        by_obj = annotate_lines_detailed(
            self.repo.object_store, c4, b"app2.txt", ignore_revisions={c2_commit}
        )
        self.assertEqual(by_id, by_obj)

    def test_annotate_lines_compat_with_ignore(self) -> None:
        """annotate_lines with ignore_revisions keeps the legacy shape."""
        c1, c2, c3, c4 = self._build_format_history()

        result = annotate_lines(
            self.repo.object_store, c4, b"app2.txt", ignore_revisions={c2}
        )

        self.assertEqual(
            [None, c1, c1, c1, c1, c1, c3],
            [
                commit.id if commit is not None else None
                for (commit, _entry), _line in result
            ],
        )
        # The line only added by the ignored commit has no source.
        (commit, entry), line = result[0]
        self.assertIsNone(commit)
        self.assertIsNone(entry)
        self.assertEqual(b"# formatted\n", line)

    def test_annotate_lines_default_unchanged_by_ignore_param(self) -> None:
        """Default annotate_lines output is unaffected by the new parameter."""
        _c1, _c2, _c3, c4 = self._build_format_history()

        legacy = annotate_lines(self.repo.object_store, c4, b"app2.txt")
        detailed = annotate_lines_detailed(self.repo.object_store, c4, b"app2.txt")

        self.assertEqual(
            [(commit.id, line) for (commit, _entry), line in legacy],
            [
                (annotated.commit.id if annotated.commit else None, annotated.line)
                for annotated in detailed
            ],
        )

    def test_ignore_revisions_merge_parent_selection(self) -> None:
        """Merge commits still participate in parent selection when ignoring."""
        c1 = self._make_commit(
            {
                b"app.txt": b"header = 1\ntotal = compute(a, b)\ndone\ndone\n"
                b"footer = 2\n",
                b"other.txt": b"other\n",
            },
            "business: add app",
            time=1000000000,
        )
        c2 = self._make_commit(
            {
                b"app.txt": b"# formatted\nheader = 1\ntotal = compute(\n"
                b"    a, b)\ndone\ndone\nfooter = 2\n",
                b"other.txt": b"other\n",
            },
            "style: reformat",
            parents=(c1,),
            time=1000000003,
        )
        c3 = self._make_commit(
            {
                b"app.txt": b"header = 1\ntotal = compute(a, b)\ndone\ndone\n"
                b"footer = 3\n",
                b"other.txt": b"other\n",
            },
            "business: bump footer",
            parents=(c1,),
            time=1000000002,
        )
        # Merge takes the formatted file with the business change on top.
        c4 = self._make_commit(
            {
                b"app.txt": b"# formatted\nheader = 1\ntotal = compute(\n"
                b"    a, b)\ndone\ndone\nfooter = 3\n",
                b"other.txt": b"other\n",
            },
            "merge",
            parents=(c2, c3),
            time=1000000001,
        )

        result = annotate_lines_detailed(
            self.repo.object_store, c4, b"app.txt", ignore_revisions={c2}
        )

        self.assertEqual(
            [
                (None, b"# formatted\n"),
                (c1, b"header = 1\n"),
                (c1, b"total = compute(\n"),
                (c1, b"    a, b)\n"),
                (c1, b"done\n"),
                (c1, b"done\n"),
                # The business change came from the second parent; with the
                # fixed first-parent tie-break the merge is credited.
                (c4, b"footer = 3\n"),
            ],
            [
                (
                    annotated.commit.id if annotated.commit else None,
                    annotated.line,
                )
                for annotated in result
            ],
        )
        self.assertEqual([True] + [False] * 6, [a.unblamable for a in result])

    def test_binary_blob_reported(self) -> None:
        """Binary blobs are reported explicitly."""
        commit_id = self._make_commit({b"bin.dat": b"\x00\x01\x02\x03"}, "binary")

        with self.assertRaises(BinaryFileError):
            annotate_lines_detailed(self.repo.object_store, commit_id, b"bin.dat")
        with self.assertRaises(BinaryFileError):
            annotate_lines(
                self.repo.object_store,
                commit_id,
                b"bin.dat",
                ignore_revisions={commit_id},
            )

    def test_missing_blob_reported(self) -> None:
        """Missing blobs are reported explicitly."""
        commit_id = self._make_commit(
            {b"test.txt": b"content\n"}, "missing blob", add_blobs=False
        )

        with self.assertRaises(MissingBlobError):
            annotate_lines_detailed(self.repo.object_store, commit_id, b"test.txt")

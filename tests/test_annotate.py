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
    BinaryBlobError,
    MissingBlobError,
    _find_source_line,
    _LineSource,
    annotate_lines,
    annotate_lines_detailed,
    update_lines,
)
from dulwich.objects import Blob, Commit, Tree
from dulwich.repo import Repo
from dulwich.walk import ORDER_TOPO


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


class IgnoreRevisionsTestCase(TestCase):
    """Tests for annotate with ignore_revisions."""

    ALICE = b"Alice <alice@example.com>"
    FORMATTER = b"Formatter Bot <formatter@example.com>"
    BOB = b"Bob <bob@example.com>"
    CAROL = b"Carol <carol@example.com>"

    BUSINESS_V1 = (
        b"def foo():\n"
        b"    return 1\n"
        b"\n"
        b"def bar():\n"
        b"    return 2\n"
        b"\n"
        b"def baz():\n"
        b"    return 1\n"
        b"# move me\n"
        b"total = (\n"
        b"    42\n"
        b")\n"
        b"value = 10\n"
    )
    FORMATTED = (
        b"# move me\n"
        b"# formatted by black\n"
        b"def foo():\n"
        b"    return (\n"
        b"        1\n"
        b"    )\n"
        b"\n"
        b"def bar():\n"
        b"    return 2\n"
        b"\n"
        b"def baz():\n"
        b"    return 1\n"
        b"total = (42)\n"
        b"value = 10\n"
    )
    BUSINESS_V2 = FORMATTED.replace(b"    return 2\n", b"    return 3\n")

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Repo.init(self.temp_dir)

    def tearDown(self) -> None:
        self.repo.close()
        shutil.rmtree(self.temp_dir)

    def _make_commit(
        self,
        files: dict[bytes, bytes],
        message: bytes,
        author: bytes,
        parents: list[bytes] | None = None,
    ) -> bytes:
        # All commits share the same timestamp; the tests use ORDER_TOPO
        # so that they do not rely on commit times increasing.
        tree = Tree()
        for path, content in files.items():
            blob = Blob()
            blob.data = content
            self.repo.object_store.add_object(blob)
            tree.add(path, 0o100644, blob.id)
        self.repo.object_store.add_object(tree)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = author
        commit.author_time = commit.commit_time = 1000000000
        commit.author_timezone = commit.commit_timezone = 0
        commit.encoding = b"UTF-8"
        commit.message = message
        commit.parents = parents or []
        self.repo.object_store.add_object(commit)
        return commit.id

    def _make_history(self) -> tuple[bytes, bytes, bytes, bytes]:
        """Create a business/format/business/rename history."""
        commit1 = self._make_commit(
            {b"code.py": self.BUSINESS_V1}, b"business: add code", self.ALICE
        )
        commit2 = self._make_commit(
            {b"code.py": self.FORMATTED},
            b"style: reformat",
            self.FORMATTER,
            parents=[commit1],
        )
        commit3 = self._make_commit(
            {b"code.py": self.BUSINESS_V2},
            b"business: change bar",
            self.BOB,
            parents=[commit2],
        )
        commit4 = self._make_commit(
            {b"main.py": self.BUSINESS_V2},
            b"rename code.py to main.py",
            self.CAROL,
            parents=[commit3],
        )
        return commit1, commit2, commit3, commit4

    def test_without_ignore_revisions(self) -> None:
        """Without ignore_revisions the format commit takes the blame."""
        commit1, commit2, commit3, commit4 = self._make_history()

        result = annotate_lines(
            self.repo.object_store, commit4, b"main.py", order=ORDER_TOPO
        )

        self.assertEqual(
            [
                commit2,  # # move me
                commit2,  # # formatted by black
                commit1,  # def foo():
                commit2,  #     return (
                commit2,  #         1
                commit2,  #     )
                commit1,  # <blank>
                commit1,  # def bar():
                commit3,  #     return 3
                commit1,  # <blank>
                commit1,  # def baz():
                commit1,  #     return 1
                commit2,  # total = (42)
                commit1,  # value = 10
            ],
            [commit.id for (commit, entry), line in result],
        )

    def test_ignore_revisions_detailed(self) -> None:
        """Ignored format commit lines map back or become unblamable."""
        commit1, commit2, commit3, commit4 = self._make_history()

        result = annotate_lines_detailed(
            self.repo.object_store,
            commit4,
            b"main.py",
            order=ORDER_TOPO,
            ignore_revisions=[commit2],
        )

        # Contents and order of the queried commit are preserved.
        self.assertEqual(self.BUSINESS_V2, b"".join(r.line for r in result))
        self.assertEqual(list(range(1, 15)), [r.line_no for r in result])

        expected = [
            (commit1, 9),  # # move me (moved by the format commit)
            (None, None),  # # formatted by black (only added by it)
            (commit1, 1),  # def foo():
            (None, None),  #     return ( (split by the format commit)
            (None, None),  #         1
            (None, None),  #     )
            (commit1, 3),  # <blank>
            (commit1, 4),  # def bar():
            (commit3, 9),  #     return 3
            (commit1, 6),  # <blank>
            (commit1, 7),  # def baz():
            (commit1, 8),  #     return 1
            (None, None),  # total = (42) (joined by the format commit)
            (commit1, 13),  # value = 10
        ]
        self.assertEqual(
            expected,
            [
                (
                    r.commit.id if r.commit is not None else None,
                    r.source_line_no,
                )
                for r in result
            ],
        )
        self.assertEqual(
            [commit_id is None for commit_id, _ in expected],
            [r.unblamable for r in result],
        )
        # Rename following: lines from the first commit carry the old path.
        self.assertEqual(b"code.py", result[0].entry.path)

    def test_ignore_revisions_legacy_return(self) -> None:
        """annotate_lines stays tuple-shaped; unblamable is (None, None)."""
        commit1, commit2, _commit3, commit4 = self._make_history()

        result = annotate_lines(
            self.repo.object_store,
            commit4,
            b"main.py",
            order=ORDER_TOPO,
            ignore_revisions=[commit2],
        )

        self.assertEqual(14, len(result))
        (commit, entry), line = result[0]
        self.assertEqual(b"# move me\n", line)
        self.assertEqual(commit1, commit.id)
        (commit, entry), line = result[1]
        self.assertEqual(b"# formatted by black\n", line)
        self.assertIsNone(commit)
        self.assertIsNone(entry)

    def test_ignore_revisions_duplicate_lines(self) -> None:
        """Duplicate lines map to the nearest candidate source line."""
        original = b"header\ndup\nbody1\nbody2\nbody3\ndup\nfooter1\nfooter2\n"
        moved_to_top = b"dup\nheader\nbody1\nbody2\nbody3\nfooter1\nfooter2\n"
        moved_to_bottom = b"header\nbody1\nbody2\nbody3\nfooter1\nfooter2\ndup\n"

        base_top = self._make_commit({b"top.py": original}, b"base", self.ALICE)
        format_top = self._make_commit(
            {b"top.py": moved_to_top},
            b"format",
            self.FORMATTER,
            parents=[base_top],
        )
        result = annotate_lines_detailed(
            self.repo.object_store,
            format_top,
            b"top.py",
            order=ORDER_TOPO,
            ignore_revisions=[format_top],
        )
        # The dup moved to the top is closest to the dup at line 2.
        self.assertEqual(b"dup\n", result[0].line)
        self.assertEqual(base_top, result[0].commit.id)
        self.assertEqual(2, result[0].source_line_no)

        base_bottom = self._make_commit({b"bottom.py": original}, b"base", self.ALICE)
        format_bottom = self._make_commit(
            {b"bottom.py": moved_to_bottom},
            b"format",
            self.FORMATTER,
            parents=[base_bottom],
        )
        result = annotate_lines_detailed(
            self.repo.object_store,
            format_bottom,
            b"bottom.py",
            order=ORDER_TOPO,
            ignore_revisions=[format_bottom],
        )
        # The dup moved to the bottom is closest to the dup at line 6.
        self.assertEqual(b"dup\n", result[-1].line)
        self.assertEqual(base_bottom, result[-1].commit.id)
        self.assertEqual(6, result[-1].source_line_no)

    def test_find_source_line_breaks_ties_by_lowest_line_number(self) -> None:
        """Equidistant duplicate candidates resolve to the lowest line."""
        annotated = [
            (_LineSource(None, None, 1), b"x"),
            (_LineSource(None, None, 2), b"dup"),
            (_LineSource(None, None, 3), b"y"),
            (_LineSource(None, None, 4), b"dup"),
        ]
        # Anchor 2 is equidistant from the dup lines at indices 1 and 3.
        self.assertEqual(1, _find_source_line(annotated, [1, 3], b"dup", 2))
        self.assertIsNone(_find_source_line(annotated, [1, 3], b"other", 2))

    def test_binary_blob_raises(self) -> None:
        """Binary blobs are reported explicitly."""
        commit = self._make_commit(
            {b"bin.dat": b"\x00\x01\x02\x03"}, b"add binary", self.ALICE
        )
        with self.assertRaises(BinaryBlobError):
            annotate_lines_detailed(
                self.repo.object_store, commit, b"bin.dat", order=ORDER_TOPO
            )
        with self.assertRaises(BinaryBlobError):
            annotate_lines(self.repo.object_store, commit, b"bin.dat", order=ORDER_TOPO)

    def test_missing_blob_raises(self) -> None:
        """Missing objects are reported explicitly."""
        blob = Blob()
        blob.data = b"content\n"
        # Deliberately do not add the blob to the object store.
        tree = Tree()
        tree.add(b"file.txt", 0o100644, blob.id)
        self.repo.object_store.add_object(tree)
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = self.ALICE
        commit.author_time = commit.commit_time = 1000000000
        commit.author_timezone = commit.commit_timezone = 0
        commit.message = b"commit with missing blob"
        commit.parents = []
        self.repo.object_store.add_object(commit)

        with self.assertRaises(MissingBlobError):
            annotate_lines_detailed(
                self.repo.object_store, commit.id, b"file.txt", order=ORDER_TOPO
            )

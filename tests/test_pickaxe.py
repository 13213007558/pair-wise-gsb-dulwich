# test_pickaxe.py -- Tests for pickaxe queries over commit history.
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

"""Tests for pickaxe queries over commit history."""

import re

from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob
from dulwich.pickaxe import MERGE_ALL, MERGE_FIRST, pickaxe
from dulwich.tests.utils import build_commit_graph, make_object

from . import TestCase


def make_blob(data):
    return make_object(Blob, data=data)


class RecordingStore(MemoryObjectStore):
    """MemoryObjectStore that records which blobs were loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded_blobs = []

    def __getitem__(self, sha):
        obj = super().__getitem__(sha)
        if isinstance(obj, Blob):
            self.loaded_blobs.append(obj.id)
        return obj


class PickaxeTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = MemoryObjectStore()

    def make_commits(self, commit_spec, trees):
        return build_commit_graph(self.store, commit_spec, trees=trees)

    def assertPickaxe(self, expected_commits, *args, **kwargs):
        results = list(pickaxe(self.store, *args, **kwargs))
        self.assertEqual(
            [c.id for c in expected_commits], [r.commit.id for r in results]
        )
        return results


class PickaxeStringTest(PickaxeTestCase):
    """Tests for the literal string occurrence counting mode."""

    def make_history(self):
        # c2 only moves a line around; c3 removes one line containing the
        # needle and adds another, so the occurrence count is unchanged;
        # c4 adds another occurrence; c5 removes them all.
        return self.make_commits(
            [[1], [2, 1], [3, 2], [4, 3], [5, 4]],
            {
                1: [(b"f.txt", make_blob(b"alpha\nbeta\n"))],
                2: [(b"f.txt", make_blob(b"beta\nalpha\n"))],
                3: [(b"f.txt", make_blob(b"beta\nalpha gamma\n"))],
                4: [(b"f.txt", make_blob(b"beta\nalpha gamma\nalpha\n"))],
                5: [(b"f.txt", make_blob(b"beta\n"))],
            },
        )

    def test_counts_occurrences(self):
        c1, _c2, _c3, c4, c5 = self.make_history()
        results = self.assertPickaxe([c5, c4, c1], [c5.id], string=b"alpha")
        # The root commit is compared against the empty tree.
        self.assertEqual(
            [(None, b"f.txt")], [(m.parent, m.path) for m in results[-1].matches]
        )

    def test_moved_text_does_not_match(self):
        c1, c2, _, _, _ = self.make_history()
        self.assertPickaxe([], [c2.id], exclude=[c1.id], string=b"alpha")

    def test_cancelled_out_changes_do_not_match(self):
        _, c2, c3, _, _ = self.make_history()
        self.assertPickaxe([], [c3.id], exclude=[c2.id], string=b"alpha")

    def test_str_string_accepted(self):
        c1, _, _, c4, c5 = self.make_history()
        results = list(pickaxe(self.store, [c5.id], string="alpha"))
        self.assertEqual([c5.id, c4.id, c1.id], [r.commit.id for r in results])

    def test_non_overlapping_occurrences(self):
        # "aa" occurs non-overlapping: once in "aaa", twice in "aaaa".
        c1, _c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {
                1: [(b"f.txt", make_blob(b"aa"))],
                2: [(b"f.txt", make_blob(b"aaa"))],
                3: [(b"f.txt", make_blob(b"aaaa"))],
            },
        )
        self.assertPickaxe([c3, c1], [c3.id], string=b"aa")

    def test_exclude(self):
        c1, _c2, _c3, c4, c5 = self.make_history()
        results = list(pickaxe(self.store, [c5.id], exclude=[c4.id], string=b"alpha"))
        self.assertEqual([c5.id], [r.commit.id for r in results])
        results = list(pickaxe(self.store, [c5.id], exclude=[c1.id], string=b"alpha"))
        self.assertEqual([c5.id, c4.id], [r.commit.id for r in results])


class PickaxeRegexTest(PickaxeTestCase):
    """Tests for the regex-on-changed-lines mode."""

    def make_history(self):
        # c2 only moves a line around; c3 removes one matching line and
        # adds another; c4 changes a line that does not match while the
        # matching lines stay untouched; c5 adds a matching line.
        return self.make_commits(
            [[1], [2, 1], [3, 2], [4, 3], [5, 4]],
            {
                1: [(b"f.txt", make_blob(b"l1\nl2\nalpha\nl3\nl4\n"))],
                2: [(b"f.txt", make_blob(b"l1\nl2\nl3\nl4\nalpha\n"))],
                3: [(b"f.txt", make_blob(b"l1\nl2\nl3\nl4\nalpha gamma\n"))],
                4: [(b"f.txt", make_blob(b"L1\nl2\nl3\nl4\nalpha gamma\n"))],
                5: [(b"f.txt", make_blob(b"L1\nl2\nl3\nl4\nalpha gamma\nalpha\n"))],
            },
        )

    def test_matches_added_and_removed_lines(self):
        c1, c2, c3, _c4, c5 = self.make_history()
        # "alpha" appears on lines added/removed by c1, c2, c3 and c5.
        self.assertPickaxe([c5, c3, c2, c1], [c5.id], pattern=rb"alpha")

    def test_untouched_matching_content_does_not_match(self):
        # c4 still contains "alpha gamma", but its diff only touches the
        # "l1" line, so it must not match.
        _, _, c3, c4, _ = self.make_history()
        self.assertPickaxe([], [c4.id], exclude=[c3.id], pattern=rb"alpha")

    def test_anchored_pattern(self):
        _, _, c3, _, c5 = self.make_history()
        # Only c3 adds a line ending in "gamma"; c5 merely keeps it as
        # context, so it does not match.
        results = list(pickaxe(self.store, [c5.id], pattern=rb"gamma$"))
        self.assertEqual([c3.id], [r.commit.id for r in results])

    def test_str_pattern_accepted(self):
        c1, _, _, _, c5 = self.make_history()
        results = list(pickaxe(self.store, [c5.id], pattern="alpha"))
        self.assertIn(c1.id, [r.commit.id for r in results])

    def test_invalid_pattern_raises_before_walking(self):
        c1, _, _, _, _ = self.make_history()
        # The pattern is compiled before any history is traversed, so the
        # error surfaces even if the start revision does not exist.
        missing = b"0" * 40
        self.assertRaises(re.error, pickaxe, self.store, [missing], pattern=b"(")
        self.assertRaises(re.error, pickaxe, self.store, [c1.id], pattern=b"[")


class PickaxeChangeTypesTest(PickaxeTestCase):
    """Tests for adds, deletes, renames and missing trailing newlines."""

    def test_added_file(self):
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {
                1: [(b"f.txt", make_blob(b"alpha\n"))],
                2: [
                    (b"f.txt", make_blob(b"alpha\n")),
                    (b"g.txt", make_blob(b"more alpha\n")),
                ],
            },
        )
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"alpha")
        self.assertEqual(
            [(c1.id, b"g.txt")], [(m.parent, m.path) for m in results[0].matches]
        )

    def test_deleted_file(self):
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {
                1: [
                    (b"f.txt", make_blob(b"alpha\n")),
                    (b"g.txt", make_blob(b"alpha\n")),
                ],
                2: [(b"f.txt", make_blob(b"alpha\n"))],
            },
        )
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"alpha")
        # Deletions are reported with the old path.
        self.assertEqual(
            [(c1.id, b"g.txt")], [(m.parent, m.path) for m in results[0].matches]
        )

    def test_rename_without_content_change_does_not_match(self):
        blob = make_blob(b"alpha\nbeta\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {1: [(b"f.txt", blob)], 2: [(b"g.txt", blob)]},
        )
        self.assertPickaxe([], [c2.id], exclude=[c1.id], string=b"alpha")
        self.assertPickaxe([], [c2.id], exclude=[c1.id], pattern=rb"alpha")

    def test_rename_with_content_change_matches(self):
        # Without content-based rename detection this shows up as a
        # delete plus an add, both of which change the occurrence count.
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {
                1: [(b"f.txt", make_blob(b"alpha\nbeta\n"))],
                2: [(b"g.txt", make_blob(b"alpha\nbeta\nalpha again\n"))],
            },
        )
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"alpha")
        self.assertEqual(
            [(c1.id, b"f.txt"), (c1.id, b"g.txt")],
            [(m.parent, m.path) for m in results[0].matches],
        )

    def test_no_trailing_newline(self):
        _c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {
                1: [(b"f.txt", make_blob(b"alpha"))],
                2: [(b"f.txt", make_blob(b"alpha\nbeta"))],
                3: [(b"f.txt", make_blob(b"alpha"))],
            },
        )
        # The last line has no trailing newline in either revision.
        self.assertPickaxe([c2], [c2.id], string=b"beta")
        self.assertPickaxe([c2], [c2.id], pattern=rb"^beta$")
        self.assertPickaxe([c3, c2], [c3.id], string=b"beta")
        self.assertPickaxe([c3, c2], [c3.id], pattern=rb"^beta$")


class PickaxeMergeTest(PickaxeTestCase):
    """Tests for merge commit handling."""

    def make_merge(self, merge_tree):
        # c2 does unrelated work on the main line; c3 adds an occurrence
        # of the needle on the side branch; c4 merges the two.
        return self.make_commits(
            [[1], [2, 1], [3, 1], [4, 2, 3]],
            {
                1: [(b"f.txt", make_blob(b"alpha\nbeta\n"))],
                2: [
                    (b"f.txt", make_blob(b"alpha\nbeta\n")),
                    (b"other.txt", make_blob(b"main\n")),
                ],
                3: [(b"f.txt", make_blob(b"alpha\nbeta\nside alpha\n"))],
                4: merge_tree,
            },
        )

    def test_merge_first_parent(self):
        # The merge takes the side branch's change, so it differs from
        # its first parent.
        c1, c2, c3, c4 = self.make_merge(
            [
                (b"f.txt", make_blob(b"alpha\nbeta\nside alpha\n")),
                (b"other.txt", make_blob(b"main\n")),
            ]
        )
        results = self.assertPickaxe(
            [c4, c3, c1], [c4.id], string=b"alpha", merge=MERGE_FIRST
        )
        self.assertEqual(
            [(c2.id, b"f.txt")], [(m.parent, m.path) for m in results[0].matches]
        )

    def test_merge_all_parents(self):
        # The merge keeps the first parent's tree, so it only differs
        # from the second parent.
        c1, _c2, c3, c4 = self.make_merge(
            [
                (b"f.txt", make_blob(b"alpha\nbeta\n")),
                (b"other.txt", make_blob(b"main\n")),
            ]
        )
        self.assertPickaxe([c3, c1], [c4.id], string=b"alpha", merge=MERGE_FIRST)
        results = self.assertPickaxe(
            [c4, c3, c1], [c4.id], string=b"alpha", merge=MERGE_ALL
        )
        self.assertEqual(
            [(c3.id, b"f.txt")], [(m.parent, m.path) for m in results[0].matches]
        )

    def test_merge_differs_from_both_parents(self):
        c1, c2, c3, c4 = self.make_merge(
            [
                (b"f.txt", make_blob(b"alpha\nbeta\nside alpha\nmore alpha\n")),
                (b"other.txt", make_blob(b"main\n")),
            ]
        )
        results = self.assertPickaxe(
            [c4, c3, c1], [c4.id], string=b"alpha", merge=MERGE_ALL
        )
        # One result for the merge commit, listing both parents.
        self.assertEqual(
            [(c2.id, b"f.txt"), (c3.id, b"f.txt")],
            [(m.parent, m.path) for m in results[0].matches],
        )


class PickaxePathsTest(PickaxeTestCase):
    """Tests for path filtering."""

    def make_history(self):
        return self.make_commits(
            [[1], [2, 1]],
            {
                1: [
                    (b"a.txt", make_blob(b"foo\n")),
                    (b"dir/b.txt", make_blob(b"foo\n")),
                ],
                2: [
                    (b"a.txt", make_blob(b"foo\nfoo\n")),
                    (b"dir/b.txt", make_blob(b"foo\nfoo\nfoo\n")),
                ],
            },
        )

    def test_paths_filter(self):
        c1, c2 = self.make_history()
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"foo", paths=[b"a.txt"])
        for result in results:
            self.assertEqual([b"a.txt"], [m.path for m in result.matches])

    def test_paths_filter_directory(self):
        c1, c2 = self.make_history()
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"foo", paths=[b"dir"])
        for result in results:
            self.assertEqual([b"dir/b.txt"], [m.path for m in result.matches])

    def test_commit_not_duplicated_for_multiple_matching_files(self):
        c1, c2 = self.make_history()
        results = self.assertPickaxe([c2, c1], [c2.id], string=b"foo")
        self.assertEqual([b"a.txt", b"dir/b.txt"], [m.path for m in results[0].matches])

    def test_unrequested_blobs_not_loaded(self):
        store = RecordingStore()
        blob_a1 = make_blob(b"foo\n")
        blob_b1 = make_blob(b"bar\n")
        blob_a2 = make_blob(b"foo\nfoo\n")
        blob_b2 = make_blob(b"bar\nbar\nbar\n")
        c1, c2 = build_commit_graph(
            store,
            [[1], [2, 1]],
            trees={
                1: [(b"a.txt", blob_a1), (b"b.txt", blob_b1)],
                2: [(b"a.txt", blob_a2), (b"b.txt", blob_b2)],
            },
        )
        results = list(pickaxe(store, [c2.id], string=b"foo", paths=[b"a.txt"]))
        self.assertEqual([c2.id, c1.id], [r.commit.id for r in results])
        self.assertNotIn(blob_b1.id, store.loaded_blobs)
        self.assertNotIn(blob_b2.id, store.loaded_blobs)
        self.assertIn(blob_a1.id, store.loaded_blobs)
        self.assertIn(blob_a2.id, store.loaded_blobs)


class PickaxeValidationTest(PickaxeTestCase):
    def test_exactly_one_mode(self):
        (c1,) = self.make_commits([[1]], {1: [(b"f.txt", make_blob(b"x"))]})
        self.assertRaises(ValueError, pickaxe, self.store, [c1.id])
        self.assertRaises(
            ValueError,
            pickaxe,
            self.store,
            [c1.id],
            string=b"x",
            pattern=b"x",
        )

    def test_empty_string_rejected(self):
        (c1,) = self.make_commits([[1]], {1: [(b"f.txt", make_blob(b"x"))]})
        self.assertRaises(ValueError, pickaxe, self.store, [c1.id], string=b"")

    def test_unknown_merge_mode_rejected(self):
        (c1,) = self.make_commits([[1]], {1: [(b"f.txt", make_blob(b"x"))]})
        self.assertRaises(
            ValueError, pickaxe, self.store, [c1.id], string=b"x", merge="combined"
        )


class PickaxeOrderTest(PickaxeTestCase):
    def make_history(self):
        return self.make_commits(
            [[1], [2, 1], [3, 2]],
            {
                1: [(b"f.txt", make_blob(b"foo\n"))],
                2: [(b"f.txt", make_blob(b"foo\nfoo\n"))],
                3: [(b"f.txt", make_blob(b"foo\nfoo\nfoo\n"))],
            },
        )

    def test_walker_order(self):
        c1, c2, c3 = self.make_history()
        self.assertPickaxe([c3, c2, c1], [c3.id], string=b"foo")

    def test_reverse(self):
        c1, c2, c3 = self.make_history()
        results = list(pickaxe(self.store, [c3.id], string=b"foo", reverse=True))
        self.assertEqual([c1.id, c2.id, c3.id], [r.commit.id for r in results])

    def test_max_entries(self):
        _c1, c2, c3 = self.make_history()
        results = list(pickaxe(self.store, [c3.id], string=b"foo", max_entries=2))
        self.assertEqual([c3.id, c2.id], [r.commit.id for r in results])

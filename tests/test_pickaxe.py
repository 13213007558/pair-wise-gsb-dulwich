# test_pickaxe.py -- Tests for pickaxe-style history filtering.
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

"""Tests for pickaxe-style content filtering of commit history."""

import re

from dulwich.diff_tree import RenameDetector
from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob
from dulwich.pickaxe import (
    BinaryPolicy,
    DiffRegexFilter,
    OccurrenceFilter,
)
from dulwich.repo import MemoryRepo
from dulwich.tests.utils import build_commit_graph, make_object
from dulwich.walk import Walker

from . import TestCase


class PickaxeWalkerTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = MemoryObjectStore()

    def make_commits(self, commit_spec, **kwargs):
        return build_commit_graph(self.store, commit_spec, **kwargs)

    def walk(self, include, pickaxe, **kwargs):
        return list(Walker(self.store, include, pickaxe=pickaxe, **kwargs))

    def walk_ids(self, include, pickaxe, **kwargs):
        return [entry.commit.id for entry in self.walk(include, pickaxe, **kwargs)]

    def test_occurrence_count_added(self) -> None:
        blob1 = make_object(Blob, data=b"hello\n")
        blob2 = make_object(Blob, data=b"hello needle\nhello\n")
        blob3 = make_object(Blob, data=b"hello needle\nworld\n")
        c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            trees={
                1: [(b"a", blob1)],
                2: [(b"a", blob2)],
                3: [(b"a", blob3)],
            },
        )
        # Only c2 changes the occurrence count of the needle.
        entries = self.walk([c3.id], OccurrenceFilter(b"needle"))
        self.assertEqual([c2.id], [e.commit.id for e in entries])
        (match,) = entries[0].pickaxe_matches
        self.assertEqual(c1.id, match.parent)
        self.assertEqual([b"a"], [c.new.path for c in match.changes])

    def test_occurrence_count_removed(self) -> None:
        blob1 = make_object(Blob, data=b"needle\nneedle\n")
        blob2 = make_object(Blob, data=b"needle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        # c2 removes an occurrence; the root commit c1 introduces them.
        self.assertEqual(
            [c2.id, c1.id], self.walk_ids([c2.id], OccurrenceFilter(b"needle"))
        )

    def test_occurrence_count_move_no_match(self) -> None:
        blob1 = make_object(Blob, data=b"line1\nneedle here\nline2\n")
        blob2 = make_object(Blob, data=b"needle here\nline1\nline2\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        # The text was only moved to another line; the occurrence count
        # is unchanged, so the count filter does not match c2. The root
        # commit c1 introduces the text and does match.
        self.assertEqual([c1.id], self.walk_ids([c2.id], OccurrenceFilter(b"needle")))

    def test_diff_regex_move_matches(self) -> None:
        blob1 = make_object(Blob, data=b"line1\nneedle here\nline2\n")
        blob2 = make_object(Blob, data=b"needle here\nline1\nline2\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        # The moved line is removed and re-added, so the changed-line
        # regex filter matches c2 (and the root commit c1).
        self.assertEqual(
            [c2.id, c1.id], self.walk_ids([c2.id], DiffRegexFilter(b"needle"))
        )

    def test_diff_regex_matches_changed_lines_only(self) -> None:
        blob1 = make_object(Blob, data=b"needle\nfoo\n")
        blob2 = make_object(Blob, data=b"needle\nbar\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        # The line containing the needle is unchanged in c2; only the
        # root commit c1 matches.
        self.assertEqual([c1.id], self.walk_ids([c2.id], DiffRegexFilter(b"needle")))
        # A pattern matching the changed line does match.
        self.assertEqual([c2.id], self.walk_ids([c2.id], DiffRegexFilter(b"^bar$")))

    def test_diff_regex_compiled_pattern(self) -> None:
        blob1 = make_object(Blob, data=b"foo\n")
        blob2 = make_object(Blob, data=b"foo\nbar123\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        pattern = re.compile(b"bar[0-9]+")
        self.assertEqual([c2.id], self.walk_ids([c2.id], DiffRegexFilter(pattern)))

    def test_root_commit(self) -> None:
        blob1 = make_object(Blob, data=b"needle\n")
        (c1,) = self.make_commits([[1]], trees={1: [(b"a", blob1)]})
        entries = self.walk([c1.id], OccurrenceFilter(b"needle"))
        self.assertEqual([c1.id], [e.commit.id for e in entries])
        # A root commit is compared against the empty tree.
        (match,) = entries[0].pickaxe_matches
        self.assertIsNone(match.parent)

    def test_empty_commit(self) -> None:
        blob1 = make_object(Blob, data=b"needle\n")
        c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            trees={
                1: [(b"a", blob1)],
                2: [(b"a", blob1)],
                3: [(b"a", blob1)],
            },
        )
        # Only the root commit c1, which introduces the needle, matches.
        self.assertEqual([c1.id], self.walk_ids([c3.id], OccurrenceFilter(b"needle")))
        self.assertEqual([c1.id], self.walk_ids([c3.id], DiffRegexFilter(b"needle")))

    def test_rename_unchanged_content(self) -> None:
        blob1 = make_object(Blob, data=b"needle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a.txt", blob1)], 2: [(b"b.txt", blob1)]},
        )
        rename_detector = RenameDetector(self.store)
        # With rename detection, a pure rename has no content change and
        # does not match either filter; only the root commit c1, which
        # introduces the needle, matches.
        self.assertEqual(
            [c1.id],
            self.walk_ids(
                [c2.id],
                OccurrenceFilter(b"needle"),
                rename_detector=rename_detector,
            ),
        )
        self.assertEqual(
            [c1.id],
            self.walk_ids(
                [c2.id],
                DiffRegexFilter(b"needle"),
                rename_detector=RenameDetector(self.store),
            ),
        )
        # Without rename detection the rename shows up as a delete plus
        # an add, both of which change the occurrence count.
        self.assertEqual(
            [c2.id, c1.id], self.walk_ids([c2.id], OccurrenceFilter(b"needle"))
        )

    def test_merge_matches_single_parent(self) -> None:
        blob1 = make_object(Blob, data=b"x\n")
        blob2 = make_object(Blob, data=b"x needle\n")
        c1, c2, c3, c4 = self.make_commits(
            [[1], [2, 1], [3, 1], [4, 2, 3]],
            trees={
                1: [(b"a", blob1)],
                2: [(b"a", blob2)],
                3: [(b"a", blob1)],
                4: [(b"a", blob2)],
            },
        )
        entries = self.walk([c4.id], OccurrenceFilter(b"needle"))
        self.assertEqual([c4.id, c2.id], [e.commit.id for e in entries])
        # The merge only differs from its second parent.
        (match,) = entries[0].pickaxe_matches
        self.assertEqual(c3.id, match.parent)
        # The branch commit differs from the root commit.
        (match,) = entries[1].pickaxe_matches
        self.assertEqual(c1.id, match.parent)

    def test_merge_matches_multiple_parents(self) -> None:
        blob1 = make_object(Blob, data=b"x\n")
        blob2 = make_object(Blob, data=b"y\n")
        blob3 = make_object(Blob, data=b"needle\n")
        c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 1, 2]],
            trees={
                1: [(b"a", blob1)],
                2: [(b"a", blob2)],
                3: [(b"a", blob3)],
            },
        )
        entries = self.walk([c3.id], OccurrenceFilter(b"needle"))
        self.assertEqual([c3.id], [e.commit.id for e in entries])
        # The merge introduces the needle relative to both parents.
        self.assertEqual([c1.id, c2.id], [m.parent for m in entries[0].pickaxe_matches])

    def test_paths_combined(self) -> None:
        blob1 = make_object(Blob, data=b"foo\n")
        blob2 = make_object(Blob, data=b"foo needle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={
                1: [(b"a.txt", blob1), (b"b.txt", blob1)],
                2: [(b"a.txt", blob2), (b"b.txt", blob2)],
            },
        )
        entries = self.walk([c2.id], OccurrenceFilter(b"needle"), paths=[b"a.txt"])
        self.assertEqual([c2.id], [e.commit.id for e in entries])
        (match,) = entries[0].pickaxe_matches
        self.assertEqual([b"a.txt"], [c.new.path for c in match.changes])
        # No matching change under this path.
        self.assertEqual(
            [],
            self.walk_ids([c2.id], OccurrenceFilter(b"needle"), paths=[b"c.txt"]),
        )

    def test_binary_skip_by_default(self) -> None:
        blob1 = make_object(Blob, data=b"\x00binary\n")
        blob2 = make_object(Blob, data=b"\x00binary needle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        self.assertEqual([], self.walk_ids([c2.id], OccurrenceFilter(b"needle")))
        self.assertEqual([], self.walk_ids([c2.id], DiffRegexFilter(b"needle")))

    def test_binary_match_policy(self) -> None:
        blob1 = make_object(Blob, data=b"\x00binary\n")
        blob2 = make_object(Blob, data=b"\x00binary needle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        self.assertEqual(
            [c2.id],
            self.walk_ids([c2.id], OccurrenceFilter(b"needle", BinaryPolicy.MATCH)),
        )
        self.assertEqual(
            [c2.id],
            self.walk_ids([c2.id], DiffRegexFilter(b"needle", BinaryPolicy.MATCH)),
        )

    def test_lazy_iteration(self) -> None:
        blob1 = make_object(Blob, data=b"needle\n")
        c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob1)], 3: [(b"a", blob1)]},
        )
        walker = Walker(self.store, [c3.id], pickaxe=OccurrenceFilter(b"needle"))
        entries = iter(walker)
        # Iteration is lazy; entries are produced one at a time.
        self.assertFalse(isinstance(entries, list))
        entry = next(entries)
        self.assertEqual(c1.id, entry.commit.id)
        self.assertRaises(StopIteration, next, entries)

    def test_get_walker_passes_pickaxe(self) -> None:
        repo = MemoryRepo()
        blob1 = make_object(Blob, data=b"foo\n")
        blob2 = make_object(Blob, data=b"foo needle\n")
        c1, c2 = build_commit_graph(
            repo.object_store,
            [[1], [2, 1]],
            trees={1: [(b"a", blob1)], 2: [(b"a", blob2)]},
        )
        entries = list(repo.get_walker([c2.id], pickaxe=OccurrenceFilter(b"needle")))
        self.assertEqual([c2.id], [e.commit.id for e in entries])
        (match,) = entries[0].pickaxe_matches
        self.assertEqual(c1.id, match.parent)


class OccurrenceFilterTest(TestCase):
    def test_empty_needle_rejected(self) -> None:
        self.assertRaises(ValueError, OccurrenceFilter, b"")

    def test_str_needle_encoded(self) -> None:
        filter = OccurrenceFilter("needle")
        self.assertEqual(b"needle", filter.needle)


class DiffRegexFilterTest(TestCase):
    def test_str_pattern_encoded(self) -> None:
        filter = DiffRegexFilter("needle")
        self.assertEqual(b"needle", filter.pattern.pattern)

# test_pickaxe.py -- Tests for pickaxe history search.
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

"""Tests for pickaxe-style history search."""

import os
import re
import tempfile

from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob
from dulwich.pickaxe import MERGE_ALL, MERGE_FIRST_PARENT, pickaxe
from dulwich.repo import Repo
from dulwich.tests.utils import build_commit_graph, make_object

from . import TestCase
from .compat.utils import require_git_version, rmtree_ro, run_git_or_fail


def _blob(data):
    return make_object(Blob, data=data)


class _NoAccessStore(MemoryObjectStore):
    """Store that fails on any object access."""

    def __getitem__(self, sha):
        raise AssertionError("object store accessed before traversal")


class _TrackingStore:
    """Store proxy recording which objects get accessed."""

    def __init__(self, base):
        self._base = base
        self.accessed = []

    def __getitem__(self, sha):
        self.accessed.append(sha)
        return self._base[sha]

    def __getattr__(self, name):
        return getattr(self._base, name)


class PickaxeTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = MemoryObjectStore()

    def make_commits(self, spec, trees):
        return build_commit_graph(self.store, spec, trees=trees)

    def pickaxe_ids(self, *args, **kwargs):
        return [m.commit.id for m in pickaxe(self.store, *args, **kwargs)]

    def test_string_counts_occurrences(self):
        b1 = _blob(b"alpha\nkey = on\nomega\n")
        b2 = _blob(b"key = on\nalpha\nomega\n")  # moved, count unchanged
        b3 = _blob(b"key = off\nalpha\nkey = on\n")  # one add, one removal
        b4 = _blob(b"key = off\nalpha\nkey = on\nkey = on\n")  # real increase
        c1, _c2, _c3, c4, c5 = self.make_commits(
            [[1], [2, 1], [3, 2], [4, 3], [5, 4]],
            {
                1: [(b"c.txt", b1)],
                2: [(b"c.txt", b2)],
                3: [(b"c.txt", b3)],
                4: [(b"c.txt", b4)],
                5: [],  # file deleted
            },
        )
        self.assertEqual(
            [c5.id, c4.id, c1.id],
            self.pickaxe_ids(c5.id, string=b"key = on"),
        )

    def test_string_move_not_matched(self):
        b1 = _blob(b"alpha\nkey = on\nomega\n")
        b2 = _blob(b"key = on\nalpha\nomega\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]], {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)]}
        )
        # Only the root commit (which adds the string) matches.
        self.assertEqual(
            [c1.id], self.pickaxe_ids(c2.id, string=b"key = on", paths=[b"c.txt"])
        )
        # The moved line does show up in the line diff, so -G style matches.
        self.assertEqual(
            [c2.id, c1.id],
            self.pickaxe_ids(c2.id, regex=b"^key = on$", paths=[b"c.txt"]),
        )

    def test_regex_matches_added_and_removed_lines(self):
        # One matching line is removed and another added, so the
        # occurrence count cancels out but the diff lines match.
        b1 = _blob(b"alpha\nkey = on\nomega\n")
        b2 = _blob(b"alpha\nkey = once\nomega\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]], {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)]}
        )
        self.assertEqual(
            [c1.id], self.pickaxe_ids(c2.id, string=b"key = on", paths=[b"c.txt"])
        )
        self.assertEqual(
            [c2.id, c1.id], self.pickaxe_ids(c2.id, regex=b"key = on", paths=[b"c.txt"])
        )

    def test_regex_does_not_search_unchanged_content(self):
        b1 = _blob(b"needle\nkeep\n")
        b2 = _blob(b"needle\nchanged\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]], {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)]}
        )
        # "needle" is present in both versions but its line is untouched,
        # so only the root commit that introduces it matches.
        self.assertEqual([c1.id], self.pickaxe_ids(c2.id, regex=b"needle"))
        self.assertEqual([c2.id, c1.id], self.pickaxe_ids(c2.id, regex=b"keep|changed"))

    def test_pure_rename_not_matched(self):
        blob = _blob(b"key = on\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {1: [(b"a.txt", blob)], 2: [(b"b.txt", blob)]},
        )
        # Only the root commit (which adds the string) matches.
        self.assertEqual([c1.id], self.pickaxe_ids(c2.id, string=b"key = on"))
        self.assertEqual([c1.id], self.pickaxe_ids(c2.id, regex=b"key = on"))

    def test_no_trailing_newline(self):
        b1 = _blob(b"key = on")  # no trailing newline
        b2 = _blob(b"key = on\nother")  # still no trailing newline
        b3 = _blob(b"key = on\nother\nkey = on")
        c1, _c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)], 3: [(b"c.txt", b3)]},
        )
        self.assertEqual([c3.id, c1.id], self.pickaxe_ids(c3.id, string=b"key = on"))
        self.assertEqual([c3.id, c1.id], self.pickaxe_ids(c3.id, regex=b"^key = on$"))

    def test_deleted_file_matched(self):
        b1 = _blob(b"key = on\n")
        c1, c2 = self.make_commits([[1], [2, 1]], {1: [(b"c.txt", b1)], 2: []})
        results = list(pickaxe(self.store, c2.id, string=b"key = on"))
        self.assertEqual([c2.id, c1.id], [m.commit.id for m in results])
        self.assertEqual(
            [(c1.id, b"c.txt")], [(m.parent, m.path) for m in results[0].matches]
        )

    def test_paths_filter(self):
        b1 = _blob(b"key = on\n")
        b2 = _blob(b"key = on\nkey = on\n")
        c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {
                1: [(b"a.txt", b1), (b"b.txt", b1)],
                2: [(b"a.txt", b2), (b"b.txt", b1)],
                3: [(b"a.txt", b2), (b"b.txt", b2)],
            },
        )
        self.assertEqual(
            [c2.id, c1.id],
            self.pickaxe_ids(c3.id, string=b"key = on", paths=[b"a.txt"]),
        )
        self.assertEqual(
            [c3.id, c2.id, c1.id],
            self.pickaxe_ids(c3.id, string=b"key = on"),
        )

    def test_paths_filter_directory(self):
        b1 = _blob(b"key = on\n")
        b2 = _blob(b"key = on\nkey = on\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {
                1: [(b"dir/c.txt", b1), (b"other.txt", b1)],
                2: [(b"dir/c.txt", b2), (b"other.txt", b1)],
            },
        )
        self.assertEqual(
            [c2.id, c1.id],
            self.pickaxe_ids(c2.id, string=b"key = on", paths=[b"dir"]),
        )

    def test_unrequested_blobs_not_loaded(self):
        wanted1 = _blob(b"key = on\n")
        wanted2 = _blob(b"key = on\nkey = on\n")
        unwanted1 = _blob(b"key = on\n" * 100)
        unwanted2 = _blob(b"key = on\n" * 3)
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {
                1: [(b"wanted.txt", wanted1), (b"unwanted.txt", unwanted1)],
                2: [(b"wanted.txt", wanted2), (b"unwanted.txt", unwanted2)],
            },
        )
        tracking = _TrackingStore(self.store)
        results = list(
            pickaxe(tracking, c2.id, string=b"key = on", paths=[b"wanted.txt"])
        )
        self.assertEqual([c2.id, c1.id], [m.commit.id for m in results])
        self.assertNotIn(unwanted1.id, tracking.accessed)
        self.assertNotIn(unwanted2.id, tracking.accessed)

    def test_merge_first_parent(self):
        base_a = _blob(b"base\n")
        base_b = _blob(b"one\n")
        needle_a = _blob(b"base\nneedle\n")
        needle_b = _blob(b"one\nneedle\n")
        _c1, c2, c3, c4 = self.make_commits(
            [[1], [2, 1], [3, 1], [4, 2, 3]],
            {
                1: [(b"a.txt", base_a), (b"b.txt", base_b)],
                2: [(b"a.txt", needle_a), (b"b.txt", base_b)],
                3: [(b"a.txt", base_a), (b"b.txt", needle_b)],
                4: [(b"a.txt", needle_a), (b"b.txt", needle_b)],
            },
        )
        results = {m.commit.id: m for m in pickaxe(self.store, c4.id, string=b"needle")}
        self.assertEqual([c4.id, c3.id, c2.id], [m.commit.id for m in results.values()])
        # The merge introduces b.txt relative to its first parent only.
        self.assertEqual(
            [(c2.id, b"b.txt")], [(m.parent, m.path) for m in results[c4.id].matches]
        )

    def test_merge_all_parents(self):
        base_a = _blob(b"base\n")
        base_b = _blob(b"one\n")
        needle_a = _blob(b"base\nneedle\n")
        needle_b = _blob(b"one\nneedle\n")
        _c1, c2, c3, c4 = self.make_commits(
            [[1], [2, 1], [3, 1], [4, 2, 3]],
            {
                1: [(b"a.txt", base_a), (b"b.txt", base_b)],
                2: [(b"a.txt", needle_a), (b"b.txt", base_b)],
                3: [(b"a.txt", base_a), (b"b.txt", needle_b)],
                4: [(b"a.txt", needle_a), (b"b.txt", needle_b)],
            },
        )
        results = {
            m.commit.id: m
            for m in pickaxe(self.store, c4.id, string=b"needle", merge_mode=MERGE_ALL)
        }
        # The merge matches against both parents, but is returned once.
        self.assertEqual(
            [(c2.id, b"b.txt"), (c3.id, b"a.txt")],
            [(m.parent, m.path) for m in results[c4.id].matches],
        )

    def test_multiple_files_single_entry(self):
        blob = _blob(b"needle\n")
        (c1,) = self.make_commits([[1]], {1: [(b"a.txt", blob), (b"b.txt", blob)]})
        results = list(pickaxe(self.store, c1.id, string=b"needle"))
        self.assertEqual(1, len(results))
        self.assertEqual(
            [(None, b"a.txt"), (None, b"b.txt")],
            [(m.parent, m.path) for m in results[0].matches],
        )

    def test_max_entries(self):
        b1 = _blob(b"needle\n")
        b2 = _blob(b"needle\nneedle\n")
        b3 = _blob(b"needle\nneedle\nneedle\n")
        _c1, _c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)], 3: [(b"c.txt", b3)]},
        )
        self.assertEqual(
            [c3.id], self.pickaxe_ids(c3.id, string=b"needle", max_entries=1)
        )

    def test_reverse(self):
        b1 = _blob(b"needle\n")
        b2 = _blob(b"needle\nneedle\n")
        c1, c2 = self.make_commits(
            [[1], [2, 1]],
            {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)]},
        )
        self.assertEqual(
            [c1.id, c2.id],
            self.pickaxe_ids(c2.id, string=b"needle", reverse=True),
        )

    def test_exclude(self):
        b1 = _blob(b"needle\n")
        b2 = _blob(b"needle\nneedle\n")
        b3 = _blob(b"needle\nneedle\nneedle\n")
        _c1, c2, c3 = self.make_commits(
            [[1], [2, 1], [3, 2]],
            {1: [(b"c.txt", b1)], 2: [(b"c.txt", b2)], 3: [(b"c.txt", b3)]},
        )
        self.assertEqual(
            [c3.id],
            self.pickaxe_ids(c3.id, exclude=[c2.id], string=b"needle"),
        )

    def test_string_and_regex_mutually_exclusive(self):
        store = _NoAccessStore()
        self.assertRaises(ValueError, pickaxe, store, [b"1" * 40])
        self.assertRaises(
            ValueError, pickaxe, store, [b"1" * 40], string=b"a", regex=b"b"
        )
        self.assertRaises(ValueError, pickaxe, store, [b"1" * 40], string=b"")

    def test_invalid_merge_mode(self):
        store = _NoAccessStore()
        self.assertRaises(
            ValueError, pickaxe, store, [b"1" * 40], string=b"a", merge_mode="bogus"
        )

    def test_invalid_regex_raises_before_traversal(self):
        store = _NoAccessStore()
        self.assertRaises(re.error, pickaxe, store, [b"1" * 40], regex=b"(")


class PickaxeGitCompatTest(TestCase):
    """Compare pickaxe results against git log -S/-G on a real repo."""

    KEY = b"key = on"

    def setUp(self) -> None:
        super().setUp()
        require_git_version((2, 28))
        self.repo_dir = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, self.repo_dir)
        self._commit_time = 1700000000
        self.git("init", "-q", "-b", "main")

    def git(self, *args, env=None):
        return run_git_or_fail(list(args), cwd=self.repo_dir, env=env or {})

    def write(self, name, data):
        with open(os.path.join(self.repo_dir, name), "wb") as f:
            f.write(data)

    def commit(self, message):
        self._commit_time += 60
        date = f"@{self._commit_time} +0000"
        self.git("add", "-A")
        self.git(
            "commit",
            "-qm",
            message,
            env={"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date},
        )
        return self.git("rev-parse", "HEAD").strip().decode("ascii")

    def git_log_pickaxe(self, *args):
        out = self.git("log", "--format=%H", *args)
        # -m may list a merge commit once per parent; deduplicate.
        result = []
        for line in out.decode("ascii").splitlines():
            if line and line not in result:
                result.append(line)
        return result

    def build_repo(self):
        ids = {}
        self.write("config.txt", b"alpha\nbeta\nkey = on\ngamma\nomega\n")
        ids["add"] = self.commit("add config")
        # Move the line within the file; occurrence count is unchanged.
        self.write("config.txt", b"alpha\nbeta\ngamma\nomega\nkey = on\n")
        ids["move"] = self.commit("move line")
        # Add one occurrence and remove another; the count cancels out.
        self.write("config.txt", b"alpha\nkey = on\ngamma\nomega\nzeta\n")
        ids["cancel"] = self.commit("cancel out")
        # Genuinely add another occurrence.
        self.write("config.txt", b"alpha\nkey = on\ngamma\nomega\nzeta\nkey = on\n")
        ids["increase"] = self.commit("real increase")
        # Pure rename, content unchanged.
        self.git("mv", "config.txt", "settings.txt")
        ids["rename"] = self.commit("rename file")
        # Append an occurrence without a trailing newline.
        self.write(
            "settings.txt", b"alpha\nkey = on\ngamma\nomega\nzeta\nkey = on\nkey = on"
        )
        ids["no_newline"] = self.commit("no trailing newline")
        # Merge two branches with different changes on each side.
        self.git("checkout", "-q", "-b", "side")
        self.write("side.txt", b"key = on\n")
        ids["side"] = self.commit("side add")
        self.git("checkout", "-q", "main")
        self.write("other.txt", b"unrelated\n")
        ids["main2"] = self.commit("main work")
        self.git("merge", "-q", "--no-ff", "side", "-m", "merge side")
        ids["merge"] = self.git("rev-parse", "HEAD").strip().decode("ascii")
        return ids

    def open_repo(self):
        repo = Repo(self.repo_dir)
        self.addCleanup(repo.close)
        return repo

    def test_string_matches_git(self):
        self.build_repo()
        repo = self.open_repo()
        expected = self.git_log_pickaxe("-m", "-Skey = on")
        results = pickaxe(
            repo.object_store, repo.head(), string=self.KEY, merge_mode=MERGE_ALL
        )
        self.assertEqual(expected, [m.commit.id.decode("ascii") for m in results])

    def test_regex_matches_git(self):
        self.build_repo()
        repo = self.open_repo()
        expected = self.git_log_pickaxe("-m", "-Gkey = on")
        results = pickaxe(
            repo.object_store, repo.head(), regex=self.KEY, merge_mode=MERGE_ALL
        )
        self.assertEqual(expected, [m.commit.id.decode("ascii") for m in results])

    def test_string_first_parent(self):
        ids = self.build_repo()
        repo = self.open_repo()
        results = {
            m.commit.id.decode("ascii"): m
            for m in pickaxe(
                repo.object_store,
                repo.head(),
                string=self.KEY,
                merge_mode=MERGE_FIRST_PARENT,
            )
        }
        # The merge introduces side.txt relative to its first parent.
        merge_match = results[ids["merge"]]
        self.assertEqual(
            [(ids["main2"], "side.txt")],
            [(p.decode("ascii"), pa.decode("ascii")) for p, pa in merge_match.matches],
        )
        # Non-merge commits are unaffected by the merge mode.
        all_ids = {
            m.commit.id.decode("ascii")
            for m in pickaxe(
                repo.object_store, repo.head(), string=self.KEY, merge_mode=MERGE_ALL
            )
        }
        self.assertEqual(all_ids, set(results))

    def test_string_paths_filter_matches_git(self):
        self.build_repo()
        repo = self.open_repo()
        expected = self.git_log_pickaxe(
            "-m", "-Skey = on", "--", "config.txt", "settings.txt"
        )
        results = pickaxe(
            repo.object_store,
            repo.head(),
            string=self.KEY,
            paths=[b"config.txt", b"settings.txt"],
            merge_mode=MERGE_ALL,
        )
        self.assertEqual(expected, [m.commit.id.decode("ascii") for m in results])

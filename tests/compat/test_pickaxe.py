# test_pickaxe.py -- Compatibility tests for pickaxe queries.
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

"""Compatibility tests for pickaxe queries against C git."""

import os
import re
import shutil
import tempfile

from dulwich.pickaxe import MERGE_ALL, MERGE_FIRST, pickaxe
from dulwich.repo import Repo

from .utils import CompatTestCase, run_git_or_fail

_HEX_RE = re.compile(r"^[0-9a-f]{40}$")


class CompatPickaxeTestCase(CompatTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        self.repo_path = os.path.join(self.test_dir, "repo")
        run_git_or_fail(["init", "-q", "-b", "main", self.repo_path])
        self.repo = Repo(self.repo_path)
        self.addCleanup(self.repo.close)
        self._commit_time = 1700000000

    def _git(self, *args, **kwargs):
        return run_git_or_fail(list(args), cwd=self.repo_path, **kwargs)

    def _write(self, name, content):
        path = os.path.join(self.repo_path, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def _next_date_env(self):
        self._commit_time += 100
        date = f"{self._commit_time} +0000"
        return {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}

    def _commit(self, message):
        self._git("add", "-A")
        self._git("commit", "-qm", message, env=self._next_date_env())

    def _commit_file(self, name, content, message):
        self._write(name, content)
        self._commit(message)

    def _git_log_ids(self, *args):
        out = self._git("log", "--format=%H", *args).decode()
        return [line for line in out.splitlines() if _HEX_RE.match(line)]

    def _pickaxe_ids(self, **kwargs):
        results = pickaxe(self.repo.object_store, [self.repo.head()], **kwargs)
        return [r.commit.id.decode("ascii") for r in results]

    def _pickaxe_results(self, **kwargs):
        return list(pickaxe(self.repo.object_store, [self.repo.head()], **kwargs))


class PickaxeStringCompatTest(CompatPickaxeTestCase):
    def make_history(self):
        # move: text moves around, occurrence count unchanged.
        # cancel: a line with the needle is removed and another added,
        # so the count is unchanged.
        # increase: another occurrence is added.
        self._commit_file("f.txt", b"l1\nl2\nalpha\nl3\nl4\n", "base")
        self._commit_file("f.txt", b"l1\nl2\nl3\nl4\nalpha\n", "move")
        self._commit_file("f.txt", b"l1\nl2\nl3\nl4\nalpha gamma\n", "cancel")
        self._commit_file("f.txt", b"l1\nl2\nl3\nl4\nalpha gamma\nalpha\n", "increase")

    def test_string_mode_matches_git(self):
        self.make_history()
        expected = self._git_log_ids("-S", "alpha")
        self.assertEqual(expected, self._pickaxe_ids(string=b"alpha"))
        # The moved and cancelled-out commits must not show up.
        messages = self._git("log", "--format=%H %s").decode().splitlines()
        by_id = dict(line.split(" ", 1) for line in messages)
        for subject in ("move", "cancel"):
            commit_id = next(k for k, v in by_id.items() if v == subject)
            self.assertNotIn(commit_id, expected)

    def test_regex_mode_matches_git(self):
        self.make_history()
        expected = self._git_log_ids("-G", "alpha")
        self.assertEqual(expected, self._pickaxe_ids(pattern=rb"alpha"))

    def test_regex_anchored_matches_git(self):
        self.make_history()
        expected = self._git_log_ids("-G", "^alpha$")
        self.assertEqual(expected, self._pickaxe_ids(pattern=rb"^alpha$"))

    def test_paths_filter_matches_git(self):
        self._commit_file("a.txt", b"foo\n", "base a")
        self._commit_file("dir/b.txt", b"foo\n", "base b")
        self._commit_file("a.txt", b"foo\nfoo\n", "more a")
        self._commit_file("dir/b.txt", b"foo\nfoo\nfoo\n", "more b")
        expected = self._git_log_ids("-S", "foo", "--", "a.txt")
        self.assertEqual(expected, self._pickaxe_ids(string=b"foo", paths=[b"a.txt"]))
        expected = self._git_log_ids("-S", "foo", "--", "dir")
        self.assertEqual(expected, self._pickaxe_ids(string=b"foo", paths=[b"dir"]))

    def test_revision_range_matches_git(self):
        self.make_history()
        stop = self._git("rev-parse", "HEAD~2").decode().strip()
        expected = self._git_log_ids("-S", "alpha", "HEAD~2..HEAD")
        self.assertEqual(
            expected,
            self._pickaxe_ids(string=b"alpha", exclude=[stop.encode("ascii")]),
        )


class PickaxeNoNewlineCompatTest(CompatPickaxeTestCase):
    def make_history(self):
        self._commit_file("f.txt", b"alpha", "no newline")
        self._commit_file("f.txt", b"alpha\nbeta", "add line, still no newline")
        self._commit_file("f.txt", b"alpha\nbeta\n", "terminate last line")

    def test_string_mode_matches_git(self):
        self.make_history()
        expected = self._git_log_ids("-S", "beta")
        self.assertEqual(expected, self._pickaxe_ids(string=b"beta"))

    def test_regex_mode_matches_git(self):
        self.make_history()
        expected = self._git_log_ids("-G", "^beta$")
        self.assertEqual(expected, self._pickaxe_ids(pattern=rb"^beta$"))

    def test_regex_mode_matches_git_for_newline_change(self):
        # A line whose newline termination changed counts as
        # removed and re-added.
        self.make_history()
        expected = self._git_log_ids("-G", "^alpha$")
        self.assertEqual(expected, self._pickaxe_ids(pattern=rb"^alpha$"))


class PickaxeRenameCompatTest(CompatPickaxeTestCase):
    def test_rename_matches_git(self):
        self._commit_file("f.txt", b"alpha\nbeta\n", "base")
        self._git("mv", "f.txt", "g.txt")
        self._commit("pure rename")
        self._commit_file("g.txt", b"alpha\nbeta\nalpha\n", "modify after rename")
        expected = self._git_log_ids("-S", "alpha")
        self.assertEqual(expected, self._pickaxe_ids(string=b"alpha"))
        expected = self._git_log_ids("-G", "alpha")
        self.assertEqual(expected, self._pickaxe_ids(pattern=rb"alpha"))


class PickaxeMergeCompatTest(CompatPickaxeTestCase):
    def make_merge(self):
        # A merge that keeps its first parent's tree ("ours"): it
        # introduces no change relative to the first parent, but
        # discards an occurrence of the needle relative to the second.
        self._commit_file("f.txt", b"alpha\nbeta\n", "base")
        self._commit_file("other.txt", b"main\n", "main work")
        self._git("checkout", "-q", "-b", "side", "HEAD~1")
        self._commit_file("f.txt", b"alpha\nbeta\nside alpha\n", "side add")
        side_id = self._git("rev-parse", "HEAD").decode().strip()
        self._git("checkout", "-q", "main")
        self._git(
            "merge", "-q", "-s", "ours", "--no-edit", "side", env=self._next_date_env()
        )
        return side_id

    def test_merge_first_parent_matches_git(self):
        self.make_merge()
        expected = self._git_log_ids("--diff-merges=first-parent", "-S", "alpha")
        self.assertEqual(
            expected, self._pickaxe_ids(string=b"alpha", merge=MERGE_FIRST)
        )

    def test_merge_all_parents_matches_git(self):
        side_id = self.make_merge()
        # git -m lists a merge commit once per matching parent; dulwich
        # lists it once with all matching parents attached.
        expected = []
        for commit_id in self._git_log_ids("-m", "-S", "alpha"):
            if commit_id not in expected:
                expected.append(commit_id)
        results = self._pickaxe_results(string=b"alpha", merge=MERGE_ALL)
        self.assertEqual(expected, [r.commit.id.decode("ascii") for r in results])
        # The merge commit is reported with the side parent it differs
        # from.
        merge_result = results[0]
        self.assertEqual(2, len(merge_result.commit.parents))
        self.assertEqual(
            [(side_id.encode("ascii"), b"f.txt")],
            [(m.parent, m.path) for m in merge_result.matches],
        )

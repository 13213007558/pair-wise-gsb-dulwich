# tests/test_replace.py -- Tests for refs/replace/* replacement views
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

"""Tests for refs/replace/* replacement views."""

import shutil
import tempfile

from dulwich.graph import find_merge_base
from dulwich.objects import Blob, Commit, ObjectID, Tree
from dulwich.replace import (
    ReplacementCycleError,
    ReplacementMissingError,
    ReplacementObjectStore,
    ReplacementTypeMismatchError,
    parse_replace_refs,
)
from dulwich.repo import MemoryRepo, Repo

from . import TestCase


def _make_tree(repo, entries):
    tree = Tree()
    for name, data in sorted(entries.items()):
        blob = Blob()
        blob.data = data
        repo.object_store.add_object(blob)
        tree.add(name, 0o100644, blob.id)
    repo.object_store.add_object(tree)
    return tree


def _make_commit(repo, message, tree, parents=(), timestamp=100):
    commit = Commit()
    commit.message = message
    commit.committer = b"Test Committer <test@nodomain.com>"
    commit.author = b"Test Author <test@nodomain.com>"
    commit.commit_time = timestamp
    commit.commit_timezone = 0
    commit.author_time = timestamp
    commit.author_timezone = 0
    commit.tree = tree.id
    commit.parents = list(parents)
    repo.object_store.add_object(commit)
    return commit


class ReplaceRepoMixin:
    """Build a repository where commit A is replaced by B, and B by C."""

    def _build_history(self):
        r = self._repo
        self._base = _make_commit(
            r, b"base", _make_tree(r, {b"keep": b"keep"}), timestamp=100
        )
        self._a = _make_commit(
            r,
            b"commit A",
            _make_tree(r, {b"keep": b"keep", b"only-a": b"a"}),
            parents=[self._base.id],
            timestamp=200,
        )
        self._tip = _make_commit(
            r,
            b"tip",
            _make_tree(r, {b"keep": b"keep", b"only-a": b"a", b"other": b"o"}),
            parents=[self._a.id],
            timestamp=400,
        )
        # B and C are not referenced by any ordinary ref.
        self._b = _make_commit(
            r,
            b"commit B",
            _make_tree(r, {b"keep": b"keep", b"via-b": b"b"}),
            parents=[self._base.id],
            timestamp=300,
        )
        self._c = _make_commit(
            r,
            b"commit C",
            _make_tree(r, {b"keep": b"keep", b"via-c": b"c"}),
            parents=[],
            timestamp=250,
        )
        # D shares history with A through base.
        self._d = _make_commit(
            r,
            b"commit D",
            _make_tree(r, {b"keep": b"keep", b"d-file": b"d"}),
            parents=[self._base.id],
            timestamp=350,
        )
        r.refs[b"HEAD"] = self._tip.id

    def _add_replace_refs(self):
        self._repo.refs[b"refs/replace/" + self._a.id] = self._b.id
        self._repo.refs[b"refs/replace/" + self._b.id] = self._c.id


class ParseReplaceRefsTests(ReplaceRepoMixin, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._repo = MemoryRepo()
        self.addCleanup(self._repo.close)
        self._build_history()

    def test_no_replace_refs(self) -> None:
        self.assertEqual({}, parse_replace_refs(self._repo.refs))

    def test_replace_refs(self) -> None:
        self._add_replace_refs()
        self.assertEqual(
            {self._a.id: self._b.id, self._b.id: self._c.id},
            parse_replace_refs(self._repo.refs),
        )


class ReplacementObjectStoreTests(ReplaceRepoMixin, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._repo = MemoryRepo()
        self.addCleanup(self._repo.close)
        self._build_history()
        self._add_replace_refs()
        self._view = self._repo.replacement_object_store()

    def test_view_is_replacement_store(self) -> None:
        self.assertIsInstance(self._view, ReplacementObjectStore)
        self.assertIs(self._view.physical_store, self._repo.object_store)

    def test_resolve_follows_multiple_hops(self) -> None:
        self.assertEqual(self._c.id, self._view.resolve(self._a.id))
        self.assertEqual(self._c.id, self._view.resolve(self._b.id))
        self.assertEqual(self._tip.id, self._view.resolve(self._tip.id))

    def test_is_replaced(self) -> None:
        self.assertTrue(self._view.is_replaced(self._a.id))
        self.assertTrue(self._view.is_replaced(self._b.id))
        self.assertFalse(self._view.is_replaced(self._c.id))
        self.assertFalse(self._view.is_replaced(self._tip.id))

    def test_getitem_preserves_requested_identity(self) -> None:
        obj = self._view[self._a.id]
        # Displayed identity is the original commit ...
        self.assertEqual(self._a.id, obj.id)
        # ... while the contents come from the resolved replacement.
        self.assertEqual(b"commit C", obj.message)
        self.assertEqual(self._c.tree, obj.tree)
        self.assertEqual(self._c.as_raw_string(), obj.as_raw_string())

    def test_getitem_intermediate_hop(self) -> None:
        obj = self._view[self._b.id]
        self.assertEqual(self._b.id, obj.id)
        self.assertEqual(b"commit C", obj.message)

    def test_get_raw_applies_replacement(self) -> None:
        type_num, raw = self._view.get_raw(self._a.id)
        self.assertEqual(Commit.type_num, type_num)
        self.assertEqual(self._c.as_raw_string(), raw)

    def test_unreplaced_object_read_directly(self) -> None:
        obj = self._view[self._tip.id]
        self.assertEqual(self._tip.id, obj.id)
        self.assertEqual(b"tip", obj.message)

    def test_iteration_is_physical(self) -> None:
        self.assertEqual(set(iter(self._repo.object_store)), set(iter(self._view)))

    def test_contains(self) -> None:
        self.assertIn(self._a.id, self._view)
        self.assertIn(self._tip.id, self._view)
        self.assertNotIn(ObjectID(b"9" * 40), self._view)

    def test_cycle_detected(self) -> None:
        r = self._repo
        r.refs[b"refs/replace/" + self._c.id] = self._a.id
        view = r.replacement_object_store()
        with self.assertRaises(ReplacementCycleError) as cm:
            view[self._a.id]
        self.assertEqual(self._a.id, cm.exception.original)
        self.assertIn(self._a.id.decode("ascii"), str(cm.exception))

    def test_missing_target_detected(self) -> None:
        missing = ObjectID(b"8" * 40)
        self._repo.refs[b"refs/replace/" + self._c.id] = missing
        view = self._repo.replacement_object_store()
        with self.assertRaises(ReplacementMissingError) as cm:
            view[self._a.id]
        self.assertEqual(self._a.id, cm.exception.original)
        self.assertEqual(missing, cm.exception.missing)
        self.assertIn(self._a.id.decode("ascii"), str(cm.exception))

    def test_type_mismatch_detected(self) -> None:
        blob = Blob()
        blob.data = b"not a commit"
        self._repo.object_store.add_object(blob)
        self._repo.refs[b"refs/replace/" + self._c.id] = blob.id
        view = self._repo.replacement_object_store()
        with self.assertRaises(ReplacementTypeMismatchError) as cm:
            view[self._a.id]
        self.assertEqual(self._a.id, cm.exception.original)
        self.assertEqual(blob.id, cm.exception.replacement)
        self.assertIn(self._a.id.decode("ascii"), str(cm.exception))


class ReplacementViewRepoTests(ReplaceRepoMixin, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._repo = MemoryRepo()
        self.addCleanup(self._repo.close)
        self._build_history()
        self._add_replace_refs()

    def _walked_ids(self, repo, **kwargs):
        return [e.commit.id for e in repo.get_walker(**kwargs)]

    def test_repo_get_object_uses_view(self) -> None:
        obj = self._repo[self._a.id]
        self.assertEqual(self._a.id, obj.id)
        self.assertEqual(b"commit C", obj.message)
        # The physical channel still returns the original object.
        physical = self._repo.physical_object_store[self._a.id]
        self.assertEqual(b"commit A", physical.message)
        self.assertIs(self._repo.physical_object_store, self._repo.object_store)

    def test_walker_follows_replacement_history(self) -> None:
        # C has no parents, so the walk stops at A.
        self.assertEqual([self._tip.id, self._a.id], self._walked_ids(self._repo))
        messages = [e.commit.message for e in self._repo.get_walker()]
        self.assertEqual([b"tip", b"commit C"], messages)

    def test_walker_path_filtering_uses_replacement_trees(self) -> None:
        # only-a is introduced by the physical A, but the replacement C
        # does not touch it; instead tip introduces it relative to C.
        self.assertEqual(
            [self._tip.id], self._walked_ids(self._repo, paths=[b"only-a"])
        )
        # via-c only exists in the replacement chain: A (showing C's
        # contents) introduces it, and tip deletes it again.
        self.assertEqual(
            [self._tip.id, self._a.id],
            self._walked_ids(self._repo, paths=[b"via-c"]),
        )
        # via-b is shadowed by the second hop (B is itself replaced by C).
        self.assertEqual([], self._walked_ids(self._repo, paths=[b"via-b"]))

    def test_walker_path_filtering_single_hop(self) -> None:
        # With only A -> B, path filtering sees B's tree.
        del self._repo.refs[b"refs/replace/" + self._b.id]
        self.assertEqual(
            [self._tip.id, self._a.id],
            self._walked_ids(self._repo, paths=[b"via-b"]),
        )
        self.assertEqual([], self._walked_ids(self._repo, paths=[b"via-c"]))

    def test_merge_base_uses_replacement_parents(self) -> None:
        # Physically tip and D share base; through the replacement view
        # A has no parents (C has none), so there is no common ancestor.
        self.assertEqual([], find_merge_base(self._repo, [self._tip.id, self._d.id]))

    def test_disable_restores_physical_history(self) -> None:
        self._repo.use_replace_refs = False
        self.assertEqual(
            [self._tip.id, self._a.id, self._base.id],
            self._walked_ids(self._repo),
        )
        self.assertEqual(
            [self._base.id],
            find_merge_base(self._repo, [self._tip.id, self._d.id]),
        )
        self.assertEqual([self._a.id], self._walked_ids(self._repo, paths=[b"only-a"]))
        obj = self._repo[self._a.id]
        self.assertEqual(self._a.id, obj.id)
        self.assertEqual(b"commit A", obj.message)

    def test_toggle_does_not_modify_objects_or_refs(self) -> None:
        before_raw = self._repo.object_store[self._a.id].as_raw_string()
        before_refs = dict(self._repo.refs.as_dict())
        self._repo.use_replace_refs = False
        self._repo.use_replace_refs = True
        self.assertEqual(
            before_raw, self._repo.object_store[self._a.id].as_raw_string()
        )
        self.assertEqual(before_refs, dict(self._repo.refs.as_dict()))
        self.assertEqual(
            {self._a.id: self._b.id, self._b.id: self._c.id},
            parse_replace_refs(self._repo.refs),
        )

    def test_no_replace_refs_returns_physical_store(self) -> None:
        repo = MemoryRepo()
        self.addCleanup(repo.close)
        self.assertIs(repo.object_store, repo.replacement_object_store())

    def test_disabled_returns_physical_store(self) -> None:
        self._repo.use_replace_refs = False
        self.assertIs(self._repo.object_store, self._repo.replacement_object_store())


class ReplacementPhysicalChannelTests(ReplaceRepoMixin, TestCase):
    """Packing and connectivity checks read physical objects only."""

    def setUp(self) -> None:
        super().setUp()
        self._repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._repo_dir)
        self._repo = Repo.init(self._repo_dir)
        self.addCleanup(self._repo.close)
        self._build_history()
        self._add_replace_refs()

    def test_physical_store_reads_original_objects(self) -> None:
        store = self._repo.physical_object_store
        self.assertEqual(b"commit A", store[self._a.id].message)
        self.assertEqual(b"commit B", store[self._b.id].message)
        self.assertEqual(b"commit C", store[self._c.id].message)

    def test_pack_data_contains_physical_objects(self) -> None:
        from dulwich.objects import hex_to_sha

        store = self._repo.physical_object_store
        _count, objects = store.generate_pack_data(have=[], want=[self._tip.id])
        packed = {unpacked.sha(): unpacked.sha_file() for unpacked in objects}
        # The physical closure of tip includes A and base ...
        self.assertIn(hex_to_sha(self._a.id), packed)
        self.assertIn(hex_to_sha(self._base.id), packed)
        # ... but not the replacement-only commits B and C.
        self.assertNotIn(hex_to_sha(self._b.id), packed)
        self.assertNotIn(hex_to_sha(self._c.id), packed)
        # The packed copy of A carries the original contents, not the
        # replacement contents.
        self.assertEqual(b"commit A", packed[hex_to_sha(self._a.id)].message)

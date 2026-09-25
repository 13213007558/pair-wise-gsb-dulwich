# test_replace.py -- Tests for refs/replace/ read views
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

"""Tests for the refs/replace/ replacement view."""

from dulwich.graph import find_merge_base
from dulwich.objects import Blob, Commit, hex_to_sha
from dulwich.replace import (
    ReplacementCycleError,
    ReplacementObjectStore,
    ReplacementTargetMissingError,
    ReplacementTypeMismatchError,
    read_replace_refs,
)
from dulwich.repo import MemoryRepo
from dulwich.tests.utils import build_commit_graph, make_object

from . import TestCase


def _blob(data: bytes) -> Blob:
    return make_object(Blob, data=data)


class ReplacementViewTestCase(TestCase):
    """Build a history where commit A is replaced by B, and B by C.

    Physical topology (numbers are commit spec ids):

      c1 (1) <-- A (3) <-- head (4)
      c0 (2) <-- o1 (5)
           ^--- C (7)
      c1 (1) <-- B (6)

    Replacement refs: A -> B, B -> C, so reads of A resolve to C.
    """

    def setUp(self) -> None:
        super().setUp()
        self.repo = MemoryRepo()
        commits = build_commit_graph(
            self.repo.object_store,
            [[1], [2], [3, 1], [4, 3], [5, 2], [6, 1], [7, 2]],
            trees={
                1: [(b"base", _blob(b"base\n"))],
                2: [(b"c0file", _blob(b"c0\n"))],
                3: [(b"base", _blob(b"base\n")), (b"x", _blob(b"x\n"))],
                4: [
                    (b"base", _blob(b"base\n")),
                    (b"y", _blob(b"y\n")),
                    (b"z", _blob(b"z\n")),
                    (b"top", _blob(b"top\n")),
                ],
                5: [(b"c0file", _blob(b"c0\n")), (b"o", _blob(b"o\n"))],
                6: [(b"base", _blob(b"base\n")), (b"y", _blob(b"y\n"))],
                7: [(b"c0file", _blob(b"c0\n")), (b"z", _blob(b"z\n"))],
            },
            attrs={
                1: {"message": b"c1", "commit_time": 1},
                2: {"message": b"c0", "commit_time": 2},
                3: {"message": b"original A", "commit_time": 3},
                4: {"message": b"head", "commit_time": 6},
                5: {"message": b"other", "commit_time": 5},
                6: {"message": b"replacement B", "commit_time": 4},
                7: {"message": b"replacement C", "commit_time": 4},
            },
        )
        (self.c1, self.c0, self.c_a, self.head, self.o1, self.c_b, self.c_c) = commits
        self.repo.refs[b"refs/heads/master"] = self.head.id
        self.repo.refs[b"refs/heads/other"] = self.o1.id
        self.repo.refs[b"refs/replace/" + self.c_a.id] = self.c_b.id
        self.repo.refs[b"refs/replace/" + self.c_b.id] = self.c_c.id
        self.view = self.repo.replacement_view()

    def walk_ids(self, repo, **kwargs):
        return [
            entry.commit.id
            for entry in repo.get_walker(include=[self.head.id], **kwargs)
        ]


class ReadReplaceRefsTests(ReplacementViewTestCase):
    def test_read_replace_refs(self) -> None:
        self.assertEqual(
            {self.c_a.id: self.c_b.id, self.c_b.id: self.c_c.id},
            read_replace_refs(self.repo.refs),
        )

    def test_read_replace_refs_empty(self) -> None:
        repo = MemoryRepo()
        self.assertEqual({}, read_replace_refs(repo.refs))


class ResolveTests(ReplacementViewTestCase):
    def test_resolve_multi_hop(self) -> None:
        # A -> B -> C resolves all the way to C.
        self.assertEqual(self.c_c.id, self.view.object_store.resolve(self.c_a.id))
        self.assertEqual(self.c_c.id, self.view.object_store.resolve(self.c_b.id))

    def test_resolve_unreplaced(self) -> None:
        self.assertEqual(self.c1.id, self.view.object_store.resolve(self.c1.id))

    def test_resolve_raw_sha(self) -> None:
        self.assertEqual(
            self.c_c.id, self.view.object_store.resolve(hex_to_sha(self.c_a.id))
        )

    def test_resolve_disabled(self) -> None:
        self.view.enabled = False
        self.assertEqual(self.c_a.id, self.view.object_store.resolve(self.c_a.id))

    def test_cycle(self) -> None:
        self.repo.refs[b"refs/replace/" + self.c1.id] = self.o1.id
        self.repo.refs[b"refs/replace/" + self.o1.id] = self.c1.id
        view = self.repo.replacement_view()
        with self.assertRaises(ReplacementCycleError) as cm:
            view.object_store.resolve(self.c1.id)
        self.assertEqual(self.c1.id, cm.exception.original)
        self.assertIn(self.c1.id.decode("ascii"), str(cm.exception))

    def test_type_mismatch(self) -> None:
        blob = _blob(b"not a commit")
        self.repo.object_store.add_object(blob)
        self.repo.refs[b"refs/replace/" + self.c1.id] = blob.id
        view = self.repo.replacement_view()
        with self.assertRaises(ReplacementTypeMismatchError) as cm:
            view.object_store.resolve(self.c1.id)
        self.assertEqual(self.c1.id, cm.exception.original)
        self.assertEqual(blob.id, cm.exception.target)
        self.assertIn(self.c1.id.decode("ascii"), str(cm.exception))

    def test_missing_target(self) -> None:
        missing = b"1" * 40
        self.repo.refs[b"refs/replace/" + self.c1.id] = missing
        view = self.repo.replacement_view()
        with self.assertRaises(ReplacementTargetMissingError) as cm:
            view.object_store.resolve(self.c1.id)
        self.assertEqual(self.c1.id, cm.exception.original)
        self.assertEqual(missing, cm.exception.missing)
        self.assertIn(self.c1.id.decode("ascii"), str(cm.exception))

    def test_missing_target_not_in_store(self) -> None:
        missing = b"1" * 40
        self.repo.refs[b"refs/replace/" + self.c1.id] = missing
        view = self.repo.replacement_view()
        self.assertNotIn(self.c1.id, view.object_store)


class ObjectReadTests(ReplacementViewTestCase):
    def test_get_object_keeps_original_identity(self) -> None:
        obj = self.view.get_object(self.c_a.id)
        # The displayed identity is the original commit...
        self.assertEqual(self.c_a.id, obj.id)
        self.assertIsInstance(obj, Commit)
        # ...but the contents come from the transitively resolved replacement.
        self.assertEqual(b"replacement C", obj.message)
        self.assertEqual([self.c0.id], obj.parents)
        # The identity of the object actually read is available separately.
        self.assertEqual(self.c_c.id, self.view.object_store.resolve(obj.id))

    def test_get_object_physical_unaffected(self) -> None:
        obj = self.repo.get_object(self.c_a.id)
        self.assertEqual(self.c_a.id, obj.id)
        self.assertEqual(b"original A", obj.message)
        self.assertEqual([self.c1.id], obj.parents)

    def test_get_raw_follows_replacement(self) -> None:
        self.assertEqual(
            self.repo.object_store.get_raw(self.c_c.id),
            self.view.object_store.get_raw(self.c_a.id),
        )

    def test_getitem_raw_sha(self) -> None:
        obj = self.view.object_store[hex_to_sha(self.c_a.id)]
        self.assertEqual(self.c_a.id, obj.id)
        self.assertEqual(b"replacement C", obj.message)

    def test_contains(self) -> None:
        self.assertIn(self.c_a.id, self.view.object_store)
        self.assertIn(self.c_c.id, self.view.object_store)

    def test_disable_restores_physical_reads(self) -> None:
        self.view.enabled = False
        obj = self.view.get_object(self.c_a.id)
        self.assertEqual(self.c_a.id, obj.id)
        self.assertEqual(b"original A", obj.message)
        self.assertEqual([self.c1.id], obj.parents)
        # Re-enabling applies replacements again.
        self.view.enabled = True
        self.assertEqual(b"replacement C", self.view.get_object(self.c_a.id).message)


class WalkTests(ReplacementViewTestCase):
    def test_walk_shows_original_ids_with_replacement_content(self) -> None:
        entries = list(self.view.get_walker(include=[self.head.id]))
        self.assertEqual(
            [self.head.id, self.c_a.id, self.c0.id],
            [entry.commit.id for entry in entries],
        )
        replaced = entries[1]
        # Identity is the replaced commit, contents are the replacement's.
        self.assertEqual(b"replacement C", replaced.commit.message)
        self.assertEqual([self.c0.id], replaced.commit.parents)

    def test_walk_physical_unaffected(self) -> None:
        entries = list(self.repo.get_walker(include=[self.head.id]))
        self.assertEqual(
            [self.head.id, self.c_a.id, self.c1.id],
            [entry.commit.id for entry in entries],
        )
        self.assertEqual(b"original A", entries[1].commit.message)

    def test_walk_path_filtering(self) -> None:
        # Physically, only head touches z (A only touches x).
        self.assertEqual([self.head.id], self.walk_ids(self.repo, paths=[b"z"]))
        # In the view, A (read as C) is the commit that touches z.
        self.assertEqual([self.c_a.id], self.walk_ids(self.view, paths=[b"z"]))
        # x is only touched in the physical history.
        self.assertEqual(
            [self.head.id, self.c_a.id], self.walk_ids(self.repo, paths=[b"x"])
        )
        self.assertEqual([], self.walk_ids(self.view, paths=[b"x"]))

    def test_walk_disabled_view_matches_physical(self) -> None:
        self.view.enabled = False
        self.assertEqual(
            self.walk_ids(self.repo, paths=[b"z"]),
            self.walk_ids(self.view, paths=[b"z"]),
        )
        self.assertEqual(
            [self.head.id, self.c_a.id, self.c1.id], self.walk_ids(self.view)
        )


class MergeBaseTests(ReplacementViewTestCase):
    def test_merge_base_physical(self) -> None:
        # head and o1 are on disjoint physical histories.
        self.assertEqual([], find_merge_base(self.repo, [self.head.id, self.o1.id]))

    def test_merge_base_view(self) -> None:
        # In the view, A resolves to C, whose parent is c0 -- an ancestor of
        # o1 -- so the histories share merge base c0.
        self.assertEqual(
            [self.c0.id], find_merge_base(self.view, [self.head.id, self.o1.id])
        )

    def test_merge_base_disabled_view(self) -> None:
        self.view.enabled = False
        self.assertEqual([], find_merge_base(self.view, [self.head.id, self.o1.id]))


class PhysicalChannelTests(ReplacementViewTestCase):
    def test_iteration_is_physical(self) -> None:
        # Pack generation and connectivity checks enumerate objects by
        # iteration; the view must enumerate the physical object graph.
        self.assertEqual(set(self.repo.object_store), set(self.view.object_store))

    def test_physical_attribute(self) -> None:
        self.assertIs(self.repo.object_store, self.view.object_store.physical)
        self.assertEqual(
            self.repo.object_store.get_raw(self.c_a.id),
            self.view.object_store.physical.get_raw(self.c_a.id),
        )
        self.assertNotEqual(
            self.view.object_store.get_raw(self.c_a.id),
            self.view.object_store.physical.get_raw(self.c_a.id),
        )

    def test_pack_generation_is_physical(self) -> None:
        # Packing through the view delegates to the physical store, so
        # replacement content is never sent in place of the original objects.
        count, pack_objects = self.view.object_store.generate_pack_data(
            [], [self.head.id]
        )
        packed = {obj.sha() for obj in pack_objects}
        pcount, ppack_objects = self.repo.object_store.generate_pack_data(
            [], [self.head.id]
        )
        self.assertEqual(pcount, count)
        self.assertEqual({obj.sha() for obj in ppack_objects}, packed)
        # The physical history is packed: the original A is included, and
        # the replacement commits B and C are not reachable from head.
        self.assertIn(hex_to_sha(self.c_a.id), packed)
        self.assertNotIn(hex_to_sha(self.c_b.id), packed)
        self.assertNotIn(hex_to_sha(self.c_c.id), packed)

    def test_writes_go_to_physical_store(self) -> None:
        blob = _blob(b"written through the view")
        self.view.object_store.add_object(blob)
        self.assertIn(blob.id, self.repo.object_store)
        self.assertEqual(
            blob.data,
            Blob.from_raw_string(*self.repo.object_store.get_raw(blob.id)).data,
        )

    def test_connectivity_check_uses_physical_content(self) -> None:
        # A connectivity check walks the physical store; replaced objects
        # must still validate as their physical selves.
        for sha in self.repo.object_store:
            obj = self.repo.object_store[sha]
            self.assertEqual(sha, obj.id)
        self.assertEqual(b"original A", self.repo.object_store[self.c_a.id].message)

    def test_view_does_not_mutate_repo(self) -> None:
        refs_before = self.repo.refs.as_dict()
        objects_before = set(self.repo.object_store)
        raw_a_before = self.repo.object_store.get_raw(self.c_a.id)
        # Exercise the view thoroughly.
        list(self.view.get_walker(include=[self.head.id]))
        find_merge_base(self.view, [self.head.id, self.o1.id])
        self.view.get_object(self.c_a.id)
        self.view.enabled = False
        list(self.view.get_walker(include=[self.head.id]))
        self.view.enabled = True
        self.assertEqual(refs_before, self.repo.refs.as_dict())
        self.assertEqual(objects_before, set(self.repo.object_store))
        self.assertEqual(raw_a_before, self.repo.object_store.get_raw(self.c_a.id))


class ReplacementObjectStoreStandaloneTests(TestCase):
    def test_explicit_replacements_mapping(self) -> None:
        repo = MemoryRepo()
        (c1,) = build_commit_graph(repo.object_store, [[1]])
        blob = _blob(b"data")
        repo.object_store.add_object(blob)
        store = ReplacementObjectStore(repo.object_store, {c1.id: c1.id}, enabled=False)
        self.assertEqual(c1.id, store.resolve(c1.id))
        self.assertFalse(store.enabled)

    def test_refresh_requires_refs(self) -> None:
        repo = MemoryRepo()
        store = ReplacementObjectStore(repo.object_store, {})
        with self.assertRaises(ValueError):
            store.refresh()

    def test_refresh_picks_up_new_refs(self) -> None:
        repo = MemoryRepo()
        c1, c2 = build_commit_graph(repo.object_store, [[1], [2, 1]])
        store = ReplacementObjectStore(repo.object_store, refs=repo.refs)
        self.assertEqual(c1.id, store.resolve(c1.id))
        repo.refs[b"refs/replace/" + c1.id] = c2.id
        store.refresh()
        self.assertEqual(c2.id, store.resolve(c1.id))

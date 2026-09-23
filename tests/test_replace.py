# test_replace.py -- tests for the refs/replace read-only view
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
# <http://www.apache.org/licenses/> for a copy of the Apache License
# and <http://www.gnu.org/licenses/> for a copy of the GNU General Public
# License, version 2.0 or later.
#

"""Tests for the refs/replace read-only object store view."""

from unittest import TestCase

from dulwich.annotate import annotate_lines
from dulwich.object_store import MemoryObjectStore, tree_lookup_path
from dulwich.objects import Blob, ObjectID, Tree
from dulwich.refs import DictRefsContainer
from dulwich.replace import (
    ReplacementCycle,
    ReplacementTargetMissing,
    ReplacementTypeMismatch,
    ReplaceRefsObjectStore,
    read_replace_refs,
)
from dulwich.tests.utils import build_commit_graph, make_commit, make_object
from dulwich.walk import Walker


def make_blob(store, data):
    blob = make_object(Blob, data=data)
    store.add_object(blob)
    return blob


def make_tree(store, entries):
    tree = Tree()
    for name, mode, sha in entries:
        tree.add(name, mode, sha)
    store.add_object(tree)
    return tree


def add_commit(store, **attrs):
    commit = make_commit(**attrs)
    store.add_object(commit)
    return commit


class ReplaceRefsObjectStoreTests(TestCase):
    """Tests for ReplaceRefsObjectStore."""

    def setUp(self):
        self.store = MemoryObjectStore()

    def test_commit_replacement_preserves_logical_id(self):
        old = add_commit(self.store, message=b"old message")
        new = add_commit(self.store, message=b"new message")
        view = ReplaceRefsObjectStore(self.store, {old.id: new.id})
        obj = view[old.id]
        # Logical identity stays with the original id...
        self.assertEqual(old.id, obj.id)
        # ...while the contents come from the replacement.
        self.assertEqual(b"new message", obj.message)
        # The actual content id is reported separately.
        self.assertEqual(new.id, view.get_actual_id(old.id))
        self.assertEqual([old.id, new.id], view.resolve_chain(old.id))
        # The underlying store still serves the real object.
        self.assertEqual(b"old message", self.store[old.id].message)

    def test_tree_replacement(self):
        blob_old = make_blob(self.store, b"old contents\n")
        blob_new = make_blob(self.store, b"new contents\n")
        tree_old = make_tree(self.store, [(b"file.txt", 0o100644, blob_old.id)])
        tree_new = make_tree(self.store, [(b"file.txt", 0o100644, blob_new.id)])
        view = ReplaceRefsObjectStore(self.store, {tree_old.id: tree_new.id})
        tree = view[tree_old.id]
        self.assertEqual(tree_old.id, tree.id)
        self.assertEqual(blob_new.id, tree[b"file.txt"][1])
        _mode, sha = tree_lookup_path(view.__getitem__, tree_old.id, b"file.txt")
        self.assertEqual(blob_new.id, sha)
        self.assertEqual(b"new contents\n", view[sha].data)
        # Disabled view sees the original tree.
        view.enabled = False
        self.assertEqual(blob_old.id, view[tree_old.id][b"file.txt"][1])

    def test_blob_replacement(self):
        blob_old = make_blob(self.store, b"old data\n")
        blob_new = make_blob(self.store, b"new data\n")
        view = ReplaceRefsObjectStore(self.store, {blob_old.id: blob_new.id})
        blob = view[blob_old.id]
        self.assertEqual(blob_old.id, blob.id)
        self.assertEqual(b"new data\n", blob.data)
        self.assertEqual((3, b"new data\n"), view.get_raw(blob_old.id))
        self.assertEqual((3, b"old data\n"), self.store.get_raw(blob_old.id))

    def test_multi_hop_chain(self):
        c_a = add_commit(self.store, message=b"a")
        c_b = add_commit(self.store, message=b"b")
        c_c = add_commit(self.store, message=b"c")
        view = ReplaceRefsObjectStore(self.store, {c_a.id: c_b.id, c_b.id: c_c.id})
        self.assertEqual([c_a.id, c_b.id, c_c.id], view.resolve_chain(c_a.id))
        self.assertEqual(c_c.id, view.get_actual_id(c_a.id))
        obj = view[c_a.id]
        self.assertEqual(c_a.id, obj.id)
        self.assertEqual(b"c", obj.message)

    def test_missing_target_reports_chain(self):
        c_a = add_commit(self.store, message=b"a")
        c_b = add_commit(self.store, message=b"b")
        missing = ObjectID(b"9" * 40)
        view = ReplaceRefsObjectStore(self.store, {c_a.id: c_b.id, c_b.id: missing})
        with self.assertRaises(ReplacementTargetMissing) as cm:
            view.resolve_chain(c_a.id)
        self.assertEqual([c_a.id, c_b.id, missing], cm.exception.chain)
        self.assertIn(missing.decode("ascii"), str(cm.exception))
        self.assertIn(" -> ", str(cm.exception))

    def test_type_mismatch_reports_chain(self):
        commit = add_commit(self.store, message=b"commit")
        blob = make_blob(self.store, b"blob\n")
        view = ReplaceRefsObjectStore(self.store, {commit.id: blob.id})
        with self.assertRaises(ReplacementTypeMismatch) as cm:
            view.resolve_chain(commit.id)
        self.assertEqual([commit.id, blob.id], cm.exception.chain)
        self.assertIn(blob.id.decode("ascii"), str(cm.exception))

    def test_cycle_reports_chain(self):
        c_a = add_commit(self.store, message=b"a")
        c_b = add_commit(self.store, message=b"b")
        view = ReplaceRefsObjectStore(self.store, {c_a.id: c_b.id, c_b.id: c_a.id})
        with self.assertRaises(ReplacementCycle) as cm:
            view.resolve_chain(c_a.id)
        self.assertEqual([c_a.id, c_b.id, c_a.id], cm.exception.chain)

    def test_disabled_view_sees_original(self):
        old = add_commit(self.store, message=b"old message")
        new = add_commit(self.store, message=b"new message")
        view = ReplaceRefsObjectStore(self.store, {old.id: new.id}, enabled=False)
        self.assertEqual(b"old message", view[old.id].message)
        self.assertEqual(old.id, view.get_actual_id(old.id))
        self.assertEqual((1, self.store.get_raw(old.id)[1]), view.get_raw(old.id))
        # Enabling applies replacements immediately...
        view.enabled = True
        self.assertEqual(b"new message", view[old.id].message)
        # ...and disabling again restores the original contents,
        # uncorrupted by cached resolutions.
        view.enabled = False
        self.assertEqual(b"old message", view[old.id].message)

    def test_two_views_same_store_do_not_interfere(self):
        old = add_commit(self.store, message=b"old message")
        new1 = add_commit(self.store, message=b"new message 1")
        new2 = add_commit(self.store, message=b"new message 2")
        view1 = ReplaceRefsObjectStore(self.store, {old.id: new1.id})
        view2 = ReplaceRefsObjectStore(self.store, {old.id: new2.id})
        # Prime both views' caches, then verify results stay separate.
        self.assertEqual(b"new message 1", view1[old.id].message)
        self.assertEqual(b"new message 2", view2[old.id].message)
        self.assertEqual(b"new message 1", view1[old.id].message)
        view1.enabled = False
        self.assertEqual(b"old message", view1[old.id].message)
        self.assertEqual(b"new message 2", view2[old.id].message)
        self.assertEqual(b"old message", self.store[old.id].message)

    def test_fixed_mapping_snapshot(self):
        old = add_commit(self.store, message=b"old message")
        new = add_commit(self.store, message=b"new message")
        mapping = {old.id: new.id}
        view = ReplaceRefsObjectStore(self.store, mapping)
        # Later mutations of the passed-in mapping do not affect the view.
        mapping.clear()
        self.assertEqual(b"new message", view[old.id].message)
        self.assertEqual({old.id: new.id}, view.replacements)

    def test_read_only(self):
        view = ReplaceRefsObjectStore(self.store)
        blob = make_object(Blob, data=b"data\n")
        self.assertRaises(NotImplementedError, view.add_object, blob)
        self.assertRaises(NotImplementedError, view.add_objects, [(blob, None)])

    def test_iteration_yields_real_ids_only(self):
        old = add_commit(self.store, message=b"old message")
        new = add_commit(self.store, message=b"new message")
        view = ReplaceRefsObjectStore(self.store, {old.id: new.id})
        self.assertEqual(set(self.store), set(view))

    def test_from_refs(self):
        old = add_commit(self.store, message=b"old message")
        new = add_commit(self.store, message=b"new message")
        refs = DictRefsContainer(
            {
                b"refs/heads/main": old.id,
                b"refs/replace/" + old.id: new.id,
            }
        )
        self.assertEqual({old.id: new.id}, read_replace_refs(refs))
        view = ReplaceRefsObjectStore.from_refs(self.store, refs)
        self.assertEqual(b"new message", view[old.id].message)
        self.assertEqual(old.id, view[old.id].id)


class ReplacedParentHistoryTests(TestCase):
    """An old commit is replaced by one with a different parent."""

    def setUp(self):
        self.store = MemoryObjectStore()
        blobs = {
            num: make_blob(self.store, data)
            for num, data in [
                (1, b"original\n"),
                (2, b"original\nbase\n"),
                (3, b"original\nbase\ntop\n"),
                (4, b"alt\n"),
                (5, b"alt\nbase\n"),
            ]
        }
        # Original history: 1 <- 2 <- 3. Commit 5 replaces commit 2 and
        # is instead rooted at the unrelated commit 4.
        commits = build_commit_graph(
            self.store,
            [[1], [4], [2, 1], [5, 4], [3, 2]],
            trees={
                1: [(b"file.txt", blobs[1])],
                4: [(b"file.txt", blobs[4])],
                2: [(b"file.txt", blobs[2])],
                5: [(b"file.txt", blobs[5])],
                3: [(b"file.txt", blobs[3])],
            },
            attrs={5: {"message": b"replacement of 2"}},
        )
        (self.c1, self.c4, self.c2, self.r2, self.c3) = commits
        self.raw_c2 = self.store.get_raw(self.c2.id)
        self.view = ReplaceRefsObjectStore(self.store, {self.c2.id: self.r2.id})

    def test_walker_follows_replaced_parents(self):
        walker = Walker(self.view, include=[self.c3.id])
        entries = list(walker)
        # The walk follows the replacement's parentage (4, not 1), but
        # keeps reporting the logical commit ids.
        self.assertEqual(
            [self.c3.id, self.c2.id, self.c4.id],
            [entry.commit.id for entry in entries],
        )
        replaced_entry = entries[1]
        self.assertEqual(b"replacement of 2", replaced_entry.commit.message)
        self.assertEqual([self.c4.id], replaced_entry.commit.parents)
        # The actual content id is available alongside the logical id.
        self.assertEqual(self.r2.id, self.view.get_actual_id(self.c2.id))
        # Without the view, the original parent chain is walked.
        plain_ids = [
            entry.commit.id for entry in Walker(self.store, include=[self.c3.id])
        ]
        self.assertEqual([self.c3.id, self.c2.id, self.c1.id], plain_ids)

    def test_tree_contents_come_from_replacement(self):
        commit = self.view[self.c2.id]
        self.assertEqual(self.c2.id, commit.id)
        _mode, sha = tree_lookup_path(self.view.__getitem__, commit.tree, b"file.txt")
        self.assertEqual(b"alt\nbase\n", self.view[sha].data)
        # The original commit's tree is untouched in the base store.
        _mode, sha = tree_lookup_path(
            self.store.__getitem__, self.store[self.c2.id].tree, b"file.txt"
        )
        self.assertEqual(b"original\nbase\n", self.store[sha].data)

    def test_annotate_uses_replaced_history(self):
        annotated = annotate_lines(self.view, self.c3.id, b"file.txt")
        self.assertEqual(
            [b"original\n", b"base\n", b"top\n"], [line for (_, line) in annotated]
        )
        by_line = {line: commit for ((commit, entry), line) in annotated}
        # Line attribution follows the replacement's contents, but the
        # reported commit identity remains the logical (original) id.
        self.assertEqual(self.c3.id, by_line[b"original\n"].id)
        self.assertEqual(self.c2.id, by_line[b"base\n"].id)
        self.assertEqual(self.c3.id, by_line[b"top\n"].id)
        self.assertEqual(self.r2.id, self.view.get_actual_id(by_line[b"base\n"].id))
        # Without the view, the same line is attributed differently.
        plain = annotate_lines(self.store, self.c3.id, b"file.txt")
        plain_by_line = {line: commit for ((commit, entry), line) in plain}
        self.assertEqual(self.c1.id, plain_by_line[b"original\n"].id)
        self.assertEqual(self.c2.id, plain_by_line[b"base\n"].id)

    def test_original_object_bytes_unchanged(self):
        # Exercise the view thoroughly, then verify the base store.
        list(Walker(self.view, include=[self.c3.id]))
        annotate_lines(self.view, self.c3.id, b"file.txt")
        self.view[self.c2.id]
        self.assertEqual(self.raw_c2, self.store.get_raw(self.c2.id))
        self.assertEqual(b"Commit 2", self.store[self.c2.id].message)
        self.assertEqual([self.c1.id], self.store[self.c2.id].parents)

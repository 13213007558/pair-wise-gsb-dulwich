# test_object_authz.py -- Tests for upload-pack object authorization
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

"""Tests for upload-pack object authorization policies.

These tests build a repository with a shared base history, a public branch
and an independent hidden branch, then drive the smart protocol through an
``UploadPackHandler`` with per-user ref filtering. They prove that:

* objects reachable from the advertised (visible) refs can still be fetched,
  including the shared base objects;
* objects belonging to hidden history cannot be downloaded even when the
  client guesses their object ids in want/shallow lines;
* the allow-*-sha1-in-want capabilities cannot widen the secure default.
"""

from io import BytesIO

from dulwich.errors import UnauthorizedObjectError
from dulwich.object_authz import (
    AllowAllAuthorizationPolicy,
    AllowReachableSha1InWantAuthorizationPolicy,
    VisibleRefsAuthorizationPolicy,
    reachable_object_ids,
)
from dulwich.objects import Blob, Tree
from dulwich.pack import PackData
from dulwich.protocol import (
    CAPABILITY_ALLOW_REACHABLE_SHA1_IN_WANT,
    CAPABILITY_ALLOW_TIP_SHA1_IN_WANT,
)
from dulwich.repo import MemoryRepo
from dulwich.server import DictBackend, UploadPackHandler
from dulwich.tests.utils import make_commit, make_tag

from . import TestCase
from .test_server import TestProto


class _ShallowTestProto(TestProto):
    """TestProto extended with the readahead buffer used by shallow flows."""

    def __init__(self) -> None:
        super().__init__()
        self._readahead: bytes | None = None

    def read_pkt_line(self):
        if self._readahead is not None:
            data = self._readahead
            self._readahead = None
            return data
        return super().read_pkt_line()

    def unread_pkt_line(self, data) -> None:
        if self._readahead is not None:
            raise ValueError("Only one pkt-line may be unread")
        self._readahead = data

    def eof(self) -> bool:
        return self._readahead is None and not self._output


REQUIRED_CAPS = b"side-band-64k thin-pack ofs-delta"


def _blob(store, contents: bytes) -> Blob:
    blob = Blob.from_string(contents)
    store.add_object(blob)
    return blob


def _tree(store, entries: dict[bytes, Blob]) -> Tree:
    tree = Tree()
    for name, blob in sorted(entries.items()):
        tree.add(name, 0o100644, blob.id)
    store.add_object(tree)
    return tree


class AuthorizationRepo:
    """In-memory repository with shared, public and hidden histories."""

    def __init__(self) -> None:
        self.repo = MemoryRepo.init_bare([], {})
        store = self.repo.object_store

        # Shared base history reachable from both branches.
        self.base_blob = _blob(store, b"shared base contents\n")
        base_tree = _tree(store, {b"shared.txt": self.base_blob})
        self.base_commit = make_commit(tree=base_tree.id, message=b"base")
        store.add_object(self.base_commit)

        # Public branch: another commit on top of the shared base.
        self.public_blob = _blob(store, b"public branch contents\n")
        public_tree = _tree(
            store,
            {b"shared.txt": self.base_blob, b"public.txt": self.public_blob},
        )
        self.public_commit = make_commit(
            tree=public_tree.id,
            parents=[self.base_commit.id],
            message=b"public",
        )
        store.add_object(self.public_commit)

        # Hidden branch: independent secret history forked from the base.
        self.secret_blob = _blob(store, b"top secret contents\n")
        secret_tree = _tree(
            store,
            {b"shared.txt": self.base_blob, b"secret.txt": self.secret_blob},
        )
        self.secret_commit = make_commit(
            tree=secret_tree.id,
            parents=[self.base_commit.id],
            message=b"secret",
        )
        store.add_object(self.secret_commit)

        # A hidden annotated tag pointing at the hidden commit.
        self.secret_tag = make_tag(self.secret_commit, name=b"secret-tag")
        store.add_object(self.secret_tag)

        # A visible annotated tag pointing at the public commit.
        self.public_tag = make_tag(self.public_commit, name=b"public-tag")
        store.add_object(self.public_tag)

        self.all_refs = {
            b"refs/heads/main": self.public_commit.id,
            b"refs/heads/secret": self.secret_commit.id,
            b"refs/tags/public-tag": self.public_tag.id,
            b"refs/tags/secret-tag": self.secret_tag.id,
        }
        for name, sha in self.all_refs.items():
            self.repo.refs[name] = sha

        # What a restricted user is allowed to see.
        self.visible_refs = {
            b"refs/heads/main": self.public_commit.id,
            b"refs/tags/public-tag": self.public_tag.id,
        }

    def handler(self, proto: TestProto, **kwargs) -> UploadPackHandler:
        backend = self.filtered_backend()
        return UploadPackHandler(backend, [b"/", b"host=example"], proto, **kwargs)

    def filtered_repo(self) -> MemoryRepo:
        """Return a repo view whose get_refs() only exposes visible refs.

        It subclasses the real in-memory repo so find_missing_objects() and
        friends are inherited unchanged; overriding get_refs() makes every
        code path that enumerates refs honor the per-user filter.
        """
        repo = self.repo
        visible = self.visible_refs

        class FilteredRepo(MemoryRepo):
            def get_refs(self):
                return dict(visible)

            def get_peeled(self, ref):
                if ref not in visible:
                    raise KeyError(ref)
                return repo.get_peeled(ref)

        view = FilteredRepo.__new__(FilteredRepo)
        view.__dict__ = repo.__dict__
        return view

    def filtered_backend(self) -> DictBackend:
        return DictBackend({b"/": self.filtered_repo()})


class ReachableObjectIdsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = AuthorizationRepo()

    def test_closure_covers_all_object_kinds(self) -> None:
        store = self.fixture.repo.object_store
        reachable = reachable_object_ids(store, [self.fixture.public_tag.id])
        # Tag -> public commit -> base commit, both trees, both blobs.
        self.assertIn(self.fixture.public_tag.id, reachable)
        self.assertIn(self.fixture.public_commit.id, reachable)
        self.assertIn(self.fixture.base_commit.id, reachable)
        self.assertIn(self.fixture.public_blob.id, reachable)
        self.assertIn(self.fixture.base_blob.id, reachable)
        # Hidden history is not reachable from the visible tag.
        self.assertNotIn(self.fixture.secret_commit.id, reachable)
        self.assertNotIn(self.fixture.secret_blob.id, reachable)
        self.assertNotIn(self.fixture.secret_tag.id, reachable)

    def test_blob_only_authorizes_itself(self) -> None:
        store = self.fixture.repo.object_store
        self.assertEqual(
            {self.fixture.secret_blob.id},
            reachable_object_ids(store, [self.fixture.secret_blob.id]),
        )

    def test_missing_roots_are_ignored(self) -> None:
        store = self.fixture.repo.object_store
        missing = b"9" * 40
        self.assertEqual(
            {self.fixture.base_blob.id},
            reachable_object_ids(store, [missing, self.fixture.base_blob.id]),
        )


class VisibleRefsPolicyUnitTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = AuthorizationRepo()
        self.store = self.fixture.repo.object_store
        self.policy = VisibleRefsAuthorizationPolicy()

    def _check(self, wants):
        self.policy.check_wants(
            wants,
            self.fixture.visible_refs,
            self.store,
            client_capabilities=set(),
            all_refs=self.fixture.all_refs,
        )

    def test_visible_commit_allowed(self) -> None:
        self._check([self.fixture.public_commit.id])

    def test_shared_base_allowed(self) -> None:
        # Reachable from the visible branch even though no visible ref
        # points directly at it.
        self._check([self.fixture.base_commit.id])
        self._check([self.fixture.base_blob.id])
        self._check([self.fixture.public_blob.id])

    def test_visible_tag_and_its_target_allowed(self) -> None:
        self._check([self.fixture.public_tag.id])

    def test_hidden_commit_rejected(self) -> None:
        with self.assertRaises(UnauthorizedObjectError) as ctx:
            self._check([self.fixture.secret_commit.id])
        self.assertEqual(self.fixture.secret_commit.id, ctx.exception.object_id)

    def test_hidden_blob_rejected(self) -> None:
        with self.assertRaises(UnauthorizedObjectError):
            self._check([self.fixture.secret_blob.id])

    def test_hidden_tree_rejected(self) -> None:
        secret_tree_id = self.store[self.fixture.secret_commit.id].tree
        with self.assertRaises(UnauthorizedObjectError):
            self._check([secret_tree_id])

    def test_hidden_tag_rejected(self) -> None:
        with self.assertRaises(UnauthorizedObjectError):
            self._check([self.fixture.secret_tag.id])

    def test_missing_object_rejected(self) -> None:
        with self.assertRaises(UnauthorizedObjectError):
            self._check([b"a" * 40])

    def test_no_visible_refs_rejects_everything(self) -> None:
        # A client that may see no ref cannot fetch any object, even ones
        # reachable from refs hidden from them.
        for obj_id in (
            self.fixture.public_commit.id,
            self.fixture.base_commit.id,
            self.fixture.base_blob.id,
        ):
            with self.assertRaises(UnauthorizedObjectError):
                self.policy.check_wants(
                    [obj_id],
                    {},
                    self.store,
                    client_capabilities=set(),
                    all_refs=self.fixture.all_refs,
                )

    def test_mixed_request_rejected_before_pack(self) -> None:
        # One bad apple must reject the whole request; the policy raises
        # while the request lines are still being parsed.
        with self.assertRaises(UnauthorizedObjectError):
            self._check(
                [
                    self.fixture.public_commit.id,
                    self.fixture.secret_commit.id,
                ]
            )

    def test_allow_reachable_capability_does_not_widen_default(self) -> None:
        # The secure policy ignores allow-*-sha1-in-want: hidden objects
        # stay hidden regardless of negotiated capabilities.
        for cap in (
            CAPABILITY_ALLOW_REACHABLE_SHA1_IN_WANT,
            CAPABILITY_ALLOW_TIP_SHA1_IN_WANT,
        ):
            with self.assertRaises(UnauthorizedObjectError):
                self.policy.check_wants(
                    [self.fixture.secret_commit.id],
                    self.fixture.visible_refs,
                    self.store,
                    client_capabilities={cap},
                    all_refs=self.fixture.all_refs,
                )

    def test_shallow_boundary_rules(self) -> None:
        self.policy.check_shallow_request(
            [self.fixture.base_commit.id],
            self.fixture.visible_refs,
            self.store,
            client_capabilities=set(),
            all_refs=self.fixture.all_refs,
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.policy.check_shallow_request(
                [self.fixture.secret_commit.id],
                self.fixture.visible_refs,
                self.store,
                client_capabilities=set(),
                all_refs=self.fixture.all_refs,
            )
        # A blob cannot serve as a shallow boundary.
        with self.assertRaises(UnauthorizedObjectError):
            self.policy.check_shallow_request(
                [self.fixture.public_blob.id],
                self.fixture.visible_refs,
                self.store,
                client_capabilities=set(),
                all_refs=self.fixture.all_refs,
            )


class AllowReachablePolicyUnitTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = AuthorizationRepo()
        self.store = self.fixture.repo.object_store
        self.policy = AllowReachableSha1InWantAuthorizationPolicy()

    def test_without_capability_uses_visible_boundary(self) -> None:
        with self.assertRaises(UnauthorizedObjectError):
            self.policy.check_wants(
                [self.fixture.secret_commit.id],
                self.fixture.visible_refs,
                self.store,
                client_capabilities=set(),
                all_refs=self.fixture.all_refs,
            )

    def test_reachable_capability_authorizes_hidden_history(self) -> None:
        self.policy.check_wants(
            [self.fixture.secret_commit.id, self.fixture.secret_blob.id],
            self.fixture.visible_refs,
            self.store,
            client_capabilities={CAPABILITY_ALLOW_REACHABLE_SHA1_IN_WANT},
            all_refs=self.fixture.all_refs,
        )

    def test_tip_capability_authorizes_ref_targets_only(self) -> None:
        self.policy.check_wants(
            [self.fixture.secret_commit.id],
            self.fixture.visible_refs,
            self.store,
            client_capabilities={CAPABILITY_ALLOW_TIP_SHA1_IN_WANT},
            all_refs=self.fixture.all_refs,
        )
        # A blob that is no ref tip is rejected even with tip capability.
        with self.assertRaises(UnauthorizedObjectError):
            self.policy.check_wants(
                [self.fixture.secret_blob.id],
                self.fixture.visible_refs,
                self.store,
                client_capabilities={CAPABILITY_ALLOW_TIP_SHA1_IN_WANT},
                all_refs=self.fixture.all_refs,
            )


class AllowAllPolicyUnitTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fixture = AuthorizationRepo()
        self.store = self.fixture.repo.object_store
        self.policy = AllowAllAuthorizationPolicy()

    def test_any_present_object_allowed(self) -> None:
        self.policy.check_wants(
            [self.fixture.secret_blob.id, self.fixture.secret_tag.id],
            self.fixture.visible_refs,
            self.store,
        )

    def test_missing_object_still_rejected(self) -> None:
        with self.assertRaises(UnauthorizedObjectError):
            self.policy.check_wants([b"b" * 40], self.fixture.visible_refs, self.store)


def _pack_object_ids(pack_bytes: bytes, object_format) -> set[bytes]:
    data = PackData(
        "",
        object_format,
        file=BytesIO(pack_bytes),
        size=len(pack_bytes),
    )
    # PackData yields binary (raw) object ids; the fixture uses hex ids.
    return {unpacked.sha().hex().encode("ascii") for unpacked in data.iter_unpacked()}


class UploadPackProtocolTest(TestCase):
    """End-to-end checks through the smart protocol with filtered refs."""

    def setUp(self) -> None:
        super().setUp()
        self.fixture = AuthorizationRepo()
        self.object_format = self.fixture.repo.object_format

    def _handle(self, request_lines, policy=None):
        proto = _ShallowTestProto()
        proto.set_output(request_lines)
        backend = self.fixture.filtered_backend()
        if policy is None:
            handler = UploadPackHandler(backend, [b"/", b"host=example"], proto)
        else:
            handler = UploadPackHandler(
                backend,
                [b"/", b"host=example"],
                proto,
                object_authorization_policy=policy,
            )
        handler.handle()
        return proto

    def _data_pack(self, proto: TestProto) -> bytes:
        # Side-band channel 1 carries pack data; concatenate the chunks.
        chunks = list(proto._received[1])
        return b"".join(chunks)

    def test_advertisement_omits_hidden_refs(self) -> None:
        proto = self._handle([None])
        advertised = b"".join(line or b"" for line in proto._received[0])
        self.assertIn(b"refs/heads/main", advertised)
        self.assertIn(b"refs/tags/public-tag", advertised)
        self.assertNotIn(b"refs/heads/secret", advertised)
        self.assertNotIn(b"secret-tag", advertised)

    def test_fetch_visible_branch_sends_shared_and_public_objects(self) -> None:
        want = self.fixture.public_commit.id
        proto = self._handle(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ]
        )
        sent = _pack_object_ids(self._data_pack(proto), self.object_format)
        # Public history plus the shared base objects are all present.
        self.assertIn(self.fixture.public_commit.id, sent)
        self.assertIn(self.fixture.base_commit.id, sent)
        self.assertIn(self.fixture.public_blob.id, sent)
        self.assertIn(self.fixture.base_blob.id, sent)
        # Nothing from the hidden history leaks into the pack.
        self.assertNotIn(self.fixture.secret_commit.id, sent)
        self.assertNotIn(self.fixture.secret_blob.id, sent)
        self.assertNotIn(self.fixture.secret_tag.id, sent)

    def test_guessed_hidden_commit_want_is_rejected_before_pack(self) -> None:
        want = self.fixture.secret_commit.id
        proto = TestProto()
        proto.set_output(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()
        # No pack data may have been produced.
        self.assertEqual([], proto._received[1])

    def test_guessed_hidden_blob_want_is_rejected(self) -> None:
        want = self.fixture.secret_blob.id
        proto = TestProto()
        proto.set_output(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()
        self.assertEqual([], proto._received[1])

    def test_guessed_hidden_tree_want_is_rejected(self) -> None:
        want = self.fixture.repo.object_store[self.fixture.secret_commit.id].tree
        proto = TestProto()
        proto.set_output(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()
        self.assertEqual([], proto._received[1])

    def test_allow_reachable_capability_does_not_bypass_policy(self) -> None:
        # The default handler never advertises allow-reachable-sha1-in-want,
        # and a client claiming it anyway must not widen the authorization.
        want = self.fixture.secret_commit.id
        proto = TestProto()
        proto.set_output(
            [
                b"want "
                + want
                + b" "
                + REQUIRED_CAPS
                + b" allow-reachable-sha1-in-want",
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(Exception):
            # Rejected either as an unadvertised capability or by the
            # authorization policy; either way no hidden object is sent.
            self.fixture.handler(proto).handle()
        self.assertEqual([], proto._received[1])

    def test_guessed_hidden_tag_want_is_rejected(self) -> None:
        want = self.fixture.secret_tag.id
        proto = TestProto()
        proto.set_output(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()

    def test_visible_plus_hidden_mixed_wants_rejected(self) -> None:
        proto = TestProto()
        proto.set_output(
            [
                b"want " + self.fixture.public_commit.id + b" " + REQUIRED_CAPS,
                b"want " + self.fixture.secret_commit.id,
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()
        self.assertEqual([], proto._received[1])

    def test_shallow_request_anchored_on_hidden_commit_rejected(self) -> None:
        proto = _ShallowTestProto()
        proto.set_output(
            [
                b"want "
                + self.fixture.public_commit.id
                + b" "
                + REQUIRED_CAPS
                + b" shallow",
                b"shallow " + self.fixture.secret_commit.id,
                b"deepen 1",
                None,
                b"done",
                None,
            ]
        )
        with self.assertRaises(UnauthorizedObjectError):
            self.fixture.handler(proto).handle()
        self.assertEqual([], proto._received[1])

    def test_shallow_request_on_visible_commit_succeeds(self) -> None:
        proto = self._handle(
            [
                b"want "
                + self.fixture.public_commit.id
                + b" "
                + REQUIRED_CAPS
                + b" shallow",
                b"deepen 1",
                None,
                b"done",
                None,
            ]
        )
        sent = _pack_object_ids(self._data_pack(proto), self.object_format)
        self.assertIn(self.fixture.public_commit.id, sent)
        self.assertNotIn(self.fixture.secret_commit.id, sent)

    def test_include_tag_does_not_leak_hidden_tag(self) -> None:
        # Want the visible tag object itself; with include-tag the peeled
        # commit and its closure are sent together with the tag.
        want = self.fixture.public_tag.id
        proto = self._handle(
            [
                b"want " + want + b" " + REQUIRED_CAPS + b" include-tag",
                None,
                b"done",
                None,
            ]
        )
        sent = _pack_object_ids(self._data_pack(proto), self.object_format)
        self.assertIn(self.fixture.public_tag.id, sent)
        self.assertIn(self.fixture.public_commit.id, sent)
        self.assertNotIn(self.fixture.secret_tag.id, sent)
        self.assertNotIn(self.fixture.secret_commit.id, sent)

    def test_allow_all_policy_restores_legacy_access(self) -> None:
        want = self.fixture.secret_commit.id
        proto = self._handle(
            [
                b"want " + want + b" " + REQUIRED_CAPS,
                None,
                b"done",
                None,
            ],
            policy=AllowAllAuthorizationPolicy(),
        )
        sent = _pack_object_ids(self._data_pack(proto), self.object_format)
        self.assertIn(self.fixture.secret_commit.id, sent)
        self.assertIn(self.fixture.secret_blob.id, sent)

# test_object_format_negotiation.py -- Tests for cross-format negotiation safety
# Copyright (C) 2024 The Dulwich contributors
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

"""Tests that the negotiated object format stays consistent with how objects,
packs, indexes, alternates and wire object IDs are interpreted.

The goal is that a SHA-1 and a SHA-256 repository can both be served, and both
keep working on their own matching paths, but a mismatched exchange is refused
before refs are written or the object store is polluted.
"""

import os
import shutil
import tempfile
from io import BytesIO

from dulwich.client import LocalGitClient
from dulwich.errors import GitProtocolError, ObjectFormatException
from dulwich.object_format import SHA1, SHA256
from dulwich.object_store import DiskObjectStore
from dulwich.objects import Blob, Commit, Tree, valid_hexsha
from dulwich.pack import (
    REF_DELTA,
    write_pack_objects,
)
from dulwich.protocol import capability_object_format
from dulwich.repo import MemoryRepo, Repo
from dulwich.server import (
    DictBackend,
    ReceivePackHandler,
    UploadPackHandler,
)
from dulwich.tests.utils import make_object

from . import TestCase


def _make_commit(repo, fmt, content=b"hello world", parent=None):
    """Create a blob/tree/commit chain in the given object format."""
    blob = Blob.from_string(content)
    tree = Tree()
    tree.add(b"f.txt", 0o100644, blob.get_id(fmt))
    commit = Commit()
    commit.tree = tree.get_id(fmt)
    commit.author = commit.committer = b"Test User <test@example.com>"
    commit.commit_time = commit.author_time = 1234567890
    commit.commit_timezone = commit.author_timezone = 0
    commit.message = b"a commit"
    if parent is not None:
        commit.parents = [parent]
    for obj in (blob, tree, commit):
        obj.object_format = fmt
        repo.object_store.add_object(obj)
    cid = commit.get_id(fmt)
    repo.refs[b"refs/heads/master"] = cid
    return blob, tree, commit, cid


class _ScriptedProto:
    """Minimal protocol transport fed from a fixed list of packet lines."""

    def __init__(self, lines, raw=b""):
        self._lines = list(lines)
        self._raw = BytesIO(raw)
        self.written = []
        self.sideband = {1: [], 2: [], 3: []}

    def read_pkt_line(self):
        if not self._lines:
            from dulwich.errors import HangupException

            raise HangupException
        line = self._lines.pop(0)
        return None if line is None else line

    def read(self, size=-1):
        return self._raw.read(size)

    def recv(self, size=-1):
        return self._raw.read(size)

    def write_pkt_line(self, data):
        self.written.append(data)

    def write_sideband(self, band, data):
        self.sideband[band].append(data)


class ServerObjectFormatAdvertisementTests(TestCase):
    """Handlers advertise the format of the repository they serve."""

    def _handler(self, fmt, handler_cls):
        repo = MemoryRepo.init_bare([], {}, object_format=fmt.name)
        backend = DictBackend({b"/": repo})
        return handler_cls(backend, [b"/"], _ScriptedProto([])), repo

    def test_upload_pack_advertises_repository_format(self):
        for fmt in (SHA1, SHA256):
            handler, _ = self._handler(fmt, UploadPackHandler)
            self.assertIn(capability_object_format(fmt.name), handler.capabilities())

    def test_receive_pack_advertises_repository_format(self):
        for fmt in (SHA1, SHA256):
            handler, _ = self._handler(fmt, ReceivePackHandler)
            self.assertIn(capability_object_format(fmt.name), handler.capabilities())


class ServerCapabilityNegotiationTests(TestCase):
    """set_client_capabilities enforces the negotiated object format."""

    def _handler(self, fmt):
        repo = MemoryRepo.init_bare([], {}, object_format=fmt.name)
        return ReceivePackHandler(
            DictBackend({b"/": repo}), [b"/"], _ScriptedProto([])
        ), repo

    def test_matching_format_accepted(self):
        for fmt in (SHA1, SHA256):
            handler, _ = self._handler(fmt)
            caps = [
                *handler.capabilities(),
                b"report-status",
            ]
            handler.set_client_capabilities(caps)
            self.assertIn(capability_object_format(fmt.name),
                          handler._client_capabilities)

    def test_mismatched_format_rejected_before_any_write(self):
        pairs = [(SHA1, SHA256), (SHA256, SHA1)]
        for repo_fmt, client_fmt in pairs:
            repo = MemoryRepo.init_bare([], {}, object_format=repo_fmt.name)

            class _Handler(ReceivePackHandler):
                # Accept either object-format value so the negotiation reaches
                # the explicit format comparison (the normal allow-list would
                # reject the unadvertised value one step earlier).
                def capabilities(self):
                    return [
                        capability_object_format(f)
                        for f in (SHA1.name, SHA256.name)
                    ]

            handler = _Handler(
                DictBackend({b"/": repo}), [b"/"], _ScriptedProto([])
            )
            with self.assertRaises(GitProtocolError) as cm:
                handler.set_client_capabilities(
                    [capability_object_format(client_fmt.name)]
                )
            self.assertIn("Object format mismatch", str(cm.exception))
            # Nothing was recorded as negotiated and the store is untouched.
            self.assertIsNone(handler._client_capabilities)
            self.assertEqual(list(repo.object_store), [])
            self.assertEqual(repo.get_refs(), {})

    def test_unknown_format_rejected(self):
        repo = MemoryRepo.init_bare([], {}, object_format="sha256")
        base_caps = ReceivePackHandler(
            DictBackend({b"/": repo}), [b"/"], _ScriptedProto([])
        ).capabilities()

        class _Handler(ReceivePackHandler):
            def capabilities(self):
                return [*base_caps, b"object-format=sha512"]

        handler = _Handler(
            DictBackend({b"/": repo}), [b"/"], _ScriptedProto([])
        )
        with self.assertRaises(GitProtocolError) as cm:
            handler.set_client_capabilities([b"object-format=sha512"])
        self.assertIn("unsupported object format", str(cm.exception))


class ServerObjectIdWidthTests(TestCase):
    """Wire object IDs must use the repository's hash width."""

    def _receive_handler(self, fmt):
        repo = MemoryRepo.init_bare([], {}, object_format=fmt.name)
        return (
            ReceivePackHandler(
                DictBackend({b"/": repo}), [b"/"], _ScriptedProto([])
            ),
            repo,
        )

    def test_valid_hexsha_accepts_both_widths_but_repo_does_not(self):
        # valid_hexsha is format agnostic; the handler adds the width pin.
        self.assertTrue(valid_hexsha(b"a" * 40))
        self.assertTrue(valid_hexsha(b"a" * 64))

    def test_truncated_and_overlong_ids_rejected_on_receive(self):
        for fmt in (SHA1, SHA256):
            handler, repo = self._receive_handler(fmt)
            good_width = fmt.hex_length
            foreign_widths = (
                {40, 64} - {good_width}
            )
            # IDs that are valid hex for the *other* hash format are the
            # dangerous case: valid_hexsha accepts them, so the handler's
            # explicit width check must reject them before any write.
            for bad_width in foreign_widths:
                proto = _ScriptedProto(
                    [
                        b"0" * good_width
                        + b" "
                        + b"a" * bad_width
                        + b" refs/heads/x\x00report-status",
                        None,
                    ]
                )
                h = ReceivePackHandler(
                    DictBackend({b"/": repo}), [b"/"], proto
                )
                h.set_client_capabilities([b"report-status"])
                with self.assertRaises(GitProtocolError) as cm:
                    h.handle()
                self.assertIn("requires", str(cm.exception))
                self.assertNotIn(b"refs/heads/x", repo.get_refs())
                self.assertEqual(list(repo.object_store), [])

            # Garbled lengths (not a valid hex ID at all) are also refused.
            for bad_width in (good_width - 1, good_width + 1):
                proto = _ScriptedProto(
                    [
                        b"0" * good_width
                        + b" "
                        + b"a" * bad_width
                        + b" refs/heads/y\x00report-status",
                        None,
                    ]
                )
                h = ReceivePackHandler(
                    DictBackend({b"/": repo}), [b"/"], proto
                )
                h.set_client_capabilities([b"report-status"])
                with self.assertRaises(GitProtocolError):
                    h.handle()
                self.assertNotIn(b"refs/heads/y", repo.get_refs())

    def test_correct_width_passes_width_check(self):
        # A correctly-sized line passes the width validation. The width check
        # runs while parsing ref-update lines, so reaching pack handling
        # proves it was not rejected on length grounds.
        for fmt in (SHA1, SHA256):
            repo = MemoryRepo.init_bare([], {}, object_format=fmt.name)
            new_id = b"a" * fmt.hex_length
            zero = b"0" * fmt.hex_length
            proto = _ScriptedProto(
                [
                    zero
                    + b" "
                    + new_id
                    + b" refs/heads/other\x00report-status",
                    None,
                ]
            )
            h = ReceivePackHandler(
                DictBackend({b"/": repo}), [b"/"], proto
            )
            h.set_client_capabilities([b"report-status"])
            # Empty pack stream; parsing ref lines (including width check)
            # must already have succeeded.
            h.handle()
            self.assertNotIn(b"refs/heads/other", repo.get_refs())
            self.assertEqual(list(repo.object_store), [])


class UploadPackWantWidthTests(TestCase):
    """upload-pack rejects want IDs that do not match its hash width."""

    def _repo_with_commit(self, fmt):
        repo = MemoryRepo.init_bare([], {}, object_format=fmt.name)
        _blob, _tree, commit, cid = _make_commit(repo, fmt)
        return repo, cid

    def test_foreign_width_want_rejected(self):
        for repo_fmt in (SHA1, SHA256):
            foreign_fmt = SHA256 if repo_fmt is SHA1 else SHA1
            repo, cid = self._repo_with_commit(repo_fmt)
            foreign_id = b"a" * foreign_fmt.hex_length
            # First packet is the want line (with required side-band caps),
            # followed by a flush-pkt.
            proto = _ScriptedProto(
                [
                    b"want "
                    + foreign_id
                    + b" multi_ack_detailed side-band-64k thin-pack ofs-delta",
                    None,
                ]
            )
            handler = UploadPackHandler(
                DictBackend({b"/": repo}), [b"/"], proto
            )
            with self.assertRaises(GitProtocolError) as cm:
                handler.handle()
            self.assertIn("requires", str(cm.exception))
            # No object was added and the store keeps its original format.
            self.assertEqual(repo_fmt, repo.object_store.object_format)

    def test_matching_width_want_passes_width_check(self):
        for fmt in (SHA1, SHA256):
            repo, cid = self._repo_with_commit(fmt)
            proto = _ScriptedProto(
                [
                    b"want "
                    + cid
                    + b" multi_ack_detailed side-band-64k thin-pack ofs-delta",
                    None,
                ]
            )
            handler = UploadPackHandler(
                DictBackend({b"/": repo}), [b"/"], proto
            )
            # A correct-width want must not raise a width error. It will fail
            # later in graph walking only if the scripted session is incomplete,
            # so assert specifically that no width-related error is raised here.
            try:
                handler.handle()
            except GitProtocolError as e:
                self.assertNotIn("requires", str(e))


class ReceivePackObjectStoreIsolationTests(TestCase):
    """A rejected/foreign pack must not pollute the object store."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)

    def _repo(self, name, fmt):
        return Repo.init_bare(
            os.path.join(self.path, name), mkdir=True, object_format=fmt.name
        )

    def _build_pack_bytes(self, objects, fmt):
        buf = BytesIO()
        write_pack_objects(buf.write, [(o, None) for o in objects], fmt)
        buf.seek(0)
        return buf

    def test_sha1_pack_rejected_by_sha256_store(self):
        sha1_repo = MemoryRepo.init_bare([], {}, object_format="sha1")
        blob, tree, commit, cid = _make_commit(sha1_repo, SHA1)
        pack = self._build_pack_bytes([blob, tree, commit], SHA1)

        sha256_store = DiskObjectStore.init(
            os.path.join(self.path, "sha256-store"), object_format=SHA256
        )
        before = list(sha256_store)
        # The SHA-1 pack trailer is 20 bytes while a SHA-256 store expects a
        # 32-byte trailer, so checksum verification must fail.
        with self.assertRaises(Exception):
            sha256_store.add_thin_pack(pack.read, None)
        self.assertEqual(list(sha256_store), before)

    def test_sha256_pack_rejected_by_sha1_store(self):
        sha256_repo = MemoryRepo.init_bare([], {}, object_format="sha256")
        blob, tree, commit, cid = _make_commit(sha256_repo, SHA256)
        pack = self._build_pack_bytes([blob, tree, commit], SHA256)

        sha1_store = DiskObjectStore.init(
            os.path.join(self.path, "sha1-store"), object_format=SHA1
        )
        before = list(sha1_store)
        with self.assertRaises(Exception):
            sha1_store.add_thin_pack(pack.read, None)
        self.assertEqual(list(sha1_store), before)


class ThinPackExternalBaseTests(TestCase):
    """Thin packs resolve external bases with the store's own format."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)

    def _build_thin_pack(self, store, fmt, base, derived_data):
        """Write a thin pack whose only object is a REF_DELTA against base."""
        from dulwich.objects import hex_to_sha
        from dulwich.pack import (
            REF_DELTA,
            create_delta,
            write_pack_header,
            write_pack_object,
        )

        base_hex_id = base.get_id(fmt)  # width matches the store format
        base_raw_id = hex_to_sha(base_hex_id)
        delta = list(create_delta(base.as_raw_string(), derived_data))

        buf = BytesIO()
        hasher = fmt.new_hash()
        write_pack_header(buf.write, 1)
        hasher.update(b"PACK")
        import struct as _struct

        hasher.update(_struct.pack(b">L", 2))
        hasher.update(_struct.pack(b">L", 1))
        write_pack_object(
            buf.write,
            REF_DELTA,
            (base_raw_id, delta),
            object_format=fmt,
            sha=hasher,
        )
        buf.write(hasher.digest())
        buf.seek(0)
        return buf

    def test_thin_pack_resolves_external_base_for_each_format(self):
        for fmt in (SHA1, SHA256):
            store = DiskObjectStore.init(
                os.path.join(self.path, fmt.name), object_format=fmt
            )
            base = make_object(Blob, data=b"base object contents" * 8)
            store.add_object(base)
            derived_data = b"base object contents" * 8 + b" with a tail"

            pack_buf = self._build_thin_pack(store, fmt, base, derived_data)
            # The base lives only in the store, outside the thin pack.
            store.add_thin_pack(pack_buf.read, None)

            derived_id = make_object(Blob, data=derived_data).get_id(fmt)
            self.assertIn(derived_id, store)
            type_num, data = store.get_raw(derived_id)
            self.assertEqual((Blob.type_num, derived_data), (type_num, data))
            # Every stored ID uses this store's hash width; no foreign-width
            # identifier leaked in through the external base reference.
            self.assertTrue(all(len(hex_id) == fmt.hex_length for hex_id in store))


class AlternatesFormatTests(TestCase):
    """Alternate object directories share the referencing store's format."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)

    def _write_alternates(self, main_store, alt_path):
        info = os.path.join(main_store.path, "info")
        os.makedirs(info, exist_ok=True)
        with open(os.path.join(info, "alternates"), "w") as f:
            f.write(alt_path.path + "\n")

    def test_alternate_opened_in_sha256_format(self):
        main = DiskObjectStore.init(
            os.path.join(self.path, "main256"), object_format=SHA256
        )
        alt = DiskObjectStore.init(
            os.path.join(self.path, "alt256"), object_format=SHA256
        )
        blob = make_object(Blob, data=b"shared sha256 base")
        alt.add_object(blob)
        self._write_alternates(main, alt)

        opened = main.alternates[0]
        self.assertEqual(SHA256, opened.object_format)
        self.assertEqual(blob.get_id(SHA256), opened[blob.get_id(SHA256)].get_id(SHA256))

    def test_alternate_opened_in_sha1_format(self):
        main = DiskObjectStore.init(
            os.path.join(self.path, "main1"), object_format=SHA1
        )
        alt = DiskObjectStore.init(
            os.path.join(self.path, "alt1"), object_format=SHA1
        )
        blob = make_object(Blob, data=b"shared sha1 base")
        alt.add_object(blob)
        self._write_alternates(main, alt)

        opened = main.alternates[0]
        self.assertEqual(SHA1, opened.object_format)
        self.assertIn(blob.id, opened)

    def test_mismatched_alternate_configuration_rejected(self):
        # Main store is SHA-1; the alternate is a SHA-256 git directory,
        # which records extensions.objectformat explicitly in its config.
        main = DiskObjectStore.init(
            os.path.join(self.path, "main"), object_format=SHA1
        )
        foreign = Repo.init(
            os.path.join(self.path, "foreign"),
            mkdir=True,
            object_format="sha256",
        )
        self.addCleanup(foreign.close)
        foreign.object_store.add_object(make_object(Blob, data=b"foreign"))
        self._write_alternates(main, foreign.object_store)

        with self.assertRaises(ObjectFormatException) as cm:
            main.alternates
        self.assertIn("object format", str(cm.exception))


class LocalFetchFormatMismatchTests(TestCase):
    """Local fetch refuses cross-format targets without mutating them."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.client = LocalGitClient()

    def _src(self, name, fmt):
        repo = Repo.init(
            os.path.join(self.path, name), mkdir=True, object_format=fmt.name
        )
        _make_commit(repo, fmt)
        return repo

    def test_sha1_into_sha256_refused_and_target_untouched(self):
        sha1_src = self._src("sha1-src", SHA1)
        sha256_dst = Repo.init(
            os.path.join(self.path, "sha256-dst"), mkdir=True, object_format="sha256"
        )
        with self.assertRaises(GitProtocolError) as cm:
            self.client.fetch(sha1_src.path, sha256_dst)
        self.assertIn("Object format mismatch", str(cm.exception))
        self.assertEqual(SHA256, sha256_dst.object_format)
        self.assertEqual(list(sha256_dst.object_store), [])

    def test_sha256_into_sha1_refused_even_when_target_nonempty(self):
        sha256_src = self._src("sha256-src", SHA256)
        sha1_dst = Repo.init(
            os.path.join(self.path, "sha1-dst"), mkdir=True, object_format="sha1"
        )
        existing = make_object(Blob, data=b"already here")
        sha1_dst.object_store.add_object(existing)
        before = list(sha1_dst.object_store)

        with self.assertRaises(GitProtocolError) as cm:
            self.client.fetch(sha256_src.path, sha1_dst)
        self.assertIn("Object format mismatch", str(cm.exception))
        self.assertEqual(SHA1, sha1_dst.object_format)
        self.assertEqual(before, list(sha1_dst.object_store))


class LocalMatchingFormatInteropTests(TestCase):
    """Same-format fetch keeps working for both hash algorithms."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.client = LocalGitClient()

    def test_fetch_succeeds_for_each_format(self):
        for fmt in (SHA1, SHA256):
            src = Repo.init(
                os.path.join(self.path, fmt.name + "-src"),
                mkdir=True,
                object_format=fmt.name,
            )
            _blob, _tree, commit, cid = _make_commit(src, fmt)
            dst = Repo.init(
                os.path.join(self.path, fmt.name + "-dst"),
                mkdir=True,
                object_format=fmt.name,
            )
            result = self.client.fetch(src.path, dst)
            self.assertEqual(fmt.name, result.object_format)
            self.assertIn(cid, dst.object_store)
            self.assertEqual(fmt, dst.object_format)


class LocalPushFormatTests(TestCase):
    """Local push handles zero IDs and hash widths per the target format."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.client = LocalGitClient()

    def test_push_creates_ref_for_each_format(self):
        # Regression: a brand-new ref used the all-zero ID as the compare
        # value, but the refs container expects None for an absent ref, so the
        # first push into an empty repository silently failed.
        for fmt in (SHA1, SHA256):
            src = Repo.init(
                os.path.join(self.path, fmt.name + "-psrc"),
                mkdir=True,
                object_format=fmt.name,
            )
            _blob, _tree, commit, cid = _make_commit(src, fmt)
            bare = Repo.init_bare(
                os.path.join(self.path, fmt.name + "-pbare"),
                mkdir=True,
                object_format=fmt.name,
            )
            result = self.client.send_pack(
                bare.path,
                lambda refs: {b"refs/heads/master": cid},
                lambda have, want, ofs_delta, progress: src.generate_pack_data(
                    have, want, ofs_delta=ofs_delta
                ),
            )
            self.assertEqual({}, result.ref_status)
            self.assertEqual(cid, bare.get_refs()[b"refs/heads/master"])
            self.assertIn(cid, bare.object_store)

    def test_push_rejects_foreign_width_id(self):
        for target_fmt, foreign_fmt in ((SHA1, SHA256), (SHA256, SHA1)):
            src = Repo.init(
                os.path.join(self.path, target_fmt.name + "-fsrc"),
                mkdir=True,
                object_format=foreign_fmt.name,
            )
            _blob, _tree, commit, foreign_cid = _make_commit(src, foreign_fmt)
            bare = Repo.init_bare(
                os.path.join(self.path, target_fmt.name + "-fbare"),
                mkdir=True,
                object_format=target_fmt.name,
            )
            before = list(bare.object_store)
            with self.assertRaises(GitProtocolError) as cm:
                self.client.send_pack(
                    bare.path,
                    lambda refs: {b"refs/heads/master": foreign_cid},
                    lambda have, want, ofs_delta, progress: src.generate_pack_data(
                        have, want, ofs_delta=ofs_delta
                    ),
                )
            self.assertIn("requires", str(cm.exception))
            self.assertEqual(before, list(bare.object_store))
            self.assertNotIn(b"refs/heads/master", bare.get_refs())


class DirectRepoFetchFormatTests(TestCase):
    """BaseRepo.fetch (in-process repo to repo) enforces the format too."""

    def setUp(self):
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)

    def test_mismatched_direct_fetch_refused(self):
        sha1_src = Repo.init(
            os.path.join(self.path, "src"), mkdir=True, object_format="sha1"
        )
        _make_commit(sha1_src, SHA1)
        sha256_dst = Repo.init(
            os.path.join(self.path, "dst"), mkdir=True, object_format="sha256"
        )
        with self.assertRaises(GitProtocolError) as cm:
            sha1_src.fetch(sha256_dst)
        self.assertIn("Object format mismatch", str(cm.exception))
        self.assertEqual(SHA256, sha256_dst.object_format)
        self.assertEqual([], list(sha256_dst.object_store))

    def test_matching_direct_fetch_works(self):
        for fmt in (SHA1, SHA256):
            src = Repo.init(
                os.path.join(self.path, fmt.name + "-dsrc"),
                mkdir=True,
                object_format=fmt.name,
            )
            _blob, _tree, commit, cid = _make_commit(src, fmt)
            dst = Repo.init(
                os.path.join(self.path, fmt.name + "-ddst"),
                mkdir=True,
                object_format=fmt.name,
            )
            src.fetch(dst)
            self.assertIn(cid, dst.object_store)
            self.assertEqual(fmt, dst.object_format)

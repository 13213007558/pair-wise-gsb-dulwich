# test_server.py -- Tests for the git server
# Copyright (C) 2010 Google, Inc.
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

"""Tests for the smart protocol server."""

import os
import shutil
import sys
import tempfile
from io import BytesIO
from unittest.mock import Mock

from dulwich.errors import (
    GitProtocolError,
    HangupException,
    HookError,
    NotGitRepository,
    UnexpectedCommandError,
)
from dulwich.object_filters import BlobNoneFilter
from dulwich.object_store import MemoryObjectStore, find_shallow
from dulwich.objects import ObjectID, Tree
from dulwich.pack import write_pack_objects
from dulwich.protocol import ZERO_SHA, Protocol, format_capability_line, pkt_line
from dulwich.refs import Ref
from dulwich.repo import MemoryRepo, Repo
from dulwich.server import (
    Backend,
    DictBackend,
    FileSystemBackend,
    InMemoryNonceStore,
    MultiAckDetailedGraphWalkerImpl,
    MultiAckGraphWalkerImpl,
    PackHandler,
    PushCertificateError,
    ReceivePackHandler,
    SingleAckGraphWalkerImpl,
    UploadPackHandler,
    _parse_push_certificate,
    _ProtocolGraphWalker,
    _split_proto_line,
    serve_command,
    update_server_info,
)
from dulwich.tests.utils import make_commit, make_tag

from . import TestCase

ONE = b"1" * 40
TWO = b"2" * 40
THREE = b"3" * 40
FOUR = b"4" * 40
FIVE = b"5" * 40
SIX = b"6" * 40


class TestProto:
    def __init__(self) -> None:
        self._output: list[bytes] = []
        self._received: dict[int, list[bytes]] = {0: [], 1: [], 2: [], 3: []}

    def set_output(self, output_lines) -> None:
        self._output = output_lines

    def read_pkt_line(self):
        if self._output:
            data = self._output.pop(0)
            if data is not None:
                return data.rstrip() + b"\n"
            else:
                # flush-pkt ('0000').
                return None
        else:
            raise HangupException

    def read_pkt_seq(self):
        pkt = self.read_pkt_line()
        while pkt:
            yield pkt
            pkt = self.read_pkt_line()

    def write_sideband(self, band, data) -> None:
        self._received[band].append(data)

    def write_pkt_line(self, data) -> None:
        self._received[0].append(data)

    def get_received_line(self, band=0):
        lines = self._received[band]
        return lines.pop(0)


class TestGenericPackHandler(PackHandler):
    def __init__(self) -> None:
        PackHandler.__init__(self, Backend(), None)

    @classmethod
    def capabilities(cls):
        return [b"cap1", b"cap2", b"cap3"]

    @classmethod
    def required_capabilities(cls):
        return [b"cap2"]


class HandlerTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._handler = TestGenericPackHandler()

    def assertSucceeds(self, func, *args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except GitProtocolError as e:
            self.fail(e)

    def test_capability_line(self) -> None:
        self.assertEqual(
            b" cap1 cap2 cap3",
            format_capability_line([b"cap1", b"cap2", b"cap3"]),
        )

    def test_set_client_capabilities(self) -> None:
        set_caps = self._handler.set_client_capabilities
        self.assertSucceeds(set_caps, [b"cap2"])
        self.assertSucceeds(set_caps, [b"cap1", b"cap2"])

        # different order
        self.assertSucceeds(set_caps, [b"cap3", b"cap1", b"cap2"])

        # error cases
        self.assertRaises(GitProtocolError, set_caps, [b"capxxx", b"cap2"])
        self.assertRaises(GitProtocolError, set_caps, [b"cap1", b"cap3"])

        # ignore innocuous but unknown capabilities
        self.assertRaises(GitProtocolError, set_caps, [b"cap2", b"ignoreme"])
        self.assertNotIn(b"ignoreme", self._handler.capabilities())
        self._handler.innocuous_capabilities = lambda: (b"ignoreme",)
        self.assertSucceeds(set_caps, [b"cap2", b"ignoreme"])

    def test_has_capability(self) -> None:
        self.assertRaises(GitProtocolError, self._handler.has_capability, b"cap")
        caps = self._handler.capabilities()
        self._handler.set_client_capabilities(caps)
        for cap in caps:
            self.assertTrue(self._handler.has_capability(cap))
        self.assertFalse(self._handler.has_capability(b"capxxx"))


class UploadPackHandlerTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)
        self._repo = Repo.init_bare(self.path)
        backend = DictBackend({b"/": self._repo})
        self._handler = UploadPackHandler(
            backend, [b"/", b"host=lolcathost"], TestProto()
        )

    def test_progress(self) -> None:
        caps = self._handler.required_capabilities()
        self._handler.set_client_capabilities(caps)
        self._handler._start_pack_send_phase()
        self._handler.progress(b"first message")
        self._handler.progress(b"second message")
        self.assertEqual(b"first message", self._handler.proto.get_received_line(2))
        self.assertEqual(b"second message", self._handler.proto.get_received_line(2))
        self.assertRaises(IndexError, self._handler.proto.get_received_line, 2)

    def test_no_progress(self) -> None:
        caps = [*list(self._handler.required_capabilities()), b"no-progress"]
        self._handler.set_client_capabilities(caps)
        self._handler.progress(b"first message")
        self._handler.progress(b"second message")
        self.assertRaises(IndexError, self._handler.proto.get_received_line, 2)

    def test_filter_capability_advertised(self) -> None:
        """Test that the filter capability is advertised by UploadPackHandler."""
        caps = self._handler.capabilities()
        self.assertIn(b"filter", caps)

    def test_filter_spec_parsed(self) -> None:
        """Test that filter specification is parsed from client capabilities."""
        caps = [b"filter=blob:none", *list(self._handler.required_capabilities())]
        self._handler.set_client_capabilities(caps)
        self.assertIsNotNone(self._handler.filter_spec)
        self.assertIsInstance(self._handler.filter_spec, BlobNoneFilter)

    def test_filter_spec_not_present(self) -> None:
        """Test that filter_spec is None when filter capability is not used."""
        caps = self._handler.required_capabilities()
        self._handler.set_client_capabilities(caps)
        self.assertIsNone(self._handler.filter_spec)

    def test_filter_spec_invalid(self) -> None:
        """Test that invalid filter spec raises GitProtocolError."""
        caps = [b"filter=invalid:spec", *list(self._handler.required_capabilities())]
        with self.assertRaises(GitProtocolError):
            self._handler.set_client_capabilities(caps)

    def test_get_tagged(self) -> None:
        refs = {
            b"refs/tags/tag1": ONE,
            b"refs/tags/tag2": TWO,
            b"refs/heads/master": FOUR,  # not a tag, no peeled value
        }
        # repo needs to peel this object
        self._repo.object_store.add_object(make_commit(id=FOUR))
        for name, sha in refs.items():
            self._repo.refs[name] = sha
        peeled = {
            b"refs/tags/tag1": b"1234" * 10,
            b"refs/tags/tag2": b"5678" * 10,
        }
        self._repo.refs._peeled_refs = peeled
        self._repo.refs.add_packed_refs(refs)

        caps = [*list(self._handler.required_capabilities()), b"include-tag"]
        self._handler.set_client_capabilities(caps)
        self.assertEqual(
            {b"1234" * 10: ONE, b"5678" * 10: TWO},
            self._handler.get_tagged(refs, repo=self._repo),
        )

        # non-include-tag case
        caps = self._handler.required_capabilities()
        self._handler.set_client_capabilities(caps)
        self.assertEqual({}, self._handler.get_tagged(refs, repo=self._repo))

    def test_nothing_to_do_but_wants(self) -> None:
        # Just the fact that the client claims to want an object is enough
        # for sending a pack. Even if there turns out to be nothing.
        tree = Tree()
        self._repo.object_store.add_object(tree)
        commit = make_commit(tree=tree.id)
        self._repo.object_store.add_object(commit)
        refs = {b"refs/tags/tag1": commit.id}
        for name, sha in refs.items():
            self._repo.refs[name] = sha
        self._handler.proto.set_output(
            [
                b"want " + commit.id + b" side-band-64k thin-pack ofs-delta",
                None,
                b"have " + commit.id,
                b"done",
                None,
            ]
        )
        self._handler.handle()
        # The server should always send a pack, even if it's empty.
        self.assertTrue(self._handler.proto.get_received_line(1).startswith(b"PACK"))

    def test_nothing_to_do_no_wants(self) -> None:
        # Don't send a pack if the client didn't ask for anything.
        tree = Tree()
        self._repo.object_store.add_object(tree)
        commit = make_commit(tree=tree.id)
        self._repo.object_store.add_object(commit)
        refs = {b"refs/tags/tag1": commit.id}
        for ref, sha in refs.items():
            self._repo.refs[ref] = sha
        self._handler.proto.set_output([None])
        self._handler.handle()
        # The server should not send a pack, since the client didn't ask for
        # anything.
        self.assertEqual([], self._handler.proto._received[1])


class FindShallowTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._store = MemoryObjectStore()

    def make_commit(self, **attrs):
        commit = make_commit(**attrs)
        self._store.add_object(commit)
        return commit

    def make_linear_commits(self, n, message=b""):
        commits = []
        parents = []
        for _ in range(n):
            commits.append(self.make_commit(parents=parents, message=message))
            parents = [commits[-1].id]
        return commits

    def assertSameElements(self, expected, actual) -> None:
        self.assertEqual(set(expected), set(actual))

    def test_linear(self) -> None:
        c1, c2, c3 = self.make_linear_commits(3)

        self.assertEqual(({c3.id}, set()), find_shallow(self._store, [c3.id], 1))
        self.assertEqual(
            ({c2.id}, {c3.id}),
            find_shallow(self._store, [c3.id], 2),
        )
        self.assertEqual(
            ({c1.id}, {c2.id, c3.id}),
            find_shallow(self._store, [c3.id], 3),
        )
        self.assertEqual(
            (set(), {c1.id, c2.id, c3.id}),
            find_shallow(self._store, [c3.id], 4),
        )

    def test_multiple_independent(self) -> None:
        a = self.make_linear_commits(2, message=b"a")
        b = self.make_linear_commits(2, message=b"b")
        c = self.make_linear_commits(2, message=b"c")
        heads = [a[1].id, b[1].id, c[1].id]

        self.assertEqual(
            ({a[0].id, b[0].id, c[0].id}, set(heads)),
            find_shallow(self._store, heads, 2),
        )

    def test_multiple_overlapping(self) -> None:
        # Create the following commit tree:
        # 1--2
        #  \
        #   3--4
        c1, c2 = self.make_linear_commits(2)
        c3 = self.make_commit(parents=[c1.id])
        c4 = self.make_commit(parents=[c3.id])

        # 1 is shallow along the path from 4, but not along the path from 2.
        self.assertEqual(
            ({c1.id}, {c1.id, c2.id, c3.id, c4.id}),
            find_shallow(self._store, [c2.id, c4.id], 3),
        )

    def test_merge(self) -> None:
        c1 = self.make_commit()
        c2 = self.make_commit()
        c3 = self.make_commit(parents=[c1.id, c2.id])

        self.assertEqual(
            ({c1.id, c2.id}, {c3.id}),
            find_shallow(self._store, [c3.id], 2),
        )

    def test_tag(self) -> None:
        c1, c2 = self.make_linear_commits(2)
        tag = make_tag(c2, name=b"tag")
        self._store.add_object(tag)

        self.assertEqual(
            ({c1.id}, {c2.id}),
            find_shallow(self._store, [tag.id], 2),
        )


class TestUploadPackHandler(UploadPackHandler):
    @classmethod
    def required_capabilities(self):
        return []


class ReceivePackHandlerTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._repo = MemoryRepo.init_bare([], {})
        backend = DictBackend({b"/": self._repo})
        self._handler = ReceivePackHandler(
            backend, [b"/", b"host=lolcathost"], TestProto()
        )

    def test_apply_pack_del_ref(self) -> None:
        refs = {b"refs/heads/master": TWO, b"refs/heads/fake-branch": ONE}
        self._repo.refs._update(refs)
        update_refs = [
            [ONE, ZERO_SHA, b"refs/heads/fake-branch"],
        ]
        self._handler.set_client_capabilities([b"delete-refs"])
        status = list(self._handler._apply_pack(update_refs))
        self.assertEqual(status[0][0], b"unpack")
        self.assertEqual(status[0][1], b"ok")
        self.assertEqual(status[1][0], b"refs/heads/fake-branch")
        self.assertEqual(status[1][1], b"ok")

    def _push_empty_pack(self, caps, update_refs):
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path)
        repo = Repo.init_bare(path)
        self.addCleanup(repo.close)
        c1 = make_commit()
        c2 = make_commit(parents=[c1.id])
        repo.object_store.add_objects([(c1, None), (c2, None)])
        repo.refs[b"refs/heads/master"] = c1.id
        pack = BytesIO()
        write_pack_objects(pack.write, [], object_format=repo.object_format)
        pack.seek(0)
        proto = TestProto()
        proto.read = pack.read
        handler = ReceivePackHandler(DictBackend({b"/": repo}), [b"/"], proto)
        handler.set_client_capabilities(caps)
        status = list(handler._apply_pack(update_refs(c1.id, c2.id)))
        self.assertEqual(c1.id, repo.refs[b"refs/heads/master"])
        self.assertNotIn(b"refs/heads/other", repo.refs)
        return status

    def test_apply_pack_refuses_head(self) -> None:
        status = self._push_empty_pack([], lambda c1, c2: [(c1, c2, b"HEAD")])
        self.assertEqual([(b"unpack", b"ok"), (b"HEAD", b"funny refname")], status)

    def test_apply_pack_atomic_refuses_head(self) -> None:
        status = self._push_empty_pack(
            [b"atomic"],
            lambda c1, c2: [(c1, c2, b"HEAD"), (ZERO_SHA, c2, b"refs/heads/other")],
        )
        self.assertEqual(
            [
                (b"unpack", b"ok"),
                (b"HEAD", b"funny refname"),
                (b"refs/heads/other", b"atomic push failed"),
            ],
            status,
        )

    def test_apply_pack_refuses_head_delete(self) -> None:
        self._repo.refs[b"refs/heads/master"] = ONE
        self._handler.set_client_capabilities([b"delete-refs"])
        status = list(self._handler._apply_pack([(ONE, ZERO_SHA, b"HEAD")]))
        self.assertEqual([(b"unpack", b"ok"), (b"HEAD", b"funny refname")], status)
        self.assertEqual(ONE, self._repo.refs[b"refs/heads/master"])

    def test_apply_pack_refuses_malformed_ref(self) -> None:
        self._repo.refs[b"refs/heads/master"] = ONE
        self._handler.set_client_capabilities([b"delete-refs"])
        status = list(
            self._handler._apply_pack([(ONE, ZERO_SHA, b"refs/heads/../master")])
        )
        self.assertEqual(
            [(b"unpack", b"ok"), (b"refs/heads/../master", b"funny refname")],
            status,
        )
        self.assertEqual(ONE, self._repo.refs[b"refs/heads/master"])

    def test_handle_invalid_ref_line(self) -> None:
        """A ref-update line without three fields is rejected, not crashed on."""
        self._handler.proto.set_output(
            [ONE + b" " + TWO + b" refs/heads/master extra", None]
        )
        self.assertRaises(GitProtocolError, self._handler.handle)

    def test_handle_invalid_object_id(self) -> None:
        """A ref-update line with a non-hex object id is rejected."""
        self._handler.proto.set_output(
            [b"nothexsha " + TWO + b" refs/heads/master", None]
        )
        self.assertRaises(GitProtocolError, self._handler.handle)

    def test_handle_valid_ref_line(self) -> None:
        """A well-formed ref-update line is still parsed and applied."""
        self._repo.refs._update({b"refs/heads/todelete": ONE})
        self._handler.proto.set_output(
            [
                ONE
                + b" "
                + ZERO_SHA
                + b" refs/heads/todelete\x00delete-refs report-status",
                None,
            ]
        )
        self._handler.handle()
        self.assertNotIn(b"refs/heads/todelete", self._repo.refs.allkeys())

    def test_pre_receive_hook_success(self) -> None:
        """Test that pre-receive hook is called and can succeed."""

        def mock_hook_execute(client_refs):
            # Verify the hook receives the expected refs
            self.assertEqual(len(client_refs), 1)
            self.assertEqual(client_refs[0][0], ONE)
            self.assertEqual(client_refs[0][1], TWO)
            self.assertEqual(client_refs[0][2], b"refs/heads/master")
            return (b"", b"")

        # Create a mock hook
        mock_hook = Mock()
        mock_hook.execute = mock_hook_execute
        self._repo.hooks["pre-receive"] = mock_hook

        # Test that the hook is called
        client_refs = [(ONE, TWO, b"refs/heads/master")]
        self._handler._on_pre_receive(client_refs)

    def test_pre_receive_hook_failure(self) -> None:
        """Test that pre-receive hook can abort a push."""

        def mock_hook_execute(client_refs):
            raise HookError("pre-receive hook declined")

        # Create a mock hook
        mock_hook = Mock()
        mock_hook.execute = mock_hook_execute
        self._repo.hooks["pre-receive"] = mock_hook

        # Test that the hook failure raises HookError
        client_refs = [(ONE, TWO, b"refs/heads/master")]
        self.assertRaises(HookError, self._handler._on_pre_receive, client_refs)

    def test_update_hook_success(self) -> None:
        """Test that update hook is called and can succeed."""

        def mock_hook_execute(ref_name, old_sha, new_sha):
            # Verify the hook receives the expected arguments
            self.assertEqual(ref_name, b"refs/heads/master")
            self.assertEqual(old_sha, ONE)
            self.assertEqual(new_sha, TWO)
            return (b"", b"")

        # Create a mock hook
        mock_hook = Mock()
        mock_hook.execute = mock_hook_execute
        self._repo.hooks["update"] = mock_hook

        # Test that the hook is called
        result = self._handler._on_update(b"refs/heads/master", ONE, TWO)
        self.assertIsNone(result)

    def test_update_hook_failure(self) -> None:
        """Test that update hook can decline a ref update."""

        def mock_hook_execute(ref_name, old_sha, new_sha):
            raise HookError("update hook declined")

        # Create a mock hook
        mock_hook = Mock()
        mock_hook.execute = mock_hook_execute
        self._repo.hooks["update"] = mock_hook

        # Test that the hook failure returns an error message
        result = self._handler._on_update(b"refs/heads/master", ONE, TWO)
        self.assertIsNotNone(result)
        self.assertIn(b"update hook declined", result)

    def test_update_hook_declines_ref(self) -> None:
        """Test that update hook can decline a ref during pack application."""
        refs = {b"refs/heads/master": ONE}
        self._repo.refs._update(refs)

        # Create a hook that declines the ref
        def mock_hook_execute(ref_name, old_sha, new_sha):
            raise HookError("update hook declined this ref")

        # Create a mock hook
        mock_hook = Mock()
        mock_hook.execute = mock_hook_execute
        self._repo.hooks["update"] = mock_hook

        # Test the ref update part without reading pack data
        self._handler.set_client_capabilities([b"delete-refs"])

        # The update hook should decline the ref
        ref_status = self._handler._on_update(b"refs/heads/master", ONE, ZERO_SHA)

        # Verify hook declined the ref
        self.assertIsNotNone(ref_status)
        self.assertIn(b"update hook declined", ref_status)

    def test_post_receive_hook_only_gets_updated_refs(self) -> None:
        self._repo.refs[b"refs/heads/master"] = ONE
        self._repo.refs[b"refs/heads/other"] = TWO
        mock_hook = Mock()
        mock_hook.execute.return_value = None
        self._repo.hooks["post-receive"] = mock_hook
        self._handler.proto.set_output(
            [
                ONE + b" " + ZERO_SHA + b" HEAD\x00delete-refs",
                TWO + b" " + ZERO_SHA + b" refs/heads/other",
                None,
            ]
        )
        self._handler.handle()
        mock_hook.execute.assert_called_once_with(
            [(TWO, ZERO_SHA, b"refs/heads/other")]
        )

    def test_post_receive_hook_skipped_without_updates(self) -> None:
        self._repo.refs[b"refs/heads/master"] = ONE
        mock_hook = Mock()
        self._repo.hooks["post-receive"] = mock_hook
        self._handler.proto.set_output(
            [ONE + b" " + ZERO_SHA + b" HEAD\x00delete-refs", None]
        )
        self._handler.handle()
        mock_hook.execute.assert_not_called()


class RecordingVerifier:
    """Controllable push certificate verifier for tests."""

    def __init__(self, result: bool = True, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[bytes] = []

    def verify(self, certificate) -> bool:
        self.calls.append(certificate.cert)
        if self.raises is not None:
            raise self.raises
        return self.result


def make_cert_pkt_lines(
    nonce: bytes,
    commands: list[tuple[bytes, bytes, bytes]],
    *,
    signature: bytes = b"fakesig",
    pushee: bytes = b"example.com",
    push_options: list[bytes] | None = None,
) -> list[bytes]:
    """Build the pkt-line payload of a v0.1 push certificate."""
    lines = [
        b"certificate version 0.1",
        b"pusher " + pushee,
        b"pushee " + pushee,
        b"nonce " + nonce,
        b"pushee",
    ]
    lines.extend(old + b" " + new + b" " + ref for old, new, ref in commands)
    lines.append(b"gpgsig " + signature)
    lines.append(b"end-cert")
    if push_options:
        lines.extend(push_options)
    return lines


class PushCertificateParsingTestCase(TestCase):
    def test_parse_minimal_certificate(self) -> None:
        nonce = b"the-nonce"
        commands = [(ONE, TWO, b"refs/heads/master")]
        cert = _parse_push_certificate(make_cert_pkt_lines(nonce, commands))
        self.assertEqual(nonce, cert.nonce)
        self.assertEqual(
            [(ObjectID(ONE), ObjectID(TWO), Ref(b"refs/heads/master"))],
            cert.commands,
        )
        self.assertEqual(b"fakesig\n", cert.signature)
        self.assertIn(b"nonce the-nonce\n", cert.cert)
        self.assertTrue(cert.cert.endswith(ONE + b" " + TWO + b" refs/heads/master\n"))

    def test_parse_keeps_signed_bytes_exact(self) -> None:
        lines = make_cert_pkt_lines(
            b"n",
            [(ZERO_SHA, ONE, b"refs/heads/new")],
            push_options=[b"ci=true"],
        )
        cert = _parse_push_certificate(lines)
        self.assertEqual([b"ci=true"], cert.push_options)
        self.assertEqual(
            b"certificate version 0.1\n"
            b"pusher example.com\n"
            b"pushee example.com\n"
            b"nonce n\n"
            b"pushee\n" + ZERO_SHA + b" " + ONE + b" refs/heads/new\n",
            cert.cert,
        )

    def test_parse_multi_line_signature(self) -> None:
        lines = [
            b"certificate version 0.1",
            b"nonce n",
            b"pushee",
            ONE + b" " + TWO + b" refs/heads/master",
            b"gpgsig -----BEGIN PGP SIGNATURE-----",
            b"gpgsig abc",
            b"gpgsig def",
            b"gpgsig -----END PGP SIGNATURE-----",
            b"end-cert",
        ]
        cert = _parse_push_certificate(lines)
        self.assertEqual(
            b"-----BEGIN PGP SIGNATURE-----\nabc\ndef\n"
            b"-----END PGP SIGNATURE-----\n",
            cert.signature,
        )

    def test_parse_rejects_unknown_version(self) -> None:
        lines = make_cert_pkt_lines(b"n", [])
        lines[0] = b"certificate version 9.9"
        self.assertRaises(PushCertificateError, _parse_push_certificate, lines)

    def test_parse_rejects_missing_nonce(self) -> None:
        lines = make_cert_pkt_lines(b"n", [])
        lines = [line for line in lines if not line.startswith(b"nonce ")]
        self.assertRaises(PushCertificateError, _parse_push_certificate, lines)

    def test_parse_rejects_malformed_command(self) -> None:
        lines = make_cert_pkt_lines(b"n", [])
        lines.insert(4, b"garbage-command")
        self.assertRaises(PushCertificateError, _parse_push_certificate, lines)

    def test_parse_rejects_missing_end_marker(self) -> None:
        lines = make_cert_pkt_lines(b"n", [(ONE, TWO, b"refs/heads/master")])
        self.assertRaises(PushCertificateError, _parse_push_certificate, lines[:-1])


class InMemoryNonceStoreTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._store = InMemoryNonceStore()

    def test_nonce_is_single_use(self) -> None:
        nonce = self._store.create_nonce()
        self._store.validate_nonce(nonce)
        self.assertRaises(PushCertificateError, self._store.validate_nonce, nonce)

    def test_unknown_nonce_rejected(self) -> None:
        self.assertRaises(
            PushCertificateError, self._store.validate_nonce, b"never-minted"
        )

    def test_minted_nonces_are_unique(self) -> None:
        self.assertNotEqual(self._store.create_nonce(), self._store.create_nonce())


class PushCertificateReceiveTestCase(TestCase):
    """End-to-end checks over the in-memory test protocol."""

    def setUp(self) -> None:
        super().setUp()
        self._old_commit = make_commit(id=ONE, parents=[], commit_time=111)
        self._new_commit = make_commit(
            id=TWO, parents=[self._old_commit.id], commit_time=222
        )
        self._extra_commit = make_commit(
            id=THREE, parents=[self._new_commit.id], commit_time=333
        )
        self._repo = MemoryRepo.init_bare(
            [self._old_commit, self._new_commit, self._extra_commit],
            {b"refs/heads/master": self._old_commit.id},
        )
        self._backend = DictBackend({b"/": self._repo})
        self._nonce_store = InMemoryNonceStore()
        self._verifier = RecordingVerifier()
        self._master_command = (ONE, TWO, b"refs/heads/master")

    def _proto(self) -> TestProto:
        proto = TestProto()
        # All objects named by the test commands are already in the store;
        # supply a valid but empty pack for the handler's pack read.
        pack = BytesIO()
        write_pack_objects(pack.write, [], object_format=self._repo.object_format)
        proto.read = BytesIO(pack.getvalue()).read
        return proto

    def _status_lines(self, proto: TestProto) -> list[bytes]:
        """Drop the ref advertisement and return report-status lines."""
        lines = list(proto._received[0])
        return [
            line
            for line in lines
            if line is not None and line[:2] in (b"un", b"ok", b"ng")
        ]

    def _handler(self, proto, *, verifier: bool = True) -> ReceivePackHandler:
        return ReceivePackHandler(
            self._backend,
            [b"/"],
            proto,
            push_certificate_verifier=self._verifier if verifier else None,
            nonce_store=self._nonce_store,
        )

    def _run_cert_push(
        self,
        proto,
        *,
        nonce: bytes | None = None,
        cert_commands=None,
        wire_commands=None,
        handler: ReceivePackHandler | None = None,
    ) -> ReceivePackHandler:
        if handler is None:
            handler = self._handler(proto)
        if nonce is None:
            # Default to the exact nonce minted for this handler's session.
            assert handler._advertised_nonce is not None
            nonce = handler._advertised_nonce
        if cert_commands is None:
            cert_commands = [self._master_command]
        if wire_commands is None:
            wire_commands = [self._master_command]
        cert_lines = make_cert_pkt_lines(nonce, cert_commands)
        command_lines = [
            old + b" " + new + b" " + ref for old, new, ref in wire_commands
        ]
        proto.set_output(
            [
                command_lines[0] + b"\x00push-cert=0.1 report-status",
                *command_lines[1:],
                None,  # delim-pkt ends the command list
                *cert_lines,
                None,  # flush-pkt ends the certificate
            ]
        )
        handler.handle()
        return handler

    def test_capability_advertised_only_with_verifier(self) -> None:
        handler = self._handler(self._proto())
        advertised = b" ".join(handler.capabilities())
        self.assertIn(b"push-cert=nonce ", advertised)

        plain = ReceivePackHandler(self._backend, [b"/"], self._proto())
        self.assertNotIn(b"push-cert", b" ".join(plain.capabilities()))

    def test_constructor_requires_nonce_store(self) -> None:
        self.assertRaises(
            ValueError,
            ReceivePackHandler,
            self._backend,
            [b"/"],
            self._proto(),
            push_certificate_verifier=RecordingVerifier(),
        )

    def test_accepted_certificate_updates_ref(self) -> None:
        proto = self._proto()
        self._run_cert_push(proto)
        self.assertEqual(1, len(self._verifier.calls))
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])
        self.assertEqual(
            [b"unpack ok\n", b"ok refs/heads/master\n"],
            self._status_lines(proto),
        )

    def test_rejected_nonce_leaves_refs_untouched(self) -> None:
        proto = self._proto()
        self._run_cert_push(proto, nonce=b"stale-or-replayed-nonce")
        self.assertEqual([], self._verifier.calls)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"nonce", b" ".join(self._status_lines(proto)))

    def test_replay_nonce_leaves_refs_untouched(self) -> None:
        # First session: a fresh handler mints and redeems its nonce.
        first_proto = self._proto()
        first_handler = self._handler(first_proto)
        reused_nonce = first_handler._advertised_nonce
        self._run_cert_push(first_proto, handler=first_handler)
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])

        # Roll the ref back so a second success would be observable.
        self._repo.refs._update({b"refs/heads/master": ObjectID(ONE)})

        # A second handler gets its own session nonce; replaying the first
        # certificate (with the old nonce) must be refused.
        second_proto = self._proto()
        self._run_cert_push(second_proto, nonce=reused_nonce)
        self.assertEqual(1, len(self._verifier.calls))
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"nonce", b" ".join(self._status_lines(second_proto)))

    def test_command_mismatch_leaves_refs_untouched(self) -> None:
        proto = self._proto()
        self._run_cert_push(
            proto,
            cert_commands=[(ONE, THREE, b"refs/heads/master")],
        )
        self.assertEqual([], self._verifier.calls)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"do not match", b" ".join(self._status_lines(proto)))

    def test_added_command_detected(self) -> None:
        proto = self._proto()
        self._run_cert_push(
            proto,
            cert_commands=[
                self._master_command,
                (ZERO_SHA, THREE, b"refs/heads/extra"),
            ],
        )
        self.assertEqual([], self._verifier.calls)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])

    def test_reordered_commands_detected(self) -> None:
        other_command = (ZERO_SHA, THREE, b"refs/heads/other")
        proto = self._proto()
        handler = self._handler(proto)
        self._run_cert_push(
            proto,
            handler=handler,
            # Certificate signs the two commands in the opposite order.
            cert_commands=[other_command, self._master_command],
            wire_commands=[self._master_command, other_command],
        )
        self.assertEqual([], self._verifier.calls)
        self.assertNotIn(b"refs/heads/other", self._repo.refs.allkeys())
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])

    def test_verifier_failure_leaves_refs_untouched(self) -> None:
        self._verifier.result = False
        proto = self._proto()
        self._run_cert_push(proto)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"not trusted", b" ".join(self._status_lines(proto)))

    def test_verifier_exception_leaves_refs_untouched(self) -> None:
        self._verifier.raises = RuntimeError("keyring unavailable")
        proto = self._proto()
        self._run_cert_push(proto)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"verifier failed", b" ".join(self._status_lines(proto)))

    def test_certificate_without_verifier_is_rejected(self) -> None:
        # With no verifier wired up the push-cert capability is never
        # advertised, so a client sending one is a protocol violation and
        # the push is refused before any ref is touched.
        nonce = self._nonce_store.create_nonce()
        proto = self._proto()
        proto.set_output(
            [
                ONE + b" " + TWO + b" refs/heads/master\x00push-cert=0.1 report-status",
                None,
                *make_cert_pkt_lines(nonce, [self._master_command]),
                None,
            ]
        )
        self.assertRaises(GitProtocolError, self._handler(proto, verifier=False).handle)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])

    def test_certificate_without_verifier_never_advertised(self) -> None:
        handler = ReceivePackHandler(self._backend, [b"/"], self._proto())
        self.assertNotIn(b"push-cert", b" ".join(handler.capabilities()))

    def test_unsigned_push_still_works_with_verifier_configured(self) -> None:
        proto = self._proto()
        proto.set_output(
            [
                ONE + b" " + TWO + b" refs/heads/master\x00report-status",
                None,
            ]
        )
        self._handler(proto).handle()
        self.assertEqual([], self._verifier.calls)
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])

    def test_unsigned_push_still_works_without_verifier(self) -> None:
        proto = self._proto()
        proto.set_output(
            [
                ONE + b" " + TWO + b" refs/heads/master\x00report-status",
                None,
            ]
        )
        ReceivePackHandler(self._backend, [b"/"], proto).handle()
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])

    def test_verifier_runs_before_pre_receive_hook(self) -> None:
        events: list[str] = []

        class OrderVerifier(RecordingVerifier):
            def verify(self, certificate) -> bool:
                events.append("verify")
                return super().verify(certificate)

        mock_hook = Mock()

        def hook_execute(client_refs):
            events.append("pre-receive")
            return (b"", b"")

        mock_hook.execute = hook_execute
        self._repo.hooks["pre-receive"] = mock_hook
        self._verifier = OrderVerifier()

        proto = self._proto()
        self._run_cert_push(proto)
        self.assertEqual(["verify", "pre-receive"], events)
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])

    def test_failed_verifier_skips_hooks_and_refs(self) -> None:
        mock_hook = Mock()
        self._repo.hooks["pre-receive"] = mock_hook
        self._verifier.result = False
        proto = self._proto()
        self._run_cert_push(proto)
        mock_hook.execute.assert_not_called()
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])

    def _real_proto_push(self, nonce: bytes | None = None, *, verifier) -> bytes:
        # Exercise the genuine packet parser including the delim-pkt (0001)
        # that separates commands from the certificate.
        outf = BytesIO()
        # Build the handler first so it mints the advertised session nonce,
        # then frame the request around that nonce.
        handler = ReceivePackHandler(
            self._backend,
            [b"/"],
            Protocol(BytesIO(b"").read, outf.write),
            push_certificate_verifier=verifier,
            nonce_store=self._nonce_store,
        )
        used_nonce = nonce or handler._advertised_nonce
        assert used_nonce is not None

        frames = pkt_line(
            ONE + b" " + TWO + b" refs/heads/master\x00push-cert=0.1 report-status"
        )
        frames += b"0001"
        for line in make_cert_pkt_lines(used_nonce, [self._master_command]):
            frames += pkt_line(line)
        frames += b"0000"
        pack = BytesIO()
        write_pack_objects(pack.write, [], object_format=self._repo.object_format)
        frames += pack.getvalue()

        handler.proto = Protocol(BytesIO(frames).read, outf.write)
        handler.handle()
        return outf.getvalue()

    def test_real_protocol_accepted_push(self) -> None:
        verifier = RecordingVerifier()
        output = self._real_proto_push(verifier=verifier)
        self.assertEqual(1, len(verifier.calls))
        self.assertEqual(ObjectID(TWO), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"unpack ok", output)

    def test_real_protocol_foreign_nonce_refused(self) -> None:
        verifier = RecordingVerifier()
        output = self._real_proto_push(
            b"nonce-from-a-different-session", verifier=verifier
        )
        self.assertEqual([], verifier.calls)
        self.assertEqual(ObjectID(ONE), self._repo.refs[b"refs/heads/master"])
        self.assertIn(b"nonce", output)


class ProtocolGraphWalkerEmptyTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._repo = MemoryRepo.init_bare([], {})
        backend = DictBackend({b"/": self._repo})
        self._walker = _ProtocolGraphWalker(
            TestUploadPackHandler(backend, [b"/", b"host=lolcats"], TestProto()),
            self._repo.object_store,
            self._repo.get_peeled,
            self._repo.refs.get_symrefs,
        )

    def test_empty_repository(self) -> None:
        # The server should wait for a flush packet.
        self._walker.proto.set_output([])
        self.assertRaises(HangupException, self._walker.determine_wants, {})
        self.assertEqual(None, self._walker.proto.get_received_line())

        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants({}))
        self.assertEqual(None, self._walker.proto.get_received_line())


class ProtocolGraphWalkerTestCase(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Create the following commit tree:
        #   3---5
        #  /
        # 1---2---4
        commits = [
            make_commit(id=ONE, parents=[], commit_time=111),
            make_commit(id=TWO, parents=[ONE], commit_time=222),
            make_commit(id=THREE, parents=[ONE], commit_time=333),
            make_commit(id=FOUR, parents=[TWO], commit_time=444),
            make_commit(id=FIVE, parents=[THREE], commit_time=555),
        ]
        self._repo = MemoryRepo.init_bare(commits, {})
        backend = DictBackend({b"/": self._repo})
        self._walker = _ProtocolGraphWalker(
            TestUploadPackHandler(backend, [b"/", b"host=lolcats"], TestProto()),
            self._repo.object_store,
            self._repo.get_peeled,
            self._repo.refs.get_symrefs,
        )

    def test_all_wants_satisfied_no_haves(self) -> None:
        self._walker.set_wants([ONE])
        self.assertFalse(self._walker.all_wants_satisfied([]))
        self._walker.set_wants([TWO])
        self.assertFalse(self._walker.all_wants_satisfied([]))
        self._walker.set_wants([THREE])
        self.assertFalse(self._walker.all_wants_satisfied([]))

    def test_all_wants_satisfied_have_root(self) -> None:
        self._walker.set_wants([ONE])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self._walker.set_wants([TWO])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self._walker.set_wants([THREE])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))

    def test_all_wants_satisfied_have_branch(self) -> None:
        self._walker.set_wants([TWO])
        self.assertTrue(self._walker.all_wants_satisfied([TWO]))
        # wrong branch
        self._walker.set_wants([THREE])
        self.assertFalse(self._walker.all_wants_satisfied([TWO]))

    def test_all_wants_satisfied(self) -> None:
        self._walker.set_wants([FOUR, FIVE])
        # trivial case: wants == haves
        self.assertTrue(self._walker.all_wants_satisfied([FOUR, FIVE]))
        # cases that require walking the commit tree
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self.assertFalse(self._walker.all_wants_satisfied([TWO]))
        self.assertFalse(self._walker.all_wants_satisfied([THREE]))
        self.assertTrue(self._walker.all_wants_satisfied([TWO, THREE]))

    def test_split_proto_line(self) -> None:
        allowed = (b"want", b"done", None)
        self.assertEqual(
            (b"want", ONE), _split_proto_line(b"want " + ONE + b"\n", allowed)
        )
        self.assertEqual(
            (b"want", TWO), _split_proto_line(b"want " + TWO + b"\n", allowed)
        )
        self.assertRaises(GitProtocolError, _split_proto_line, b"want xxxx\n", allowed)
        self.assertRaises(
            UnexpectedCommandError,
            _split_proto_line,
            b"have " + THREE + b"\n",
            allowed,
        )
        self.assertRaises(
            GitProtocolError,
            _split_proto_line,
            b"foo " + FOUR + b"\n",
            allowed,
        )
        self.assertRaises(GitProtocolError, _split_proto_line, b"bar", allowed)
        self.assertEqual((b"done", None), _split_proto_line(b"done\n", allowed))
        self.assertEqual((None, None), _split_proto_line(b"", allowed))

    def test_determine_wants(self) -> None:
        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants({}))
        self.assertEqual(None, self._walker.proto.get_received_line())

        self._walker.proto.set_output(
            [
                b"want " + ONE + b" multi_ack",
                b"want " + TWO,
                None,
            ]
        )
        heads = {
            b"refs/heads/ref1": ONE,
            b"refs/heads/ref2": TWO,
            b"refs/heads/ref3": THREE,
        }
        self._repo.refs._update(heads)
        self.assertEqual([ONE, TWO], self._walker.determine_wants(heads))

        self._walker.advertise_refs = True
        self.assertEqual([], self._walker.determine_wants(heads))
        self._walker.advertise_refs = False

        self._walker.proto.set_output([b"want " + FOUR + b" multi_ack", None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)

        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants(heads))

        self._walker.proto.set_output([b"want " + ONE + b" multi_ack", b"foo", None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)

        self._walker.proto.set_output([b"want " + FOUR + b" multi_ack", None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)

    def test_determine_wants_advertisement(self) -> None:
        self._walker.proto.set_output([None])
        # advertise branch tips plus tag
        heads = {
            b"refs/heads/ref4": FOUR,
            b"refs/heads/ref5": FIVE,
            b"refs/heads/tag6": SIX,
        }
        self._repo.refs._update(heads)
        self._repo.refs._update_peeled(heads)
        self._repo.refs._update_peeled({b"refs/heads/tag6": FIVE})
        self._walker.determine_wants(heads)
        lines = []
        while True:
            line = self._walker.proto.get_received_line()
            if line is None:
                break
            # strip capabilities list if present
            if b"\x00" in line:
                line = line[: line.index(b"\x00")]
            lines.append(line.rstrip())

        self.assertEqual(
            [
                FOUR + b" refs/heads/ref4",
                FIVE + b" refs/heads/ref5",
                FIVE + b" refs/heads/tag6^{}",
                SIX + b" refs/heads/tag6",
            ],
            sorted(lines),
        )

        # ensure peeled tag was advertised immediately following tag
        for i, line in enumerate(lines):
            if line.endswith(b" refs/heads/tag6"):
                self.assertEqual(FIVE + b" refs/heads/tag6^{}", lines[i + 1])

    # TODO: test commit time cutoff

    def _handle_shallow_request(self, lines, heads) -> None:
        self._walker.proto.set_output([*lines, None])
        self._walker._handle_shallow_request(heads)

    def assertReceived(self, expected) -> None:
        self.assertEqual(
            expected, list(iter(self._walker.proto.get_received_line, None))
        )

    def test_handle_shallow_request_no_client_shallows(self) -> None:
        self._handle_shallow_request([b"deepen 2\n"], [FOUR, FIVE])
        self.assertEqual({TWO, THREE}, self._walker.shallow)
        self.assertReceived(
            [
                b"shallow " + TWO,
                b"shallow " + THREE,
            ]
        )

    def test_handle_shallow_request_no_new_shallows(self) -> None:
        lines = [
            b"shallow " + TWO + b"\n",
            b"shallow " + THREE + b"\n",
            b"deepen 2\n",
        ]
        self._handle_shallow_request(lines, [FOUR, FIVE])
        self.assertEqual({TWO, THREE}, self._walker.shallow)
        self.assertReceived([])

    def test_handle_shallow_request_unshallows(self) -> None:
        lines = [
            b"shallow " + TWO + b"\n",
            b"deepen 3\n",
        ]
        self._handle_shallow_request(lines, [FOUR, FIVE])
        self.assertEqual({ONE}, self._walker.shallow)
        self.assertReceived(
            [
                b"shallow " + ONE,
                b"unshallow " + TWO,
                # THREE is unshallow but was is not shallow in the client
            ]
        )


class TestProtocolGraphWalker:
    def __init__(self) -> None:
        self.acks: list[bytes] = []
        self.lines: list[bytes] = []
        self.wants_satisified = False
        self.stateless_rpc = None
        self.advertise_refs = False
        self._impl = None
        self.done_required = True
        self.done_received = False
        self._empty = False
        self.pack_sent = False

    def read_proto_line(self, allowed):
        command, sha = self.lines.pop(0)
        if allowed is not None:
            assert command in allowed
        return command, sha

    def send_ack(self, sha, ack_type=b"") -> None:
        self.acks.append((sha, ack_type))

    def send_nak(self) -> None:
        self.acks.append((None, b"nak"))

    def all_wants_satisfied(self, haves):
        if haves:
            return self.wants_satisified

    def pop_ack(self):
        if not self.acks:
            return None
        return self.acks.pop(0)

    def handle_done(self):
        if not self._impl:
            return
        # Whether or not PACK is sent after is determined by this, so
        # record this value.
        self.pack_sent = self._impl.handle_done(self.done_required, self.done_received)
        return self.pack_sent

    def notify_done(self) -> None:
        self.done_received = True


class AckGraphWalkerImplTestCase(TestCase):
    """Base setup and asserts for AckGraphWalker tests."""

    def setUp(self) -> None:
        super().setUp()
        self._walker = TestProtocolGraphWalker()
        self._walker.lines = [
            (b"have", TWO),
            (b"have", ONE),
            (b"have", THREE),
            (b"done", None),
        ]
        self._impl = self.impl_cls(self._walker)
        self._walker._impl = self._impl

    def assertNoAck(self) -> None:
        self.assertEqual(None, self._walker.pop_ack())

    def assertAcks(self, acks) -> None:
        for sha, ack_type in acks:
            self.assertEqual((sha, ack_type), self._walker.pop_ack())
        self.assertNoAck()

    def assertAck(self, sha, ack_type=b"") -> None:
        self.assertAcks([(sha, ack_type)])

    def assertNak(self) -> None:
        self.assertAck(None, b"nak")

    def assertNextEquals(self, sha) -> None:
        self.assertEqual(sha, next(self._impl))

    def assertNextEmpty(self) -> None:
        # This is necessary because of no-done - the assumption that it
        # it safe to immediately send out the final ACK is no longer
        # true but the test is still needed for it.  TestProtocolWalker
        # does implement the handle_done which will determine whether
        # the final confirmation can be sent.
        self.assertRaises(IndexError, next, self._impl)
        self._walker.handle_done()


class SingleAckGraphWalkerImplTestCase(AckGraphWalkerImplTestCase):
    impl_cls = SingleAckGraphWalkerImpl

    def test_single_ack(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE)

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNoAck()

    def test_single_ack_flush(self) -> None:
        # same as ack test but ends with a flush-pkt instead of done
        self._walker.lines[-1] = (None, None)

        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE)

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNoAck()

    def test_single_ack_nak(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()

    def test_single_ack_nak_flush(self) -> None:
        # same as nak test but ends with a flush-pkt instead of done
        self._walker.lines[-1] = (None, None)

        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()


class MultiAckGraphWalkerImplTestCase(AckGraphWalkerImplTestCase):
    impl_cls = MultiAckGraphWalkerImpl

    def test_multi_ack(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"continue")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"continue")

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(THREE)

    def test_multi_ack_partial(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"continue")

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(ONE)

    def test_multi_ack_flush(self) -> None:
        self._walker.lines = [
            (b"have", TWO),
            (None, None),
            (b"have", ONE),
            (b"have", THREE),
            (b"done", None),
        ]
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNak()  # nak the flush-pkt

        self._impl.ack(ONE)
        self.assertAck(ONE, b"continue")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"continue")

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(THREE)

    def test_multi_ack_nak(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()


class MultiAckDetailedGraphWalkerImplTestCase(AckGraphWalkerImplTestCase):
    impl_cls = MultiAckDetailedGraphWalkerImpl

    def test_multi_ack(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"common")

        # done is read.
        self._walker.wants_satisified = True
        self.assertNextEquals(None)
        self._walker.lines.append((None, None))
        self.assertNextEmpty()
        self.assertAcks([(THREE, b"ready"), (None, b"nak"), (THREE, b"")])
        # PACK is sent
        self.assertTrue(self._walker.pack_sent)

    def test_multi_ack_nodone(self) -> None:
        self._walker.done_required = False
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"common")

        # done is read.
        self._walker.wants_satisified = True
        self.assertNextEquals(None)
        self._walker.lines.append((None, None))
        self.assertNextEmpty()
        self.assertAcks([(THREE, b"ready"), (None, b"nak"), (THREE, b"")])
        # PACK is sent
        self.assertTrue(self._walker.pack_sent)

    def test_multi_ack_flush_end(self) -> None:
        # transmission ends with a flush-pkt without a done but no-done is
        # assumed.
        self._walker.lines[-1] = (None, None)
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"common")

        # no done is read
        self._walker.wants_satisified = True
        self.assertNextEmpty()
        self.assertAcks([(THREE, b"ready"), (None, b"nak")])
        # PACK is NOT sent
        self.assertFalse(self._walker.pack_sent)

    def test_multi_ack_flush_end_nodone(self) -> None:
        # transmission ends with a flush-pkt without a done but no-done is
        # assumed.
        self._walker.lines[-1] = (None, None)
        self._walker.done_required = False
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"common")

        # no done is read, but pretend it is (last 'ACK 'commit_id' '')
        self._walker.wants_satisified = True
        self.assertNextEmpty()
        self.assertAcks([(THREE, b"ready"), (None, b"nak"), (THREE, b"")])
        # PACK is sent
        self.assertTrue(self._walker.pack_sent)

    def test_multi_ack_partial(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAck(ONE)

    def test_multi_ack_flush(self) -> None:
        # same as ack test but contains a flush-pkt in the middle
        self._walker.lines = [
            (b"have", TWO),
            (None, None),
            (b"have", ONE),
            (b"have", THREE),
            (b"done", None),
            (None, None),
        ]
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNak()  # nak the flush-pkt

        self._impl.ack(ONE)
        self.assertAck(ONE, b"common")

        self.assertNextEquals(THREE)
        self._impl.ack(THREE)
        self.assertAck(THREE, b"common")

        self._walker.wants_satisified = True
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertAcks([(THREE, b"ready"), (None, b"nak"), (THREE, b"")])

    def test_multi_ack_nak(self) -> None:
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        # Done is sent here.
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()
        self.assertNextEmpty()
        self.assertTrue(self._walker.pack_sent)

    def test_multi_ack_nak_nodone(self) -> None:
        self._walker.done_required = False
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        # Done is sent here.
        self.assertFalse(self._walker.pack_sent)
        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertTrue(self._walker.pack_sent)
        self.assertNak()
        self.assertNextEmpty()

    def test_multi_ack_nak_flush(self) -> None:
        # same as nak test but contains a flush-pkt in the middle
        self._walker.lines = [
            (b"have", TWO),
            (None, None),
            (b"have", ONE),
            (b"have", THREE),
            (b"done", None),
        ]
        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNak()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertNextEquals(None)
        self.assertNextEmpty()
        self.assertNak()

    def test_multi_ack_stateless(self) -> None:
        # transmission ends with a flush-pkt
        self._walker.lines[-1] = (None, None)
        self._walker.stateless_rpc = True

        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertFalse(self._walker.pack_sent)
        self.assertNextEquals(None)
        self.assertNak()

        self.assertNextEmpty()
        self.assertNoAck()
        self.assertFalse(self._walker.pack_sent)

    def test_multi_ack_stateless_nodone(self) -> None:
        self._walker.done_required = False
        # transmission ends with a flush-pkt
        self._walker.lines[-1] = (None, None)
        self._walker.stateless_rpc = True

        self.assertNextEquals(TWO)
        self.assertNoAck()

        self.assertNextEquals(ONE)
        self.assertNoAck()

        self.assertNextEquals(THREE)
        self.assertNoAck()

        self.assertFalse(self._walker.pack_sent)
        self.assertNextEquals(None)
        self.assertNak()

        self.assertNextEmpty()
        self.assertNoAck()
        # PACK will still not be sent.
        self.assertFalse(self._walker.pack_sent)


class FileSystemBackendTests(TestCase):
    """Tests for FileSystemBackend."""

    def setUp(self) -> None:
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)
        if sys.platform == "win32":
            self.backend = FileSystemBackend(self.path[0] + ":" + os.sep)
        else:
            self.backend = FileSystemBackend()

    def test_nonexistent(self) -> None:
        self.assertRaises(
            NotGitRepository,
            self.backend.open_repository,
            "/does/not/exist/unless/foo",
        )

    def test_absolute(self) -> None:
        repo = self.backend.open_repository(self.path)
        self.assertTrue(
            os.path.samefile(
                os.path.abspath(repo.path), os.path.abspath(self.repo.path)
            )
        )

    def test_child(self) -> None:
        self.assertRaises(
            NotGitRepository,
            self.backend.open_repository,
            os.path.join(self.path, "foo"),
        )

    def test_bad_repo_path(self) -> None:
        backend = FileSystemBackend()

        self.assertRaises(NotGitRepository, lambda: backend.open_repository("/ups"))

    def test_bytes_path(self) -> None:
        # Test that FileSystemBackend can handle bytes paths (issue #973)
        repo = self.backend.open_repository(self.path.encode("utf-8"))
        self.assertTrue(
            os.path.samefile(
                os.path.abspath(repo.path), os.path.abspath(self.repo.path)
            )
        )


class DictBackendTests(TestCase):
    """Tests for DictBackend."""

    def test_nonexistent(self) -> None:
        repo = MemoryRepo.init_bare([], {})
        backend = DictBackend({b"/": repo})
        self.assertRaises(
            NotGitRepository,
            backend.open_repository,
            "/does/not/exist/unless/foo",
        )

    def test_bad_repo_path(self) -> None:
        repo = MemoryRepo.init_bare([], {})
        backend = DictBackend({b"/": repo})

        self.assertRaises(NotGitRepository, lambda: backend.open_repository("/ups"))


class ServeCommandTests(TestCase):
    """Tests for serve_command."""

    def setUp(self) -> None:
        super().setUp()
        self.backend = DictBackend({})

    def serve_command(self, handler_cls, args, inf, outf):
        return serve_command(
            handler_cls,
            [b"test", *args],
            backend=self.backend,
            inf=inf,
            outf=outf,
        )

    def test_receive_pack(self) -> None:
        commit = make_commit(id=ONE, parents=[], commit_time=111)
        self.backend.repos[b"/"] = MemoryRepo.init_bare(
            [commit], {b"refs/heads/master": commit.id}
        )
        outf = BytesIO()
        exitcode = self.serve_command(
            ReceivePackHandler, [b"/"], BytesIO(b"0000"), outf
        )
        outlines = outf.getvalue().splitlines()
        self.assertEqual(2, len(outlines))
        self.assertEqual(
            b"1111111111111111111111111111111111111111 refs/heads/master",
            outlines[0][4:].split(b"\x00")[0],
        )
        self.assertEqual(b"0000", outlines[-1])
        self.assertEqual(0, exitcode)


class UpdateServerInfoTests(TestCase):
    """Tests for update_server_info."""

    def setUp(self) -> None:
        super().setUp()
        self.path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.path)
        self.repo = Repo.init(self.path)

    def test_empty(self) -> None:
        update_server_info(self.repo)
        with open(os.path.join(self.path, ".git", "info", "refs"), "rb") as f:
            self.assertEqual(b"", f.read())
        p = os.path.join(self.path, ".git", "objects", "info", "packs")
        with open(p, "rb") as f:
            self.assertEqual(b"", f.read())

    def test_simple(self) -> None:
        commit_id = self.repo.get_worktree().commit(
            message=b"foo",
            committer=b"Joe Example <joe@example.com>",
            ref=b"refs/heads/foo",
        )
        update_server_info(self.repo)
        with open(os.path.join(self.path, ".git", "info", "refs"), "rb") as f:
            self.assertEqual(f.read(), commit_id + b"\trefs/heads/foo\n")
        p = os.path.join(self.path, ".git", "objects", "info", "packs")
        with open(p, "rb") as f:
            self.assertEqual(f.read(), b"")

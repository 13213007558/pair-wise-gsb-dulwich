# test_sha256_pack.py -- Tests for SHA256 pack support
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

"""Tests for SHA256 pack support in Dulwich."""

import shutil
import tempfile
import unittest
from hashlib import sha1, sha256
from io import BytesIO

from dulwich.object_format import SHA1, SHA256
from dulwich.objects import Blob
from dulwich.pack import (
    PackData,
    UnpackedObject,
    full_unpacked_object,
    load_pack_index_file,
    pack_object_header,
    unpack_object,
    write_pack_data,
    write_pack_index,
    write_pack_index_v1,
    write_pack_index_v2,
    write_pack_index_v3,
    write_pack_objects,
)


class SHA256PackTests(unittest.TestCase):
    """Tests for SHA256 pack support."""

    def setUp(self):
        """Set up test repository directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test repository."""
        shutil.rmtree(self.test_dir)

    def test_pack_index_v2_with_sha256(self):
        """Test that pack index v2 correctly handles SHA256 hashes."""
        # Create SHA256 entries manually (simulating what would happen in a SHA256 repo)
        entries = []
        for i in range(5):
            # Create a fake SHA256 hash
            sha256_hash = SHA256.hash_func(f"test object {i}".encode()).digest()
            offset = i * 1000  # Fake offsets
            crc32 = i  # Fake CRC32
            entries.append((sha256_hash, offset, crc32))

        # Sort entries by SHA (required for pack index)
        entries.sort(key=lambda e: e[0])

        # Write SHA256 pack index with SHA1 pack checksum (Git always uses SHA1 for pack checksums)
        index_buf = BytesIO()
        pack_checksum = sha1(b"fake pack data").digest()
        write_pack_index_v2(index_buf, entries, pack_checksum)

        # Load and verify the index
        index_buf.seek(0)
        pack_idx = load_pack_index_file("<memory>", index_buf, SHA256)
        self.addCleanup(pack_idx.close)

        # Check that the index loaded correctly
        self.assertEqual(len(pack_idx), 5)
        self.assertEqual(pack_idx.version, 2)

        # Verify hash_size detection
        self.assertEqual(pack_idx.hash_size, 32)

        # Verify we can look up objects by SHA256
        for sha256_hash, offset, _ in entries:
            # This should not raise KeyError
            found_offset = pack_idx.object_offset(sha256_hash)
            self.assertEqual(found_offset, offset)

    def test_pack_index_v1_with_sha256(self):
        """Test that pack index v1 correctly rejects SHA256 hashes."""
        # Create SHA256 entries manually
        entries = []
        for i in range(5):
            # Create a fake SHA256 hash
            sha256_hash = SHA256.hash_func(f"test v1 object {i}".encode()).digest()
            offset = i * 1000  # Fake offsets
            crc32 = None  # v1 doesn't store CRC32
            entries.append((sha256_hash, offset, crc32))

        # Sort entries by SHA (required for pack index)
        entries.sort(key=lambda e: e[0])

        # Write SHA256 pack index v1 with SHA1 pack checksum
        index_buf = BytesIO()
        pack_checksum = sha1(b"fake v1 pack data").digest()

        # Pack index v1 only supports SHA-1, so this should raise TypeError
        with self.assertRaises(TypeError) as cm:
            write_pack_index_v1(index_buf, entries, pack_checksum)

        self.assertIn("pack index v1 only supports SHA-1", str(cm.exception))


class SHA256PackWriteTests(unittest.TestCase):
    """Tests for writing SHA256 packs."""

    def setUp(self):
        """Set up test repository directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test repository."""
        shutil.rmtree(self.test_dir)

    def _make_blobs(self, count=3):
        return [Blob.from_string(f"sha256 blob {i}".encode()) for i in range(count)]

    def test_write_pack_objects_sha256(self):
        """write_pack_objects keys entries by 32-byte ids and writes a
        32-byte trailer checksum."""
        blobs = self._make_blobs()
        pack_buf = BytesIO()
        entries, pack_checksum = write_pack_objects(
            pack_buf.write, blobs, object_format=SHA256
        )
        self.assertEqual(32, len(pack_checksum))
        self.assertEqual(len(blobs), len(entries))
        expected_ids = {blob.get_id(SHA256) for blob in blobs}
        self.assertEqual(expected_ids, {k.hex().encode("ascii") for k in entries})
        # The trailer is the SHA-256 of everything that precedes it
        pack_bytes = pack_buf.getvalue()
        self.assertEqual(sha256(pack_bytes[:-32]).digest(), pack_bytes[-32:])
        self.assertEqual(pack_checksum, pack_bytes[-32:])
        # The pack can be read back with the SHA-256 object format
        data = PackData("test.pack", file=BytesIO(pack_bytes), object_format=SHA256)
        self.addCleanup(data.close)
        self.assertEqual(len(blobs), len(data))
        self.assertEqual(pack_checksum, data.get_stored_checksum())

    def test_write_pack_data_sha256_ref_delta(self):
        """REF_DELTA records use 32-byte base ids in SHA-256 packs."""
        blob = self._make_blobs(1)[0]
        base = full_unpacked_object(blob, object_format=SHA256)
        # A delta whose base is not itself written to the pack is encoded
        # as a REF_DELTA with the raw base id
        delta_base = SHA256.hash_object(b"some external base")
        delta = UnpackedObject(
            blob.type_num,
            delta_base=delta_base,
            decomp_chunks=[b"\x90\x80\x04"],
            hash_func=SHA256.hash_func,
        )
        pack_buf = BytesIO()
        entries, pack_checksum = write_pack_data(
            pack_buf.write,
            iter([base, delta]),
            num_records=2,
            object_format=SHA256,
        )
        self.assertEqual(2, len(entries))
        # Read the second object back and check the base id length
        pack_bytes = pack_buf.getvalue()
        delta_offset = next(
            offset for sha, (offset, _crc) in entries.items() if sha == delta.sha()
        )
        unpacked, _unused = unpack_object(
            BytesIO(pack_bytes[delta_offset:]).read, hash_func=SHA256.hash_func
        )
        self.assertEqual(delta_base, unpacked.delta_base)
        self.assertEqual(32, len(unpacked.delta_base))

    def test_pack_object_header_ref_delta_sha256(self):
        """pack_object_header requires a 32-byte base for SHA-256 REF_DELTA."""
        # size 7 fits in a single header byte
        header = pack_object_header(7, SHA256.hash_object(b"base"), 7, SHA256)
        self.assertEqual(1 + 32, len(header))
        # size 42 needs a continuation byte
        header = pack_object_header(7, SHA256.hash_object(b"base"), 42, SHA256)
        self.assertEqual(2 + 32, len(header))
        with self.assertRaises(AssertionError):
            # A 20-byte (SHA-1) base must not be accepted
            pack_object_header(7, sha1(b"base").digest(), 42, SHA256)

    def test_write_pack_index_v2_object_format(self):
        """write_pack_index validates object_format before writing."""
        entries = [(SHA256.hash_object(b"object %d" % i), i * 100, i) for i in range(3)]
        entries.sort(key=lambda e: e[0])
        pack_checksum = sha256(b"pack data").digest()

        # Matching object format works
        index_buf = BytesIO()
        write_pack_index(index_buf, entries, pack_checksum, object_format=SHA256)
        index_buf.seek(0)
        idx = load_pack_index_file("<memory>", index_buf, SHA256)
        self.addCleanup(idx.close)
        self.assertEqual(3, len(idx))
        self.assertEqual(pack_checksum, idx.get_pack_checksum())

        # Mismatched object format raises before anything is written
        for bad_format, bad_checksum, bad_entries in (
            (SHA1, pack_checksum, entries),  # sha1 format, sha256 checksum
            (SHA256, sha1(b"pack").digest(), entries),  # sha256 format, sha1 checksum
            (SHA256, pack_checksum, [(sha1(b"x").digest(), 0, 0)]),  # sha1 name
        ):
            buf = BytesIO()
            with self.assertRaises((ValueError, TypeError)):
                write_pack_index(
                    buf, bad_entries, bad_checksum, object_format=bad_format
                )
            self.assertEqual(b"", buf.getvalue())

    def test_write_pack_index_v1_rejects_sha256_format(self):
        """write_pack_index with version=1 rejects the SHA-256 format."""
        entries = [(SHA256.hash_object(b"object"), 0, 0)]
        buf = BytesIO()
        with self.assertRaises(ValueError):
            write_pack_index(
                buf,
                entries,
                sha256(b"pack").digest(),
                version=1,
                object_format=SHA256,
            )
        self.assertEqual(b"", buf.getvalue())

    def test_write_pack_index_v3_sha256(self):
        """write_pack_index writes SHA-256 v3 indexes."""
        entries = [(SHA256.hash_object(b"object %d" % i), i * 100, i) for i in range(3)]
        entries.sort(key=lambda e: e[0])
        pack_checksum = sha256(b"pack data").digest()
        index_buf = BytesIO()
        write_pack_index(
            index_buf, entries, pack_checksum, version=3, object_format=SHA256
        )
        index_buf.seek(0)
        idx = load_pack_index_file("<memory>", index_buf, SHA256)
        self.addCleanup(idx.close)
        self.assertEqual(3, idx.version)
        self.assertEqual(2, idx.hash_format)
        self.assertEqual(32, idx.hash_size)
        self.assertEqual(entries, list(idx.iterentries()))
        self.assertEqual(pack_checksum, idx.get_pack_checksum())

    def test_write_pack_index_v3_sha256_direct(self):
        """write_pack_index_v3 with hash_format=2 writes SHA-256 indexes."""
        entries = [(SHA256.hash_object(b"object"), 42, 7)]
        pack_checksum = sha256(b"pack data").digest()
        index_buf = BytesIO()
        write_pack_index_v3(index_buf, entries, pack_checksum, hash_format=2)
        index_buf.seek(0)
        idx = load_pack_index_file("<memory>", index_buf, SHA256)
        self.addCleanup(idx.close)
        self.assertEqual(2, idx.hash_format)
        self.assertEqual(entries, list(idx.iterentries()))

    def test_full_unpacked_object_sha256(self):
        """full_unpacked_object uses the requested object format for ids."""
        blob = self._make_blobs(1)[0]
        unpacked = full_unpacked_object(blob, object_format=SHA256)
        self.assertEqual(blob.get_id(SHA256), unpacked.sha().hex().encode("ascii"))
        self.assertEqual(32, len(unpacked.sha()))
        # Default remains SHA-1
        unpacked_sha1 = full_unpacked_object(blob)
        self.assertEqual(20, len(unpacked_sha1.sha()))
        self.assertEqual(blob.id, unpacked_sha1.sha().hex().encode("ascii"))


if __name__ == "__main__":
    unittest.main()

# test_sha256_packs.py -- Compatibility tests for SHA256 pack files
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

"""Compatibility tests for SHA256 pack files with git command line tools."""

import os
import shutil
import tempfile
from io import BytesIO

from dulwich.object_format import SHA256
from dulwich.objects import Blob, Commit, Tree
from dulwich.pack import (
    OFS_DELTA,
    REF_DELTA,
    PackData,
    UnpackedObject,
    create_delta,
    load_pack_index_file,
    write_pack,
    write_pack_data,
)
from dulwich.repo import Repo

from .utils import CompatTestCase, rmtree_ro, run_git_or_fail


class GitSHA256PackCompatibilityTests(CompatTestCase):
    """Test SHA256 pack file compatibility with git command line tools."""

    min_git_version = (2, 29, 0)

    def _run_git(self, args, cwd=None):
        """Run git command in the specified directory."""
        return run_git_or_fail(args, cwd=cwd)

    def test_git_pack_readable_by_dulwich(self):
        """Test that git-created SHA256 pack files are readable by dulwich."""
        # Create SHA256 repo with git
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        self._run_git(["init", "--object-format=sha256", repo_path])

        # Create multiple files to ensure pack creation
        for i in range(20):
            test_file = os.path.join(repo_path, f"file{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Content for file {i}\n")

        self._run_git(["add", "."], cwd=repo_path)
        self._run_git(["commit", "-m", "Add 20 files"], cwd=repo_path)

        # Force pack creation
        self._run_git(["gc"], cwd=repo_path)

        # Open with dulwich
        repo = Repo(repo_path)
        self.assertEqual(repo.object_format, SHA256)

        # Find pack files
        pack_dir = os.path.join(repo_path, ".git", "objects", "pack")
        pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".pack")]
        self.assertGreater(len(pack_files), 0, "No pack files created")

        # Read pack with dulwich
        for pack_file in pack_files:
            pack_path = os.path.join(pack_dir, pack_file)
            idx_path = pack_path[:-5] + ".idx"

            # Load pack index with SHA256 algorithm
            with open(idx_path, "rb") as f:
                pack_idx = load_pack_index_file(idx_path, f, SHA256)

            # Verify it's detected as SHA256
            self.assertEqual(pack_idx.hash_size, 32)

            # Verify we can iterate objects
            obj_count = 0
            for sha, offset, crc32 in pack_idx.iterentries():
                self.assertEqual(len(sha), 32)  # SHA256
                obj_count += 1

            self.assertGreater(obj_count, 20)  # At least our files + trees + commit

        # Verify we can read all objects through the repo interface
        head_ref = repo.refs[b"refs/heads/master"]
        commit = repo[head_ref]
        self.assertIsInstance(commit, Commit)

        # Read the tree
        tree = repo[commit.tree]
        self.assertIsInstance(tree, Tree)

        # Verify all files are there
        file_count = 0
        for name, mode, sha in tree.items():
            if name.startswith(b"file") and name.endswith(b".txt"):
                file_count += 1
                # Read the blob
                blob = repo[sha]
                self.assertIsInstance(blob, Blob)

        self.assertEqual(file_count, 20)
        repo.close()

    def test_dulwich_objects_readable_by_git(self):
        """Test that dulwich-created SHA256 objects are readable by git."""
        # Create SHA256 repo with dulwich
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        repo = Repo.init(repo_path, mkdir=False, object_format="sha256")

        # Create objects
        blobs = []
        for i in range(10):
            blob = Blob.from_string(f"Dulwich blob content {i}".encode())
            repo.object_store.add_object(blob)
            blobs.append(blob)

        # Create a tree with all blobs
        tree = Tree()
        for i, blob in enumerate(blobs):
            tree.add(f"blob{i}.txt".encode(), 0o100644, blob.get_id(SHA256))
        repo.object_store.add_object(tree)

        # Create a commit
        commit = Commit()
        commit.tree = tree.get_id(SHA256)
        commit.author = commit.committer = b"Dulwich Test <test@dulwich.org>"
        commit.commit_time = commit.author_time = 1234567890
        commit.commit_timezone = commit.author_timezone = 0
        commit.message = b"Test commit with blobs"
        repo.object_store.add_object(commit)

        # Update HEAD
        repo.refs[b"refs/heads/master"] = commit.get_id(SHA256)
        repo.close()

        # Verify git can read all objects
        output = self._run_git(["rev-parse", "HEAD"], cwd=repo_path)
        self.assertEqual(len(output.strip()), 64)  # SHA256

        # List tree contents
        tree_output = self._run_git(["ls-tree", "HEAD"], cwd=repo_path)
        # Count lines instead of occurrences of "blob" since "blob" appears twice per line
        lines = tree_output.strip().split(b"\n")
        self.assertEqual(len(lines), 10)

        # Verify git can check out the content
        self._run_git(["checkout", "HEAD", "--", "."], cwd=repo_path)

        # Verify files exist with correct content
        for i in range(10):
            file_path = os.path.join(repo_path, f"blob{i}.txt")
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "rb") as f:
                content = f.read()
                self.assertEqual(content, f"Dulwich blob content {i}".encode())

    def test_large_pack_interop(self):
        """Test large pack file interoperability."""
        # Create repo with dulwich
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        repo = Repo.init(repo_path, mkdir=False, object_format="sha256")

        # Create a large file that will use delta compression
        large_content = b"A" * 10000
        blobs = []

        # Create similar blobs to trigger delta compression
        for i in range(10):
            content = large_content + f" variation {i}".encode()
            blob = Blob.from_string(content)
            repo.object_store.add_object(blob)
            blobs.append(blob)

        # Create tree
        tree = Tree()
        for i, blob in enumerate(blobs):
            tree.add(f"large{i}.txt".encode(), 0o100644, blob.get_id(SHA256))
        repo.object_store.add_object(tree)

        # Create commit
        commit = Commit()
        commit.tree = tree.get_id(SHA256)
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.commit_time = commit.author_time = 1234567890
        commit.commit_timezone = commit.author_timezone = 0
        commit.message = b"Large files for delta compression test"
        repo.object_store.add_object(commit)

        repo.refs[b"refs/heads/master"] = commit.get_id(SHA256)
        repo.close()

        # Run git gc to create packs with delta compression
        self._run_git(["gc", "--aggressive"], cwd=repo_path)

        # Verify git created a pack
        pack_dir = os.path.join(repo_path, ".git", "objects", "pack")
        pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".pack")]
        self.assertGreater(len(pack_files), 0)

        # Re-open with dulwich and verify we can read everything
        repo = Repo(repo_path)
        head = repo.refs[b"refs/heads/master"]
        commit = repo[head]
        tree = repo[commit.tree]

        # Read all blobs
        for i in range(10):
            name = f"large{i}.txt".encode()
            _mode, sha = tree[name]
            blob = repo[sha]
            expected = large_content + f" variation {i}".encode()
            self.assertEqual(blob.data, expected)

        repo.close()

    def test_mixed_loose_packed_objects(self):
        """Test repositories with both loose and packed objects."""
        # Create repo with git
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        self._run_git(["init", "--object-format=sha256", repo_path])

        # Create initial objects that will be packed
        for i in range(5):
            test_file = os.path.join(repo_path, f"packed{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Will be packed {i}\n")

        self._run_git(["add", "."], cwd=repo_path)
        self._run_git(["commit", "-m", "Initial packed objects"], cwd=repo_path)
        self._run_git(["gc"], cwd=repo_path)

        # Create more objects that will remain loose
        for i in range(5):
            test_file = os.path.join(repo_path, f"loose{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Will stay loose {i}\n")

        self._run_git(["add", "."], cwd=repo_path)
        self._run_git(["commit", "-m", "Loose objects"], cwd=repo_path)

        # Open with dulwich
        repo = Repo(repo_path)

        # Count objects in packs vs loose
        pack_dir = os.path.join(repo_path, ".git", "objects", "pack")
        pack_count = len([f for f in os.listdir(pack_dir) if f.endswith(".pack")])
        self.assertGreater(pack_count, 0)

        # Verify we can read all objects
        head = repo.refs[b"refs/heads/master"]
        commit = repo[head]

        # Walk the commit history
        commit_count = 0
        while commit.parents:
            commit_count += 1
            tree = repo[commit.tree]
            # Verify we can read the tree
            self.assertGreater(len(tree), 0)

            if commit.parents:
                commit = repo[commit.parents[0]]
            else:
                break

        self.assertEqual(commit_count, 1)  # We made 2 commits total
        repo.close()


class GitSHA256PackWriteCompatibilityTests(CompatTestCase):
    """Test that SHA-256 packs written by dulwich are accepted by git."""

    min_git_version = (2, 29, 0)

    def _run_git(self, args, cwd=None, input=None):
        """Run git command in the specified directory."""
        return run_git_or_fail(args, cwd=cwd, input=input)

    def _make_git_repo(self):
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        self._run_git(["init", "--object-format=sha256", repo_path])
        return repo_path

    @staticmethod
    def _pack_types(pack_path):
        pack_data = PackData(pack_path, object_format=SHA256)
        try:
            return {
                pack_data.get_unpacked_object_at(offset).pack_type_num
                for _sha, offset, _crc in pack_data.iterentries()
            }
        finally:
            pack_data.close()

    def _make_commit(self, blobs):
        tree = Tree()
        for i, blob in enumerate(blobs):
            tree.add(f"file{i}.txt".encode(), 0o100644, blob.get_id(SHA256))
        commit = Commit()
        commit.tree = tree.get_id(SHA256)
        commit.author = commit.committer = b"Dulwich Test <test@dulwich.org>"
        commit.commit_time = commit.author_time = 1234567890
        commit.commit_timezone = commit.author_timezone = 0
        commit.message = b"SHA-256 pack write test"
        return tree, commit

    def test_dulwich_pack_and_index_verified_by_git(self):
        """Dulwich-written pack (with OFS deltas) and index pass git checks."""
        blobs = [
            Blob.from_string(b"A" * 5000 + b" variation %d" % i) for i in range(10)
        ]
        tree, commit = self._make_commit(blobs)
        objects = blobs + [tree, commit]

        tmp = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, tmp)
        pack_base = os.path.join(tmp, "dulwich-pack")
        data_sum, idx_sum = write_pack(pack_base, objects, SHA256, deltify=True)
        self.assertEqual(32, len(data_sum))
        self.assertEqual(32, len(idx_sum))

        # The pack must actually contain deltas for this test to be meaningful
        self.assertIn(OFS_DELTA, self._pack_types(pack_base + ".pack"))

        # git index-pack validates the pack contents and trailer checksum
        git_repo = self._make_git_repo()
        self._run_git(["index-pack", "--strict", pack_base + ".pack"], cwd=git_repo)

        # Install the dulwich-written pack *and* index under the canonical
        # name and let git verify objects through the dulwich index
        pack_dir = os.path.join(git_repo, ".git", "objects", "pack")
        pack_name = "pack-" + data_sum.hex()
        shutil.copy(
            pack_base + ".pack", os.path.join(pack_dir, pack_name + ".pack")
        )
        shutil.copy(pack_base + ".idx", os.path.join(pack_dir, pack_name + ".idx"))

        self._run_git(
            ["update-ref", "refs/heads/master", commit.get_id(SHA256).decode()],
            cwd=git_repo,
        )
        self._run_git(["fsck", "--strict"], cwd=git_repo)

        for blob in blobs:
            out = self._run_git(
                ["cat-file", "-p", blob.get_id(SHA256).decode()], cwd=git_repo
            )
            self.assertEqual(blob.data, out)

    def test_dulwich_ref_delta_pack_verified_by_git(self):
        """REF_DELTA records with 32-byte base ids are accepted by git."""
        base = Blob.from_string(b"B" * 5000)
        derived = Blob.from_string(b"B" * 5000 + b"tail data")
        delta = b"".join(create_delta(base.as_raw_chunks(), derived.as_raw_chunks()))
        base_id = base.sha(SHA256).digest()
        derived_id = derived.sha(SHA256).digest()

        # Delta comes first, so the writer cannot use OFS_DELTA and must
        # emit a REF_DELTA record with a 32-byte base id.
        records = [
            UnpackedObject(
                REF_DELTA,
                sha=derived_id,
                delta_base=base_id,
                decomp_chunks=[delta],
                hash_func=SHA256.hash_func,
            ),
            UnpackedObject(
                Blob.type_num,
                sha=base_id,
                decomp_chunks=base.as_raw_chunks(),
                hash_func=SHA256.hash_func,
            ),
        ]

        tmp = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, tmp)
        pack_path = os.path.join(tmp, "ref-delta.pack")
        with open(pack_path, "wb") as f:
            entries, pack_sum = write_pack_data(
                f.write, iter(records), num_records=2, object_format=SHA256
            )
        self.assertEqual(32, len(pack_sum))
        self.assertEqual({derived_id, base_id}, set(entries))
        self.assertIn(REF_DELTA, self._pack_types(pack_path))

        git_repo = self._make_git_repo()
        self._run_git(["index-pack", "--strict", pack_path], cwd=git_repo)

        # Read the objects back through git
        pack_dir = os.path.join(git_repo, ".git", "objects", "pack")
        pack_name = "pack-" + pack_sum.hex()
        shutil.copy(pack_path, os.path.join(pack_dir, pack_name + ".pack"))
        shutil.copy(
            pack_path[:-5] + ".idx", os.path.join(pack_dir, pack_name + ".idx")
        )
        out = self._run_git(
            ["cat-file", "-p", derived.get_id(SHA256).decode()], cwd=git_repo
        )
        self.assertEqual(derived.data, out)

    def test_dulwich_thin_pack_completed_by_git(self):
        """Thin pack from dulwich is completed by git index-pack --fix-thin."""
        base = Blob.from_string(b"C" * 5000)
        derived = Blob.from_string(b"C" * 5000 + b"tail data")
        delta = b"".join(create_delta(base.as_raw_chunks(), derived.as_raw_chunks()))
        base_id = base.sha(SHA256).digest()
        derived_id = derived.sha(SHA256).digest()

        records = [
            UnpackedObject(
                REF_DELTA,
                sha=derived_id,
                delta_base=base_id,
                decomp_chunks=[delta],
                hash_func=SHA256.hash_func,
            ),
        ]

        git_repo = self._make_git_repo()
        # The receiving side already has the base object
        out = self._run_git(
            ["hash-object", "-w", "--stdin"], cwd=git_repo, input=base.data
        )
        self.assertEqual(base.get_id(SHA256), out.strip())

        pack_out = BytesIO()
        write_pack_data(
            pack_out.write, iter(records), num_records=1, object_format=SHA256
        )

        # git completes the thin pack, stores it in the repo and validates it
        self._run_git(
            ["index-pack", "--fix-thin", "--strict", "--stdin"],
            cwd=git_repo,
            input=pack_out.getvalue(),
        )

        out = self._run_git(
            ["cat-file", "-p", derived.get_id(SHA256).decode()], cwd=git_repo
        )
        self.assertEqual(derived.data, out)
        self._run_git(["fsck", "--strict"], cwd=git_repo)

    def test_dulwich_generated_pack_data_verified_by_git(self):
        """Pack data generated the way the server does is accepted by git."""
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        repo = Repo.init(repo_path, mkdir=False, object_format="sha256")
        self.addCleanup(repo.close)

        blobs = [
            Blob.from_string(b"D" * 5000 + b" variation %d" % i) for i in range(5)
        ]
        for blob in blobs:
            repo.object_store.add_object(blob)
        tree, commit = self._make_commit(blobs)
        repo.object_store.add_object(tree)
        repo.object_store.add_object(commit)
        commit_id = commit.get_id(SHA256)
        repo.refs[b"refs/heads/master"] = commit_id

        # This is the code path the server uses to answer a fetch
        count, records = repo.object_store.generate_pack_data([], [commit_id])
        tmp = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, tmp)
        pack_path = os.path.join(tmp, "server.pack")
        with open(pack_path, "wb") as f:
            entries, pack_sum = write_pack_data(
                f.write,
                records,
                num_records=count,
                object_format=repo.object_format,
            )
        self.assertEqual(32, len(pack_sum))
        # Index entries must use full-length SHA-256 object ids
        for oid in entries:
            self.assertEqual(32, len(oid))
        self.assertIn(commit.sha(SHA256).digest(), entries)

        git_repo = self._make_git_repo()
        self._run_git(["index-pack", "--strict", pack_path], cwd=git_repo)

    def test_git_thin_pack_ingested_by_dulwich(self):
        """Thin pack produced by git is completed and indexed by dulwich."""
        for delta_flags in ([], ["--delta-base-offset"]):
            with self.subTest(delta_flags=delta_flags):
                self._git_thin_pack_roundtrip(delta_flags)

    def _git_thin_pack_roundtrip(self, delta_flags):
        git_repo = self._make_git_repo()
        big_file = os.path.join(git_repo, "bigfile")
        with open(big_file, "wb") as f:
            f.write(b"E" * 5000)
        self._run_git(["add", "."], cwd=git_repo)
        self._run_git(["commit", "-m", "base"], cwd=git_repo)
        with open(big_file, "ab") as f:
            f.write(b"appended data")
        self._run_git(["add", "."], cwd=git_repo)
        self._run_git(["commit", "-m", "extend"], cwd=git_repo)

        def object_ids(rev):
            out = self._run_git(["rev-list", "--objects", rev], cwd=git_repo)
            return [line.split()[0] for line in out.splitlines()]

        # Full pack with everything up to HEAD~1
        have_ids = object_ids("HEAD~1")
        full_pack = self._run_git(
            ["pack-objects", "--stdout"],
            cwd=git_repo,
            input=b"\n".join(have_ids) + b"\n",
        )

        # Thin pack with just the new objects, delta against the haves
        new_ids = object_ids("HEAD~1..HEAD")
        thin_input = b"\n".join(new_ids + [b"^" + i for i in have_ids]) + b"\n"
        thin_pack = self._run_git(
            ["pack-objects", "--thin", "--stdout"] + delta_flags,
            cwd=git_repo,
            input=thin_input,
        )

        # Ingest both packs into a dulwich repository
        dul_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, dul_path)
        repo = Repo.init(dul_path, mkdir=False, object_format="sha256")
        try:
            repo.object_store.add_thin_pack(BytesIO(full_pack).read, None)
            repo.object_store.add_thin_pack(BytesIO(thin_pack).read, None)
            head = self._run_git(["rev-parse", "HEAD"], cwd=git_repo).strip()
            repo.refs[b"refs/heads/master"] = head

            # Dulwich can read the objects back
            commit = repo[head]
            self.assertIsInstance(commit, Commit)
        finally:
            repo.close()

        # The resulting on-disk repository passes git's integrity check
        self._run_git(["fsck", "--strict"], cwd=dul_path)
        out = self._run_git(["show", "HEAD:bigfile"], cwd=dul_path)
        self.assertEqual(b"E" * 5000 + b"appended data", out)


if __name__ == "__main__":
    import unittest

    unittest.main()

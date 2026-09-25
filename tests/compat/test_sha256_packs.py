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
import sys
import tempfile

import dulwich
from dulwich.object_format import SHA256
from dulwich.object_store import MissingObjectFinder
from dulwich.objects import Blob, Commit, Tree
from dulwich.pack import (
    load_pack_index_file,
    write_pack_from_container,
    write_pack_index,
)
from dulwich.repo import Repo

from .utils import CompatTestCase, rmtree_ro, run_git_or_fail

# Root of the dulwich source tree, so that git's ext:: transport can run
# "python -m dulwich ..." against this checkout.
_DULWICH_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(dulwich.__file__)))


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
    """Test that dulwich-written SHA256 packs and indexes are accepted by git."""

    min_git_version = (2, 29, 0)

    def _run_git(self, args, cwd=None, env=None, **kwargs):
        """Run git command in the specified directory."""
        return run_git_or_fail(args, cwd=cwd, env=env or {}, **kwargs)

    def _make_git_repo(self):
        """Create a git SHA-256 repo with several commits and packed objects."""
        repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, repo_path)
        self._run_git(["init", "-b", "main", "--object-format=sha256", repo_path])
        big = "x" * 2000
        for i in range(5):
            with open(os.path.join(repo_path, "big.txt"), "w") as f:
                f.write(big + f" variation {i}\n")
            with open(os.path.join(repo_path, f"file{i}.txt"), "w") as f:
                f.write(f"content {i}\n")
            self._run_git(["add", "."], cwd=repo_path)
            self._run_git(["commit", "-m", f"commit {i}"], cwd=repo_path)
        # Pack everything so that delta reuse is exercised
        self._run_git(["gc", "--aggressive"], cwd=repo_path)
        return repo_path

    def _write_pack(self, repo, object_ids, pack_path, idx_path, **kwargs):
        """Write a pack and its index with dulwich."""
        with open(pack_path, "wb") as f:
            entries, pack_checksum = write_pack_from_container(
                f.write,
                repo.object_store,
                object_ids,
                object_format=SHA256,
                **kwargs,
            )
        index_entries = sorted((k, v[0], v[1]) for k, v in entries.items())
        with open(idx_path, "wb") as f:
            write_pack_index(f, index_entries, pack_checksum, object_format=SHA256)
        return pack_checksum

    def _all_object_ids(self, repo):
        head = repo.refs[b"refs/heads/main"]
        mof = MissingObjectFinder(repo.object_store, haves=set(), wants=[head])
        return list(mof)

    def test_dulwich_pack_passes_git_index_pack_strict(self):
        """git index-pack --strict verifies a dulwich-written SHA256 pack."""
        src_path = self._make_git_repo()
        repo = Repo(src_path)
        self.addCleanup(repo.close)

        out_dir = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, out_dir)
        pack_path = os.path.join(out_dir, "dulwich.pack")
        idx_path = os.path.join(out_dir, "dulwich.idx")
        # deltify exercises OFS_DELTA/REF_DELTA encoding
        self._write_pack(
            repo, self._all_object_ids(repo), pack_path, idx_path, deltify=True
        )

        # git verifies the pack and writes its own index next to a copy.
        # GIT_CEILING_DIRECTORIES keeps git from discovering an enclosing
        # repository, so index-pack runs in standalone mode.
        git_idx_path = os.path.join(out_dir, "git.idx")
        self._run_git(
            [
                "index-pack",
                "--strict",
                "--object-format=sha256",
                "-o",
                git_idx_path,
                pack_path,
            ],
            cwd=out_dir,
            env={"GIT_CEILING_DIRECTORIES": out_dir},
        )
        # The index dulwich wrote is byte-identical to the one git writes
        with open(idx_path, "rb") as f:
            dulwich_idx = f.read()
        with open(git_idx_path, "rb") as f:
            git_idx = f.read()
        self.assertEqual(git_idx, dulwich_idx)

    def test_dulwich_pack_and_index_pass_git_fsck(self):
        """git fsck accepts a repo whose pack and index were written by dulwich."""
        src_path = self._make_git_repo()
        repo = Repo(src_path)
        self.addCleanup(repo.close)
        head = repo.refs[b"refs/heads/main"]

        # Fresh git repo; the only objects it has come from the dulwich pack
        dst_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, dst_path)
        self._run_git(["init", "-b", "main", "--object-format=sha256", dst_path])
        pack_dir = os.path.join(dst_path, ".git", "objects", "pack")
        object_ids = self._all_object_ids(repo)
        pack_base = os.path.join(pack_dir, "pack-dulwich")
        self._write_pack(repo, object_ids, pack_base + ".pack", pack_base + ".idx")

        self._run_git(
            ["update-ref", "refs/heads/main", head.decode("ascii")], cwd=dst_path
        )
        self._run_git(["fsck", "--strict"], cwd=dst_path)
        # git can check out the content from the dulwich-written pack
        self._run_git(["checkout", "-f", "main"], cwd=dst_path)
        with open(os.path.join(dst_path, "file4.txt"), "rb") as f:
            self.assertEqual(b"content 4\n", f.read())

    def test_dulwich_thin_pack_completed_by_git(self):
        """git index-pack --fix-thin completes a dulwich-written thin pack."""
        src_path = self._make_git_repo()
        repo = Repo(src_path)
        self.addCleanup(repo.close)

        head = repo.refs[b"refs/heads/main"]
        first_commit = head
        while repo[first_commit].parents:
            first_commit = repo[first_commit].parents[0]
        # Send everything but the first commit's objects, which the
        # receiving side already has
        mof = MissingObjectFinder(repo.object_store, haves={first_commit}, wants=[head])
        object_ids = list(mof)
        remote_has = mof.get_remote_has()

        out_dir = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, out_dir)
        thin_pack_path = os.path.join(out_dir, "thin.pack")
        with open(thin_pack_path, "wb") as f:
            write_pack_from_container(
                f.write,
                repo.object_store,
                object_ids,
                object_format=SHA256,
                other_haves=remote_has,
            )

        with open(thin_pack_path, "rb") as f:
            thin_pack = f.read()
        # git completes the thin pack against the objects in the source repo
        self._run_git(
            ["index-pack", "--strict", "--fix-thin", "--stdin"],
            cwd=src_path,
            input=thin_pack,
        )

    def test_bidirectional_exchange_with_git_client(self):
        """git client clones, fetches and pushes against dulwich servers."""
        src_path = self._make_git_repo()
        env = {"PYTHONPATH": _DULWICH_PARENT}
        upload_pack = f"ext::{sys.executable} -m dulwich upload-pack {src_path}"

        # Clone from dulwich upload-pack; git verifies the pack with
        # index-pack as it receives it
        work_dir = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, work_dir)
        clone_path = os.path.join(work_dir, "clone")
        self._run_git(
            ["-c", "protocol.ext.allow=always", "clone", upload_pack, clone_path],
            cwd=work_dir,
            env=env,
        )
        output = self._run_git(["rev-parse", "--show-object-format"], cwd=clone_path)
        self.assertEqual(b"sha256", output.strip())
        self._run_git(["fsck", "--strict"], cwd=clone_path)

        # Incremental fetch: the dulwich server sends a thin pack against
        # the objects the clone already has
        with open(os.path.join(src_path, "big.txt"), "a") as f:
            f.write("more data\n")
        self._run_git(["add", "."], cwd=src_path)
        self._run_git(["commit", "-m", "another commit"], cwd=src_path)
        self._run_git(
            ["-c", "protocol.ext.allow=always", "fetch", "origin"],
            cwd=clone_path,
            env=env,
        )
        self._run_git(["merge", "--ff-only", "origin/main"], cwd=clone_path)
        self._run_git(["fsck", "--strict"], cwd=clone_path)

        # Push to a bare repository served by dulwich receive-pack
        bare_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, bare_path)
        self._run_git(
            ["init", "--bare", "-b", "main", "--object-format=sha256", bare_path]
        )
        receive_pack_url = f"ext::{sys.executable} -m dulwich receive-pack {bare_path}"
        with open(os.path.join(clone_path, "pushed.txt"), "w") as f:
            f.write("pushed from git\n")
        self._run_git(["add", "."], cwd=clone_path)
        self._run_git(["commit", "-m", "pushed commit"], cwd=clone_path)
        self._run_git(
            ["-c", "protocol.ext.allow=always", "push", receive_pack_url, "main:main"],
            cwd=clone_path,
            env=env,
        )

        # The ref was updated and git fsck is happy with the pack and
        # index dulwich wrote while receiving the push
        pushed_head = self._run_git(["rev-parse", "main"], cwd=clone_path).strip()
        refs = self._run_git(["show-ref"], cwd=bare_path)
        self.assertIn(pushed_head + b" refs/heads/main", refs)
        self._run_git(["fsck", "--strict"], cwd=bare_path)

        # git index-pack --strict verifies the pack dulwich stored
        pack_dir = os.path.join(bare_path, "objects", "pack")
        pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".pack")]
        self.assertEqual(1, len(pack_files))
        self._run_git(
            [
                "index-pack",
                "--strict",
                "--object-format=sha256",
                "-o",
                os.path.join(pack_dir, "verify.idx"),
                os.path.join(pack_dir, pack_files[0]),
            ],
            cwd=bare_path,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

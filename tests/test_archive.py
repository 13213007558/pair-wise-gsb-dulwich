# test_archive.py -- tests for archive
# Copyright (C) 2015 Jelmer Vernooij <jelmer@jelmer.uk>
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

"""Tests for archive support."""

import datetime
import struct
import tarfile
from io import BytesIO
from unittest.mock import patch

from dulwich.archive import (
    ChunkedBytesIO,
    UnsafeArchivePathError,
    _is_unsafe_archive_path,
    tar_stream,
)
from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob, Commit, Tree
from dulwich.tests.utils import build_commit_graph

from . import TestCase


class ArchiveTests(TestCase):
    def test_empty(self) -> None:
        store = MemoryObjectStore()
        _c1, _c2, c3 = build_commit_graph(store, [[1], [2, 1], [3, 1, 2]])
        tree = store[c3.tree]
        stream = b"".join(tar_stream(store, tree, 10))
        out = BytesIO(stream)
        tf = tarfile.TarFile(fileobj=out)
        self.addCleanup(tf.close)
        self.assertEqual([], tf.getnames())

    def _get_example_tar_stream(
        self, mtime: int, prefix: bytes = b"", format: str = ""
    ) -> BytesIO:
        store = MemoryObjectStore()
        b1 = Blob.from_string(b"somedata")
        store.add_object(b1)
        t1 = Tree()
        t1.add(b"somename", 0o100644, b1.id)
        store.add_object(t1)
        stream = b"".join(tar_stream(store, t1, mtime, prefix, format))
        return BytesIO(stream)

    def test_simple(self) -> None:
        stream = self._get_example_tar_stream(mtime=0)
        tf = tarfile.TarFile(fileobj=stream)
        self.addCleanup(tf.close)
        self.assertEqual(["somename"], tf.getnames())

    def test_unicode(self) -> None:
        store = MemoryObjectStore()
        b1 = Blob.from_string(b"somedata")
        store.add_object(b1)
        t1 = Tree()
        t1.add("ő".encode(), 0o100644, b1.id)
        store.add_object(t1)
        stream = b"".join(tar_stream(store, t1, mtime=0))
        tf = tarfile.TarFile(fileobj=BytesIO(stream))
        self.addCleanup(tf.close)
        self.assertEqual(["ő"], tf.getnames())

    def test_prefix(self) -> None:
        stream = self._get_example_tar_stream(mtime=0, prefix=b"blah")
        tf = tarfile.TarFile(fileobj=stream)
        self.addCleanup(tf.close)
        self.assertEqual(["blah/somename"], tf.getnames())

    def test_gzip_mtime(self) -> None:
        stream = self._get_example_tar_stream(mtime=1234, format="gz")
        expected_mtime = struct.pack("<L", 1234)
        self.assertEqual(stream.getvalue()[4:8], expected_mtime)

    def test_same_file(self) -> None:
        contents: list[bytes | None] = [None, None]
        for format in ["", "gz", "bz2"]:
            for i in [0, 1]:
                with patch("time.time", return_value=i):
                    stream = self._get_example_tar_stream(mtime=0, format=format)
                    contents[i] = stream.getvalue()
            self.assertEqual(
                contents[0],
                contents[1],
                f"Different file contents for format {format!r}",
            )

    def test_tar_stream_with_directory(self) -> None:
        """Test tar_stream with a tree containing directories."""
        store = MemoryObjectStore()

        # Create a blob for a file
        b1 = Blob.from_string(b"file in subdir")
        store.add_object(b1)

        # Create a subtree
        subtree = Tree()
        subtree.add(b"file.txt", 0o100644, b1.id)
        store.add_object(subtree)

        # Create root tree with a directory
        root_tree = Tree()
        root_tree.add(b"subdir", 0o040000, subtree.id)
        store.add_object(root_tree)

        # Generate tar stream
        stream = b"".join(tar_stream(store, root_tree, 0))
        tf = tarfile.TarFile(fileobj=BytesIO(stream))
        self.addCleanup(tf.close)

        # Should contain the file in the subdirectory
        self.assertEqual(["subdir/file.txt"], tf.getnames())

    def test_unsafe_paths_rejected(self) -> None:
        """Tree entry names that git would reject as invalid are refused.

        Mirrors git's ``error: invalid path`` from ``git archive`` for
        absolute paths, parent-traversal, ``.git`` aliases, embedded
        backslashes, and NTFS alternate-data-stream separators.
        """
        # NUL is excluded here because Tree serialization uses it as a name
        # terminator, so a NUL-containing name cannot be round-tripped via the
        # public Tree API. _is_unsafe_archive_path still rejects it on its own.
        unsafe_names = [
            b"../evil.txt",
            b"/absolute.txt",
            b".git/hooks/pre-commit",
            b"..\\evil.txt",
            b".git\\hooks\\pre-commit",
            b"visible.txt:hidden",
            b"..",
            b".git",
            b".GIT",
            b".Git.",
        ]
        for name in unsafe_names:
            store = MemoryObjectStore()
            b1 = Blob.from_string(b"x")
            store.add_object(b1)
            t = Tree()
            t.add(name, 0o100644, b1.id)
            store.add_object(t)
            with self.assertRaises(UnsafeArchivePathError, msg=repr(name)) as cm:
                b"".join(tar_stream(store, t, mtime=0))
            self.assertEqual(name, cm.exception.path)

    def test_is_unsafe_archive_path_nul(self) -> None:
        """NUL bytes in a path are rejected even though they can't be stored in a Tree."""
        self.assertTrue(_is_unsafe_archive_path(b"foo\x00bar"))
        self.assertTrue(_is_unsafe_archive_path(b""))
        self.assertFalse(_is_unsafe_archive_path(b"good.txt"))
        self.assertFalse(_is_unsafe_archive_path(b"sub/dir/good.txt"))

    def test_unsafe_path_nested_in_subtree(self) -> None:
        """A ``.git`` directory hidden under a benign parent is still rejected."""
        store = MemoryObjectStore()
        b1 = Blob.from_string(b"payload")
        store.add_object(b1)
        dotgit = Tree()
        dotgit.add(b"config", 0o100644, b1.id)
        store.add_object(dotgit)
        root = Tree()
        root.add(b"subdir", 0o040000, dotgit.id)
        store.add_object(root)
        # benign so far - now add a malicious sibling
        outer = Tree()
        outer.add(b".git", 0o040000, dotgit.id)
        store.add_object(outer)
        with self.assertRaises(UnsafeArchivePathError):
            b"".join(tar_stream(store, outer, mtime=0))

    def test_mode_canonicalized(self) -> None:
        """setuid/setgid/sticky and other non-canonical mode bits are dropped.

        A crafted tree can hold a regular-file mode such as 0o104755. git's
        archive writer canonicalizes these before emitting the tar permission
        field, so an extracted file never ends up setuid.
        """
        store = MemoryObjectStore()
        b1 = Blob.from_string(b"payload")
        store.add_object(b1)
        t = Tree()
        t.add(b"suid_exec", 0o104755, b1.id)
        t.add(b"sgid_sticky", 0o106644, b1.id)
        t.add(b"plain", 0o100644, b1.id)
        t.add(b"exec", 0o100755, b1.id)
        store.add_object(t)
        stream = b"".join(tar_stream(store, t, mtime=0))
        tf = tarfile.TarFile(fileobj=BytesIO(stream))
        self.addCleanup(tf.close)
        modes = {m.name: m.mode for m in tf.getmembers()}
        self.assertEqual(0o755, modes["suid_exec"])
        self.assertEqual(0o644, modes["sgid_sticky"])
        self.assertEqual(0o644, modes["plain"])
        self.assertEqual(0o755, modes["exec"])
        for mode in modes.values():
            self.assertEqual(0, mode & 0o7000)

    def test_tar_stream_with_submodule(self) -> None:
        """Test tar_stream handles missing objects (submodules) gracefully."""
        store = MemoryObjectStore()

        # Create a tree with an entry that doesn't exist in the store
        # (simulating a submodule reference)
        root_tree = Tree()
        # Use a valid hex SHA (40 hex chars = 20 bytes)
        nonexistent_sha = b"a" * 40
        root_tree.add(b"submodule", 0o160000, nonexistent_sha)
        store.add_object(root_tree)

        # Should not raise, just skip the missing entry
        stream = b"".join(tar_stream(store, root_tree, 0))
        tf = tarfile.TarFile(fileobj=BytesIO(stream))
        self.addCleanup(tf.close)

        # Submodule should be skipped
        self.assertEqual([], tf.getnames())


class ChunkedBytesIOTests(TestCase):
    """Tests for ChunkedBytesIO class."""

    def test_read_all(self) -> None:
        """Test reading all bytes from ChunkedBytesIO."""
        chunks = [b"hello", b" ", b"world"]
        chunked = ChunkedBytesIO(chunks)

        result = chunked.read()
        self.assertEqual(b"hello world", result)

    def test_read_with_limit(self) -> None:
        """Test reading limited bytes from ChunkedBytesIO."""
        chunks = [b"hello", b" ", b"world"]
        chunked = ChunkedBytesIO(chunks)

        # Read first 5 bytes
        result = chunked.read(5)
        self.assertEqual(b"hello", result)

        # Read next 3 bytes
        result = chunked.read(3)
        self.assertEqual(b" wo", result)

        # Read remaining
        result = chunked.read()
        self.assertEqual(b"rld", result)

    def test_read_negative_maxbytes(self) -> None:
        """Test reading with negative maxbytes reads all."""
        chunks = [b"hello", b" ", b"world"]
        chunked = ChunkedBytesIO(chunks)

        result = chunked.read(-1)
        self.assertEqual(b"hello world", result)

    def test_read_across_chunks(self) -> None:
        """Test reading across multiple chunks."""
        chunks = [b"abc", b"def", b"ghi"]
        chunked = ChunkedBytesIO(chunks)

        # Read 7 bytes (spans three chunks)
        result = chunked.read(7)
        self.assertEqual(b"abcdefg", result)

        # Read remaining
        result = chunked.read()
        self.assertEqual(b"hi", result)

    def test_read_empty_chunks(self) -> None:
        """Test reading from empty chunks list."""
        chunked = ChunkedBytesIO([])

        result = chunked.read()
        self.assertEqual(b"", result)

    def test_read_with_empty_chunks_mixed(self) -> None:
        """Test reading with some empty chunks in the list."""
        chunks = [b"hello", b"", b"world", b""]
        chunked = ChunkedBytesIO(chunks)

        result = chunked.read()
        self.assertEqual(b"helloworld", result)

    def test_read_exact_chunk_boundary(self) -> None:
        """Test reading exactly to a chunk boundary."""
        chunks = [b"abc", b"def", b"ghi"]
        chunked = ChunkedBytesIO(chunks)

        # Read exactly first chunk
        result = chunked.read(3)
        self.assertEqual(b"abc", result)

        # Read exactly second chunk
        result = chunked.read(3)
        self.assertEqual(b"def", result)

        # Read exactly third chunk
        result = chunked.read(3)
        self.assertEqual(b"ghi", result)

        # Should be at end
        result = chunked.read()
        self.assertEqual(b"", result)


class ArchiveGitAttributesTests(TestCase):
    """Tests for export-ignore / export-subst handling in tar_stream."""

    def _blob(self, store: MemoryObjectStore, data: bytes) -> Blob:
        blob = Blob.from_string(data)
        store.add_object(blob)
        return blob

    def _commit(
        self,
        store: MemoryObjectStore,
        tree: Tree,
        message: bytes = b"Release v1\n\nLonger body\n",
    ) -> Commit:
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"A U Thor <author@example.com>"
        commit.author_time = commit.commit_time = 1234567890
        commit.author_timezone = commit.commit_timezone = 8 * 3600
        commit.message = message
        store.add_object(commit)
        return commit

    def _archive(
        self, store: MemoryObjectStore, tree: Tree, **kwargs
    ) -> tarfile.TarFile:
        stream = b"".join(tar_stream(store, tree, mtime=0, **kwargs))
        tf = tarfile.TarFile(fileobj=BytesIO(stream))
        self.addCleanup(tf.close)
        return tf

    def test_export_ignore(self) -> None:
        store = MemoryObjectStore()
        attrs_root = self._blob(
            store,
            b"testdata/ export-ignore\n"
            b"*.log export-ignore\n"
            b"keep.txt export-ignore\n"
            b"secret.txt export-ignore\n",
        )
        # Deeper .gitattributes files override the root one: "-attr" unsets
        # the attribute, "!attr" resets it to unspecified; both re-include
        # the file.
        attrs_sub = self._blob(
            store,
            b"keep.txt -export-ignore\nsecret.txt !export-ignore\n",
        )
        file_txt = self._blob(store, b"file\n")
        keep = self._blob(store, b"keep\n")
        secret = self._blob(store, b"secret\n")
        log = self._blob(store, b"log\n")
        data = self._blob(store, b"data\n")
        testdata = Tree()
        testdata.add(b"data.txt", 0o100644, data.id)
        store.add_object(testdata)
        subdir = Tree()
        subdir.add(b".gitattributes", 0o100644, attrs_sub.id)
        subdir.add(b"keep.txt", 0o100644, keep.id)
        subdir.add(b"other.log", 0o100644, log.id)
        subdir.add(b"secret.txt", 0o100644, secret.id)
        store.add_object(subdir)
        tree = Tree()
        tree.add(b".gitattributes", 0o100644, attrs_root.id)
        tree.add(b"error.log", 0o100644, log.id)
        tree.add(b"file.txt", 0o100644, file_txt.id)
        tree.add(b"keep.txt", 0o100644, keep.id)
        tree.add(b"subdir", 0o040000, subdir.id)
        tree.add(b"testdata", 0o040000, testdata.id)
        store.add_object(tree)

        tf = self._archive(store, tree)
        # error.log and keep.txt match root patterns; testdata is dropped
        # with its whole subtree; subdir/other.log still matches the root
        # *.log pattern, while the deeper file re-includes keep.txt and
        # resets secret.txt to unspecified.
        self.assertEqual(
            [
                ".gitattributes",
                "file.txt",
                "subdir/.gitattributes",
                "subdir/keep.txt",
                "subdir/secret.txt",
            ],
            tf.getnames(),
        )

    def test_export_subst(self) -> None:
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(
            store,
            b"full=$Format:%H$\n"
            b"short=$Format:%h$\n"
            b"date=$Format:%aI$\n"
            b"subject=$Format:%s$\n"
            b"unknown=$Format:%Q$\n"
            b"mixed=$Format:commit %H -> %s$\n",
        )
        tree = Tree()
        tree.add(b".gitattributes", 0o100644, attrs.id)
        tree.add(b"VERSION", 0o100644, version.id)
        store.add_object(tree)
        commit = self._commit(store, tree)

        tf = self._archive(store, tree, commit=commit)
        data = tf.extractfile("VERSION").read()
        expected_date = (
            datetime.datetime.fromtimestamp(
                1234567890, datetime.timezone(datetime.timedelta(hours=8))
            )
            .isoformat()
            .encode("ascii")
        )
        self.assertEqual(
            b"full=" + commit.id + b"\n"
            b"short=" + commit.id[:7] + b"\n"
            b"date=" + expected_date + b"\n"
            b"subject=Release v1\n"
            b"unknown=$Format:%Q$\n"
            b"mixed=commit " + commit.id + b" -> Release v1\n",
            data,
        )
        # The stored blob is not modified by the substitution.
        self.assertEqual(version.data, store[version.id].data)

    def test_export_subst_without_commit_left_verbatim(self) -> None:
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(store, b"commit=$Format:%H$\n")
        tree = Tree()
        tree.add(b".gitattributes", 0o100644, attrs.id)
        tree.add(b"VERSION", 0o100644, version.id)
        store.add_object(tree)

        tf = self._archive(store, tree)
        # Tree-only archiving has no commit to take values from, so the
        # placeholders are archived unexpanded.
        self.assertEqual(b"commit=$Format:%H$\n", tf.extractfile("VERSION").read())

    def test_file_types_and_symlink_not_substituted(self) -> None:
        store = MemoryObjectStore()
        attrs = self._blob(store, b"* export-subst\n")
        plain = self._blob(store, b"plain\n")
        exe = self._blob(store, b"#!/bin/sh\n")
        link = self._blob(store, b"target-$Format:%H$")
        tree = Tree()
        tree.add(b".gitattributes", 0o100644, attrs.id)
        tree.add(b"link", 0o120000, link.id)
        tree.add(b"plain", 0o100644, plain.id)
        tree.add(b"run.sh", 0o100755, exe.id)
        store.add_object(tree)
        commit = self._commit(store, tree)

        tf = self._archive(store, tree, commit=commit)
        members = {m.name: m for m in tf.getmembers()}
        self.assertTrue(members["plain"].isreg())
        self.assertEqual(0o644, members["plain"].mode)
        self.assertTrue(members["run.sh"].isreg())
        self.assertEqual(0o755, members["run.sh"].mode)
        # The symlink keeps its tar type and its target is neither
        # dereferenced nor substituted, despite "* export-subst".
        self.assertTrue(members["link"].issym())
        self.assertEqual("target-$Format:%H$", members["link"].linkname)
        self.assertEqual(0o777, members["link"].mode)
        self.assertEqual(0, members["link"].size)

    def test_prefix_does_not_affect_attributes(self) -> None:
        store = MemoryObjectStore()
        attrs = self._blob(store, b"secret.txt export-ignore\ntestdata export-ignore\n")
        secret = self._blob(store, b"secret\n")
        visible = self._blob(store, b"visible\n")
        tree = Tree()
        tree.add(b".gitattributes", 0o100644, attrs.id)
        tree.add(b"secret.txt", 0o100644, secret.id)
        tree.add(b"visible.txt", 0o100644, visible.id)
        store.add_object(tree)

        # The prefix matches the "testdata" ignore pattern, but it is not
        # part of the repository and must not affect attribute matching;
        # secret.txt is still ignored via its repo-relative path.
        tf = self._archive(store, tree, prefix=b"testdata")
        self.assertEqual(
            ["testdata/.gitattributes", "testdata/visible.txt"], tf.getnames()
        )

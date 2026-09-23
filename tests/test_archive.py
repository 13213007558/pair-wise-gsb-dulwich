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
from dulwich.objects import Blob, Tree
from dulwich.tests.utils import build_commit_graph, make_commit

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


class ArchiveAttributesTests(TestCase):
    """Tests for export-ignore / export-subst handling in tar_stream."""

    def _blob(self, store: MemoryObjectStore, data: bytes) -> Blob:
        blob = Blob.from_string(data)
        store.add_object(blob)
        return blob

    def _tree(self, store: MemoryObjectStore, entries) -> Tree:
        tree = Tree()
        for name, mode, sha in entries:
            tree.add(name, mode, sha)
        store.add_object(tree)
        return tree

    def _tar(self, stream) -> tarfile.TarFile:
        tf = tarfile.TarFile(fileobj=BytesIO(b"".join(stream)))
        self.addCleanup(tf.close)
        return tf

    def test_export_ignore_file_and_directory(self) -> None:
        """Ignored files are skipped; ignored directories are pruned wholesale."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"secret.txt export-ignore\ntestdata export-ignore\n")
        secret = self._blob(store, b"secret")
        keep = self._blob(store, b"keep")
        data = self._blob(store, b"test data")
        sub = self._tree(store, [(b"data.txt", 0o100644, data.id)])
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"keep.txt", 0o100644, keep.id),
                (b"secret.txt", 0o100644, secret.id),
                (b"testdata", 0o040000, sub.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        self.assertEqual([".gitattributes", "keep.txt"], tf.getnames())

    def test_export_ignore_nested_override_and_unspecified(self) -> None:
        """Deeper .gitattributes files override (-attr) and reset (!attr)."""
        store = MemoryObjectStore()
        root_attrs = self._blob(store, b"*.log export-ignore\n")
        sub_attrs = self._blob(
            store, b"keep.log -export-ignore\nunset.log !export-ignore\n"
        )
        content = self._blob(store, b"x")
        sub = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, sub_attrs.id),
                (b"keep.log", 0o100644, content.id),
                (b"other.log", 0o100644, content.id),
                (b"unset.log", 0o100644, content.id),
            ],
        )
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, root_attrs.id),
                (b"sub", 0o040000, sub.id),
                (b"top.log", 0o100644, content.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        self.assertEqual(
            [
                ".gitattributes",
                "sub/.gitattributes",
                "sub/keep.log",
                "sub/unset.log",
            ],
            tf.getnames(),
        )

    def test_export_ignore_unset_does_not_ignore(self) -> None:
        """-export-ignore explicitly keeps a file that would else be ignored."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"*.txt export-ignore\nkeep.txt -export-ignore\n")
        content = self._blob(store, b"x")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"drop.txt", 0o100644, content.id),
                (b"keep.txt", 0o100644, content.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        self.assertEqual([".gitattributes", "keep.txt"], tf.getnames())

    def test_export_subst(self) -> None:
        """Supported $Format:...$ placeholders expand; unknown ones are kept."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(
            store,
            b"full=$Format:%H$\n"
            b"short=$Format:%h$\n"
            b"iso=$Format:%aI$\n"
            b"ts=$Format:%at$\n"
            b"date=$Format:%ad$\n"
            b"subject=$Format:%s$\n"
            b"unknown=$Format:%NoSuchDirective$\n",
        )
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"VERSION", 0o100644, version.id),
            ],
        )
        commit = make_commit(
            tree=root.id,
            author_time=1262304000,
            author_timezone=0,
            message=b"Release v1\n\nLonger body.\n",
        )
        store.add_object(commit)
        tf = self._tar(tar_stream(store, root, mtime=0, commit=commit))
        data = tf.extractfile("VERSION").read()
        self.assertEqual(
            b"full=" + commit.id + b"\n"
            b"short=" + commit.id[:7] + b"\n"
            b"iso=2010-01-01T00:00:00+00:00\n"
            b"ts=1262304000\n"
            b"date=Fri Jan 1 00:00:00 2010 +0000\n"
            b"subject=Release v1\n"
            b"unknown=$Format:%NoSuchDirective$\n",
            data,
        )

    def test_export_subst_author_timezone(self) -> None:
        """Author dates are rendered in the commit's author timezone."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(store, b"$Format:%aI$ $Format:%ad$\n")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"VERSION", 0o100644, version.id),
            ],
        )
        commit = make_commit(
            tree=root.id,
            author_time=1262304000,
            author_timezone=8 * 3600,
            message=b"subject\n",
        )
        store.add_object(commit)
        tf = self._tar(tar_stream(store, root, mtime=0, commit=commit))
        data = tf.extractfile("VERSION").read()
        self.assertEqual(
            b"2010-01-01T08:00:00+08:00 Fri Jan 1 08:00:00 2010 +0800\n",
            data,
        )

    def test_export_subst_without_commit_context(self) -> None:
        """With only a tree (no commit), placeholders are left as-is."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(store, b"commit=$Format:%H$\n")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"VERSION", 0o100644, version.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        self.assertEqual(b"commit=$Format:%H$\n", tf.extractfile("VERSION").read())

    def test_export_subst_only_where_attribute_set(self) -> None:
        """Files without export-subst keep their placeholders."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"VERSION export-subst\n")
        version = self._blob(store, b"$Format:%H$\n")
        other = self._blob(store, b"$Format:%H$\n")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"VERSION", 0o100644, version.id),
                (b"other.txt", 0o100644, other.id),
            ],
        )
        commit = make_commit(tree=root.id)
        store.add_object(commit)
        tf = self._tar(tar_stream(store, root, mtime=0, commit=commit))
        self.assertEqual(commit.id + b"\n", tf.extractfile("VERSION").read())
        self.assertEqual(b"$Format:%H$\n", tf.extractfile("other.txt").read())

    def test_symlink_entry_not_substituted_not_dereferenced(self) -> None:
        """Symlinks keep their tar type; targets are never substituted."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"* export-subst\n")
        target = self._blob(store, b"placeholder $Format:%H$\n")
        link = self._blob(store, b"$Format:%H$.txt")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"link", 0o120000, link.id),
                (b"target.txt", 0o100644, target.id),
            ],
        )
        commit = make_commit(tree=root.id)
        store.add_object(commit)
        tf = self._tar(tar_stream(store, root, mtime=0, commit=commit))
        members = {m.name: m for m in tf.getmembers()}
        link_member = members["link"]
        self.assertTrue(link_member.issym())
        # The link target is the raw blob content: no substitution, and the
        # link is not dereferenced to target.txt.
        self.assertEqual("$Format:%H$.txt", link_member.linkname)
        self.assertEqual(0o777, link_member.mode)
        self.assertEqual(0, link_member.size)
        # The regular file the link points at was substituted, though.
        self.assertEqual(
            b"placeholder " + commit.id + b"\n",
            tf.extractfile("target.txt").read(),
        )

    def test_file_types_and_modes(self) -> None:
        """Regular files, executables and symlinks keep their tar types."""
        store = MemoryObjectStore()
        content = self._blob(store, b"x")
        link = self._blob(store, b"plain.txt")
        root = self._tree(
            store,
            [
                (b"exec", 0o100755, content.id),
                (b"link", 0o120000, link.id),
                (b"plain.txt", 0o100644, content.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        members = {m.name: m for m in tf.getmembers()}
        self.assertTrue(members["plain.txt"].isreg())
        self.assertEqual(0o644, members["plain.txt"].mode)
        self.assertTrue(members["exec"].isreg())
        self.assertEqual(0o755, members["exec"].mode)
        self.assertTrue(members["link"].issym())
        self.assertEqual("plain.txt", members["link"].linkname)

    def test_prefix_does_not_affect_attribute_matching(self) -> None:
        """The archive prefix is not part of the repo path used for matching."""
        store = MemoryObjectStore()
        attrs = self._blob(store, b"secret.txt export-ignore\n")
        content = self._blob(store, b"x")
        root = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, attrs.id),
                (b"keep.txt", 0o100644, content.id),
                (b"secret.txt", 0o100644, content.id),
            ],
        )
        # A prefix that itself matches an ignore pattern must not cause
        # anything to be ignored; matching uses repo-relative paths only.
        tf = self._tar(tar_stream(store, root, mtime=0, prefix=b"secret.txt"))
        self.assertEqual(
            ["secret.txt/.gitattributes", "secret.txt/keep.txt"], tf.getnames()
        )
        # And an ignored file stays ignored regardless of the prefix.
        tf = self._tar(tar_stream(store, root, mtime=0, prefix=b"pkg"))
        self.assertEqual(["pkg/.gitattributes", "pkg/keep.txt"], tf.getnames())

    def test_subdirectory_gitattributes_scoped_to_subtree(self) -> None:
        """Patterns in a subdirectory .gitattributes don't leak outside it."""
        store = MemoryObjectStore()
        sub_attrs = self._blob(store, b"*.txt export-ignore\n")
        content = self._blob(store, b"x")
        sub = self._tree(
            store,
            [
                (b".gitattributes", 0o100644, sub_attrs.id),
                (b"drop.txt", 0o100644, content.id),
            ],
        )
        root = self._tree(
            store,
            [
                (b"keep.txt", 0o100644, content.id),
                (b"sub", 0o040000, sub.id),
            ],
        )
        tf = self._tar(tar_stream(store, root, mtime=0))
        self.assertEqual(["keep.txt", "sub/.gitattributes"], tf.getnames())


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

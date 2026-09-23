# test_worddiff.py -- Tests for word-level diff functionality.
# Copyright (C) 2026 Dulwich contributors
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

"""Tests for word-level diff functionality."""

import re
from itertools import pairwise

from dulwich.diff_tree import CHANGE_MODIFY, TreeChange
from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob, TreeEntry
from dulwich.worddiff import (
    tokenize_text,
    word_diff,
    word_diff_blobs,
    word_diff_tree_change,
)

from . import TestCase


def make_blob(data: bytes) -> Blob:
    blob = Blob()
    blob.data = data
    return blob


def old_slice(segments, data: bytes, op: str) -> bytes:
    return b"".join(data[s.old_start : s.old_end] for s in segments if s.op == op)


def new_slice(segments, data: bytes, op: str) -> bytes:
    return b"".join(data[s.new_start : s.new_end] for s in segments if s.op == op)


class TokenizeTextTests(TestCase):
    def test_words_and_gaps_tile_input(self):
        text = "foo, bar  baz!"
        tokens = tokenize_text(text, r"\w+")
        self.assertEqual(
            [t[0] for t in tokens],
            ["foo", ", ", "bar", "  ", "baz", "!"],
        )
        self.assertEqual("".join(t[0] for t in tokens), text)
        for token, start, end in tokens:
            self.assertEqual(text[start:end], token)

    def test_zero_length_match_rejected(self):
        self.assertRaises(ValueError, tokenize_text, "abc", r"\w*")
        self.assertRaises(ValueError, tokenize_text, "abc", r"(?:a|)")
        self.assertRaises(ValueError, tokenize_text, "abc", re.compile(r"x*"))

    def test_match_spanning_newline_is_split(self):
        tokens = tokenize_text("ab\ncd", r"[a-z\n]+")
        self.assertEqual([t[0] for t in tokens], ["ab", "\n", "cd"])
        for token, _, _ in tokens:
            if "\n" in token:
                self.assertEqual(token, "\n")

    def test_crlf_kept_together(self):
        tokens = tokenize_text("ab\r\ncd", r"[a-z]+")
        self.assertEqual([t[0] for t in tokens], ["ab", "\r\n", "cd"])

    def test_gap_spanning_newline_is_split(self):
        tokens = tokenize_text("a, \n b", r"\w+")
        self.assertEqual([t[0] for t in tokens], ["a", ", ", "\n", " ", "b"])


class WordDiffTests(TestCase):
    def assert_roundtrip(self, old: bytes, new: bytes, segments) -> None:
        self.assertEqual(old, b"".join(old[s.old_start : s.old_end] for s in segments))
        self.assertEqual(new, b"".join(new[s.new_start : s.new_end] for s in segments))
        for prev, cur in pairwise(segments):
            self.assertEqual(prev.old_end, cur.old_start)
            self.assertEqual(prev.new_end, cur.new_start)

    def diff(self, old: bytes, new: bytes, pattern=r"\w+"):
        segments = word_diff(old, new, pattern)
        self.assert_roundtrip(old, new, segments)
        return segments

    def ops(self, segments):
        return [s.op for s in segments]

    def test_single_word_replacement(self):
        old = b"hello world"
        new = b"hello there"
        segments = self.diff(old, new)
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert"])
        self.assertEqual(b"hello ", old[segments[0].old_start : segments[0].old_end])
        self.assertEqual(b"world", old[segments[1].old_start : segments[1].old_end])
        self.assertEqual(b"there", new[segments[2].new_start : segments[2].new_end])

    def test_whitespace_only_change(self):
        old = b"a  b"
        new = b"a b"
        segments = self.diff(old, new)
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert", "equal"])
        self.assertEqual(b"  ", old[segments[1].old_start : segments[1].old_end])
        self.assertEqual(b" ", new[segments[2].new_start : segments[2].new_end])

    def test_insert_word(self):
        new = b"a b c"
        segments = self.diff(b"a c", new)
        self.assertEqual(self.ops(segments), ["equal", "insert", "equal"])
        self.assertEqual(b"b ", new[segments[1].new_start : segments[1].new_end])

    def test_multibyte_characters(self):
        old = "你好 world 再见".encode()
        new = "你好 word 再见".encode()
        segments = self.diff(old, new)
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert", "equal"])
        self.assertEqual(b"world", old[segments[1].old_start : segments[1].old_end])
        self.assertEqual(b"word", new[segments[2].new_start : segments[2].new_end])
        self.assertEqual(segments[1].old_start, len("你好 ".encode()))

    def test_mixed_chinese_english(self):
        old = "修复了login页面的bug".encode()
        new = "修复了logout页面的bug".encode()
        segments = self.diff(old, new, r"[a-zA-Z]+|[^\x00-\x7F]+")
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert", "equal"])
        self.assertEqual(b"login", old_slice(segments, old, "delete"))
        self.assertEqual(b"logout", new_slice(segments, new, "insert"))

    def test_repeated_phrases(self):
        old = b"foo bar foo bar foo bar"
        new = b"foo bar foo baz foo bar"
        segments = self.diff(old, new)
        self.assertEqual(b"bar", old_slice(segments, old, "delete"))
        self.assertEqual(b"baz", new_slice(segments, new, "insert"))

    def test_crlf_and_no_trailing_newline(self):
        old = b"line one\r\nline two\r\nline three"
        new = b"line one\r\nline 2\r\nline three"
        segments = self.diff(old, new)
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert", "equal"])
        self.assertEqual(b"two", old[segments[1].old_start : segments[1].old_end])
        self.assertEqual(b"2", new[segments[2].new_start : segments[2].new_end])

    def test_trailing_newline_added(self):
        new = b"no newline\n"
        segments = self.diff(b"no newline", new)
        self.assertEqual(self.ops(segments), ["equal", "insert"])
        self.assertEqual(b"\n", new[segments[1].new_start : segments[1].new_end])

    def test_empty_inputs(self):
        self.assertEqual([], self.diff(b"", b""))
        segments = self.diff(b"", b"all new")
        self.assertEqual(self.ops(segments), ["insert"])
        segments = self.diff(b"all gone", b"")
        self.assertEqual(self.ops(segments), ["delete"])

    def test_identical_inputs(self):
        segments = self.diff(b"same text\n", b"same text\n")
        self.assertEqual(self.ops(segments), ["equal"])

    def test_invalid_utf8_raises(self):
        self.assertRaises(UnicodeDecodeError, word_diff, b"\xff\xfe", b"ok")

    def test_compiled_pattern(self):
        old = b"a-1 b-2"
        new = b"a-1 b-3"
        segments = word_diff(old, new, re.compile(r"[a-z]-\d"))
        self.assert_roundtrip(old, new, segments)
        self.assertEqual(self.ops(segments), ["equal", "delete", "insert"])


class WordDiffBlobsTests(TestCase):
    def test_text_blobs(self):
        old = b"foo bar"
        segments = word_diff_blobs(make_blob(old), make_blob(b"foo baz"))
        self.assertIsNotNone(segments)
        self.assertEqual(b"bar", old_slice(segments, old, "delete"))

    def test_binary_blob_not_applicable(self):
        self.assertIsNone(word_diff_blobs(make_blob(b"\x00binary"), make_blob(b"text")))
        self.assertIsNone(word_diff_blobs(make_blob(b"text"), make_blob(b"\x00binary")))

    def test_missing_side(self):
        segments = word_diff_blobs(None, make_blob(b"added"))
        self.assertEqual(["insert"], [s.op for s in segments])
        segments = word_diff_blobs(make_blob(b"removed"), None)
        self.assertEqual(["delete"], [s.op for s in segments])


class WordDiffTreeChangeTests(TestCase):
    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()

    def add_blob(self, data: bytes) -> Blob:
        blob = make_blob(data)
        self.store.add_object(blob)
        return blob

    def make_change(self, old_data, new_data) -> TreeChange:
        old_entry = None
        new_entry = None
        if old_data is not None:
            old_entry = TreeEntry(b"file.txt", 0o100644, self.add_blob(old_data).id)
        if new_data is not None:
            new_entry = TreeEntry(b"file.txt", 0o100644, self.add_blob(new_data).id)
        return TreeChange(CHANGE_MODIFY, old_entry, new_entry)

    def test_modify(self):
        old = b"the quick fox"
        new = b"the quick foxes"
        change = self.make_change(old, new)
        segments = word_diff_tree_change(self.store, change)
        self.assertIsNotNone(segments)
        self.assertEqual(old, b"".join(old[s.old_start : s.old_end] for s in segments))
        self.assertEqual(new, b"".join(new[s.new_start : s.new_end] for s in segments))

    def test_add_and_delete(self):
        change = self.make_change(None, b"brand new")
        segments = word_diff_tree_change(self.store, change)
        self.assertEqual(["insert"], [s.op for s in segments])
        change = self.make_change(b"gone", None)
        segments = word_diff_tree_change(self.store, change)
        self.assertEqual(["delete"], [s.op for s in segments])

    def test_binary_not_applicable(self):
        change = self.make_change(b"\x00\x01\x02", b"\x00\x03")
        self.assertIsNone(word_diff_tree_change(self.store, change))

    def test_gitlink_not_applicable(self):
        old_entry = TreeEntry(b"sub", 0o160000, b"0" * 40)
        new_entry = TreeEntry(b"sub", 0o160000, b"1" * 40)
        change = TreeChange(CHANGE_MODIFY, old_entry, new_entry)
        self.assertIsNone(word_diff_tree_change(self.store, change))

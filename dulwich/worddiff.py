# worddiff.py -- Word-level diff functionality for Dulwich
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

r"""Structured word-level diffs between two UTF-8 texts.

Where :mod:`dulwich.patch` produces line-oriented unified diffs, this module
compares two byte strings token by token and reports equal/delete/insert
segments as byte ranges into the original inputs. Tokenization is driven by a
caller-supplied regular expression; text between matches (whitespace,
punctuation, ...) is kept as tokens of its own, so concatenating the old-side
(resp. new-side) slices of all segments reproduces the old (resp. new) input
byte for byte.

Example usage:
    from dulwich.worddiff import word_diff

    segments = word_diff(b"hello world", b"hello there", r"\w+")
    for seg in segments:
        print(seg.op, old[seg.old_start:seg.old_end],
              new[seg.new_start:seg.new_end])
"""

__all__ = [
    "DEFAULT_TOKEN_PATTERN",
    "WordDiffSegment",
    "tokenize_text",
    "word_diff",
    "word_diff_blobs",
    "word_diff_tree_change",
]

import re
from difflib import SequenceMatcher
from re import Pattern
from typing import TYPE_CHECKING, NamedTuple

from .objects import S_ISGITLINK, Blob
from .patch import is_binary

if TYPE_CHECKING:
    from .diff_tree import TreeChange
    from .object_store import BaseObjectStore
    from .objects import TreeEntry

DEFAULT_TOKEN_PATTERN = r"\w+"

_OP_EQUAL = "equal"
_OP_DELETE = "delete"
_OP_INSERT = "insert"


class WordDiffSegment(NamedTuple):
    """One segment of a word-level diff.

    All offsets are byte offsets into the original inputs. Concatenating
    `old[old_start:old_end]` over every segment yields the old input, and
    concatenating `new[new_start:new_end]` yields the new input.
    """

    op: str  # "equal", "delete" or "insert"
    old_start: int
    old_end: int
    new_start: int
    new_end: int


def _split_at_newlines(piece: str, start: int) -> list[tuple[str, int, int]]:
    r"""Split a token so no token spans a newline.

    Line content and line terminators become separate tokens; a `\r`
    immediately before a `\n` is kept with the terminator so CRLF stays
    atomic.
    """
    tokens: list[tuple[str, int, int]] = []
    line_start = 0
    for i, ch in enumerate(piece):
        if ch != "\n":
            continue
        content_end = i
        nl_start = i
        if content_end > line_start and piece[content_end - 1] == "\r":
            content_end -= 1
            nl_start = content_end
        if content_end > line_start:
            tokens.append(
                (piece[line_start:content_end], start + line_start, start + content_end)
            )
        tokens.append((piece[nl_start : i + 1], start + nl_start, start + i + 1))
        line_start = i + 1
    if line_start < len(piece):
        tokens.append((piece[line_start:], start + line_start, start + len(piece)))
    return tokens


def tokenize_text(
    text: str, pattern: "str | Pattern[str]"
) -> list[tuple[str, int, int]]:
    """Split text into tokens using a caller-supplied regular expression.

    Regex matches become tokens, and the gaps between matches (whitespace,
    punctuation, ...) become tokens of their own, so the tokens tile the
    whole input. Tokens never span a newline: matches or gaps containing a
    line terminator are split, with the terminator kept as its own token.

    Args:
      text: The text to tokenize.
      pattern: Regex (string or compiled) selecting word tokens. It must not
        match the empty string.

    Returns:
      List of (token, char_start, char_end) tuples covering the text.

    Raises:
      ValueError: If the pattern can match the empty string.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    empty = compiled.match("")
    if empty is not None and empty.end() == 0:
        raise ValueError("token pattern must not match the empty string")
    tokens: list[tuple[str, int, int]] = []
    pos = 0
    for match in compiled.finditer(text):
        if match.end() == match.start():
            raise ValueError("token pattern must not match the empty string")
        if match.start() > pos:
            tokens.extend(_split_at_newlines(text[pos : match.start()], pos))
        tokens.extend(_split_at_newlines(match.group(), match.start()))
        pos = match.end()
    if pos < len(text):
        tokens.extend(_split_at_newlines(text[pos:], pos))
    return tokens


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        total += len(ch.encode("utf-8"))
        offsets[i + 1] = total
    return offsets


def _token_byte_bounds(
    tokens: list[tuple[str, int, int]], char_offsets: list[int]
) -> list[int]:
    # Tokens tile the text, so token k starts where token k-1 ended.
    bounds = [char_offsets[token[1]] for token in tokens]
    bounds.append(char_offsets[tokens[-1][2]] if tokens else 0)
    return bounds


def word_diff(
    old: bytes,
    new: bytes,
    pattern: "str | Pattern[str]" = DEFAULT_TOKEN_PATTERN,
) -> list[WordDiffSegment]:
    """Compute a word-level diff between two UTF-8 byte strings.

    Args:
      old: Old content, UTF-8 encoded.
      new: New content, UTF-8 encoded.
      pattern: Regex selecting word tokens (see tokenize_text).

    Returns:
      List of WordDiffSegment with byte ranges into old and new.

    Raises:
      UnicodeDecodeError: If either input is not valid UTF-8.
      ValueError: If the pattern can match the empty string.
    """
    old_text = old.decode("utf-8")
    new_text = new.decode("utf-8")
    old_tokens = tokenize_text(old_text, pattern)
    new_tokens = tokenize_text(new_text, pattern)
    old_bounds = _token_byte_bounds(old_tokens, _char_to_byte_offsets(old_text))
    new_bounds = _token_byte_bounds(new_tokens, _char_to_byte_offsets(new_text))

    matcher = SequenceMatcher(
        None,
        [token for token, _, _ in old_tokens],
        [token for token, _, _ in new_tokens],
        autojunk=False,
    )
    segments: list[WordDiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(
                WordDiffSegment(
                    _OP_EQUAL,
                    old_bounds[i1],
                    old_bounds[i2],
                    new_bounds[j1],
                    new_bounds[j2],
                )
            )
        elif tag == "delete":
            segments.append(
                WordDiffSegment(
                    _OP_DELETE,
                    old_bounds[i1],
                    old_bounds[i2],
                    new_bounds[j1],
                    new_bounds[j1],
                )
            )
        elif tag == "insert":
            segments.append(
                WordDiffSegment(
                    _OP_INSERT,
                    old_bounds[i1],
                    old_bounds[i1],
                    new_bounds[j1],
                    new_bounds[j2],
                )
            )
        else:  # replace
            segments.append(
                WordDiffSegment(
                    _OP_DELETE,
                    old_bounds[i1],
                    old_bounds[i2],
                    new_bounds[j1],
                    new_bounds[j1],
                )
            )
            segments.append(
                WordDiffSegment(
                    _OP_INSERT,
                    old_bounds[i2],
                    old_bounds[i2],
                    new_bounds[j1],
                    new_bounds[j2],
                )
            )
    merged: list[WordDiffSegment] = []
    for seg in segments:
        if (
            merged
            and merged[-1].op == seg.op
            and merged[-1].old_end == seg.old_start
            and merged[-1].new_end == seg.new_start
        ):
            merged[-1] = WordDiffSegment(
                seg.op,
                merged[-1].old_start,
                seg.old_end,
                merged[-1].new_start,
                seg.new_end,
            )
        else:
            merged.append(seg)
    return merged


def word_diff_blobs(
    old_blob: "Blob | None",
    new_blob: "Blob | None",
    pattern: "str | Pattern[str]" = DEFAULT_TOKEN_PATTERN,
) -> "list[WordDiffSegment] | None":
    """Compute a word-level diff between two blobs.

    Args:
      old_blob: Old blob, or None if the file was added.
      new_blob: New blob, or None if the file was deleted.
      pattern: Regex selecting word tokens (see tokenize_text).

    Returns:
      List of WordDiffSegment, or None if either side is binary (word diffs
      are not applicable to binary content).
    """
    old_data = old_blob.data if old_blob is not None else b""
    new_data = new_blob.data if new_blob is not None else b""
    if is_binary(old_data) or is_binary(new_data):
        return None
    return word_diff(old_data, new_data, pattern)


def word_diff_tree_change(
    store: "BaseObjectStore",
    change: "TreeChange",
    pattern: "str | Pattern[str]" = DEFAULT_TOKEN_PATTERN,
) -> "list[WordDiffSegment] | None":
    """Compute a word-level diff for a tree change.

    Args:
      store: Object store to fetch the blobs from.
      change: The TreeChange to diff.
      pattern: Regex selecting word tokens (see tokenize_text).

    Returns:
      List of WordDiffSegment, or None if the change involves binary content
      or gitlinks (word diffs are not applicable there).
    """
    if (change.old is not None and S_ISGITLINK(change.old.mode)) or (
        change.new is not None and S_ISGITLINK(change.new.mode)
    ):
        return None

    def get_blob(entry: "TreeEntry | None") -> "Blob | None":
        if entry is None:
            return None
        return store[entry.sha]  # type: ignore[return-value]

    return word_diff_blobs(get_blob(change.old), get_blob(change.new), pattern)

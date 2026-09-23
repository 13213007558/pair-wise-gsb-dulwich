# test_pathspec.py -- tests for Git pathspec compilation and matching
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

"""Tests for Git pathspec compilation and matching."""

from dulwich.object_store import MemoryObjectStore
from dulwich.objects import Blob, Tree
from dulwich.pathspec import (
    PathspecError,
    compile_pathspecs,
    iter_tree_matches,
)

from . import TestCase

PATHS = [
    b"README.md",
    b"src/a.py",
    b"src/abc",
    b"src/deep/deeper/x.py",
    b"src/deep/notes.txt",
    b"SRC/B.py",
    b"star*file.txt",
    b"docs/guide.md",
    b"weird-\xff.py",
]


def matched(pathspecs, paths=PATHS, **kwargs):
    return list(compile_pathspecs(pathspecs, **kwargs).match_paths(paths))


class ParseErrorTests(TestCase):
    def test_attr_magic_unsupported(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":(attr:text)src/*.py"])

    def test_attr_short_unsupported(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":(attr)src"])

    def test_unknown_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":(prefix)src"])

    def test_conflicting_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":(literal,glob)src"])

    def test_unterminated_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":(glob"])

    def test_invalid_short_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b":zsrc"])

    def test_malformed_pattern(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b"src/[unterminated"])


class DefaultMagicTests(TestCase):
    def test_plain_directory_prefix(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/abc", b"src/deep/deeper/x.py", b"src/deep/notes.txt"],
            matched([b"src"]),
        )

    def test_prefix_does_not_overrun_component(self) -> None:
        self.assertEqual([], matched([b"src/a"]))

    def test_trailing_slash(self) -> None:
        self.assertEqual(matched([b"src"]), matched([b"src/"]))

    def test_star_crosses_directories(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/deep/deeper/x.py", b"SRC/B.py", b"weird-\xff.py"],
            matched([b"*.py"]),
        )

    def test_question_mark(self) -> None:
        self.assertEqual([b"src/abc"], matched([b"src/ab?"]))

    def test_bracket_expression(self) -> None:
        self.assertEqual([b"src/abc"], matched([b"src/ab[c]"]))

    def test_escaped_star_is_literal(self) -> None:
        self.assertEqual([b"star*file.txt"], matched([b"star\\*file.txt"]))

    def test_double_colon_resets_magic(self) -> None:
        self.assertEqual([b"star*file.txt"], matched([b"::star*file.txt"]))

    def test_empty_pathspec_matches_everything(self) -> None:
        self.assertEqual(PATHS, matched([]))
        self.assertEqual(PATHS, matched([b""]))


class LiteralMagicTests(TestCase):
    def test_wildcard_characters_are_literal(self) -> None:
        self.assertEqual([b"star*file.txt"], matched([b":(literal)star*file.txt"]))

    def test_directory_prefix_still_applies(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/abc", b"src/deep/deeper/x.py", b"src/deep/notes.txt"],
            matched([b":(literal)src"]),
        )

    def test_no_component_overrun(self) -> None:
        self.assertEqual([], matched([b":(literal)src/a"]))


class GlobMagicTests(TestCase):
    def test_star_does_not_cross_directories(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/abc"],
            matched([b":(glob)src/*"]),
        )

    def test_double_star_crosses_directories(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/deep/deeper/x.py"],
            matched([b":(glob)src/**/*.py"]),
        )

    def test_leading_double_star(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/deep/deeper/x.py", b"SRC/B.py", b"weird-\xff.py"],
            matched([b":(glob)**/*.py"]),
        )

    def test_no_component_overrun(self) -> None:
        self.assertEqual([], matched([b":(glob)src/a"]))
        self.assertEqual([b"src/a.py", b"src/abc"], matched([b":(glob)src/a*"]))

    def test_short_top_form(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/abc", b"src/deep/deeper/x.py", b"src/deep/notes.txt"],
            matched([b":/src"], cwd="docs"),
        )


class IcaseMagicTests(TestCase):
    def test_literal_icase(self) -> None:
        self.assertEqual([b"SRC/B.py"], matched([b":(icase)src/b.py"]))

    def test_glob_icase(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"SRC/B.py"],
            matched([b":(glob,icase)src/*.py"]),
        )

    def test_default_magic_is_case_sensitive(self) -> None:
        self.assertEqual([], matched([b"src/b.py"]))


class ExcludeMagicTests(TestCase):
    def test_exclude_after_include(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"SRC/B.py", b"weird-\xff.py"],
            matched([b"*.py", b":!src/deep"]),
        )

    def test_exclude_only_starts_from_everything(self) -> None:
        self.assertEqual(
            [p for p in PATHS if not p.endswith(b".py")],
            matched([b":!*.py"]),
        )

    def test_exclude_caret_form(self) -> None:
        self.assertEqual(matched([b":!*.py"]), matched([b":^*.py"]))

    def test_exclude_long_form(self) -> None:
        self.assertEqual(matched([b":!*.py"]), matched([b":(exclude)*.py"]))

    def test_exclude_with_glob_magic(self) -> None:
        self.assertEqual(
            [b"src/abc"],
            matched([b":(glob)src/*", b":(exclude,glob)src/*.py"]),
        )


class CwdTests(TestCase):
    def test_relative_cwd(self) -> None:
        self.assertEqual(
            [b"src/deep/deeper/x.py", b"src/deep/notes.txt"],
            matched([b"deep"], cwd="src"),
        )

    def test_top_magic_ignores_cwd(self) -> None:
        self.assertEqual([b"docs/guide.md"], matched([b":(top)docs"], cwd="src"))

    def test_exclude_is_cwd_relative(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/abc"],
            matched([b":/src", b":!deep"], cwd="src"),
        )

    def test_absolute_cwd_under_root(self) -> None:
        self.assertEqual(
            [b"src/deep/deeper/x.py", b"src/deep/notes.txt"],
            matched([b"deep"], cwd="/repo/src", root="/repo"),
        )

    def test_relative_cwd_escaping_root(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b"x"], cwd="../elsewhere")
        self.assertRaises(
            PathspecError, compile_pathspecs, [b"x"], cwd="src/../../elsewhere"
        )

    def test_absolute_cwd_outside_root(self) -> None:
        self.assertRaises(
            PathspecError,
            compile_pathspecs,
            [b"x"],
            cwd="/elsewhere",
            root="/repo",
        )

    def test_absolute_cwd_requires_root(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [b"x"], cwd="/repo")


class BytesPathTests(TestCase):
    def test_non_utf8_path_matches_bytes_pattern(self) -> None:
        self.assertEqual([b"weird-\xff.py"], matched([b"weird-\xff.py"]))

    def test_non_utf8_path_matches_wildcard(self) -> None:
        self.assertIn(b"weird-\xff.py", matched([b"*.py"]))

    def test_non_utf8_literal_pattern(self) -> None:
        self.assertEqual([b"weird-\xff.py"], matched([b":(literal)weird-\xff.py"]))


def _build_tree(store, entries):
    """Build trees for a dict of path -> blob content, return root tree id."""
    nested = {}
    for path, content in entries.items():
        blob = Blob.from_string(content)
        store.add_object(blob)
        node = nested
        parts = path.split(b"/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = blob.id

    def make_tree(node):
        tree = Tree()
        for name, value in sorted(node.items()):
            if isinstance(value, dict):
                tree.add(name, 0o040000, make_tree(value))
            else:
                tree.add(name, 0o100644, value)
        store.add_object(tree)
        return tree.id

    return make_tree(nested)


class IterTreeMatchesTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store = MemoryObjectStore()
        entries = {path: b"content\n" for path in PATHS}
        self.tree_id = _build_tree(self.store, entries)

    def _matched_paths(self, pathspecs, **kwargs):
        pathspec = compile_pathspecs(pathspecs, **kwargs)
        return sorted(
            entry.path
            for entry in iter_tree_matches(self.store, self.tree_id, pathspec)
        )

    def test_enumerates_whole_tree(self) -> None:
        self.assertEqual(sorted(PATHS), self._matched_paths([]))

    def test_glob_magic(self) -> None:
        self.assertEqual(
            [b"src/a.py", b"src/deep/deeper/x.py"],
            self._matched_paths([b":(glob)src/**/*.py"]),
        )

    def test_exclude(self) -> None:
        self.assertEqual(
            sorted(p for p in PATHS if not p.endswith(b".py")),
            self._matched_paths([b":!*.py"]),
        )

    def test_compiled_pathspec_is_reusable_across_trees(self) -> None:
        pathspec = compile_pathspecs([b":(glob)**/*.py"])
        other_store = MemoryObjectStore()
        other_tree_id = _build_tree(
            other_store, {b"lib/util.py": b"x\n", b"lib/util.pyc": b"y\n"}
        )
        self.assertEqual(
            [b"lib/util.py"],
            [
                entry.path
                for entry in iter_tree_matches(other_store, other_tree_id, pathspec)
            ],
        )
        self.assertEqual(
            sorted(p for p in PATHS if p.endswith(b".py")),
            sorted(
                entry.path
                for entry in iter_tree_matches(self.store, self.tree_id, pathspec)
            ),
        )

# test_pathspec.py -- tests for Git pathspec compilation
# Copyright (C) 2026 Jelmer Vernooij <jelmer@jelmer.uk>
#
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# Dulwich is dual-licensed under the Apache License, Version 2.0 and the GNU
# General Public License as published by the Free Software Foundation; version 2.0
# or (at your option) any later version. You can redistribute it and/or
# modify it under the terms of either of these two licenses.
#
# You should have received a copy of the licenses; if not, see
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/LICENSE-2.0> for a copy of the Apache
# License, Version 2.0.
#

"""Tests for dulwich.pathspec."""

import os
import shutil
import stat
import tempfile

from dulwich.objects import Blob, Tree
from dulwich.pathspec import (
    PathspecError,
    compile_pathspecs,
    iter_tree_matches,
)
from dulwich.repo import MemoryRepo

from . import TestCase
from .compat.utils import require_git_version, run_git_or_fail

# A tree exercising case differences, a literal '*' in a file name, deep
# directories and a non-UTF-8 name.
PATHS = [
    b"top.py",
    b"src/a.py",
    b"src/abc.py",
    b"src/lit*er.py",
    b"src/deep/b.py",
    b"src/deep/deeper/c.txt",
    b"SRC/UP.PY",
    b"sub/c.txt",
    b"non\xc3\x28utf8.txt",
]


def matched(specs, paths=PATHS, **kwargs):
    """Compile specs and return the matching paths as a set."""
    return set(compile_pathspecs(specs, **kwargs).match_paths(paths))


class MagicParsingTests(TestCase):
    def test_attr_magic_unsupported(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [":(attr:text)src"])
        self.assertRaises(PathspecError, compile_pathspecs, [":(attr)src"])

    def test_unknown_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [":(frobnicate)src"])

    def test_conflicting_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [":(literal,glob)src/*.py"])

    def test_unterminated_magic(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [":(glob"])

    def test_empty_pathspec(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, [""])

    def test_short_forms(self) -> None:
        (item,) = compile_pathspecs(["::a*b"]).items
        self.assertEqual(frozenset({b"literal"}), item.magic)
        (item,) = compile_pathspecs([":/a"]).items
        self.assertEqual(frozenset({b"top"}), item.magic)
        (item,) = compile_pathspecs([":!a"]).items
        self.assertTrue(item.exclude)
        (item,) = compile_pathspecs([":^a"]).items
        self.assertTrue(item.exclude)

    def test_colon_without_magic_is_plain_pattern(self) -> None:
        (item,) = compile_pathspecs([":top.py"]).items
        self.assertEqual(frozenset(), item.magic)
        self.assertEqual(b"top.py", item.pattern)


class DefaultMagicTests(TestCase):
    def test_literal_file(self) -> None:
        self.assertEqual({b"src/a.py"}, matched(["src/a.py"]))

    def test_directory_prefix(self) -> None:
        self.assertEqual(
            {p for p in PATHS if p.startswith(b"src/")},
            matched(["src"]),
        )

    def test_directory_prefix_trailing_slash(self) -> None:
        self.assertEqual(matched(["src"]), matched(["src/"]))

    def test_no_partial_component_match(self) -> None:
        # 'src/a' must not match 'src/abc.py'.
        self.assertEqual(set(), matched(["src/a"]))

    def test_star_crosses_directories(self) -> None:
        # Without glob magic, '*' matches '/' too.
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py", b"src/deep/b.py"},
            matched(["src/*.py"]),
        )

    def test_bare_star_matches_everything(self) -> None:
        self.assertEqual(set(PATHS), matched(["*"]))

    def test_question_mark(self) -> None:
        self.assertIn(b"top.py", matched(["to?.py"]))

    def test_non_utf8_path(self) -> None:
        self.assertEqual({b"non\xc3\x28utf8.txt"}, matched(["non*"]))
        self.assertEqual({b"non\xc3\x28utf8.txt"}, matched([b"non\xc3\x28utf8.txt"]))


class GlobMagicTests(TestCase):
    def test_star_stays_in_directory(self) -> None:
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py"},
            matched([":(glob)src/*.py"]),
        )

    def test_double_star_spans_directories(self) -> None:
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py", b"src/deep/b.py"},
            matched([":(glob)src/**/*.py"]),
        )

    def test_glob_directory_without_wildcards_is_prefix(self) -> None:
        self.assertEqual(matched(["src"]), matched([":(glob)src"]))


class LiteralMagicTests(TestCase):
    def test_star_is_literal(self) -> None:
        self.assertEqual({b"src/lit*er.py"}, matched([":(literal)src/lit*er.py"]))
        self.assertEqual({b"src/lit*er.py"}, matched(["::src/lit*er.py"]))

    def test_literal_directory_prefix(self) -> None:
        self.assertEqual(matched(["src"]), matched([":(literal)src"]))


class IcaseMagicTests(TestCase):
    def test_case_insensitive_literal(self) -> None:
        self.assertEqual({b"SRC/UP.PY"}, matched([":(icase)src/up.py"]))

    def test_case_insensitive_wildcard(self) -> None:
        self.assertEqual(
            {
                b"SRC/UP.PY",
                b"src/a.py",
                b"src/abc.py",
                b"src/lit*er.py",
                b"src/deep/b.py",
            },
            matched([":(icase)src/*.PY"]),
        )


class ExcludeTests(TestCase):
    def test_exclude_only_starts_from_everything(self) -> None:
        self.assertEqual(
            {p for p in PATHS if not p.startswith(b"src/")},
            matched([":!src"]),
        )

    def test_positive_set_then_exclude(self) -> None:
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py"},
            matched(["src", ":(exclude)src/deep"]),
        )

    def test_exclude_with_glob(self) -> None:
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py"},
            matched(["src", ":(exclude,glob)src/deep/**"]),
        )

    def test_exclude_does_not_resurrect(self) -> None:
        # An exclude pathspec cannot add paths on its own.
        self.assertEqual(
            {b"top.py", b"SRC/UP.PY", b"non\xc3\x28utf8.txt"},
            matched([":(exclude)src", ":(exclude)sub"]),
        )


class TopAndCwdTests(TestCase):
    def test_relative_to_cwd(self) -> None:
        self.assertEqual(
            {b"src/a.py", b"src/abc.py", b"src/lit*er.py", b"src/deep/b.py"},
            matched(["*.py"], cwd="src"),
        )

    def test_top_ignores_cwd(self) -> None:
        self.assertEqual({b"top.py"}, matched([":/top.py"], cwd="sub"))
        self.assertEqual({b"top.py"}, matched([":(top)top.py"], cwd="sub"))

    def test_dotdot_within_repository(self) -> None:
        self.assertEqual({b"top.py"}, matched(["../top.py"], cwd="src"))

    def test_cwd_outside_repository_rejected(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, ["*.py"], cwd="../outside")

    def test_pattern_escaping_repository_rejected(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, ["../../x"], cwd="src")

    def test_absolute_cwd_requires_root(self) -> None:
        self.assertRaises(PathspecError, compile_pathspecs, ["*.py"], cwd="/some/where")

    def test_absolute_cwd_outside_root_rejected(self) -> None:
        self.assertRaises(
            PathspecError,
            compile_pathspecs,
            ["*.py"],
            cwd="/elsewhere",
            root="/repo",
        )

    def test_absolute_cwd_inside_root(self) -> None:
        self.assertEqual(
            {b"src/a.py"},
            matched(["a.py"], cwd="/repo/src", root="/repo"),
        )


def build_tree(store, paths):
    """Build a tree (and blobs) for paths in store, return the tree id."""
    root: dict = {}
    for path in paths:
        node = root
        parts = path.split(b"/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = path

    def emit(node):
        tree = Tree()
        for name in sorted(node):
            value = node[name]
            if isinstance(value, dict):
                tree.add(name, stat.S_IFDIR, emit(value))
            else:
                blob = Blob()
                blob.data = b"contents of " + value
                store.add_object(blob)
                tree.add(name, 0o100644, blob.id)
        store.add_object(tree)
        return tree.id

    return emit(root)


class IterTreeMatchesTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.repo = MemoryRepo()
        self.tree_a = build_tree(self.repo.object_store, PATHS)
        self.tree_b = build_tree(
            self.repo.object_store, [b"other/x.py", b"src/only.py"]
        )

    def paths(self, tree_id, matcher) -> list[bytes]:
        return [
            entry.path
            for entry in iter_tree_matches(self.repo.object_store, tree_id, matcher)
        ]

    def test_enumerates_matching_entries(self) -> None:
        matcher = compile_pathspecs([":(glob)src/**/*.py"])
        self.assertEqual(
            [b"src/a.py", b"src/abc.py", b"src/deep/b.py", b"src/lit*er.py"],
            self.paths(self.tree_a, matcher),
        )

    def test_compiled_matcher_is_reusable_across_trees(self) -> None:
        matcher = compile_pathspecs(["src/*.py"])
        self.assertEqual(
            [b"src/a.py", b"src/abc.py", b"src/deep/b.py", b"src/lit*er.py"],
            self.paths(self.tree_a, matcher),
        )
        self.assertEqual([b"src/only.py"], self.paths(self.tree_b, matcher))

    def test_non_utf8_and_case_sensitive_names(self) -> None:
        matcher = compile_pathspecs([b"non\xc3\x28utf8.txt", "SRC"])
        self.assertEqual(
            [b"SRC/UP.PY", b"non\xc3\x28utf8.txt"],
            self.paths(self.tree_a, matcher),
        )

    def test_include_trees(self) -> None:
        matcher = compile_pathspecs(["src/deep"])
        entries = list(
            iter_tree_matches(
                self.repo.object_store, self.tree_a, matcher, include_trees=True
            )
        )
        self.assertEqual(
            [
                b"src/deep",
                b"src/deep/b.py",
                b"src/deep/deeper",
                b"src/deep/deeper/c.txt",
            ],
            [entry.path for entry in entries],
        )


class GitLsFilesCompatTests(TestCase):
    """Cross-check the supported subset against git ls-files."""

    def setUp(self) -> None:
        super().setUp()
        require_git_version((2, 13))
        self.repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.repo_dir)
        run_git_or_fail(["init", "-q"], cwd=self.repo_dir)
        # Files that are safe to create on a case-insensitive filesystem.
        on_disk = [
            "top.py",
            "src/a.py",
            "src/abc.py",
            "src/lit*er.py",
            "src/deep/b.py",
            "src/deep/deeper/c.txt",
            "sub/c.txt",
        ]
        for relpath in on_disk:
            full = os.path.join(self.repo_dir, relpath)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(b"contents\n")
        run_git_or_fail(["add", "-A"], cwd=self.repo_dir)
        # Inject entries the filesystem cannot represent: a case-differing
        # directory and a non-UTF-8 file name.
        blob = run_git_or_fail(
            ["hash-object", "-w", "--stdin"], input=b"", cwd=self.repo_dir
        ).strip()
        run_git_or_fail(
            ["update-index", "--add", "--cacheinfo", b"100644," + blob + b",SRC/UP.PY"],
            cwd=self.repo_dir,
        )
        run_git_or_fail(
            [
                "update-index",
                "--add",
                "--cacheinfo",
                b"100644," + blob + b",non\xc3\x28utf8.txt",
            ],
            cwd=self.repo_dir,
        )
        out = run_git_or_fail(["ls-files", "-z", "--full-name"], cwd=self.repo_dir)
        self.all_paths = [p for p in out.split(b"\0") if p]

    def git_ls_files(self, specs, cwd=None):
        args = ["ls-files", "-z", "--full-name", "--", *specs]
        run_cwd = self.repo_dir if cwd is None else os.path.join(self.repo_dir, cwd)
        out = run_git_or_fail(args, cwd=run_cwd)
        return [p for p in out.split(b"\0") if p]

    def assert_matches_git(self, specs, cwd=None) -> None:
        expected = self.git_ls_files(specs, cwd=cwd)
        matcher = compile_pathspecs(specs, cwd=cwd)
        actual = list(matcher.match_paths(self.all_paths))
        self.assertEqual(expected, actual, f"pathspecs={specs!r} cwd={cwd!r}")

    def test_default_magic(self) -> None:
        self.assert_matches_git(["src/*.py"])
        self.assert_matches_git(["*.py"])
        self.assert_matches_git(["src"])
        self.assert_matches_git(["src/"])
        self.assert_matches_git(["src/a"])
        self.assert_matches_git(["non*"])

    def test_glob_magic(self) -> None:
        self.assert_matches_git([":(glob)src/**/*.py"])
        self.assert_matches_git([":(glob)src/*.py"])
        self.assert_matches_git([":(glob)**/*.txt"])

    def test_literal_magic(self) -> None:
        self.assert_matches_git([":(literal)src/lit*er.py"])
        self.assert_matches_git(["::src/lit*er.py"])

    def test_icase_magic(self) -> None:
        self.assert_matches_git([":(icase)src/*.PY"])
        self.assert_matches_git([":(icase)SRC/UP.PY"])

    def test_exclude_magic(self) -> None:
        self.assert_matches_git([":!src"])
        self.assert_matches_git([":^src/deep"])
        self.assert_matches_git(["src", ":(exclude)src/deep"])
        self.assert_matches_git(["src", ":(exclude,glob)src/deep/**"])

    def test_top_and_cwd(self) -> None:
        self.assert_matches_git([":/top.py"], cwd="sub")
        self.assert_matches_git(["*.py"], cwd="src")
        self.assert_matches_git(["../top.py"], cwd="src")

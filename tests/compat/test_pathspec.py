# test_pathspec.py -- Compatibility tests for git pathspec matching
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

"""Compatibility tests for pathspec matching against git itself."""

import os
import posixpath
import shutil
import tempfile
from io import StringIO

from dulwich import porcelain
from dulwich.pathspec import compile_pathspecs
from dulwich.repo import Repo

from .utils import CompatTestCase, run_git_or_fail

# Paths exercise case differences, a literal asterisk, deep directories
# and a non-UTF-8 name.
FILES = [
    b"README.md",
    b"src/a.py",
    b"src/abc",
    b"src/deep/deeper/x.py",
    b"src/deep/notes.txt",
    b"SRC/B.py",
    b"star*file.txt",
    b"docs/guide.md",
]

# Added to the index directly, since some filesystems cannot host it
NON_UTF8_FILE = b"weird-\xff.py"

PATHSPEC_SETS = [
    [b"src"],
    [b"src/"],
    [b"src/a"],
    [b"*.py"],
    [b"src/**/*.py"],
    [b"star*file.txt"],
    [b"star\\*file.txt"],
    [b":(literal)star*file.txt"],
    [b":(glob)src/*"],
    [b":(glob)src/**/*.py"],
    [b":(glob)**/*.py"],
    [b":(glob)src/a"],
    [b":(icase)src/b.py"],
    [b":(glob,icase)src/*.py"],
    [b"*.py", b":!src/deep"],
    [b":!*.py"],
    [b":(glob)src/*", b":(exclude,glob)src/*.py"],
    [b"docs", b":(exclude)*.md"],
    [b"weird-\xff.py"],
    [b":(top)docs"],
]


class PathspecCompatTestCase(CompatTestCase):
    """Compare dulwich pathspec matching with git ls-files/git grep."""

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir, True)
        run_git_or_fail(["init", "-q", "."], cwd=self.test_dir)
        for relpath in FILES:
            full = os.path.join(os.fsencode(self.test_dir), relpath)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(b"needle\n")
        run_git_or_fail(["add", "-A"], cwd=self.test_dir)
        blob_sha = run_git_or_fail(
            ["hash-object", "-w", "--stdin"], cwd=self.test_dir, input=b"needle\n"
        ).strip()
        run_git_or_fail(
            [
                "update-index",
                "--add",
                "--cacheinfo",
                b"100644," + blob_sha + b"," + NON_UTF8_FILE,
            ],
            cwd=self.test_dir,
        )
        run_git_or_fail(
            [
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "files",
            ],
            cwd=self.test_dir,
        )

    def _git_ls_files(self, pathspecs, cwd=None):
        args = ["ls-files", "-z", "--", *[os.fsdecode(p) for p in pathspecs]]
        output = run_git_or_fail(args, cwd=cwd or self.test_dir)
        return {p for p in output.split(b"\0") if p}

    def _dulwich_match(self, pathspecs, cwd=None):
        all_paths = self._git_ls_files([])
        pathspec = compile_pathspecs(pathspecs, cwd=cwd, root=self.test_dir)
        return set(pathspec.match_paths(all_paths))

    def test_ls_files_pathspec_sets(self) -> None:
        for pathspecs in PATHSPEC_SETS:
            with self.subTest(pathspecs=pathspecs):
                self.assertEqual(
                    self._git_ls_files(pathspecs),
                    self._dulwich_match(pathspecs),
                )

    def test_ls_files_from_subdirectory(self) -> None:
        subdir = os.path.join(self.test_dir, "src")
        for pathspecs in [[b"deep"], [b"*.py"], [b":(top)docs"], [b":!deep"]]:
            with self.subTest(pathspecs=pathspecs):
                # git reports paths relative to the cwd, possibly
                # escaping it with "../" for top-magic pathspecs
                expected = {
                    posixpath.normpath(b"src/" + p)
                    for p in self._git_ls_files(pathspecs, cwd=subdir)
                }
                self.assertEqual(expected, self._dulwich_match(pathspecs, cwd="src"))

    def test_grep_pathspec_sets(self) -> None:
        repo = Repo(self.test_dir)
        self.addCleanup(repo.close)
        for pathspecs in PATHSPEC_SETS:
            with self.subTest(pathspecs=pathspecs):
                expected = self._git_ls_files(pathspecs)
                outstream = StringIO()
                porcelain.grep(
                    repo,
                    "needle",
                    outstream=outstream,
                    pathspecs=pathspecs,
                    respect_ignores=False,
                )
                actual = set()
                for line in outstream.getvalue().splitlines():
                    path, _sep, _rest = line.partition(":")
                    actual.add(path.encode("utf-8", errors="replace"))
                # dulwich renders non-UTF-8 path names lossily on output;
                # apply the same decoding to git's paths for comparison
                expected = {
                    p.decode("utf-8", errors="replace").encode(
                        "utf-8", errors="replace"
                    )
                    for p in expected
                }
                self.assertEqual(expected, actual)

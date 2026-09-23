# errors.py -- errors for dulwich
# Copyright (C) 2007 James Westby <jw+debian@jameswestby.net>
# Copyright (C) 2009-2012 Jelmer Vernooij <jelmer@jelmer.uk>
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

"""Dulwich-related exception classes and utility functions."""

__all__ = [
    "ApplyDeltaError",
    "ChecksumMismatch",
    "CommitError",
    "FileFormatException",
    "GitProtocolError",
    "HangupException",
    "HookError",
    "MissingCommitError",
    "NoIndexPresent",
    "NotBlobError",
    "NotCommitError",
    "NotGitRepository",
    "NotTagError",
    "NotTreeError",
    "ObjectFormatException",
    "ObjectMissing",
    "PackedRefsException",
    "RefFormatError",
    "SendPackError",
    "UnexpectedCommandError",
    "WorkingTreeModifiedError",
    "WrongObjectException",
]

# Please do not add more errors here, but instead add them close to the code
# that raises the error.

import binascii
from collections.abc import Sequence


class ChecksumMismatch(Exception):
    """A checksum didn't match the expected contents."""

    def __init__(
        self,
        expected: bytes | str,
        got: bytes | str,
        extra: str | None = None,
    ) -> None:
        """Initialize a ChecksumMismatch exception.

        Args:
            expected: The expected checksum value (bytes or hex string).
            got: The actual checksum value (bytes or hex string).
            extra: Optional additional error information.
        """
        if isinstance(expected, bytes) and len(expected) in (20, 32):
            expected_str = binascii.hexlify(expected).decode("ascii")
        else:
            expected_str = (
                expected if isinstance(expected, str) else expected.decode("ascii")
            )
        if isinstance(got, bytes) and len(got) in (20, 32):
            got_str = binascii.hexlify(got).decode("ascii")
        else:
            got_str = got if isinstance(got, str) else got.decode("ascii")
        self.expected = expected_str
        self.got = got_str
        self.extra = extra
        message = f"Checksum mismatch: Expected {expected_str}, got {got_str}"
        if self.extra is not None:
            message += f"; {extra}"
        Exception.__init__(self, message)


class WrongObjectException(Exception):
    """Baseclass for all the _ is not a _ exceptions on objects.

    Do not instantiate directly.

    Subclasses should define a type_name attribute that indicates what
    was expected if they were raised.
    """

    type_name: str

    def __init__(self, sha: bytes, *args: object, **kwargs: object) -> None:
        """Initialize a WrongObjectException.

        Args:
            sha: The SHA of the object that was not of the expected type.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        Exception.__init__(self, f"{sha.decode('ascii')} is not a {self.type_name}")


class NotCommitError(WrongObjectException):
    """Indicates that the sha requested does not point to a commit."""

    type_name = "commit"


class NotTreeError(WrongObjectException):
    """Indicates that the sha requested does not point to a tree."""

    type_name = "tree"


class NotTagError(WrongObjectException):
    """Indicates that the sha requested does not point to a tag."""

    type_name = "tag"


class NotBlobError(WrongObjectException):
    """Indicates that the sha requested does not point to a blob."""

    type_name = "blob"


class MissingCommitError(Exception):
    """Indicates that a commit was not found in the repository."""

    def __init__(self, sha: bytes, *args: object, **kwargs: object) -> None:
        """Initialize a MissingCommitError.

        Args:
            sha: The SHA of the missing commit.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.sha = sha
        Exception.__init__(self, f"{sha.decode('ascii')} is not in the revision store")


def _format_replace_chain(chain: list[bytes]) -> str:
    """Format a replacement chain for error messages."""
    return " -> ".join(sha.decode("ascii") for sha in chain)


class ReplacementError(Exception):
    """Indicates that a refs/replace/ chain could not be resolved."""

    def __init__(self, chain: list[bytes], *args: object) -> None:
        """Initialize a ReplacementError.

        Args:
            chain: The full chain of object SHAs that was followed,
              starting with the originally requested object.
            *args: Additional positional arguments.
        """
        self.chain = list(chain)
        if not args:
            args = (f"replacement chain {_format_replace_chain(self.chain)}",)
        Exception.__init__(self, *args)


class MissingReplacementTarget(ReplacementError):
    """Indicates that a replacement ref points at a missing object."""

    def __init__(self, chain: list[bytes]) -> None:
        """Initialize a MissingReplacementTarget.

        Args:
            chain: The full chain of object SHAs that was followed; the last
              entry is the missing target.
        """
        self.chain = list(chain)
        Exception.__init__(
            self,
            f"replacement target {self.chain[-1].decode('ascii')} is missing "
            f"(chain: {_format_replace_chain(self.chain)})",
        )


class ReplacementTypeMismatch(ReplacementError):
    """Indicates that a replacement object has a different type."""

    def __init__(
        self, chain: list[bytes], expected_type: bytes, actual_type: bytes
    ) -> None:
        """Initialize a ReplacementTypeMismatch.

        Args:
            chain: The full chain of object SHAs that was followed.
            expected_type: Type name of the originally requested object.
            actual_type: Type name of the replacement object.
        """
        self.chain = list(chain)
        self.expected_type = expected_type
        self.actual_type = actual_type
        Exception.__init__(
            self,
            f"replacement object has type {actual_type.decode('ascii')}, "
            f"expected {expected_type.decode('ascii')} "
            f"(chain: {_format_replace_chain(self.chain)})",
        )


class ReplacementCycle(ReplacementError):
    """Indicates that a refs/replace/ chain contains a cycle."""

    def __init__(self, chain: list[bytes]) -> None:
        """Initialize a ReplacementCycle.

        Args:
            chain: The full chain of object SHAs that was followed; the last
              entry repeats an earlier entry.
        """
        self.chain = list(chain)
        Exception.__init__(
            self,
            f"replacement chain contains a cycle: {_format_replace_chain(self.chain)}",
        )


class ObjectMissing(Exception):
    """Indicates that a requested object is missing."""

    def __init__(self, sha: bytes, *args: object, **kwargs: object) -> None:
        """Initialize an ObjectMissing exception.

        Args:
            sha: The SHA of the missing object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        Exception.__init__(self, f"{sha.decode('ascii')} is not in the pack")


class ApplyDeltaError(Exception):
    """Indicates that applying a delta failed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize an ApplyDeltaError.

        Args:
            *args: Error message and additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        Exception.__init__(self, *args, **kwargs)


class NotGitRepository(Exception):
    """Indicates that no Git repository was found."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a NotGitRepository exception.

        Args:
            *args: Error message and additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        Exception.__init__(self, *args, **kwargs)


class GitProtocolError(Exception):
    """Git protocol exception."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a GitProtocolError.

        Args:
            *args: Error message and additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        Exception.__init__(self, *args, **kwargs)

    def __eq__(self, other: object) -> bool:
        """Check equality between GitProtocolError instances.

        Args:
            other: The object to compare with.

        Returns:
            True if both are GitProtocolError instances with same args, False otherwise.
        """
        return isinstance(other, GitProtocolError) and self.args == other.args


class SendPackError(GitProtocolError):
    """An error occurred during send_pack."""


class HangupException(GitProtocolError):
    """Hangup exception."""

    def __init__(self, stderr_lines: Sequence[bytes] | None = None) -> None:
        """Initialize a HangupException.

        Args:
            stderr_lines: Optional list of stderr output lines from the remote server.
        """
        if stderr_lines:
            super().__init__(
                "\n".join(
                    line.decode("utf-8", "surrogateescape") for line in stderr_lines
                )
            )
        else:
            super().__init__("The remote server unexpectedly closed the connection.")
        self.stderr_lines = stderr_lines

    def __eq__(self, other: object) -> bool:
        """Check equality between HangupException instances.

        Args:
            other: The object to compare with.

        Returns:
            True if both are HangupException instances with same stderr_lines, False otherwise.
        """
        return (
            isinstance(other, HangupException)
            and self.stderr_lines == other.stderr_lines
        )


class UnexpectedCommandError(GitProtocolError):
    """Unexpected command received in a proto line."""

    def __init__(self, command: str | None) -> None:
        """Initialize an UnexpectedCommandError.

        Args:
            command: The unexpected command received, or None for flush-pkt.
        """
        command_str = "flush-pkt" if command is None else f"command {command}"
        super().__init__(f"Protocol got unexpected {command_str}")


class FileFormatException(Exception):
    """Base class for exceptions relating to reading git file formats."""


class PackedRefsException(FileFormatException):
    """Indicates an error parsing a packed-refs file."""


class ObjectFormatException(FileFormatException):
    """Indicates an error parsing an object."""


class NoIndexPresent(Exception):
    """No index is present."""


class CommitError(Exception):
    """An error occurred while performing a commit."""


class RefFormatError(Exception):
    """Indicates an invalid ref name."""


class HookError(Exception):
    """An error occurred while executing a hook."""


class WorkingTreeModifiedError(Exception):
    """Indicates that the working tree has modifications that would be overwritten."""

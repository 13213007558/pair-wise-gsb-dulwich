# replace.py -- Read-through views applying refs/replace/ substitutions.
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

"""Read-through views that apply refs/replace/ substitutions.

A replacement ref (refs/replace/<sha>) declares that the object named in the
ref should be read as the object the ref points at. This module provides an
explicit opt-in view over a repository so that existing traversal APIs
(walkers, merge-base, object reads) operate on the replaced history without
modifying the underlying objects or refs.

The view is strictly read-through: writes and object enumeration always go to
the physical store, so packing and connectivity checks keep a well-defined
physical read channel and never mistake replacement content for the original
objects.
"""

__all__ = [
    "ReplacementCycleError",
    "ReplacementError",
    "ReplacementObjectStore",
    "ReplacementRepoView",
    "ReplacementTargetMissingError",
    "ReplacementTypeMismatchError",
    "read_replace_refs",
]

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from typing import TYPE_CHECKING

from .object_store import (
    BaseObjectStore,
    GraphWalker,
    ObjectReachabilityProvider,
)
from .objects import ObjectID, RawObjectID, ShaFile, object_class, sha_to_hex
from .refs import LOCAL_REPLACE_PREFIX, RefsContainer
from .repo import BaseRepo

if TYPE_CHECKING:
    from .pack import Pack, PackHint, UnpackedObject


class ReplacementError(Exception):
    """Base class for errors resolving replacement refs.

    Attributes:
      original: SHA of the object the failing resolution was started for.
    """

    def __init__(self, original: ObjectID, message: str) -> None:
        """Initialize a ReplacementError.

        Args:
          original: SHA of the object being resolved when the error occurred.
          message: Human readable error message.
        """
        self.original = original
        super().__init__(message)


class ReplacementCycleError(ReplacementError):
    """A cycle was detected while following replacement refs."""

    def __init__(self, original: ObjectID, chain: Sequence[ObjectID]) -> None:
        """Initialize a ReplacementCycleError.

        Args:
          original: SHA of the object being resolved.
          chain: The chain of object SHAs followed before the cycle was found,
            including the repeated object at the end.
        """
        self.chain = list(chain)
        path = " -> ".join(sha.decode("ascii") for sha in self.chain)
        super().__init__(
            original,
            f"replacement cycle detected resolving "
            f"{original.decode('ascii')}: {path}",
        )


class ReplacementTargetMissingError(ReplacementError):
    """A replacement ref points at an object that does not exist."""

    def __init__(self, original: ObjectID, missing: ObjectID) -> None:
        """Initialize a ReplacementTargetMissingError.

        Args:
          original: SHA of the object being resolved.
          missing: SHA of the replacement target that is missing.
        """
        self.missing = missing
        super().__init__(
            original,
            f"replacement target {missing.decode('ascii')} for "
            f"{original.decode('ascii')} is missing",
        )


class ReplacementTypeMismatchError(ReplacementError):
    """A replacement object has a different type than the object it replaces."""

    def __init__(
        self,
        original: ObjectID,
        target: ObjectID,
        expected_type: int,
        actual_type: int,
    ) -> None:
        """Initialize a ReplacementTypeMismatchError.

        Args:
          original: SHA of the object being resolved.
          target: SHA of the replacement object with the mismatched type.
          expected_type: Numeric type of the replaced object.
          actual_type: Numeric type of the replacement object.
        """
        self.target = target
        self.expected_type = expected_type
        self.actual_type = actual_type

        def _type_name(type_num: int) -> bytes:
            cls = object_class(type_num)
            return cls.type_name if cls is not None else b"unknown"

        super().__init__(
            original,
            f"replacement object {target.decode('ascii')} for "
            f"{original.decode('ascii')} has type "
            f"{_type_name(actual_type).decode('ascii')}, expected "
            f"{_type_name(expected_type).decode('ascii')}",
        )


def read_replace_refs(refs: RefsContainer) -> dict[ObjectID, ObjectID]:
    """Read the replacement refs from a refs container.

    Args:
      refs: Refs container to read refs/replace/* from.

    Returns:
      Dict mapping the SHA of each replaced object to the SHA of its
      (single hop) replacement.
    """
    replacements: dict[ObjectID, ObjectID] = {}
    for ref in refs.keys():
        if ref.startswith(LOCAL_REPLACE_PREFIX):
            replacements[ObjectID(ref[len(LOCAL_REPLACE_PREFIX) :])] = refs[ref]
    return replacements


class ReplacementObjectStore(BaseObjectStore):
    """Object store view that resolves refs/replace/ substitutions on read.

    Reads for a replaced object return the contents of the (transitively
    resolved) replacement object, while the returned object keeps the
    identity it was requested under. Use resolve() to obtain the identity of
    the object that actually backs a read.

    Object enumeration and writes are always physical: __iter__ yields the
    SHAs of the underlying store and add_object()/add_objects() write to it
    directly, so pack generation and connectivity checks can use this store
    (or the physical attribute) without replacement content leaking in.
    """

    def __init__(
        self,
        base_store: BaseObjectStore,
        replacements: Mapping[ObjectID, ObjectID] | None = None,
        *,
        refs: RefsContainer | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize a ReplacementObjectStore.

        Args:
          base_store: The physical object store to read from and write to.
          replacements: Mapping of replaced object SHA to replacement SHA.
            If None, the mapping is read from refs.
          refs: Refs container to read refs/replace/* from. Ignored if
            replacements is given.
          enabled: Whether replacements are applied. When False, all reads
            are physical.
        """
        super().__init__(object_format=base_store.object_format)
        if replacements is None:
            replacements = read_replace_refs(refs) if refs is not None else {}
        self._base_store = base_store
        self._refs = refs
        self._replacements: dict[ObjectID, ObjectID] = {
            ObjectID(original): ObjectID(target)
            for original, target in replacements.items()
        }
        self.enabled = enabled

    def __repr__(self) -> str:
        """Return a string representation of this store."""
        return (
            f"{type(self).__name__}({self._base_store!r}, "
            f"replacements={len(self._replacements)}, enabled={self.enabled})"
        )

    @property
    def physical(self) -> BaseObjectStore:
        """The underlying physical object store, bypassing all replacements."""
        return self._base_store

    @property
    def replacements(self) -> dict[ObjectID, ObjectID]:
        """The single hop replacement mapping currently in effect."""
        return dict(self._replacements)

    def refresh(self) -> None:
        """Re-read the replacement refs from the refs container.

        Raises:
          ValueError: If this store was not created with a refs container.
        """
        if self._refs is None:
            raise ValueError("no refs container associated with this store")
        self._replacements = read_replace_refs(self._refs)

    def _to_hex(self, sha: ObjectID | RawObjectID) -> ObjectID:
        if len(sha) == self.object_format.oid_length:
            return sha_to_hex(RawObjectID(sha))
        return ObjectID(sha)

    def _type_of(self, sha: ObjectID) -> int | None:
        try:
            return self._base_store.get_raw(sha)[0]
        except KeyError:
            return None

    def resolve(self, sha: ObjectID | RawObjectID) -> ObjectID:
        """Resolve the replacement chain for an object.

        Args:
          sha: SHA of the object to resolve (hex or raw).

        Returns:
          The SHA of the object that physically backs reads for sha. This is
          sha itself when replacements are disabled or no replacement applies.

        Raises:
          ReplacementCycleError: If the replacement chain contains a cycle.
          ReplacementTargetMissingError: If a replacement target is missing.
          ReplacementTypeMismatchError: If a replacement object has a
            different type than the object it replaces.
        """
        current = self._to_hex(sha)
        if not self.enabled or current not in self._replacements:
            return current
        expected_type = self._type_of(current)
        seen = {current}
        chain = [current]
        while current in self._replacements:
            target = self._replacements[current]
            if target in seen:
                raise ReplacementCycleError(self._to_hex(sha), [*chain, target])
            target_type = self._type_of(target)
            if target_type is None:
                raise ReplacementTargetMissingError(self._to_hex(sha), target)
            if expected_type is None:
                expected_type = target_type
            elif target_type != expected_type:
                raise ReplacementTypeMismatchError(
                    self._to_hex(sha), target, expected_type, target_type
                )
            seen.add(target)
            chain.append(target)
            current = target
        return current

    def get_raw(self, name: ObjectID | RawObjectID) -> tuple[int, bytes]:
        """Obtain the raw text for an object, following replacements.

        Args:
          name: SHA of the object.

        Returns:
          Tuple with numeric type and object contents of the resolved object.
        """
        return self._base_store.get_raw(self.resolve(name))

    def __getitem__(self, sha: ObjectID | RawObjectID) -> ShaFile:
        """Obtain an object by SHA, following replacements.

        The returned object carries the contents of the resolved replacement
        object but keeps the identity it was requested under, so that
        traversal output shows the original (replaced) object ids. Use
        resolve() to find the identity of the object actually read.
        """
        hexsha = self._to_hex(sha)
        resolved = self.resolve(hexsha)
        type_num, uncomp = self._base_store.get_raw(resolved)
        if resolved == hexsha:
            return ShaFile.from_raw_string(
                type_num, uncomp, verify_sha=hexsha, object_format=self.object_format
            )
        return ShaFile.from_raw_string(
            type_num, uncomp, sha=hexsha, object_format=self.object_format
        )

    def __contains__(self, sha: ObjectID) -> bool:
        """Check whether an object can be read through this view."""
        try:
            resolved = self.resolve(sha)
        except ReplacementError:
            return False
        return resolved in self._base_store

    def __iter__(self) -> Iterator[ObjectID]:
        """Iterate over the SHAs physically present in the underlying store.

        Iteration deliberately bypasses replacements so that pack generation
        and connectivity checks enumerate the physical object graph.
        """
        return iter(self._base_store)

    def contains_loose(self, sha: ObjectID) -> bool:
        """Check if the resolved object is present and loose."""
        return self._base_store.contains_loose(self.resolve(sha))

    def contains_packed(self, sha: ObjectID | RawObjectID) -> bool:
        """Check if the resolved object is present and packed."""
        return self._base_store.contains_packed(self.resolve(sha))

    @property
    def packs(self) -> list["Pack"]:
        """Iterable of pack objects in the underlying store."""
        return self._base_store.packs

    def add_object(self, obj: ShaFile) -> None:
        """Add a single object to the underlying physical store."""
        self._base_store.add_object(obj)

    def add_objects(
        self,
        objects: Sequence[tuple[ShaFile, str | None]],
        progress: Callable[..., None] | None = None,
    ) -> "Pack | None":
        """Add a set of objects to the underlying physical store."""
        return self._base_store.add_objects(objects, progress=progress)

    def generate_pack_data(
        self,
        have: Iterable[ObjectID],
        want: Iterable[ObjectID],
        *,
        shallow: "Set[ObjectID] | None" = None,
        progress: Callable[..., None] | None = None,
        ofs_delta: bool = True,
    ) -> "tuple[int, Iterator[UnpackedObject]]":
        """Generate pack data from the physical store.

        Packing always bypasses replacements so that replacement content is
        never sent in place of the original objects.
        """
        return self._base_store.generate_pack_data(
            have, want, shallow=shallow, progress=progress, ofs_delta=ofs_delta
        )

    def find_missing_objects(
        self,
        haves: Iterable[ObjectID],
        wants: Iterable[ObjectID],
        shallow: "Set[ObjectID] | None" = None,
        progress: Callable[..., None] | None = None,
        get_tagged: "Callable[[], dict[ObjectID, ObjectID]] | None" = None,
        get_parents: "Callable[..., list[ObjectID]]" = lambda commit: commit.parents,
    ) -> "Iterator[tuple[ObjectID, PackHint | None]]":
        """Find missing objects in the physical store, bypassing replacements."""
        return self._base_store.find_missing_objects(
            haves, wants, shallow, progress, get_tagged, get_parents
        )

    def find_common_revisions(self, graphwalker: "GraphWalker") -> list[ObjectID]:
        """Find common revisions in the physical store."""
        return self._base_store.find_common_revisions(graphwalker)

    def get_reachability_provider(
        self, prefer_bitmaps: bool = True
    ) -> "ObjectReachabilityProvider":
        """Get a reachability provider for the physical object graph."""
        return self._base_store.get_reachability_provider(prefer_bitmaps=prefer_bitmaps)

    def get_commit_graph(self) -> None:
        """No commit graph: the physical graph would bypass replacements."""
        return None

    def get_object_mtime(self, sha: ObjectID) -> float:
        """Get the modification time of the resolved object."""
        return self._base_store.get_object_mtime(self.resolve(sha))

    def close(self) -> None:
        """Close this view. The underlying store is not owned and stays open."""


class ReplacementRepoView(BaseRepo):
    """Read-through view of a repository with refs/replace/ applied.

    The view shares the refs and physical object store of the repository it
    wraps, but routes object reads through a ReplacementObjectStore so that
    the usual traversal APIs (get_walker, parents_provider, get_object,
    __getitem__ and helpers such as dulwich.graph.find_merge_base) operate on
    the replaced history.

    Nothing is ever written back: disabling the view (enabled = False) or
    dropping it immediately restores the physical history, and the original
    objects and refs are left untouched.
    """

    def __init__(self, repo: BaseRepo, *, enabled: bool = True) -> None:
        """Initialize a ReplacementRepoView.

        Args:
          repo: The repository to wrap.
          enabled: Whether replacements are applied initially.
        """
        self._physical_repo = repo
        self.replacement_store = ReplacementObjectStore(
            repo.object_store, refs=repo.refs, enabled=enabled
        )
        super().__init__(self.replacement_store, repo.refs)  # type: ignore[arg-type]
        # Share graft state with the physical repository.
        self._graftpoints = repo._graftpoints

    @property
    def physical_repo(self) -> BaseRepo:
        """The underlying repository, for explicit physical reads."""
        return self._physical_repo

    @property
    def enabled(self) -> bool:
        """Whether replacements are applied by this view."""
        return self.replacement_store.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.replacement_store.enabled = value

    def refresh(self) -> None:
        """Re-read the replacement refs from the repository."""
        self.replacement_store.refresh()

    def _determine_file_mode(self) -> bool:
        """Probe the physical repository for permission support."""
        return self._physical_repo._determine_file_mode()

    def _determine_symlinks(self) -> bool:
        """Probe the physical repository for symlink support."""
        return self._physical_repo._determine_symlinks()

    def get_named_file(self, path: str, basedir: str | None = None):  # type: ignore[no-untyped-def]
        """Get a named file from the physical repository."""
        return self._physical_repo.get_named_file(path)

    def get_shallow(self) -> set[ObjectID]:
        """Get the shallow commits of the physical repository."""
        return self._physical_repo.get_shallow()

    def get_config(self):  # type: ignore[no-untyped-def]
        """Get the config of the physical repository."""
        return self._physical_repo.get_config()

    def get_config_stack(self):  # type: ignore[no-untyped-def]
        """Get the config stack of the physical repository."""
        return self._physical_repo.get_config_stack()

    def get_description(self) -> bytes | None:
        """Get the description of the physical repository."""
        return self._physical_repo.get_description()

    def close(self) -> None:
        """Close this view. The wrapped repository is not owned by it."""

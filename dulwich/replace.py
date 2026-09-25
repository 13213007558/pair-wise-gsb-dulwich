# vim:ts=4:sw=4:softtabstop=4:smarttab:expandtab
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

"""Read-through support for refs/replace/* object replacements.

A replacement ref (refs/replace/<object>) declares that reads of one
object should transparently yield the contents of another.  This module
provides ReplacementObjectStore, a read-only view over a physical
object store that applies those replacements, following chains of
replacements across multiple hops.

The view never modifies the underlying objects or refs; it only affects
how objects are read.  Objects returned by the view carry the *requested*
(original) object id as their identity, while their contents come from the
resolved replacement object.  Use ReplacementObjectStore.resolve
to obtain the id of the physical object that actually backs a read.
"""

__all__ = [
    "ReplacementCycleError",
    "ReplacementError",
    "ReplacementMissingError",
    "ReplacementObjectStore",
    "ReplacementTypeMismatchError",
    "parse_replace_refs",
]

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .object_store import BaseObjectStore
from .objects import ObjectID, RawObjectID, ShaFile, object_class, sha_to_hex
from .refs import LOCAL_REPLACE_PREFIX, Ref

if TYPE_CHECKING:
    from .commit_graph import CommitGraph
    from .pack import Pack
    from .refs import RefsContainer


class ReplacementError(Exception):
    """Base class for errors resolving a replacement chain.

    Attributes:
      original: Id of the originally requested object, so callers can
        locate which object triggered the failure.
    """

    def __init__(self, original: ObjectID, message: str) -> None:
        self.original = original
        super().__init__(message)


class ReplacementCycleError(ReplacementError):
    """A replacement chain loops back on itself."""

    def __init__(self, original: ObjectID, chain: list[ObjectID]) -> None:
        self.chain = chain
        rendered = b" -> ".join(chain).decode("ascii")
        super().__init__(
            original,
            f"Replacement cycle detected for object {original.decode('ascii')}: "
            f"{rendered}",
        )


class ReplacementMissingError(ReplacementError):
    """A replacement target does not exist in the object store."""

    def __init__(self, original: ObjectID, missing: ObjectID) -> None:
        self.missing = missing
        super().__init__(
            original,
            f"Replacement target {missing.decode('ascii')} for object "
            f"{original.decode('ascii')} is missing",
        )


class ReplacementTypeMismatchError(ReplacementError):
    """A replacement object has a different type than the original object."""

    def __init__(
        self,
        original: ObjectID,
        replacement: ObjectID,
        original_type: bytes,
        replacement_type: bytes,
    ) -> None:
        self.replacement = replacement
        self.original_type = original_type
        self.replacement_type = replacement_type
        super().__init__(
            original,
            f"Replacement object {replacement.decode('ascii')} for object "
            f"{original.decode('ascii')} has type "
            f"{replacement_type.decode('ascii')}, expected "
            f"{original_type.decode('ascii')}",
        )


def parse_replace_refs(refs: "RefsContainer") -> dict[ObjectID, ObjectID]:
    """Build a replacement map from the refs/replace/* hierarchy.

    Args:
      refs: Refs container to read replacement refs from
    Returns:
      Dict mapping replaced object ids to their replacement object ids
    """
    replacements: dict[ObjectID, ObjectID] = {}
    for name, target in refs.as_dict(Ref(LOCAL_REPLACE_PREFIX.rstrip(b"/"))).items():
        replacements[ObjectID(name)] = target
    return replacements


class ReplacementObjectStore(BaseObjectStore):
    """Object store view applying refs/replace/* replacements on read.

    Reads follow replacement chains (possibly multiple hops) and return the
    replacement's contents while preserving the originally requested object
    id as the returned object's identity.  Iteration, writes and pack
    operations are delegated to the underlying physical store unchanged, so
    pack generation and connectivity checks never see replacement contents.
    """

    def __init__(
        self,
        physical_store: BaseObjectStore,
        replacements: Mapping[ObjectID, ObjectID],
    ) -> None:
        """Initialize the replacement view.

        Args:
          physical_store: The physical object store to wrap
          replacements: Map of replaced object ids to replacement object ids
        """
        super().__init__(object_format=physical_store.object_format)
        self.physical_store = physical_store
        self.replacements: dict[ObjectID, ObjectID] = dict(replacements)

    def __getattr__(self, name: str) -> Any:
        # Delegate anything not explicitly overridden (pack handling,
        # shallow info, alternates, ...) to the physical store.
        return getattr(self.physical_store, name)

    def _normalize(self, sha: ObjectID | RawObjectID) -> ObjectID:
        if len(sha) == self.object_format.oid_length:
            return sha_to_hex(RawObjectID(sha))
        return ObjectID(sha)

    def is_replaced(self, sha: ObjectID | RawObjectID) -> bool:
        """Check whether an object id is subject to replacement."""
        return self._normalize(sha) in self.replacements

    def resolve(self, sha: ObjectID | RawObjectID) -> ObjectID:
        """Resolve the replacement chain for an object id.

        Args:
          sha: Object id to resolve
        Returns:
          Id of the physical object whose contents back reads of sha;
          equal to sha itself when no replacement applies
        Raises:
          ReplacementCycleError: The replacement chain contains a cycle
          ReplacementMissingError: A replacement target is missing
        """
        original = self._normalize(sha)
        current = original
        chain = [current]
        seen = {current}
        while current in self.replacements:
            nxt = self.replacements[current]
            if nxt in seen:
                chain.append(nxt)
                raise ReplacementCycleError(original, chain)
            if nxt not in self.physical_store:
                raise ReplacementMissingError(original, nxt)
            chain.append(nxt)
            seen.add(nxt)
            current = nxt
        return current

    def get_raw(self, name: RawObjectID | ObjectID) -> tuple[int, bytes]:
        """Obtain the raw text for an object, applying replacements."""
        return self.physical_store.get_raw(self.resolve(name))

    def __getitem__(self, sha1: ObjectID | RawObjectID) -> ShaFile:
        """Obtain an object by id, applying replacements.

        The returned object keeps the requested id as its identity while
        its contents are read from the resolved replacement object.

        Raises:
          ReplacementCycleError: The replacement chain contains a cycle
          ReplacementMissingError: A replacement target is missing
          ReplacementTypeMismatchError: The replacement object has a
            different type than the original object
        """
        hexsha = self._normalize(sha1)
        final = self.resolve(hexsha)
        if final == hexsha:
            return self.physical_store[hexsha]
        type_num, uncomp = self.physical_store.get_raw(final)
        try:
            original_type, _ = self.physical_store.get_raw(hexsha)
        except KeyError:
            original_type = None
        if original_type is not None and original_type != type_num:
            raise ReplacementTypeMismatchError(
                hexsha,
                final,
                object_class(original_type).type_name,  # type: ignore[union-attr]
                object_class(type_num).type_name,  # type: ignore[union-attr]
            )
        return ShaFile.from_raw_string(
            type_num, uncomp, sha=hexsha, object_format=self.object_format
        )

    def __contains__(self, sha1: ObjectID) -> bool:
        """Check if an object is present, directly or via a replacement."""
        return sha1 in self.physical_store or sha1 in self.replacements

    def __iter__(self) -> Iterator[ObjectID]:
        """Iterate over the physical object ids present in the store."""
        return iter(self.physical_store)

    def contains_loose(self, sha: ObjectID) -> bool:
        """Check if a particular object is present and is loose."""
        return self.physical_store.contains_loose(sha)

    def contains_packed(self, sha: ObjectID | RawObjectID) -> bool:
        """Check if a particular object is present and is packed."""
        return self.physical_store.contains_packed(sha)

    @property
    def packs(self) -> "list[Pack]":
        """Iterable over pack objects of the physical store."""
        return self.physical_store.packs

    def add_object(self, obj: ShaFile) -> None:
        """Add a single object to the physical object store."""
        self.physical_store.add_object(obj)

    def add_objects(
        self,
        objects: Sequence[tuple[ShaFile, str | None]],
        progress: Callable[..., None] | None = None,
    ) -> "Pack | None":
        """Add a set of objects to the physical object store."""
        return self.physical_store.add_objects(objects, progress)

    def get_commit_graph(self) -> "CommitGraph | None":
        """Return a commit graph, if valid under replacements.

        Commit graphs record physical parentage, so they are not valid for
        commits whose contents are replaced; conservatively disable them
        whenever replacements are in effect.
        """
        if self.replacements:
            return None
        return self.physical_store.get_commit_graph()

    def close(self) -> None:
        """Close the underlying physical store."""
        self.physical_store.close()

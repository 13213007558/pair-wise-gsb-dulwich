# replace.py -- Read-only view applying refs/replace replacements
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
# <http://www.apache.org/licenses/> for a copy of the Apache License
# and <http://www.gnu.org/licenses/> for a copy of the GNU General Public
# License, version 2.0 or later.
#

"""Read-only object store view that applies refs/replace replacements.

Git's replace refs (refs/replace/<sha>) substitute one object for another
when reading history. This module provides ReplaceRefsObjectStore, an
explicitly enabled, read-only view over a real object store: log, tree
reads and annotate see the replacement contents, while the underlying
store keeps serving the real objects by their real ids.

The view never modifies packs, refs or object hashes, and since it only
ever yields objects that really exist in the base store, push/fetch
machinery driven from the base store will not propagate replacement
objects under made-up identities.
"""

__all__ = [
    "ReplaceRefsObjectStore",
    "ReplacementCycle",
    "ReplacementError",
    "ReplacementTargetMissing",
    "ReplacementTypeMismatch",
    "read_replace_refs",
]

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from .object_store import BaseObjectStore
from .objects import ObjectID, RawObjectID, ShaFile, sha_to_hex
from .refs import LOCAL_REPLACE_PREFIX, RefsContainer

if TYPE_CHECKING:
    from .pack import Pack


def _format_chain(chain: Sequence[ObjectID]) -> str:
    return " -> ".join(sha.decode("ascii") for sha in chain)


class ReplacementError(Exception):
    """Base class for errors while resolving a replacement chain.

    The full chain of object ids followed (including the offending id as
    the last element) is available as the ``chain`` attribute.
    """

    def __init__(self, chain: Sequence[ObjectID], message: str) -> None:
        """Initialize with the full chain and an error message."""
        self.chain = list(chain)
        super().__init__(f"{message}: {_format_chain(self.chain)}")


class ReplacementTargetMissing(ReplacementError):
    """A replacement target does not exist in the underlying store."""

    def __init__(self, chain: Sequence[ObjectID]) -> None:
        """Initialize with the full chain ending in the missing id."""
        super().__init__(chain, "Replacement target is missing")


class ReplacementTypeMismatch(ReplacementError):
    """A replacement object has a different type than the object it replaces."""

    def __init__(self, chain: Sequence[ObjectID]) -> None:
        """Initialize with the full chain ending in the mismatched id."""
        super().__init__(chain, "Replacement object type does not match original")


class ReplacementCycle(ReplacementError):
    """The replacement chain loops back on itself."""

    def __init__(self, chain: Sequence[ObjectID]) -> None:
        """Initialize with the full chain ending in the repeated id."""
        super().__init__(chain, "Replacement chain contains a cycle")


def read_replace_refs(refs: RefsContainer) -> dict[ObjectID, ObjectID]:
    """Read the replacement mapping from a refs container.

    Args:
      refs: RefsContainer to read refs/replace/* from.

    Returns:
      Dict mapping replaced object id -> replacement object id.
    """
    replacements = {}
    for ref in refs.allkeys():
        if ref.startswith(LOCAL_REPLACE_PREFIX):
            replaced = ObjectID(ref[len(LOCAL_REPLACE_PREFIX) :])
            replacements[replaced] = refs[ref]
    return replacements


class ReplaceRefsObjectStore(BaseObjectStore):
    """Read-only view of an object store that applies replace refs.

    Objects are looked up by their logical (original) id; when a
    replacement is registered for that id, the contents of the
    replacement object are returned instead, while the returned object's
    id remains the original one. Use get_actual_id() to find out which
    object the contents were really read from.

    The replacement mapping is fixed at construction time. All caches
    (chain resolutions and peeled tags) are private to each view instance
    and are dropped when the view is toggled, so two views over the same
    underlying store never share results.
    """

    def __init__(
        self,
        base_store: BaseObjectStore,
        replacements: Mapping[ObjectID, ObjectID]
        | Mapping[RawObjectID, RawObjectID]
        | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize a ReplaceRefsObjectStore.

        Args:
          base_store: The real object store to read objects from.
          replacements: Mapping of replaced object id -> replacement
            object id. The mapping is copied; later changes to the
            passed-in mapping do not affect this view.
          enabled: Whether replacements are applied. When False, the
            view behaves exactly like the underlying store.
        """
        super().__init__(object_format=base_store.object_format)
        self.base_store = base_store
        self._replacements: dict[ObjectID, ObjectID] = {
            self._normalize_id(old): self._normalize_id(new)
            for old, new in (replacements or {}).items()
        }
        self._enabled = enabled
        self._resolution_cache: dict[ObjectID, list[ObjectID]] = {}

    @classmethod
    def from_refs(
        cls,
        base_store: BaseObjectStore,
        refs: RefsContainer,
        *,
        enabled: bool = True,
    ) -> "ReplaceRefsObjectStore":
        """Create a view from the refs/replace/* refs in a refs container.

        Args:
          base_store: The real object store to read objects from.
          refs: RefsContainer holding the replace refs.
          enabled: Whether replacements are applied.

        Returns:
          A ReplaceRefsObjectStore with a snapshot of the current
          replacement refs.
        """
        return cls(base_store, read_replace_refs(refs), enabled=enabled)

    @property
    def replacements(self) -> dict[ObjectID, ObjectID]:
        """The fixed replacement mapping used by this view."""
        return dict(self._replacements)

    @property
    def enabled(self) -> bool:
        """Whether this view applies replacements."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        value = bool(value)
        if value != self._enabled:
            self._enabled = value
            # Cached resolutions and peels depend on the enabled state;
            # drop them so toggling takes effect immediately.
            self._resolution_cache.clear()
            self._invalidate_peel_cache()

    def _normalize_id(self, sha: ObjectID | RawObjectID) -> ObjectID:
        if len(sha) == self.object_format.oid_length:
            return ObjectID(sha_to_hex(RawObjectID(sha)))
        return ObjectID(sha)

    def resolve_chain(self, sha: ObjectID | RawObjectID) -> list[ObjectID]:
        """Resolve the replacement chain for an object id.

        Args:
          sha: Object id to resolve (hex or raw).

        Returns:
          List of object ids, starting with the requested (logical) id
          and ending with the id of the object contents are actually
          read from. Contains a single entry when the object is not
          replaced or the view is disabled.

        Raises:
          ReplacementTargetMissing: A replacement target is not present
            in the underlying store.
          ReplacementTypeMismatch: A replacement object has a different
            type than the original object.
          ReplacementCycle: The replacement chain contains a cycle.
        """
        hexsha = self._normalize_id(sha)
        if not self._enabled:
            return [hexsha]
        cached = self._resolution_cache.get(hexsha)
        if cached is not None:
            return list(cached)
        chain = [hexsha]
        seen = {hexsha}
        expected_type: int | None = None
        current = hexsha
        while current in self._replacements:
            target = self._replacements[current]
            if target in seen:
                raise ReplacementCycle([*chain, target])
            try:
                type_num, _ = self.base_store.get_raw(target)
            except KeyError:
                raise ReplacementTargetMissing([*chain, target]) from None
            if expected_type is None:
                # May raise KeyError if the original object itself is
                # missing; that is an ordinary missing object, not a
                # replacement problem.
                expected_type = self.base_store.get_raw(chain[0])[0]
            if type_num != expected_type:
                raise ReplacementTypeMismatch([*chain, target])
            seen.add(target)
            chain.append(target)
            current = target
        self._resolution_cache[hexsha] = chain
        return list(chain)

    def get_actual_id(self, sha: ObjectID | RawObjectID) -> ObjectID:
        """Return the id of the object contents are actually read from.

        Args:
          sha: Logical object id (hex or raw).

        Returns:
          The id of the replacement target at the end of the chain, or
          the id itself when it is not replaced.
        """
        return self.resolve_chain(sha)[-1]

    def get_raw(self, name: ObjectID | RawObjectID) -> tuple[int, bytes]:
        """Obtain the raw text for an object, applying replacements.

        Args:
          name: Logical sha for the object.

        Returns:
          Tuple with numeric type and object contents.
        """
        if not self._enabled:
            return self.base_store.get_raw(name)
        return self.base_store.get_raw(self.get_actual_id(name))

    def __getitem__(self, sha: ObjectID | RawObjectID) -> ShaFile:
        """Obtain an object by its logical id, applying replacements.

        The returned object carries the requested (original) id even
        when its contents come from a replacement object, so that
        parents, bookmarks and other external references keep pointing
        at the logical history.
        """
        hexsha = self._normalize_id(sha)
        actual = self.get_actual_id(hexsha)
        if actual == hexsha:
            return self.base_store[hexsha]
        type_num, uncomp = self.base_store.get_raw(actual)
        return ShaFile.from_raw_string(
            type_num, uncomp, sha=hexsha, object_format=self.object_format
        )

    def __contains__(self, sha: ObjectID) -> bool:
        """Check whether an object can be read through this view."""
        try:
            actual = self.get_actual_id(sha)
        except (ReplacementError, KeyError):
            return False
        return actual in self.base_store

    def __iter__(self) -> Iterator[ObjectID]:
        """Iterate over the real object ids in the underlying store.

        Replacement contents live under their real ids, so no synthetic
        ids are ever yielded.
        """
        return iter(self.base_store)

    def iterobjects_subset(
        self, shas: Iterable[ObjectID], *, allow_missing: bool = False
    ) -> Iterator[ShaFile]:
        """Iterate over a subset of objects, applying replacements."""
        for sha in shas:
            try:
                yield self[sha]
            except KeyError:
                if not allow_missing:
                    raise

    @property
    def packs(self) -> "list[Pack]":
        """The packs of the underlying store (real objects only)."""
        return self.base_store.packs

    def contains_loose(self, sha: ObjectID) -> bool:
        """Check if the resolved object is present and loose."""
        return self.base_store.contains_loose(self.get_actual_id(sha))

    def contains_packed(self, sha: ObjectID | RawObjectID) -> bool:
        """Check if the resolved object is present and packed."""
        return self.base_store.contains_packed(self.get_actual_id(sha))

    def add_object(self, obj: ShaFile) -> None:
        """Raise NotImplementedError; this view is read-only."""
        raise NotImplementedError(
            "ReplaceRefsObjectStore is read-only; write to the base store"
        )

    def add_objects(
        self,
        objects: Sequence[tuple[ShaFile, str | None]],
        progress: "object | None" = None,
    ) -> "Pack | None":
        """Raise NotImplementedError; this view is read-only."""
        raise NotImplementedError(
            "ReplaceRefsObjectStore is read-only; write to the base store"
        )

# object_authz.py -- Upload-pack object authorization policies
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

"""Authorization policies for objects requested through upload-pack.

Hiding a ref from the advertisement is not sufficient to protect the objects
it points at: a client may still send ``want`` (or ``shallow``) lines naming
object ids it learned or guessed. The policies in this module decide which
object ids a given client may request, independently of which refs were
advertised.

The secure default is :class:`VisibleRefsAuthorizationPolicy`, which only
authorizes objects reachable from the refs that were actually advertised to
the client. Reachability is computed with the full object graph, so the four
git object kinds are handled explicitly:

* a *tag* authorizes its peeled target and everything reachable from it;
* a *commit* authorizes its parents, its tree and everything reachable from
  that tree;
* a *tree* authorizes its nested trees and blobs (submodule gitlinks point
  at objects in other repositories and are not traversed);
* a *blob* only authorizes itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet

from .errors import UnauthorizedObjectError
from .objects import S_ISGITLINK, Commit, ObjectID, Tag, Tree
from .pack import ObjectContainer
from .protocol import (
    CAPABILITY_ALLOW_REACHABLE_SHA1_IN_WANT,
    CAPABILITY_ALLOW_TIP_SHA1_IN_WANT,
)
from .refs import Ref

__all__ = [
    "AllowAllAuthorizationPolicy",
    "AllowReachableSha1InWantAuthorizationPolicy",
    "ObjectAuthorizationPolicy",
    "VisibleRefsAuthorizationPolicy",
    "reachable_object_ids",
]


def reachable_object_ids(
    object_store: ObjectContainer, roots: Iterable[ObjectID]
) -> set[ObjectID]:
    """Return the ids of all objects reachable from ``roots``.

    Tags, commits, trees and blobs are traversed according to the git object
    model. Object ids that are not present in the store are ignored; the
    caller is responsible for rejecting requests for missing objects.

    Args:
      object_store: Container mapping object ids to
        :class:`dulwich.objects.ShaFile` objects
      roots: Object ids to start the traversal from

    Returns:
        Set of reachable object ids (including the roots that exist)
    """
    reachable: set[ObjectID] = set()
    pending: deque[ObjectID] = deque()
    for root in roots:
        if root in reachable or root not in object_store:
            continue
        reachable.add(root)
        pending.append(root)

    while pending:
        obj_id = pending.popleft()
        obj = object_store[obj_id]
        if isinstance(obj, Commit):
            for parent in obj.parents:
                if parent not in reachable and parent in object_store:
                    reachable.add(parent)
                    pending.append(parent)
            tree_id = obj.tree
            if tree_id not in reachable and tree_id in object_store:
                reachable.add(tree_id)
                pending.append(tree_id)
        elif isinstance(obj, Tree):
            for _name, mode, entry_id in obj.iteritems():
                if S_ISGITLINK(mode):
                    # Gitlink: the referenced object lives in another
                    # repository and cannot be served from this one.
                    continue
                if entry_id not in reachable and entry_id in object_store:
                    reachable.add(entry_id)
                    pending.append(entry_id)
        elif isinstance(obj, Tag):
            _cls, target_id = obj.object
            if target_id not in reachable and target_id in object_store:
                reachable.add(target_id)
                pending.append(target_id)
        # Blobs (and any other leaf type) have no outgoing references.

    return reachable


class ObjectAuthorizationPolicy:
    """Base class for upload-pack object authorization policies.

    A policy is consulted while the client's request is being parsed, so an
    unauthorized request fails with :class:`UnauthorizedObjectError` before
    any pack data is generated.
    """

    def check_wants(
        self,
        wants: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize the object ids sent in ``want`` lines.

        Args:
          wants: Object ids requested by the client
          advertised_refs: Refs advertised to this client (ref name ->
            object id), i.e. the refs after per-user filtering
          object_store: Object store backing the repository
          client_capabilities: Capabilities negotiated with the client
          all_refs: All refs in the repository before per-user filtering,
            needed by policies that honor allow-*-sha1-in-want

        Raises:
          UnauthorizedObjectError: if any requested object may not be sent
        """
        raise NotImplementedError(self.check_wants)

    def check_shallow_request(
        self,
        shallow_ids: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize object ids sent in ``shallow``/``unshallow`` lines.

        Shallow boundaries name commits that delimit the history the client
        wants to fetch. Allowing a hidden commit here would both leak its id
        in the server's ``shallow`` response and let the client anchor a
        walk on history it cannot see.

        Args:
          shallow_ids: Commit ids named by the client in shallow lines
          advertised_refs: Refs advertised to this client
          object_store: Object store backing the repository
          client_capabilities: Capabilities negotiated with the client
          all_refs: All refs in the repository before per-user filtering

        Raises:
          UnauthorizedObjectError: if any named commit may not be used as a
            shallow boundary
        """
        raise NotImplementedError(self.check_shallow_request)


class VisibleRefsAuthorizationPolicy(ObjectAuthorizationPolicy):
    """Secure default policy: only objects reachable from advertised refs.

    Every requested object must be reachable, following tag/commit/tree
    edges, from one of the refs included in the advertisement for this
    client. Objects belonging to hidden refs (including an entire hidden
    history branch) are therefore never downloadable, even if the client
    guesses their object ids.

    The ``allow-tip-sha1-in-want`` and ``allow-reachable-sha1-in-want``
    capabilities do not widen the set: per-user filtering means "reachable"
    must stay scoped to what this client can see.
    """

    def __init__(self) -> None:
        """Initialize an empty policy with a lazily filled reachability cache."""
        self._cache_key: tuple[ObjectID, ...] | None = None
        self._reachable: set[ObjectID] = set()

    def _reachable_from_refs(
        self,
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
    ) -> set[ObjectID]:
        key = tuple(sorted(advertised_refs.values()))
        if self._cache_key != key:
            self._cache_key = key
            self._reachable = reachable_object_ids(object_store, key)
        return self._reachable

    def check_wants(
        self,
        wants: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize wants against the closure of the advertised refs."""
        reachable = self._reachable_from_refs(advertised_refs, object_store)
        for want in wants:
            if want not in object_store:
                raise UnauthorizedObjectError(
                    want, "object is not present in the repository"
                )
            if want not in reachable:
                raise UnauthorizedObjectError(
                    want,
                    "object is not reachable from any ref advertised to this client",
                )

    def check_shallow_request(
        self,
        shallow_ids: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize shallow boundaries against the advertised-ref closure."""
        reachable = self._reachable_from_refs(advertised_refs, object_store)
        for shallow_id in shallow_ids:
            if shallow_id not in object_store:
                raise UnauthorizedObjectError(
                    shallow_id, "object is not present in the repository"
                )
            if not isinstance(object_store[shallow_id], Commit):
                raise UnauthorizedObjectError(
                    shallow_id, "shallow boundary must name a commit"
                )
            if shallow_id not in reachable:
                raise UnauthorizedObjectError(
                    shallow_id,
                    "shallow boundary is not reachable from any ref "
                    "advertised to this client",
                )


class AllowReachableSha1InWantAuthorizationPolicy(VisibleRefsAuthorizationPolicy):
    """Policy honoring the standard git ``allow-*-sha1-in-want`` semantics.

    This policy is opt-in and only relevant for servers that advertise one
    of the ``allow-tip-sha1-in-want`` / ``allow-reachable-sha1-in-want``
    capabilities, i.e. repositories without per-user ref filtering. When the
    client negotiated such a capability, requests are authorized against
    *all* repository refs rather than the (filtered) advertisement:

    * ``allow-reachable-sha1-in-want`` authorizes any object reachable from
      any ref;
    * ``allow-tip-sha1-in-want`` authorizes the direct targets of any ref
      (their full closure is reachable from those tips anyway).

    Without the capability, or for shallow boundaries, the secure
    advertised-refs behavior of the base class applies.
    """

    def check_wants(
        self,
        wants: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize wants, widening to all refs when a capability was negotiated."""
        caps = client_capabilities or frozenset()
        if (
            CAPABILITY_ALLOW_REACHABLE_SHA1_IN_WANT in caps
            or CAPABILITY_ALLOW_TIP_SHA1_IN_WANT in caps
        ):
            all_visible_refs = all_refs if all_refs is not None else advertised_refs
            if CAPABILITY_ALLOW_TIP_SHA1_IN_WANT in caps:
                authorized: AbstractSet[ObjectID] = set(all_visible_refs.values())
            else:
                authorized = reachable_object_ids(
                    object_store, all_visible_refs.values()
                )
            for want in wants:
                if want not in object_store:
                    raise UnauthorizedObjectError(
                        want, "object is not present in the repository"
                    )
                if want not in authorized:
                    raise UnauthorizedObjectError(
                        want,
                        "object is not reachable from any repository ref",
                    )
            return
        super().check_wants(wants, advertised_refs, object_store, client_capabilities)


class AllowAllAuthorizationPolicy(ObjectAuthorizationPolicy):
    """Policy that authorizes every object present in the store.

    This restores the historical behavior of trusting the want lines and is
    intended for repositories that advertise all refs to every client. It
    must not be combined with per-user ref filtering.
    """

    def check_wants(
        self,
        wants: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Authorize every want that exists in the object store."""
        for want in wants:
            if want not in object_store:
                raise UnauthorizedObjectError(
                    want, "object is not present in the repository"
                )

    def check_shallow_request(
        self,
        shallow_ids: Iterable[ObjectID],
        advertised_refs: Mapping[Ref, ObjectID],
        object_store: ObjectContainer,
        client_capabilities: AbstractSet[bytes] | None = None,
        all_refs: Mapping[Ref, ObjectID] | None = None,
    ) -> None:
        """Allow any existing commit as a shallow boundary."""
        for shallow_id in shallow_ids:
            if shallow_id not in object_store:
                raise UnauthorizedObjectError(
                    shallow_id, "object is not present in the repository"
                )
            if not isinstance(object_store[shallow_id], Commit):
                raise UnauthorizedObjectError(
                    shallow_id, "shallow boundary must name a commit"
                )

# commit_graph.py -- Git commit graph file format support
# Copyright (C) 2024 Jelmer Vernooij <jelmer@jelmer.uk>
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

"""Git commit graph file format support.

Git's commit graph files store commit metadata and generation numbers
for faster graph traversal operations like merge-base computation.

The commit graph format is documented at:
https://git-scm.com/docs/gitformat-commit-graph
"""

__all__ = [
    "CHUNK_BASE_GRAPHS_LIST",
    "CHUNK_BLOOM_FILTER_DATA",
    "CHUNK_BLOOM_FILTER_INDEX",
    "CHUNK_COMMIT_DATA",
    "CHUNK_EXTRA_EDGE_LIST",
    "CHUNK_GENERATION_DATA",
    "CHUNK_GENERATION_DATA_OVERFLOW",
    "CHUNK_OID_FANOUT",
    "CHUNK_OID_LOOKUP",
    "COMMIT_GRAPH_SIGNATURE",
    "COMMIT_GRAPH_VERSION",
    "GENERATION_NUMBER_INFINITY",
    "GENERATION_NUMBER_V1_MAX",
    "GENERATION_NUMBER_V2_OVERFLOW",
    "GENERATION_NUMBER_ZERO",
    "GRAPH_EXTRA_EDGES_NEEDED",
    "GRAPH_LAST_EDGE",
    "GRAPH_PARENT_MISSING",
    "GRAPH_PARENT_NONE",
    "HASH_VERSION_SHA1",
    "HASH_VERSION_SHA256",
    "CommitGraph",
    "CommitGraphChain",
    "CommitGraphChunk",
    "CommitGraphEntry",
    "find_commit_graph_chain_file",
    "find_commit_graph_file",
    "generate_commit_graph",
    "get_reachable_commits",
    "read_commit_graph",
    "read_commit_graph_chain",
    "write_commit_graph",
]

import os
import struct
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, BinaryIO

from .file import _GitFile
from .object_format import ObjectFormat

if TYPE_CHECKING:
    from .object_store import BaseObjectStore

from .objects import Commit, ObjectID, RawObjectID, hex_to_sha, sha_to_hex

# File format constants
COMMIT_GRAPH_SIGNATURE = b"CGPH"
COMMIT_GRAPH_VERSION = 1
HASH_VERSION_SHA1 = 1
HASH_VERSION_SHA256 = 2

# Chunk IDs
CHUNK_OID_FANOUT = b"OIDF"
CHUNK_OID_LOOKUP = b"OIDL"
CHUNK_COMMIT_DATA = b"CDAT"
CHUNK_GENERATION_DATA = b"GDA2"
CHUNK_GENERATION_DATA_OVERFLOW = b"GDO2"
CHUNK_EXTRA_EDGE_LIST = b"EDGE"
CHUNK_BLOOM_FILTER_INDEX = b"BIDX"
CHUNK_BLOOM_FILTER_DATA = b"BDAT"
CHUNK_BASE_GRAPHS_LIST = b"BASE"

# Generation number constants
GENERATION_NUMBER_INFINITY = 0xFFFFFFFF
GENERATION_NUMBER_ZERO = 0
GENERATION_NUMBER_V1_MAX = 0x3FFFFFFF

# Flag in the generation data chunk indicating that the corrected committer
# date is stored in the generation data overflow chunk instead.
GENERATION_NUMBER_V2_OVERFLOW = 0x80000000

# Parent encoding constants
GRAPH_PARENT_MISSING = 0x70000000
GRAPH_PARENT_NONE = 0x70000000
GRAPH_EXTRA_EDGES_NEEDED = 0x80000000
GRAPH_LAST_EDGE = 0x80000000


class CommitGraphEntry:
    """Represents a single commit entry in the commit graph."""

    def __init__(
        self,
        commit_id: ObjectID,
        tree_id: ObjectID,
        parents: list[ObjectID],
        generation: int,
        commit_time: int,
    ) -> None:
        """Initialize CommitGraphEntry.

        Args:
          commit_id: The commit object ID
          tree_id: The tree object ID
          parents: List of parent commit IDs
          generation: Generation number
          commit_time: Commit timestamp
        """
        self.commit_id = commit_id
        self.tree_id = tree_id
        self.parents = parents
        self.generation = generation
        self.commit_time = commit_time

    def __repr__(self) -> str:
        """Return string representation of CommitGraphEntry."""
        return (
            f"CommitGraphEntry(commit_id={self.commit_id!r}, "
            f"tree_id={self.tree_id!r}, parents={self.parents!r}, "
            f"generation={self.generation}, commit_time={self.commit_time})"
        )


class CommitGraphChunk:
    """Represents a chunk in the commit graph file."""

    def __init__(self, chunk_id: bytes, data: bytes) -> None:
        """Initialize CommitGraphChunk.

        Args:
          chunk_id: Chunk identifier
          data: Chunk data
        """
        self.chunk_id = chunk_id
        self.data = data

    def __repr__(self) -> str:
        """Return string representation of CommitGraphChunk."""
        return f"CommitGraphChunk(chunk_id={self.chunk_id!r}, size={len(self.data)})"


class CommitGraph:
    """Git commit graph file reader/writer."""

    def __init__(self, *, object_format: ObjectFormat | None = None) -> None:
        """Initialize CommitGraph.

        Args:
          object_format: Object format to use (defaults to SHA1)
        """
        import warnings

        from .object_format import DEFAULT_OBJECT_FORMAT, SHA256

        if object_format is None:
            warnings.warn(
                "CommitGraph() should be called with object_format parameter",
                DeprecationWarning,
                stacklevel=2,
            )
            object_format = DEFAULT_OBJECT_FORMAT
        self.object_format = object_format
        self.hash_version = (
            HASH_VERSION_SHA256 if object_format == SHA256 else HASH_VERSION_SHA1
        )
        self.chunks: dict[bytes, CommitGraphChunk] = {}
        self.entries: list[CommitGraphEntry] = []
        self._oid_to_index: dict[RawObjectID, int] = {}
        # The layer directly below this one in a commit graph chain, if any.
        self.base_graph: CommitGraph | None = None
        # Total number of commits in the layers below this one. Positions
        # below this offset refer to the base chain; positions at or above
        # it refer to this layer's own commits.
        self._num_commits_in_base = 0
        # Raw parent positions per entry, resolved against the (chain of)
        # graph(s) by _resolve_parents().
        self._parent_positions: list[list[int]] = []

    @classmethod
    def from_file(cls, f: BinaryIO) -> "CommitGraph":
        """Read commit graph from file."""
        return cls._read_from_file(f)

    @classmethod
    def _read_from_file(
        cls, f: BinaryIO, *, resolve_parents: bool = True
    ) -> "CommitGraph":
        """Read commit graph data from file.

        Args:
          f: File-like object to read from
          resolve_parents: Whether to resolve parent positions to commit IDs.
            Layers of a commit graph chain are read with this disabled so
            that cross-layer positions can be resolved once the whole chain
            has been assembled.
        """
        # Read header
        signature = f.read(4)
        if signature != COMMIT_GRAPH_SIGNATURE:
            raise ValueError(f"Invalid commit graph signature: {signature!r}")

        version = struct.unpack(">B", f.read(1))[0]
        if version != COMMIT_GRAPH_VERSION:
            raise ValueError(f"Unsupported commit graph version: {version}")

        hash_version = struct.unpack(">B", f.read(1))[0]

        # Set object_format based on hash_version from file
        from .object_format import SHA1, SHA256

        if hash_version == HASH_VERSION_SHA1:
            object_format = SHA1
        elif hash_version == HASH_VERSION_SHA256:
            object_format = SHA256
        else:
            raise ValueError(f"Unsupported hash version: {hash_version}")

        # Create instance with correct object_format
        graph = cls(object_format=object_format)
        graph.hash_version = hash_version

        num_chunks = struct.unpack(">B", f.read(1))[0]
        struct.unpack(">B", f.read(1))[0]

        # Read table of contents
        toc_entries = []
        for _ in range(num_chunks + 1):  # +1 for terminating entry
            chunk_id = f.read(4)
            offset = struct.unpack(">Q", f.read(8))[0]
            toc_entries.append((chunk_id, offset))

        # Read chunks
        # Offsets in TOC are absolute from start of file
        for i in range(num_chunks):
            chunk_id, offset = toc_entries[i]
            next_offset = toc_entries[i + 1][1]
            chunk_size = next_offset - offset

            f.seek(offset)
            chunk_data = f.read(chunk_size)
            graph.chunks[chunk_id] = CommitGraphChunk(chunk_id, chunk_data)

        # Parse chunks
        graph._parse_chunks()
        if resolve_parents:
            graph._resolve_parents()

        return graph

    def _parse_chunks(self) -> None:
        """Parse chunk data into entries."""
        if CHUNK_OID_LOOKUP not in self.chunks:
            raise ValueError("Missing required OID lookup chunk")
        if CHUNK_COMMIT_DATA not in self.chunks:
            raise ValueError("Missing required commit data chunk")

        # Parse OID lookup chunk
        oid_lookup_data = self.chunks[CHUNK_OID_LOOKUP].data
        num_commits = len(oid_lookup_data) // self.object_format.oid_length

        oids = []
        for i in range(num_commits):
            start = i * self.object_format.oid_length
            end = start + self.object_format.oid_length
            oid = RawObjectID(oid_lookup_data[start:end])
            oids.append(oid)
            self._oid_to_index[oid] = i

        # Parse commit data chunk
        commit_data = self.chunks[CHUNK_COMMIT_DATA].data
        expected_size = num_commits * (self.object_format.oid_length + 16)
        if len(commit_data) != expected_size:
            raise ValueError(
                f"Invalid commit data chunk size: {len(commit_data)}, expected {expected_size}"
            )

        self.entries = []
        for i in range(num_commits):
            offset = i * (self.object_format.oid_length + 16)

            # Tree OID
            tree_id = commit_data[offset : offset + self.object_format.oid_length]
            offset += self.object_format.oid_length

            # Parent positions (2 x 4 bytes)
            parent1_pos, parent2_pos = struct.unpack(
                ">LL", commit_data[offset : offset + 8]
            )
            offset += 8

            # Generation number and commit time (2 x 4 bytes)
            gen_and_time = struct.unpack(">LL", commit_data[offset : offset + 8])
            generation = gen_and_time[0] >> 2  # Upper 30 bits
            commit_time = gen_and_time[1] | (
                (gen_and_time[0] & 0x3) << 32
            )  # 34 bits total

            # Collect raw parent positions. Positions may refer to commits
            # in lower layers of a commit graph chain, so they are resolved
            # to commit IDs separately by _resolve_parents().
            parent_positions = []
            if parent1_pos < GRAPH_PARENT_MISSING:
                parent_positions.append(parent1_pos)

            if parent2_pos < GRAPH_PARENT_MISSING:
                parent_positions.append(parent2_pos)
            elif parent2_pos >= GRAPH_EXTRA_EDGES_NEEDED:
                # Handle extra edges (3+ parents)
                edge_index = parent2_pos & ~GRAPH_EXTRA_EDGES_NEEDED
                parent_positions.extend(self._parse_extra_edges(edge_index))

            entry = CommitGraphEntry(
                commit_id=sha_to_hex(oids[i]),
                tree_id=sha_to_hex(RawObjectID(tree_id)),
                parents=[],
                generation=generation,
                commit_time=commit_time,
            )
            self.entries.append(entry)
            self._parent_positions.append(parent_positions)

        self._parse_generation_data()

    def _parse_extra_edges(self, index: int) -> list[int]:
        """Parse extra parent edges for commits with 3+ parents.

        Args:
          index: Position in the extra edge list, as stored in the commit
            data chunk. The list holds 4-byte values, so this is scaled by 4
            to reach the corresponding byte.

        Returns: List of raw parent positions. Positions may refer to
          commits in lower layers of a commit graph chain and are resolved
          to commit IDs by _resolve_parents().
        """
        if CHUNK_EXTRA_EDGE_LIST not in self.chunks:
            return []

        edge_data = self.chunks[CHUNK_EXTRA_EDGE_LIST].data
        positions = []

        offset = index * 4
        while offset + 4 <= len(edge_data):
            parent_pos = struct.unpack(">L", edge_data[offset : offset + 4])[0]
            offset += 4

            positions.append(parent_pos & ~GRAPH_LAST_EDGE)
            if parent_pos & GRAPH_LAST_EDGE:
                break

        return positions

    def _parse_generation_data(self) -> None:
        """Parse the generation data (v2) chunks, if present.

        The generation data chunk stores, per commit, the corrected
        committer date offset: the difference between the corrected
        committer date and the committer date from the commit data chunk.
        Values with the overflow flag set index into the generation data
        overflow chunk, which holds 64-bit corrected committer dates.
        When present, the resulting corrected committer dates replace the
        v1 generation numbers from the commit data chunk.
        """
        if CHUNK_GENERATION_DATA not in self.chunks:
            return

        gen_data = self.chunks[CHUNK_GENERATION_DATA].data
        if len(gen_data) < len(self.entries) * 4:
            raise ValueError(
                f"Invalid generation data chunk size: {len(gen_data)}, "
                f"expected at least {len(self.entries) * 4}"
            )

        overflow_chunk = self.chunks.get(CHUNK_GENERATION_DATA_OVERFLOW)
        overflow_data = overflow_chunk.data if overflow_chunk is not None else b""

        for i, entry in enumerate(self.entries):
            gen_v2 = struct.unpack(">L", gen_data[i * 4 : i * 4 + 4])[0]
            if gen_v2 & GENERATION_NUMBER_V2_OVERFLOW:
                overflow_index = gen_v2 & ~GENERATION_NUMBER_V2_OVERFLOW
                overflow_offset = overflow_index * 8
                if overflow_offset + 8 > len(overflow_data):
                    raise ValueError(
                        f"Generation data overflow index out of range: "
                        f"{overflow_index}"
                    )
                entry.generation = struct.unpack(
                    ">Q", overflow_data[overflow_offset : overflow_offset + 8]
                )[0]
            else:
                # Corrected committer date = committer date + offset
                entry.generation = entry.commit_time + gen_v2

    def _resolve_position(self, position: int) -> ObjectID:
        """Resolve a commit graph position to a commit ID.

        Positions in a split (layered) graph refer to the concatenated
        position space of the chain, in which the commits of the lower
        layers come first: a position below the number of commits in the
        base layers refers to the base chain, and any higher position
        refers to this layer's own commits.

        Args:
          position: Commit graph position to resolve

        Returns: The commit ID at that position

        Raises:
          ValueError: If the position is out of range for the whole chain
        """
        graph: CommitGraph | None = self
        pos = position
        while graph is not None:
            if pos < graph._num_commits_in_base:
                graph = graph.base_graph
                continue
            own_pos = pos - graph._num_commits_in_base
            if own_pos < len(graph.entries):
                return graph.entries[own_pos].commit_id
            break
        raise ValueError(f"Invalid parent position: {position}")

    def _resolve_parents(self) -> None:
        """Resolve raw parent positions to commit IDs."""
        for entry, positions in zip(self.entries, self._parent_positions):
            entry.parents = [self._resolve_position(pos) for pos in positions]

    def get_entry_by_oid(self, oid: ObjectID) -> CommitGraphEntry | None:
        """Get commit graph entry by commit OID."""
        # Convert hex ObjectID to binary if needed for lookup
        if isinstance(oid, bytes) and len(oid) == self.object_format.hex_length:
            # Input is hex ObjectID, convert to binary for internal lookup
            lookup_oid: RawObjectID = hex_to_sha(oid)
        else:
            # Input is already binary
            lookup_oid = RawObjectID(oid)
        index = self._oid_to_index.get(lookup_oid)
        if index is not None:
            return self.entries[index]
        return None

    def get_generation_number(self, oid: ObjectID) -> int | None:
        """Get generation number for a commit."""
        entry = self.get_entry_by_oid(oid)
        return entry.generation if entry else None

    def get_parents(self, oid: ObjectID) -> list[ObjectID] | None:
        """Get parent commit IDs for a commit."""
        entry = self.get_entry_by_oid(oid)
        return entry.parents if entry else None

    def write_to_file(self, f: BinaryIO | _GitFile) -> None:
        """Write commit graph to file."""
        if not self.entries:
            raise ValueError("Cannot write empty commit graph")

        # Sort entries by commit ID for consistent output
        sorted_entries = sorted(self.entries, key=lambda e: e.commit_id)

        # Build OID lookup chunk
        oid_lookup_data = b""
        for entry in sorted_entries:
            oid_lookup_data += hex_to_sha(entry.commit_id)

        # Build commit data chunk
        commit_data = b""
        # Create OID to index mapping for parent lookups
        oid_to_index = {entry.commit_id: i for i, entry in enumerate(sorted_entries)}

        for entry in sorted_entries:
            # Tree OID (20 bytes)
            commit_data += hex_to_sha(entry.tree_id)

            # Parent positions (2 x 4 bytes)
            if len(entry.parents) == 0:
                parent1_pos = GRAPH_PARENT_MISSING
                parent2_pos = GRAPH_PARENT_MISSING
            elif len(entry.parents) == 1:
                parent1_pos = oid_to_index.get(entry.parents[0], GRAPH_PARENT_MISSING)
                parent2_pos = GRAPH_PARENT_MISSING
            elif len(entry.parents) == 2:
                parent1_pos = oid_to_index.get(entry.parents[0], GRAPH_PARENT_MISSING)
                parent2_pos = oid_to_index.get(entry.parents[1], GRAPH_PARENT_MISSING)
            else:
                # More than 2 parents - would need extra edge list chunk
                # For now, just store first two parents
                parent1_pos = oid_to_index.get(entry.parents[0], GRAPH_PARENT_MISSING)
                parent2_pos = oid_to_index.get(entry.parents[1], GRAPH_PARENT_MISSING)

            commit_data += struct.pack(">LL", parent1_pos, parent2_pos)

            # Generation and commit time (2 x 4 bytes)
            gen_and_time = (entry.generation << 2) | (entry.commit_time >> 32)
            commit_time_lower = entry.commit_time & 0xFFFFFFFF
            commit_data += struct.pack(">LL", gen_and_time, commit_time_lower)

        # Build fanout table
        fanout_data = b""
        fanout_counts = [0] * 256
        for i, entry in enumerate(sorted_entries):
            commit_oid_bytes = hex_to_sha(entry.commit_id)
            fanout_counts[commit_oid_bytes[0]] = i + 1

        # Fill in gaps - each fanout entry should be cumulative
        for i in range(1, 256):
            if fanout_counts[i] == 0:
                fanout_counts[i] = fanout_counts[i - 1]

        for count in fanout_counts:
            fanout_data += struct.pack(">L", count)

        # Calculate chunk offsets
        header_size = (
            8  # signature + version + hash_version + num_chunks + base_graph_count
        )
        toc_size = 4 * 12  # 4 entries (3 chunks + terminator) * 12 bytes each

        chunk1_offset = header_size + toc_size  # OID Fanout
        chunk2_offset = chunk1_offset + len(fanout_data)  # OID Lookup
        chunk3_offset = chunk2_offset + len(oid_lookup_data)  # Commit Data
        terminator_offset = chunk3_offset + len(commit_data)

        # Write header
        f.write(COMMIT_GRAPH_SIGNATURE)
        f.write(struct.pack(">B", COMMIT_GRAPH_VERSION))
        f.write(struct.pack(">B", self.hash_version))
        f.write(struct.pack(">B", 3))  # 3 chunks
        f.write(struct.pack(">B", 0))  # 0 base graphs

        # Write table of contents
        f.write(CHUNK_OID_FANOUT + struct.pack(">Q", chunk1_offset))
        f.write(CHUNK_OID_LOOKUP + struct.pack(">Q", chunk2_offset))
        f.write(CHUNK_COMMIT_DATA + struct.pack(">Q", chunk3_offset))
        f.write(b"\x00\x00\x00\x00" + struct.pack(">Q", terminator_offset))

        # Write chunks
        f.write(fanout_data)
        f.write(oid_lookup_data)
        f.write(commit_data)

    def __len__(self) -> int:
        """Return number of commits in the graph."""
        return len(self.entries)

    def __iter__(self) -> Iterator["CommitGraphEntry"]:
        """Iterate over commit graph entries."""
        return iter(self.entries)


class CommitGraphChain:
    """A chain of layered (split) commit graph files.

    Git maintenance (e.g. git commit-graph write --split) stores the commit
    graph as a chain of incremental layers: a commit-graph-chain file lists
    the hash of each layer, from base to tip, and each layer lives in a
    graph-{hash}.graph file next to it. Parent positions in a layer may
    refer to commits in lower layers; see CommitGraph._resolve_position().
    """

    def __init__(self, layers: list[CommitGraph]) -> None:
        """Initialize CommitGraphChain.

        Args:
          layers: Layers of the chain, ordered from base to tip
        """
        if not layers:
            raise ValueError("Commit graph chain requires at least one layer")
        self.layers = layers
        self.object_format = layers[0].object_format
        # Map each commit to the layer that contains it. Layers are
        # disjoint by construction; if a commit somehow appears in several
        # layers, the topmost layer wins.
        self._oid_to_layer: dict[RawObjectID, CommitGraph] = {}
        for layer in layers:
            for oid in layer._oid_to_index:
                self._oid_to_layer[oid] = layer

    @classmethod
    def from_file(cls, path: str | bytes) -> "CommitGraphChain":
        """Read a commit graph chain from a commit-graph-chain file.

        Args:
          path: Path to the commit-graph-chain file

        Returns: CommitGraphChain with all layers assembled

        Raises:
          OSError: If the chain file or a layer file cannot be read (e.g.
            the chain was replaced by a concurrent maintenance process)
          ValueError: If the chain or one of its layers is invalid
        """
        if isinstance(path, str):
            path = os.fsencode(path)

        with open(path, "rb") as f:
            contents = f.read()

        layer_hashes = [line.strip() for line in contents.splitlines() if line.strip()]
        if not layer_hashes:
            raise ValueError("Commit graph chain file lists no layers")

        chain_dir = os.path.dirname(path)
        layers = []
        for i, layer_hash in enumerate(layer_hashes):
            layer_path = os.path.join(chain_dir, b"graph-" + layer_hash + b".graph")
            with open(layer_path, "rb") as layer_file:
                layer = CommitGraph._read_from_file(layer_file, resolve_parents=False)
            cls._validate_layer(layer, layer_hashes[:i])
            layers.append(layer)

        for i, layer in enumerate(layers):
            layer.base_graph = layers[i - 1] if i > 0 else None
            layer._num_commits_in_base = sum(len(lower) for lower in layers[:i])
            layer._resolve_parents()

        return cls(layers)

    @staticmethod
    def _validate_layer(layer: CommitGraph, expected_base_hashes: list[bytes]) -> None:
        """Validate a layer against the hashes listed in the chain file.

        Args:
          layer: The commit graph layer to validate
          expected_base_hashes: Hex hashes of the layers below this one, as
            listed in the commit-graph-chain file

        Raises:
          ValueError: If the layer's base graphs list does not match the
            chain file
        """
        base_chunk = layer.chunks.get(CHUNK_BASE_GRAPHS_LIST)
        oid_length = layer.object_format.oid_length
        if base_chunk is None:
            actual_base_hashes: list[bytes] = []
        else:
            data = base_chunk.data
            if len(data) % oid_length:
                raise ValueError(f"Invalid base graphs chunk size: {len(data)}")
            actual_base_hashes = [
                sha_to_hex(RawObjectID(data[offset : offset + oid_length]))
                for offset in range(0, len(data), oid_length)
            ]
        if actual_base_hashes != expected_base_hashes:
            raise ValueError(
                "Commit graph layer does not match the layers listed in the "
                "chain file"
            )

    def _get_layer(self, oid: ObjectID) -> CommitGraph | None:
        """Find the layer containing a commit, if any."""
        if isinstance(oid, bytes) and len(oid) == self.object_format.hex_length:
            lookup_oid: RawObjectID = hex_to_sha(oid)
        else:
            lookup_oid = RawObjectID(oid)
        return self._oid_to_layer.get(lookup_oid)

    def get_entry_by_oid(self, oid: ObjectID) -> CommitGraphEntry | None:
        """Get commit graph entry by commit OID."""
        layer = self._get_layer(oid)
        if layer is None:
            return None
        return layer.get_entry_by_oid(oid)

    def get_generation_number(self, oid: ObjectID) -> int | None:
        """Get generation number for a commit."""
        entry = self.get_entry_by_oid(oid)
        return entry.generation if entry else None

    def get_parents(self, oid: ObjectID) -> list[ObjectID] | None:
        """Get parent commit IDs for a commit."""
        entry = self.get_entry_by_oid(oid)
        return entry.parents if entry else None

    def __len__(self) -> int:
        """Return number of commits in the chain."""
        return sum(len(layer) for layer in self.layers)

    def __iter__(self) -> Iterator["CommitGraphEntry"]:
        """Iterate over commit graph entries in all layers."""
        for layer in self.layers:
            yield from layer


def read_commit_graph(path: str | bytes) -> CommitGraph | None:
    """Read commit graph from file path."""
    if isinstance(path, str):
        path = path.encode()

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return CommitGraph.from_file(f)


def read_commit_graph_chain(path: str | bytes) -> CommitGraphChain | None:
    """Read a commit graph chain from a commit-graph-chain file path.

    Args:
      path: Path to the commit-graph-chain file

    Returns: CommitGraphChain if the chain file exists, None otherwise

    Raises:
      OSError: If a layer file cannot be read
      ValueError: If the chain or one of its layers is invalid
    """
    if isinstance(path, str):
        path = path.encode()

    if not os.path.exists(path):
        return None

    return CommitGraphChain.from_file(path)


def find_commit_graph_file(git_dir: str | bytes) -> bytes | None:
    """Find commit graph file in a Git repository."""
    if isinstance(git_dir, str):
        git_dir = git_dir.encode()

    # Standard location: .git/objects/info/commit-graph
    commit_graph_path = os.path.join(git_dir, b"objects", b"info", b"commit-graph")
    if os.path.exists(commit_graph_path):
        return commit_graph_path

    # Chain files in .git/objects/info/commit-graphs/
    commit_graphs_dir = os.path.join(git_dir, b"objects", b"info", b"commit-graphs")
    if os.path.exists(commit_graphs_dir):
        # Look for graph-{hash}.graph files
        for filename in os.listdir(commit_graphs_dir):
            if filename.startswith(b"graph-") and filename.endswith(b".graph"):
                return os.path.join(commit_graphs_dir, filename)

    return None


def find_commit_graph_chain_file(git_dir: str | bytes) -> bytes | None:
    """Find the commit graph chain file in a Git repository.

    Args:
      git_dir: Path to the git directory (or bare repository)

    Returns: Path to the commit-graph-chain file if it exists, None otherwise
    """
    if isinstance(git_dir, str):
        git_dir = git_dir.encode()

    chain_path = os.path.join(
        git_dir, b"objects", b"info", b"commit-graphs", b"commit-graph-chain"
    )
    if os.path.exists(chain_path):
        return chain_path
    return None


def generate_commit_graph(
    object_store: "BaseObjectStore", commit_ids: Sequence[ObjectID]
) -> CommitGraph:
    """Generate a commit graph from a set of commits.

    Args:
        object_store: Object store to retrieve commits from
        commit_ids: List of commit IDs to include in the graph

    Returns:
        CommitGraph object containing the specified commits
    """
    graph = CommitGraph(object_format=object_store.object_format)

    if not commit_ids:
        return graph

    # Ensure all commit_ids are in the correct format for object store access
    hex_length = object_store.object_format.hex_length
    oid_length = object_store.object_format.oid_length
    normalized_commit_ids = []
    for commit_id in commit_ids:
        if isinstance(commit_id, bytes) and len(commit_id) == hex_length:
            # Already hex ObjectID
            normalized_commit_ids.append(commit_id)
        elif isinstance(commit_id, bytes) and len(commit_id) == oid_length:
            # Binary SHA, convert to hex ObjectID
            normalized_commit_ids.append(sha_to_hex(RawObjectID(commit_id)))
        else:
            # Assume it's already correct format
            normalized_commit_ids.append(ObjectID(commit_id))

    # Build a map of all commits and their metadata
    commit_map: dict[ObjectID, Commit] = {}
    for commit_id in normalized_commit_ids:
        try:
            commit_obj = object_store[commit_id]
            if commit_obj.type_name != b"commit":
                continue
            assert isinstance(commit_obj, Commit)
            commit_map[commit_id] = commit_obj
        except KeyError:
            # Commit not found, skip
            continue

    # Calculate generation numbers using topological sort
    generation_map: dict[bytes, int] = {}

    def calculate_generation(commit_id: ObjectID) -> int:
        if commit_id in generation_map:
            return generation_map[commit_id]

        if commit_id not in commit_map:
            # Unknown commit, assume generation 0
            generation_map[commit_id] = 0
            return 0

        commit_obj = commit_map[commit_id]
        if not commit_obj.parents:
            # Root commit
            generation_map[commit_id] = 1
            return 1

        # Calculate based on parents
        max_parent_gen = 0
        for parent_id in commit_obj.parents:
            parent_gen = calculate_generation(parent_id)
            max_parent_gen = max(max_parent_gen, parent_gen)

        generation = max_parent_gen + 1
        generation_map[commit_id] = generation
        return generation

    # Calculate generation numbers for all commits
    for commit_id in commit_map:
        calculate_generation(commit_id)

    # Build commit graph entries
    for commit_id, commit_obj in commit_map.items():
        # commit_id is already hex ObjectID from normalized_commit_ids
        commit_hex: ObjectID = commit_id

        # commit_obj.tree and commit_obj.parents are already ObjectIDs
        tree_hex = commit_obj.tree
        parents_hex: list[ObjectID] = commit_obj.parents

        entry = CommitGraphEntry(
            commit_id=commit_hex,
            tree_id=tree_hex,
            parents=parents_hex,
            generation=generation_map[commit_id],
            commit_time=commit_obj.commit_time,
        )
        graph.entries.append(entry)

    # Build the OID to index mapping for lookups
    graph._oid_to_index = {}
    for i, entry in enumerate(graph.entries):
        # Convert hex ObjectID to binary RawObjectID for consistent lookup
        graph._oid_to_index[hex_to_sha(entry.commit_id)] = i

    return graph


def write_commit_graph(
    git_dir: str | bytes,
    object_store: "BaseObjectStore",
    commit_ids: Sequence[ObjectID],
) -> None:
    """Write a commit graph file for the given commits.

    Args:
        git_dir: Git directory path
        object_store: Object store to retrieve commits from
        commit_ids: List of commit IDs to include in the graph
    """
    if isinstance(git_dir, str):
        git_dir = git_dir.encode()

    # Generate the commit graph
    graph = generate_commit_graph(object_store, commit_ids)

    if not graph.entries:
        return  # Nothing to write

    # Ensure the objects/info directory exists
    info_dir = os.path.join(git_dir, b"objects", b"info")
    os.makedirs(info_dir, exist_ok=True)

    # Write using GitFile for atomic operation
    from .file import GitFile

    graph_path = os.path.join(info_dir, b"commit-graph")
    with GitFile(graph_path, "wb") as f:
        graph.write_to_file(f)


def get_reachable_commits(
    object_store: "BaseObjectStore", start_commits: Sequence[ObjectID]
) -> list[ObjectID]:
    """Get all commits reachable from the given starting commits.

    Args:
        object_store: Object store to retrieve commits from
        start_commits: List of starting commit IDs

    Returns:
        List of all reachable commit IDs (including the starting commits)
    """
    visited: set[ObjectID] = set()
    reachable: list[ObjectID] = []
    stack: list[ObjectID] = []

    hex_length = object_store.object_format.hex_length
    oid_length = object_store.object_format.oid_length

    # Normalize commit IDs for object store access and tracking
    for commit_id in start_commits:
        if isinstance(commit_id, bytes) and len(commit_id) == hex_length:
            # Hex ObjectID - use directly for object store access
            if commit_id not in visited:
                stack.append(commit_id)
        elif isinstance(commit_id, bytes) and len(commit_id) == oid_length:
            # Binary SHA, convert to hex ObjectID for object store access
            hex_id = sha_to_hex(RawObjectID(commit_id))
            if hex_id not in visited:
                stack.append(hex_id)
        else:
            # Assume it's already correct format
            oid = ObjectID(commit_id)
            if oid not in visited:
                stack.append(oid)

    while stack:
        commit_id = stack.pop()
        if commit_id in visited:
            continue

        visited.add(commit_id)

        try:
            commit_obj = object_store[commit_id]
            if not isinstance(commit_obj, Commit):
                continue

            # Add to reachable list (commit_id is already hex ObjectID)
            reachable.append(commit_id)

            # Add parents to stack
            for parent_id in commit_obj.parents:
                if parent_id not in visited:
                    stack.append(parent_id)
        except KeyError:
            # Commit not found, skip
            continue

    return reachable

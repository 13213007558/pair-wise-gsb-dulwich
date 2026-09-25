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
    "GENERATION_DATA_OVERFLOW_FLAG",
    "GENERATION_NUMBER_INFINITY",
    "GENERATION_NUMBER_V1_MAX",
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
    "find_commit_graph_file",
    "generate_commit_graph",
    "get_reachable_commits",
    "read_commit_graph",
    "read_commit_graph_chain",
    "write_commit_graph",
]

import hashlib
import io
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
GENERATION_DATA_OVERFLOW_FLAG = 0x80000000
GENERATION_NUMBER_INFINITY = 0xFFFFFFFF
GENERATION_NUMBER_ZERO = 0
GENERATION_NUMBER_V1_MAX = 0x3FFFFFFF

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
        generation_v2: int | None = None,
    ) -> None:
        """Initialize CommitGraphEntry.

        Args:
          commit_id: The commit object ID
          tree_id: The tree object ID
          parents: List of parent commit IDs
          generation: Generation number
          commit_time: Commit timestamp
          generation_v2: Corrected commit date (generation data v2), if
            the graph contained a generation data chunk
        """
        self.commit_id = commit_id
        self.tree_id = tree_id
        self.parents = parents
        self.generation = generation
        self.commit_time = commit_time
        self.generation_v2 = generation_v2

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
        self.num_base_graphs = 0

    @classmethod
    def from_file(cls, f: BinaryIO) -> "CommitGraph":
        """Read commit graph from file."""
        return cls._read_from_file(f)

    @classmethod
    def _read_from_file(
        cls, f: BinaryIO, base_oids: Sequence[RawObjectID] = ()
    ) -> "CommitGraph":
        """Read commit graph data from file.

        Args:
          f: File-like object to read the graph from
          base_oids: Commit IDs of the base layers this graph builds on, in
            chain order. Parent positions in an incremental (split) graph
            index into the concatenation of the base layers' commit IDs and
            this graph's own commit IDs.
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
        graph.num_base_graphs = struct.unpack(">B", f.read(1))[0]

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
        graph._parse_chunks(base_oids)

        return graph

    def _parse_chunks(self, base_oids: Sequence[RawObjectID] = ()) -> None:
        """Parse chunk data into entries.

        Args:
          base_oids: Commit IDs of the base layers of a split commit graph,
            in chain order. Parent positions stored in this graph index into
            the concatenation of base_oids and this graph's own commit IDs.
        """
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

        # Parent positions in an incremental graph index into the
        # concatenation of the base layers' commits and this graph's commits.
        if base_oids:
            position_oids: Sequence[RawObjectID] = list(base_oids) + oids
        else:
            position_oids = oids

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

            # Parse parents
            parents = []
            if parent1_pos < GRAPH_PARENT_MISSING:
                if parent1_pos >= len(position_oids):
                    raise ValueError(f"Invalid parent1 position: {parent1_pos}")
                parents.append(position_oids[parent1_pos])

            if parent2_pos < GRAPH_PARENT_MISSING:
                if parent2_pos >= len(position_oids):
                    raise ValueError(f"Invalid parent2 position: {parent2_pos}")
                parents.append(position_oids[parent2_pos])
            elif parent2_pos >= GRAPH_EXTRA_EDGES_NEEDED:
                # Handle extra edges (3+ parents)
                edge_index = parent2_pos & ~GRAPH_EXTRA_EDGES_NEEDED
                parents.extend(self._parse_extra_edges(edge_index, position_oids))

            entry = CommitGraphEntry(
                commit_id=sha_to_hex(oids[i]),
                tree_id=sha_to_hex(RawObjectID(tree_id)),
                parents=[sha_to_hex(p) for p in parents],
                generation=generation,
                commit_time=commit_time,
            )
            self.entries.append(entry)

        # Parse generation data chunk (corrected commit dates) if present
        if CHUNK_GENERATION_DATA in self.chunks:
            self._parse_generation_data(num_commits)

    def _parse_generation_data(self, num_commits: int) -> None:
        """Parse the GDA2/GDO2 chunks into per-entry corrected commit dates."""
        gen_data = self.chunks[CHUNK_GENERATION_DATA].data
        if len(gen_data) != num_commits * 4:
            raise ValueError(
                f"Invalid generation data chunk size: {len(gen_data)}, "
                f"expected {num_commits * 4}"
            )

        overflow_data = b""
        if CHUNK_GENERATION_DATA_OVERFLOW in self.chunks:
            overflow_data = self.chunks[CHUNK_GENERATION_DATA_OVERFLOW].data

        for i, entry in enumerate(self.entries):
            offset = struct.unpack(">L", gen_data[i * 4 : i * 4 + 4])[0]
            if offset & GENERATION_DATA_OVERFLOW_FLAG:
                overflow_index = offset & ~GENERATION_DATA_OVERFLOW_FLAG
                overflow_offset = overflow_index * 8
                if overflow_offset + 8 > len(overflow_data):
                    raise ValueError(
                        f"Invalid generation data overflow index: {overflow_index}"
                    )
                offset = struct.unpack(
                    ">Q", overflow_data[overflow_offset : overflow_offset + 8]
                )[0]
            entry.generation_v2 = entry.commit_time + offset

    def _parse_extra_edges(
        self, index: int, oids: Sequence[RawObjectID]
    ) -> list[RawObjectID]:
        """Parse extra parent edges for commits with 3+ parents.

        Args:
          index: Position in the extra edge list, as stored in the commit
            data chunk. The list holds 4-byte values, so this is scaled by 4
            to reach the corresponding byte.
          oids: Commit ids, indexed by commit graph position.
        """
        if CHUNK_EXTRA_EDGE_LIST not in self.chunks:
            return []

        edge_data = self.chunks[CHUNK_EXTRA_EDGE_LIST].data
        parents = []

        offset = index * 4
        while offset + 4 <= len(edge_data):
            parent_pos = struct.unpack(">L", edge_data[offset : offset + 4])[0]
            offset += 4

            if parent_pos & GRAPH_LAST_EDGE:
                parent_pos &= ~GRAPH_LAST_EDGE
                if parent_pos < len(oids):
                    parents.append(oids[parent_pos])
                break
            else:
                if parent_pos < len(oids):
                    parents.append(oids[parent_pos])

        return parents

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


def read_commit_graph(path: str | bytes) -> CommitGraph | None:
    """Read commit graph from file path."""
    if isinstance(path, str):
        path = path.encode()

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return CommitGraph.from_file(f)


def _verify_graph_checksum(raw: bytes) -> None:
    """Verify the trailing checksum of a serialized commit graph file.

    Args:
      raw: Full contents of the commit graph file, including the trailing
        checksum.

    Raises:
      ValueError: If the file is truncated or the checksum does not match.
    """
    if len(raw) < 8:
        raise ValueError("Commit graph file too short")
    hash_version = raw[5]
    if hash_version == HASH_VERSION_SHA1:
        hash_name = "sha1"
    elif hash_version == HASH_VERSION_SHA256:
        hash_name = "sha256"
    else:
        raise ValueError(f"Unsupported hash version: {hash_version}")
    digest_size = hashlib.new(hash_name).digest_size
    if len(raw) < digest_size:
        raise ValueError("Commit graph file too short for checksum")
    expected = hashlib.new(hash_name, raw[:-digest_size]).digest()
    if expected != raw[-digest_size:]:
        raise ValueError("Commit graph checksum mismatch")


class CommitGraphChain(CommitGraph):
    """A chain of incremental (split) commit graph files.

    Git's split commit-graph feature stores commit metadata in a stack of
    graph files listed in objects/info/commit-graphs/commit-graph-chain.
    Each incremental layer only contains new commits; parent positions in a
    layer index into the concatenation of all commits from the base layers
    up to and including that layer.

    This class assembles the layers listed in a chain file and presents the
    same interface as a single CommitGraph.
    """

    def __init__(self, layers: list[CommitGraph]) -> None:
        """Initialize CommitGraphChain.

        Args:
          layers: Graph layers ordered from base to tip, as listed in the
            commit-graph-chain file.
        """
        if not layers:
            raise ValueError("Commit graph chain requires at least one layer")
        super().__init__(object_format=layers[-1].object_format)
        self.layers = layers
        # Merge the per-layer entries into a single view. Layers are
        # disjoint in practice, but let later (tip) layers win lookups to
        # match C Git, which searches the tip layer first.
        self.entries = []
        self._oid_to_index = {}
        for layer in layers:
            for entry in layer.entries:
                self._oid_to_index[hex_to_sha(entry.commit_id)] = len(self.entries)
                self.entries.append(entry)


def read_commit_graph_chain(path: str | bytes) -> CommitGraphChain | None:
    """Read a split commit graph chain from a commit-graph-chain file.

    Args:
      path: Path to the commit-graph-chain file (usually
        objects/info/commit-graphs/commit-graph-chain).

    Returns:
      CommitGraphChain if a consistent chain was found, None if there is no
      chain or the chain disappeared while being read (e.g. replaced by a
      concurrent maintenance run). Callers should fall back to reading
      commit objects directly in that case.

    Raises:
      ValueError: If the chain or one of its layers is corrupt (bad
        checksum, missing base graph list, inconsistent layers).
    """
    if isinstance(path, str):
        path = path.encode()

    try:
        with open(path, "rb") as f:
            chain_data = f.read()
    except FileNotFoundError:
        return None

    hashes = chain_data.split()
    if not hashes:
        return None

    chain_dir = os.path.dirname(path)
    layers: list[CommitGraph] = []
    base_oids: list[RawObjectID] = []
    for index, layer_hash in enumerate(hashes):
        layer_path = os.path.join(chain_dir, b"graph-" + layer_hash + b".graph")
        try:
            with open(layer_path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            # A layer listed in the chain file is missing, most likely
            # because a concurrent maintenance process replaced the chain.
            # Fall back to object traversal rather than returning an
            # incomplete view of the history.
            return None

        _verify_graph_checksum(raw)
        layer = CommitGraph._read_from_file(io.BytesIO(raw), base_oids)

        if layer.object_format.hex_length != len(layer_hash):
            raise ValueError(
                f"Chain entry {layer_hash!r} does not match graph hash version"
            )
        if layer.num_base_graphs != index:
            raise ValueError(
                f"Graph layer {layer_hash!r} expects {layer.num_base_graphs} "
                f"base graphs, found at position {index} in chain"
            )

        base_chunk = layer.chunks.get(CHUNK_BASE_GRAPHS_LIST)
        if index == 0:
            if base_chunk is not None:
                raise ValueError("Base graph layer has a base graphs chunk")
        else:
            if base_chunk is None:
                raise ValueError(
                    f"Graph layer {layer_hash!r} is missing its base graphs chunk"
                )
            base_data = base_chunk.data
            oid_length = layer.object_format.oid_length
            if len(base_data) != index * oid_length:
                raise ValueError(
                    f"Graph layer {layer_hash!r} has invalid base graphs chunk"
                )
            for base_index in range(index):
                start = base_index * oid_length
                expected = sha_to_hex(
                    RawObjectID(base_data[start : start + oid_length])
                )
                if expected != hashes[base_index]:
                    raise ValueError(
                        f"Graph layer {layer_hash!r} does not match the "
                        "commit-graph-chain file"
                    )

        layers.append(layer)
        base_oids.extend(hex_to_sha(entry.commit_id) for entry in layer.entries)

    # The chain file may have been replaced by a concurrent maintenance
    # process while the layers were being read. Re-read it and fall back to
    # object traversal if it no longer matches what was assembled.
    try:
        with open(path, "rb") as f:
            if f.read() != chain_data:
                return None
    except FileNotFoundError:
        return None

    return CommitGraphChain(layers)


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

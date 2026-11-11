"""Shared helpers for the Dulwich delta-chain reuse benchmark.

Builds packs with a FIXED total number of objects while varying:
  * fanout      -> how many deltas share one base object
  * depth       -> how long each delta chain is
and records inflate / apply_delta / external-base-read counts.
"""
import os
import io
import tempfile

from dulwich.object_format import DEFAULT_OBJECT_FORMAT
from dulwich.objects import Blob
from dulwich.pack import (
    OFS_DELTA,
    REF_DELTA,
    Pack,
    PackData,
    MemoryPackIndex,
    write_pack_header,
    write_pack_object,
    create_delta,
)


_BODY_CACHE: dict[int, bytes] = {}


def _compressible_body(size: int) -> bytes:
    """Cached, compressible filler shared by every version of a given size."""
    body = _BODY_CACHE.get(size)
    if body is None:
        pattern = bytes(range(256))
        body = (pattern * (size // 256 + 1))[:size]
        _BODY_CACHE[size] = body
    return body


def obj_data(kind_id: int, seq: int, size: int = 4096) -> bytes:
    """Deterministic, slowly evolving payload so deltas stay small but real.

    The bulk of every version is identical, highly compressible filler, with a
    small edit at a *fixed* offset. Neighbouring versions differ by only a
    short region, so git deltas are tiny while the inflated objects stay large
    -- the "small compressed data expanding into large shared bases" shape
    that makes repeated inflation expensive.
    """
    buf = bytearray(_compressible_body(size))
    # Fixed-offset, deterministic per-version edit region -> short delta.
    edit_len = min(64, size)
    for i in range(edit_len):
        buf[16 + i] = (kind_id * 31 + seq * 7 + i * 13) & 0xFF
    prefix = f"k{kind_id:02d}s{seq:06d}-".encode()
    buf[: len(prefix)] = prefix
    return bytes(buf)


class Counters:
    """Monotonic counters installed by monkeypatching in a context."""

    def __init__(self) -> None:
        self.inflate = 0          # read_zlib_chunks_at invocations
        self.apply_delta = 0      # delta applications
        self.ext_ref = 0          # external base object reads
        self.inflate_bytes = 0    # decompressed bytes produced

    def snapshot(self) -> dict:
        return {
            "inflate": self.inflate,
            "apply_delta": self.apply_delta,
            "inflate_bytes": self.inflate_bytes,
            "ext_ref": self.ext_ref,
        }


def build_chain_pack(fanout: int, depth: int, blob_size: int = 4096):
    """Build a pack with one shared base and `fanout` chains of `depth`.

    Total objects = 1 (base) + fanout*depth (all OFS_DELTA, type blob).

    Returns (raw_bytes, expected) where expected maps offset ->
    (type_num, full_data, sha).
    """
    from dulwich.pack import obj_sha

    n_delta = fanout * depth
    # full payloads: base version 0, then versions 1..n_delta
    payloads = [obj_data(3, 0, blob_size)]
    for v in range(1, n_delta + 1):
        payloads.append(obj_data(3, v, blob_size))

    type_num = Blob.type_num
    shas = [obj_sha(type_num, [payloads[0]])]
    for v in range(1, n_delta + 1):
        shas.append(obj_sha(type_num, [payloads[v]]))

    f = io.BytesIO()
    write_pack_header(f.write, 1 + n_delta)

    # Base full object.
    base_offset = f.tell()
    write_pack_object(f.write, type_num, [payloads[0]],
                      object_format=DEFAULT_OBJECT_FORMAT)
    offsets = [base_offset]
    payload_index = [0]  # object index -> payload version

    version = 1
    for chain in range(fanout):
        prev_offset = base_offset
        prev_version = 0
        for level in range(depth):
            delta = list(create_delta(payloads[prev_version], payloads[version]))
            offset = f.tell()
            write_pack_object(
                f.write, OFS_DELTA, (offset - prev_offset, delta),
                object_format=DEFAULT_OBJECT_FORMAT,
            )
            offsets.append(offset)
            payload_index.append(version)
            prev_offset = offset
            prev_version = version
            version += 1

    body = f.getvalue()
    from hashlib import sha1
    raw = body + sha1(body).digest()

    entries = []
    for idx, off in enumerate(offsets):
        v = payload_index[idx]
        entries.append((off, type_num, payloads[v], shas[v]))
    return raw, entries


def open_pack(raw: bytes, tmpdir: str, resolve_ext_ref=None,
              cache_limit=None):
    pack_path = os.path.join(tmpdir, "test.pack")
    with open(pack_path, "wb") as fh:
        fh.write(raw)
    data = PackData(
        pack_path,
        object_format=DEFAULT_OBJECT_FORMAT,
        delta_base_cache_limit=cache_limit,
    )
    index = MemoryPackIndex.for_pack(data)
    pack = Pack.from_objects(data, index)
    if resolve_ext_ref is not None:
        pack.resolve_ext_ref = resolve_ext_ref
    return pack

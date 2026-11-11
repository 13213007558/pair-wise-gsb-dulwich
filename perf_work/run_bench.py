"""Measure delta-chain expansion cost before/after the reuse-cache work.

Counters (per full resolution of every object in the pack):
  inflate      : zlib object inflations (read_zlib_chunks_at calls)
  apply_delta  : git delta applications
  ext_ref      : external base object reads (thin packs)
  inflate_bytes: bytes produced by inflation
Also records wall time and verifies every object's payload and OID.
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

import dulwich.pack as pack_mod
from dulwich.pack import obj_sha

import bench_common


class Instrument:
    def __init__(self):
        self.c = bench_common.Counters()
        self._orig_inflate = pack_mod.read_zlib_chunks_at
        self._orig_apply = pack_mod.apply_delta

    def __enter__(self):
        c = self.c
        orig_inflate = self._orig_inflate
        orig_apply = self._orig_apply

        def counted_inflate(contents, offset, unpacked, **kw):
            c.inflate += 1
            end = orig_inflate(contents, offset, unpacked, **kw)
            c.inflate_bytes += unpacked.decomp_len or 0
            return end

        def counted_apply(base, delta):
            c.apply_delta += 1
            return orig_apply(base, delta)

        pack_mod.read_zlib_chunks_at = counted_inflate
        pack_mod.apply_delta = counted_apply
        return self

    def __exit__(self, *exc):
        pack_mod.read_zlib_chunks_at = self._orig_inflate
        pack_mod.apply_delta = self._orig_apply


def resolve_all(pack, entries):
    """Random-access path: resolve every object through Pack.get_raw by OID."""
    results = {}
    for off, type_num, data, sha in entries:
        got_type, got_raw = pack.get_raw(sha)
        results[bytes(sha)] = (got_type, got_raw)
    return results


def verify(entries, results):
    """Per-object payload + independently recomputed OID comparison."""
    mismatches = []
    for off, type_num, data, sha in entries:
        key = bytes(sha)
        if key not in results:
            mismatches.append((key.hex(), "missing"))
            continue
        got_type, got_raw = results[key]
        if got_type != type_num:
            mismatches.append((key.hex(), f"type {got_type}!={type_num}"))
        if got_raw != data:
            mismatches.append((key.hex(), "payload mismatch"))
        if obj_sha(got_type, [got_raw]) != key:
            mismatches.append((key.hex(), "OID mismatch"))
    return mismatches


def run_case(fanout, depth, cache_limit, blob_size, tmpdir):
    raw, entries = bench_common.build_chain_pack(fanout, depth, blob_size)
    pack = bench_common.open_pack(raw, tmpdir, cache_limit=cache_limit)
    try:
        with Instrument() as inst:
            gc.collect()
            t0 = time.perf_counter()
            cold_results = resolve_all(pack, entries)
            t_cold = time.perf_counter() - t0
            cold = inst.c.snapshot()

            inst.c = bench_common.Counters()
            gc.collect()
            t0 = time.perf_counter()
            warm_results = resolve_all(pack, entries)
            t_warm = time.perf_counter() - t0
            warm = inst.c.snapshot()

        mism = verify(entries, cold_results) + verify(entries, warm_results)
        resolved_cache = getattr(pack.data, "_resolved_cache", None)
        return {
            "fanout": fanout,
            "depth": depth,
            "objects": len(entries),
            "pack_bytes": len(raw),
            "cache_limit": cache_limit,
            "cold": {**cold, "seconds": round(t_cold, 5)},
            "warm": {**warm, "seconds": round(t_warm, 5)},
            "raw_cache_entries": len(pack.data._offset_cache._cache),
            "resolved_cache_entries": (
                len(resolved_cache._cache) if resolved_cache is not None else -1
            ),
            "resolved_cache_bytes": (
                resolved_cache._value_size if resolved_cache is not None else -1
            ),
            "mismatches": mism,
        }
    finally:
        pack.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob-size", type=int, default=2048)
    ap.add_argument("--product", type=int, default=256,
                    help="number of delta objects (= fanout*depth); total = product+1")
    ap.add_argument("--max-depth", type=int, default=64)
    ap.add_argument("--cache-limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = []
    depth = 1
    while depth <= min(args.product, args.max_depth):
        if args.product % depth == 0:
            cases.append((args.product // depth, depth))
        depth <<= 1

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for fanout, depth in cases:
            rows.append(run_case(fanout, depth, args.cache_limit,
                                 args.blob_size, tmpdir))
    text = json.dumps(rows, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()

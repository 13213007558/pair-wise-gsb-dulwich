"""Subset-read benchmark: resolve only `requested` leaves of a high-fanout pack.

This mirrors real fetches/checkouts that touch a few objects out of a pack
where many deltas share one base -- the case that let a small compressed pack
amplify into repeated base inflation before the fix.
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
        self.oi = pack_mod.read_zlib_chunks_at
        self.oa = pack_mod.apply_delta

    def __enter__(self):
        c = self.c
        oi, oa = self.oi, self.oa

        def ci(contents, offset, unpacked, **kw):
            c.inflate += 1
            end = oi(contents, offset, unpacked, **kw)
            c.inflate_bytes += unpacked.decomp_len or 0
            return end

        def ca(base, delta):
            c.apply_delta += 1
            return oa(base, delta)

        pack_mod.read_zlib_chunks_at = ci
        pack_mod.apply_delta = ca
        return self

    def __exit__(self, *e):
        pack_mod.read_zlib_chunks_at = self.oi
        pack_mod.apply_delta = self.oa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", type=int, default=128)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--blob-size", type=int, default=8192)
    ap.add_argument("--requested", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        raw, entries = bench_common.build_chain_pack(
            args.fanout, args.depth, args.blob_size
        )
        pack = bench_common.open_pack(raw, td)
        pack.verify_resolved = args.verify
        # Request only the first `requested` leaf objects (skip the base at 0).
        targets = entries[1 : 1 + args.requested]

        gc.collect()
        with Instrument() as inst:
            t0 = time.perf_counter()
            results = {}
            for off, tn, data, sha in targets:
                results[bytes(sha)] = pack.get_raw(sha)
            dt = time.perf_counter() - t0
        counts = inst.c.snapshot()

        mism = 0
        for off, tn, data, sha in targets:
            got_tn, got_raw = results[bytes(sha)]
            if got_tn != tn or got_raw != data or obj_sha(got_tn, [got_raw]) != sha:
                mism += 1
        print(json.dumps({
            "fanout": args.fanout, "depth": args.depth,
            "total_objects": len(entries), "requested": args.requested,
            "pack_bytes": len(raw), "verify": args.verify,
            **counts, "seconds": round(dt, 5), "mismatches": mism,
            "resolved_cache_bytes": pack.data._resolved_cache._value_size
            if hasattr(pack.data, "_resolved_cache") else -1,
        }, indent=2))


if __name__ == "__main__":
    main()

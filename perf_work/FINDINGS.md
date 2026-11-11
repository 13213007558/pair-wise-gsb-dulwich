# Delta-chain reuse: measurement and optimization

## What was wrong

`Pack.resolve_object` (random access via `Pack.get_raw`) had two defects for
packs where many deltas share a base object:

1. **Full bases were never cached.** Only the leaf object being resolved was
   written back, and only when it was itself an intermediate of another chain.
   Every leaf re-inflated its shared base, so a pack with `F` depth-1 deltas on
   one base inflated that base `F` times. Inflation work grew with fan-out.
2. **One cache keyed only by offset held two incompatible value shapes.**
   `PackData.get_object_at` returned the raw (possibly deltified) entry, but
   `resolve_object` overwrote the same offset with a fully resolved full text.
   Resolved and raw entries could not safely coexist, which is what prevented
   caching bases in the first place.

## The fix (dulwich/pack.py)

* Split the cache into two bounded `LRUSizeCache`s on `PackData`:
  * `_offset_cache`  – raw, still-deltified entries (type never changes).
  * `_resolved_cache` – fully resolved, verified full texts.
* `Pack.resolve_object` now walks each chain once, reuses any already-resolved
  internal base (early exit), and inserts resolved objects only **after** the
  whole chain applied without raising.
* Thin packs get a third bounded cache on `Pack`, `_ext_base_cache`, keyed by
  OID, so an external base is fetched once and reused.
* All three caches are byte-bounded by `delta_base_cache_limit`
  (default `DEFAULT_DELTA_BASE_CACHE_LIMIT` = 96 MiB) and are LRU-evicted.
* `Pack(verify_resolved=True)` re-hashes the requested object and compares it
  to its indexed OID before anything enters `_resolved_cache`, raising
  `ChecksumMismatch` on failure. Default off (Dulwich already verifies OIDs at
  index creation / `check()`); bad deltas are excluded from the cache either
  way because `apply_delta` raises before any cache write.

## Measurements (pure-Python `apply_delta`, Python 3.12, macOS/arm64)

Counters are installed by monkeypatching `read_zlib_chunks_at` (inflations),
`apply_delta`, and the thin-pack external resolver; every resolved object is
also re-hashed and compared to its expected payload and OID.

### Fixed total = 257 objects (1 base + 256 OFS_DELTA), 2 KiB blobs, cold full read

| fanout | depth | inflations base->opt | inflated KiB base->opt | apply_delta | per-object result |
|-------:|------:|---------------------:|-----------------------:|:-----------:|:-----------------:|
| 256 | 1 | 513 -> 257 (-50%) | 530 -> 18 (-96%) | 256 -> 256 | 0 mismatches |
| 128 | 2 | 385 -> 257 (-33%) | 276 -> 20 (-93%) | 256 -> 256 | 0 mismatches |
| 64 | 4 | 321 -> 257 (-20%) | 149 -> 21 (-86%) | 256 -> 256 | 0 mismatches |
| 32 | 8 | 289 -> 257 (-11%) | 85 -> 21 (-75%) | 256 -> 256 | 0 mismatches |
| 16 | 16 | 273 -> 257 (-6%) | 53 -> 21 (-59%) | 256 -> 256 | 0 mismatches |
| 8 | 32 | 265 -> 257 (-3%) | 37 -> 21 (-42%) | 256 -> 256 | 0 mismatches |
| 4 | 64 | 261 -> 257 (-2%) | 29 -> 21 (-27%) | 256 -> 256 | 0 mismatches |

Inflations become constant at one per object regardless of fan-out; the
amplification factor (inflations per object) drops from up to 2.0x to 1.0x.
`apply_delta` count is unchanged because each delta must still be applied once
on a cold read; the win is eliminating repeated base inflation. A second, warm
read of the same pack does zero inflations on both builds, but the optimized
build serves everything from the resolved cache.

### Realistic partial read: 16 of 128 leaves, 16 KiB bases, 50 cold repetitions

A small ~11 KB pack expands large shared bases; only 16 leaves are requested.

| metric | baseline | optimized | change |
|---|---:|---:|---:|
| inflations / read | 32 | 17 | -47% |
| inflated KiB / read | 257 | 17 | -93% |
| apply_delta / read | 16 | 16 | 0% |
| avg wall time (ms) | 0.371 | 0.276 | -26% |
| min wall time (ms) | 0.347 | 0.251 | -28% |
| wrong payload/OID | 0 | 0 | - |

### Thin pack and untrusted-input guarantees (see run_thin.py / run_bounds.py)

* Two REF_DELTA leaves on one external base: external base reads = **1**
  (was 2); both leaves still resolve to the exact payload and OID.
* Malformed delta: raises `ApplyDeltaError`; `_resolved_cache` stays empty.
* Wrong indexed OID with `verify_resolved=True`: raises `ChecksumMismatch`;
  the object is never cached.
* 16 KiB cache limit: both in-pack caches stay at or below the limit under
  LRU eviction, and every object still resolves correctly afterwards.

## Reproduce

    .venv/bin/python perf_work/run_bench.py  --product 256 --blob-size 2048
    .venv/bin/python perf_work/run_subset_rep.py --fanout 128 --requested 16 \
        --blob-size 16384 --reps 50
    .venv/bin/python perf_work/run_thin.py
    .venv/bin/python perf_work/run_bounds.py

Unit tests: `DeltaBaseReuseCacheTests` in tests/test_pack.py (7 tests).

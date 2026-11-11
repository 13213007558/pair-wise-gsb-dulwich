import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import bench_common
from dulwich.pack import LRUSizeCache

with tempfile.TemporaryDirectory() as td:
    # Tight cache limit (16 KiB) with 4 KiB blobs -> at most a few entries.
    raw, entries = bench_common.build_chain_pack(fanout=16, depth=2, blob_size=4096)
    pack = bench_common.open_pack(raw, td, cache_limit=16 * 1024)
    data = pack.data
    assert isinstance(data._resolved_cache, LRUSizeCache)
    for off, tn, payload, sha in entries:
        pack.get_raw(sha)
    rb = data._resolved_cache._value_size
    rawb = data._offset_cache._value_size
    extb = pack._ext_base_cache._value_size
    print(f"resolved cache bytes={rb} (limit 16384), entries={len(data._resolved_cache._cache)}")
    print(f"raw cache bytes={rawb} (limit 16384), entries={len(data._offset_cache._cache)}")
    print(f"ext base cache bytes={extb} (limit {pack._ext_base_cache._max_size})")
    assert rb <= 16 * 1024, "resolved cache exceeded configured byte limit"
    assert rawb <= 16 * 1024, "raw cache exceeded configured byte limit"
    assert extb <= pack._ext_base_cache._max_size
    # Correctness after eviction: every object still resolves to the right OID.
    from dulwich.pack import obj_sha
    for off, tn, payload, sha in entries:
        got_tn, got = pack.get_raw(sha)
        assert got == payload and obj_sha(got_tn, [got]) == sha
    print("all objects still correct after LRU eviction: OK")
    pack.close()

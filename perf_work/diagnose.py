import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import bench_common
from dulwich.pack import OFS_DELTA

with tempfile.TemporaryDirectory() as td:
    # Two leaves sharing one base (fanout=2, depth=1)
    raw, entries = bench_common.build_chain_pack(fanout=2, depth=1, blob_size=128)
    pack = bench_common.open_pack(raw, td)
    data = pack.data
    base_off = entries[0][0]
    leaf_offsets = [e[0] for e in entries[1:]]

    # Resolve first leaf
    t0, raw0 = data.get_object_at(leaf_offsets[0])
    rtype, rchunks = pack.resolve_object(leaf_offsets[0], t0, raw0)
    print("after leaf0 resolve, cache:")
    for off in sorted(data._offset_cache.keys()):
        tn, val = data._offset_cache[off]
        kind = "DELTA" if tn in (OFS_DELTA, 7) else f"resolved type {tn}"
        print(f"  offset={off} stored_type_num={tn} ({kind})")
    print("base offset cached? ", base_off in data._offset_cache)

    # Now resolve the second leaf, which shares the same base.
    t1, raw1 = data.get_object_at(leaf_offsets[1])
    print("\nleaf1 raw type from get_object_at:", t1,
          "(6=OFS_DELTA raw, 3=resolved blob) -- if 3 here, cache returned a"
          " resolved object where raw delta was expected")

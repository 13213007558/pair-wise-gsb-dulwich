import argparse, gc, json, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
import dulwich.pack as pack_mod
from dulwich.pack import obj_sha
import bench_common

class Inst:
    def __init__(s):
        s.c=bench_common.Counters(); s.oi=pack_mod.read_zlib_chunks_at; s.oa=pack_mod.apply_delta
    def __enter__(s):
        c=s.c; oi,oa=s.oi,s.oa
        def ci(cont,off,u,**k):
            c.inflate+=1; e=oi(cont,off,u,**k); c.inflate_bytes+=u.decomp_len or 0; return e
        def ca(b,dl): c.apply_delta+=1; return oa(b,dl)
        pack_mod.read_zlib_chunks_at=ci; pack_mod.apply_delta=ca; return s
    def __exit__(s,*e):
        pack_mod.read_zlib_chunks_at=s.oi; pack_mod.apply_delta=s.oa

ap=argparse.ArgumentParser()
ap.add_argument("--fanout",type=int,default=128)
ap.add_argument("--requested",type=int,default=16)
ap.add_argument("--blob-size",type=int,default=16384)
ap.add_argument("--reps",type=int,default=50)
a=ap.parse_args()
with tempfile.TemporaryDirectory() as td:
    raw,entries=bench_common.build_chain_pack(a.fanout,1,a.blob_size)
    pack=bench_common.open_pack(raw,td)
    targets=entries[1:1+a.requested]
    times=[]
    def reset_caches(p):
        for cname in ("_offset_cache", "_resolved_cache"):
            c = getattr(p.data, cname, None)
            if c is not None:
                c._cache.clear()
                c._value_size = 0
        ec = getattr(pack, "_ext_base_cache", None)
        if ec is not None:
            ec._cache.clear()
            ec._value_size = 0

    with Inst() as inst:
        for _ in range(a.reps):
            # fresh caches every rep => cold subset cost
            reset_caches(pack)
            gc.collect()
            t0=time.perf_counter()
            res={bytes(s):pack.get_raw(s) for _,_,_,s in targets}
            times.append(time.perf_counter()-t0)
    c=inst.c.snapshot()
    mism=sum(1 for _,tn,data,s in targets if res[bytes(s)]!=(tn,data) or obj_sha(tn,[res[bytes(s)][1]])!=s)
    print(json.dumps({"fanout":a.fanout,"requested":a.requested,"blob_size":a.blob_size,
        "reps":a.reps,"pack_bytes":len(raw),
        "inflate_per_rep":c["inflate"]//a.reps,
        "apply_per_rep":c["apply_delta"]//a.reps,
        "inflate_KiB_per_rep":c["inflate_bytes"]//1024//a.reps,
        "avg_ms":round(1000*sum(times)/len(times),4),
        "min_ms":round(1000*min(times),4),"mismatches":mism},indent=2))

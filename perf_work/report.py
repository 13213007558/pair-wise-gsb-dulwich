import json

b = {(r["fanout"], r["depth"]): r for r in json.load(open("perf_work/final_base.json"))}
o = {(r["fanout"], r["depth"]): r for r in json.load(open("perf_work/final_opt.json"))}

def d(new, old):
    return "n/a" if old == 0 else f"{(new-old)/old*100:+.0f}%"

print("FIXED TOTAL = 257 objects (1 base + 256 OFS_DELTA), 2 KiB blobs, all objects read once (cold)")
print(f"{'fanout':>6}{'depth':>6} | {'inflate base->opt':>19} | {'inflate KiB base->opt':>23} | {'apply base->opt':>17} | {'cold ms base->opt':>20} | per-object")
for k in sorted(b, key=lambda k: k[1]):
    bc, oc = b[k]["cold"], o[k]["cold"]
    print(f"{k[0]:>6}{k[1]:>6} | "
          f"{bc['inflate']:>8}->{oc['inflate']:<7}{d(oc['inflate'],bc['inflate']):>4} | "
          f"{bc['inflate_bytes']//1024:>10}->{oc['inflate_bytes']//1024:<8}{d(oc['inflate_bytes'],bc['inflate_bytes']):>4} | "
          f"{bc['apply_delta']:>7}->{oc['apply_delta']:<6}{d(oc['apply_delta'],bc['apply_delta']):>4} | "
          f"{bc['seconds']*1000:>8.2f}->{oc['seconds']*1000:<7.2f}{d(oc['seconds'],bc['seconds']):>4} | "
          f"mismatch={len(o[k]['mismatches'])}")

print("\nWarm second full read (same Pack, caches warm): inflate",
      sorted({(k, b[k]['warm']['inflate'], o[k]['warm']['inflate']) for k in b}))
print("Resolved-cache bytes held (bounded, all rows):",
      sorted({r['resolved_cache_bytes'] for r in o.values()}))

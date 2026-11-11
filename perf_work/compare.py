import json

base = {(r["fanout"], r["depth"]): r for r in json.load(open("perf_work/baseline.json"))}
opt = {(r["fanout"], r["depth"]): r for r in json.load(open("perf_work/optimized.json"))}

def pct(new, old):
    if old == 0:
        return "n/a"
    return f"{(new-old)/old*100:+.0f}%"

print(f"{'fan':>4} {'dep':>4} | {'-- inflate (cold) --':^22} | {'-- apply_delta --':^20} | "
      f"{'-- inflate MB --':^20} | {'-- cold seconds --':^20} | {'-- warm inflate --':^20}")
print(f"{'':4} {'':4} | {'base':>8} {'opt':>8} {'delta':>6} | {'base':>8} {'opt':>8} "
      f"{'delta':>6} | {'base':>8} {'opt':>8} {'delta':>6} | {'base':>8} {'opt':>8} {'delta':>6} | "
      f"{'base':>8} {'opt':>8} {'delta':>6}")
for key in sorted(base, key=lambda k: k[1]):
    b, o = base[key], opt[key]
    bc, oc = b["cold"], o["cold"]
    bw, ow = b["warm"], o["warm"]
    print(f"{key[0]:>4} {key[1]:>4} | "
          f"{bc['inflate']:>8} {oc['inflate']:>8} {pct(oc['inflate'],bc['inflate']):>6} | "
          f"{bc['apply_delta']:>8} {oc['apply_delta']:>8} {pct(oc['apply_delta'],bc['apply_delta']):>6} | "
          f"{bc['inflate_bytes']/1e6:>8.2f} {oc['inflate_bytes']/1e6:>8.2f} {pct(oc['inflate_bytes'],bc['inflate_bytes']):>6} | "
          f"{bc['seconds']:>8.4f} {oc['seconds']:>8.4f} {pct(oc['seconds'],bc['seconds']):>6} | "
          f"{bw['inflate']:>8} {ow['inflate']:>8} {pct(ow['inflate'],bw['inflate']):>6}")

print("\nCorrectness (optimized): all rows report mismatches =",
      [len(r["mismatches"]) for r in opt.values()])
print("Resolved-cache bytes held (bounded):",
      sorted({r["resolved_cache_bytes"] for r in opt.values()}))

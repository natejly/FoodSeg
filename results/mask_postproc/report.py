"""Tables + paired per-scan bootstrap from results.json -> stdout (markdown). python results/mask_postproc/report.py [names...]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SIDS = ["s000000", "s000001", "s000002", "s000007", "s000009", "s000011", "s000012", "s000013", "s000014", "s000016", "s000017"]
R = json.loads((HERE / "results.json").read_text())


def stats(rs, idx):
    ious = [x for i in idx for x in rs[SIDS[i]]["iou"]]
    tp = sum(rs[SIDS[i]]["tp"] for i in idx); fp = sum(rs[SIDS[i]]["fp"] for i in idx); fn = sum(rs[SIDS[i]]["fn"] for i in idx)
    ti = sum(rs[SIDS[i]]["tp_iou"] for i in idx)
    return np.mean(ious), ti / (tp + .5 * fp + .5 * fn)


def ci(name, ref="baseline", n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = R[name]["scans"], R[ref]["scans"]
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(SIDS), len(SIDS))
        sa, sb = stats(a, idx), stats(b, idx)
        d.append((sa[0] - sb[0], sa[1] - sb[1]))
    d = np.array(d)
    full = np.array(stats(a, range(len(SIDS)))) - np.array(stats(b, range(len(SIDS))))
    return full, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def row(name):
    rs = R[name]["scans"]
    inst, pq = stats(rs, range(len(SIDS)))
    exact = np.mean([rs[s]["count_ok"] for s in SIDS])
    ty = {t: sum(rs[s]["types"][t] for s in SIDS) for t in rs[SIDS[0]]["types"]}
    f, lo, hi = ci(name)
    sig = "+" if lo[0] > 0 else "-" if hi[0] < 0 else "0"
    log = {}
    for s in SIDS:
        for r in rs[s]["log"]:
            k = f"{r['kind']}:{r['rule']}"
            o = log.setdefault(k, [0, 0]); o[1] += 1; o[0] += int(bool(r.get("ok")))
    prec = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in log.items()) or "-"
    return (f"| {name} | {inst:.3f} | {pq:.3f} | {exact:.3f} | {ty['fragmented']} | {ty['merged']} | "
            f"{f[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}] {sig} | {f[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] | {prec} |")


HDR = ("| config | inst mIoU | PQ | exact-count | fragmented | merged | d inst mIoU vs base [95% CI] | d PQ [95% CI] | "
       "decision precision (correct/total) |\n|---|---|---|---|---|---|---|---|---|")
if __name__ == "__main__":
    print(HDR)
    for n in (sys.argv[1:] or R):
        print(row(n))

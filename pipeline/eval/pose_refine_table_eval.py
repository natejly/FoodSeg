"""Table-masked ICP sweep: metrics_table.md, frames_refined_<config>.jsonl for the best config, summary.png.

    cd pipeline && .venv/bin/python -m eval.pose_refine_table_eval
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.pose_refine_eval import OUT, ROOT, SCANS, fuse, metrics, title, traj, view
from foodscan.depth_odometry import refine_scan, write_jsonl
from foodscan.loader import _ARKIT_TO_CV, Scan

# (method, table, rot, repeats)
CONFIGS = [("geometric", "keep", "icp", 1), ("colored", "keep", "icp", 1), ("features", "keep", "icp", 1),
           ("geometric", "drop", "icp", 1), ("colored", "drop", "icp", 2), ("colored", "downweight", "icp", 2),
           ("features", "drop", "icp", 2), ("features", "downweight", "icp", 1),
           ("geometric", "drop", "arkit", 1), ("colored", "drop", "arkit", 1), ("features", "drop", "arkit", 1)]
BEST = ("features", "drop", "icp")
label = lambda c: "_".join([c[0], c[1]] + (["rotarkit"] if c[2] == "arkit" else []))


def run(job):
    name, c, rep = job
    d = ROOT / "data" / "scans" / name
    t0 = time.time()
    out = refine_scan(d, method=c[0], table=c[1], rot=c[2])
    secs = time.time() - t0
    scan = Scan(d, poses="arkit")
    m = metrics(scan, out[0], out[1], secs)
    m["fallback"] = sum("table-fallback" in w for w in out[4])
    return name, c, rep, m, out[0]


def main():
    jobs = [(n, c, r) for n in SCANS for c in CONFIGS for r in range(c[3])]
    with Pool(6) as p:
        res = p.map(run, jobs, chunksize=1)
    hdr = ["| scan | config | run | path (m) | first-vs-last NN (cm) | 1 s-gap NN (cm) | jitter (mm) | flagged | table-fallback frames | runtime (s) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    lines = [hdr[0], hdr[1]]
    for n, c, r, m, _ in res:
        lines.append(f"| {n} | {label(c)} | {r+1} | {m['path']:.2f} | {m['fl']*100:.1f} | {m['gap']*100:.2f} | {m['jit']:.1f} | {m['bad']} | {m['fallback']} | {m['secs']:.0f} |")
    (OUT / "metrics_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    # jsonl + summary for best config
    poses = {}
    for n, c, r, m, p in res:
        if c == BEST and r == 0:
            poses[n] = (p, m)
    for n in SCANS:
        d = ROOT / "data" / "scans" / n
        out = refine_scan(d, method=BEST[0], table=BEST[1], rot=BEST[2])
        write_jsonl(d, *out, method=label(BEST))
    names = ["20261017_202402_scan", "20261017_202343_scan"]
    fig, axs = plt.subplots(2, 5, figsize=(25, 11.2))
    for r, n in enumerate(names):
        scan = Scan(ROOT / "data" / "scans" / n, poses="arkit")
        ark = np.array([f.T_cw @ np.linalg.inv(_ARKIT_TO_CV) for f in scan.frames])
        look = {(n2, c): (p, m) for n2, c, rep, m, p in res if rep == 0}
        runs = {"ARKit": (ark, metrics(scan, ark)), "features": look[(n, ("features", "keep", "icp"))],
                "features+drop": look[(n, BEST)]}
        for c, (lab, (p, m)) in enumerate(runs.items()):
            P, C = fuse(scan, p); view(axs[r, c], P, C, "top"); axs[r, c].set_title(f"{n}\n{lab}\n{title(m)}", fontsize=8)
        COLORS_ = {"ARKit": "tab:red", "features": "tab:purple", "features+drop": "tab:orange"}
        ax = axs[r, 3]
        for lab, (p, m) in runs.items():
            xy = p[:, [0, 2], 3]; ax.plot(*xy.T, c=COLORS_[lab], lw=1, label=f"{lab} ({m['path']:.2f} m)"); ax.scatter(*xy[0], c=COLORS_[lab], s=30)
        ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("trajectories (top-down)")
        axs[r, 4].axis("off")
    fig.suptitle("Pose refinement: ARKit vs features vs features + table dropped", fontsize=15)
    fig.tight_layout(); fig.savefig(OUT / "summary.png", dpi=70)


if __name__ == "__main__":
    main()

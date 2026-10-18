"""Run COLMAP pose refinement on the iPad scans, write frames_refined_colmap.jsonl, metrics_colmap.md.

    cd pipeline && .venv/bin/python -m eval.pose_refine_colmap [scan ...]
Good scans: seeded from ARKit (BA, stride 3). RUNAWAY scans: unseeded SfM aligned to first 25 ARKit frames (stride 2).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from foodscan.depth_odometry import cloud_nn, gap_nn, path_length
from foodscan.loader import Scan
from foodscan.refine_poses import refine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "pose_refine"
SCANS = ["20261002_210342_scan", "20261017_202343_scan", "20261017_202402_scan",
         "20261017_203105_scan", "20261017_203137_scan"]
RUNAWAY = {"20261017_202402_scan": 25}


def interp(arkit, refined, stride):
    """Poses for every frame: slerp+lerp between registered frames. bad = outside the registered range or in a gap."""
    idx = np.array([f.index for f in arkit.frames])
    ks = np.array(sorted(refined))
    R = Rotation.from_matrix([refined[k][:3, :3] for k in ks])
    t = np.array([refined[k][:3, 3] for k in ks])
    slerp = Slerp(ks, R)
    out = np.tile(np.eye(4), (len(idx), 1, 1))
    bad = np.zeros(len(idx), bool)
    for n, i in enumerate(idx):
        c = int(np.clip(i, ks[0], ks[-1]))
        out[n, :3, :3] = slerp([c]).as_matrix()[0]
        out[n, :3, 3] = [np.interp(c, ks, t[:, d]) for d in range(3)]
        j = np.searchsorted(ks, c)
        gap = ks[min(j, len(ks) - 1)] - ks[max(j - 1, 0)]
        bad[n] = (i < ks[0]) or (i > ks[-1]) or gap > 2 * stride
    return out, bad


def stats(scan, P):
    return dict(path=path_length(P), fl=cloud_nn(scan.frames[0], P[0], scan.frames[-1], P[-1]) * 100,
                gap=gap_nn(scan, P) * 100,
                jit=float(np.median(np.linalg.norm(P[2:, :3, 3] - 2 * P[1:-1, :3, 3] + P[:-2, :3, 3], axis=1))) * 1000)


def main():
    rows = []
    for name in sys.argv[1:] or SCANS:
        d = ROOT / "data" / "scans" / name
        scan = Scan(d, poses="arkit")
        A = np.array([f.T_cw @ np.linalg.inv(np.diag([1., -1, -1, 1])) for f in scan.frames])
        seed, stride = RUNAWAY.get(name), 2 if name in RUNAWAY else 3
        t0 = time.time()
        ref = refine(scan, Path("/tmp/colmap_work") / name, stride=stride, seed_frames=seed)
        rt = time.time() - t0
        P, bad = interp(scan, ref, stride)
        lines = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l.strip()]
        with open(d / "frames_refined_colmap.jsonl", "w") as f:
            for r, T, b in zip(lines, P, bad):
                r.update(transform=T.tolist(), bad=bool(b))
                f.write(json.dumps(r) + "\n")
        n_s = len(scan.good_frames(stride))
        for lab, Q in (("ARKit", A), ("COLMAP", P)):
            s = stats(scan, Q)
            rows.append(f"| {name} | {lab} | {s['path']:.2f} | {s['fl']:.1f} | {s['gap']:.1f} | {s['jit']:.1f} | "
                        + (f"{len(ref)}/{n_s} sampled ({len(A)} total, {int(bad.sum())} bad) | {rt:.0f} |" if lab == "COLMAP" else "- | - |"))
            print(rows[-1], flush=True)
        (OUT / f"_row_{name}.md").write_text("\n".join(rows[-2:]) + "\n")
    if not sys.argv[1:]:
        hdr = ("| scan | poses | path (m) | first-vs-last NN (cm) | 1 s-gap NN (cm) | jitter (mm) | registered | runtime (s) |\n"
               "|---|---|---|---|---|---|---|---|\n")
        (OUT / "metrics_colmap.md").write_text(hdr + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

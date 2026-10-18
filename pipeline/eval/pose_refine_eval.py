"""ARKit vs depth-odometry poses on iPad scans: metrics table + before/after visuals.

    cd pipeline && .venv/bin/python -m eval.pose_refine_eval
Outputs results/pose_refine/{metrics_methods.md, <scan>_clouds.png, <scan>_traj.png, <scan>_<method>.ply, summary.png}
and data/scans/<scan>/frames_refined_<method>.jsonl.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import numpy as np
import open3d as o3d
from pathlib import Path

from foodscan.depth_odometry import METHODS, cloud_nn, gap_nn, path_length, refine_scan, world_cloud, write_jsonl
from foodscan.loader import _ARKIT_TO_CV, Scan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "pose_refine"
SCANS = ["20261002_210342_scan", "20261017_202343_scan", "20261017_202402_scan",
         "20261017_203105_scan", "20261017_203137_scan"]
SUMMARY = ["20261017_202402_scan", "20261017_202343_scan"]  # runaway + good scan


def fuse(scan: Scan, poses, step: int = 2, per_frame: int = 2):
    """World points + RGB from every `step`-th frame."""
    P, C = zip(*[world_cloud(f, T, stride=per_frame, rgb=True)
                 for f, T in list(zip(scan.frames, poses))[::step]])
    return np.concatenate(P), np.concatenate(C)


def jitter_mm(poses) -> float:
    """Median norm of the second difference of camera translations (mm)."""
    t = poses[:, :3, 3]
    return float(np.median(np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=1)) * 1000)


def metrics(scan, poses, bad=None, secs=None):
    return dict(path=path_length(poses), fl=cloud_nn(scan.frames[0], poses[0], scan.frames[-1], poses[-1]),
                gap=gap_nn(scan, poses), jit=jitter_mm(poses), bad=None if bad is None else int(bad.sum()), secs=secs)


def title(m):
    return f"path {m['path']:.2f} m, first-vs-last NN {m['fl']*100:.1f} cm, jitter {m['jit']:.1f} mm"


def view(ax, P, C, mode, n=60000, seed=0):
    """Top-down (X vs Z) or oblique orthographic scatter, axes clipped to the 2-98 percentile."""
    P, C = P[(i := np.random.default_rng(seed).permutation(len(P))[:n])], C[i]
    if mode == "top":
        x, y, d = P[:, 0], P[:, 2], P[:, 1]
    else:  # azimuth 35 deg, elevation 35 deg; world Y is up
        a, e = np.radians(35), np.radians(35)
        h, dep = P[:, 0] * np.cos(a) + P[:, 2] * np.sin(a), -P[:, 0] * np.sin(a) + P[:, 2] * np.cos(a)
        x, y, d = h, P[:, 1] * np.cos(e) + dep * np.sin(e), -dep
    o = np.argsort(d)
    ax.scatter(x[o], y[o], c=C[o] / 255, s=0.6, linewidths=0, rasterized=True)
    lo, hi = np.percentile(np.c_[x, y], 2, axis=0), np.percentile(np.c_[x, y], 98, axis=0)
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal"); ax.tick_params(labelsize=7)


COLORS = {"ARKit": "tab:red", "geometric": "tab:blue", "colored": "tab:green", "features": "tab:purple"}


def traj(ax, runs):
    """runs: {label: (poses, metrics)}"""
    for lab, (p, m) in runs.items():
        xy = p[:, [0, 2], 3]
        ax.plot(*xy.T, c=COLORS[lab], lw=1, label=f"{lab} ({m['path']:.2f} m)"); ax.scatter(*xy[0], c=COLORS[lab], marker="o", s=30)
        ax.scatter(*xy[-1], c=COLORS[lab], marker="s", s=30)
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")


def ply(path, P, C, voxel=0.004):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(C / 255)
    o3d.io.write_point_cloud(str(path), pc.voxel_down_sample(voxel))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for name in SCANS:
        d = ROOT / "data" / "scans" / name
        if not d.exists():
            print("missing:", name); continue
        scan = Scan(d, poses="arkit")
        ark = np.array([f.T_cw @ np.linalg.inv(_ARKIT_TO_CV) for f in scan.frames])
        runs = {"ARKit": (ark, metrics(scan, ark))}
        for meth in METHODS:
            t0 = time.time()
            ref, bad, fit, rmse, why = refine_scan(d, method=meth)
            secs = time.time() - t0
            write_jsonl(d, ref, bad, fit, rmse, why, meth)
            runs[meth] = (ref, metrics(scan, ref, bad, secs))
            print(name, meth, runs[meth][1], flush=True)
        fused = {lab: fuse(scan, p) for lab, (p, _) in runs.items()}
        cache[name] = (runs, fused)
        rows += [(name, lab, m) for lab, (_, m) in runs.items()]
        fig, axs = plt.subplots(2, 4, figsize=(20, 11))
        for j, (lab, (P, C)) in enumerate(fused.items()):
            for i, mode in enumerate(("top", "oblique")):
                view(axs[i, j], P, C, mode)
                axs[i, j].set_title(f"{lab} - {'top-down' if mode == 'top' else 'oblique'}" + (f"\n{title(runs[lab][1])}" if i == 0 else ""), fontsize=8)
            ply(OUT / f"{name}_{lab.lower()}.ply", P, C)
        fig.suptitle(name, fontsize=13); fig.tight_layout(); fig.savefig(OUT / f"{name}_clouds.png", dpi=90); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6)); traj(ax, runs); fig.suptitle(name)
        fig.tight_layout(); fig.savefig(OUT / f"{name}_traj.png", dpi=110); plt.close(fig)
    if all(n in cache for n in SUMMARY):
        fig, axs = plt.subplots(len(SUMMARY), 5, figsize=(25, 5.6 * len(SUMMARY)))
        for r, name in enumerate(SUMMARY):
            runs, fused = cache[name]
            for c, (lab, (P, C)) in enumerate(fused.items()):
                view(axs[r, c], P, C, "top")
                axs[r, c].set_title(f"{name}\n{lab} (top-down)\n{title(runs[lab][1])}", fontsize=8)
            traj(axs[r, 4], runs); axs[r, 4].set_title(f"{name} trajectories (top-down)", fontsize=9)
        fig.suptitle("Pose refinement: ARKit vs geometric / colored / features", fontsize=15)
        fig.tight_layout(); fig.savefig(OUT / "summary.png", dpi=70); plt.close(fig)
    hdr = ["| scan | method | path (m) | first-vs-last NN (cm) | 1 s-gap NN (cm) | jitter (mm) | flagged | runtime (s) |", "|---|---|---|---|---|---|---|---|"]
    lines = hdr + [f"| {n} | {p} | {m['path']:.2f} | {m['fl']*100:.1f} | {m['gap']*100:.2f} | {m['jit']:.1f} | "
                   f"{'-' if m['bad'] is None else m['bad']} | {'-' if m['secs'] is None else '%.0f' % m['secs']} |" for n, p, m in rows]
    (OUT / "metrics_methods.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()

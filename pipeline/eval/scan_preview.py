"""Preview images for scans: python pipeline/eval/scan_preview.py SCAN_DIR... (writes results/scan_previews/<scan>/)."""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.loader import Scan

ROOT = Path(__file__).resolve().parents[2]


def pick(n, k):
    return np.linspace(0, n - 1, min(k, n)).round().astype(int)


def preview(path):
    scan = Scan(path, poses="arkit")
    out = ROOT / "results/scan_previews" / Path(path).name
    out.mkdir(parents=True, exist_ok=True)
    F, t0 = scan.frames, scan.frames[0].timestamp

    # 1. contact sheet
    idx = pick(len(F), 12)
    fig, axs = plt.subplots(3, 4, figsize=(16, 9.6))
    for ax, i in zip(axs.flat, idx):
        ax.imshow(F[i].rgb()[::4, ::4]); ax.axis("off")
        ax.set_title(f"#{F[i].index}  t={F[i].timestamp - t0:.1f}s", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "contact_sheet.jpg", dpi=100); plt.close(fig)

    # 2. rgb | depth | conf
    idx = pick(len(F), 4)
    fig, axs = plt.subplots(len(idx), 3, figsize=(15, 3.8 * len(idx)))
    cmap3 = matplotlib.colors.ListedColormap(["#d62728", "#ffbf00", "#2ca02c"])
    for r, i in enumerate(idx):
        f = F[i]
        axs[r, 0].imshow(f.rgb()[::4, ::4]); axs[r, 0].set_title(f"RGB #{f.index}")
        d = f.depth(); d[~np.isfinite(d)] = np.nan
        im = axs[r, 1].imshow(d, cmap="turbo", vmin=0.2, vmax=1.5); axs[r, 1].set_title("depth (m)")
        fig.colorbar(im, ax=axs[r, 1], fraction=0.046)
        axs[r, 2].imshow(f.confidence(), cmap=cmap3, vmin=0, vmax=2, interpolation="nearest")
        axs[r, 2].set_title("conf: red=0 yellow=1 green=2")
        for a in axs[r]: a.axis("off")
    fig.tight_layout(); fig.savefig(out / "rgbd_frames.jpg", dpi=80); plt.close(fig)

    # 3/4. fused cloud
    pts, cols = [], []
    for f in F:
        d = f.masked_depth(max_depth=1.5)
        v, u = np.nonzero(d > 0.05)
        z = d[v, u]
        K = f.K_depth
        p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
        pts.append((f.T_cw @ p)[:3].T)
        cols.append(cv2.resize(f.rgb(), f.depth_size, interpolation=cv2.INTER_AREA)[v, u] / 255.0)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.concatenate(pts))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    pc = pc.voxel_down_sample(0.003)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pc)
    P, C = np.asarray(pc.points), np.asarray(pc.colors)
    s = np.random.default_rng(0).permutation(len(P))[:400_000]; P, C = P[s], C[s]
    cam = np.array([f.T_cw[:3, 3] for f in F])
    # ARKit world is Y-up: top-down = X/Z plane
    fig = plt.figure(figsize=(16, 8))
    a = fig.add_subplot(121); a.scatter(P[:, 0], P[:, 2], c=C, s=0.3, marker=".")
    a.plot(cam[:, 0], cam[:, 2], "r-", lw=1); a.scatter(*cam[[0], :][:, [0, 2]].T, c="k", s=40, zorder=5)
    a.set_aspect("equal"); a.set_title(f"top-down, {len(pc.points)} pts, red=camera path (black=start)")
    b = fig.add_subplot(122, projection="3d"); b.scatter(P[:, 0], P[:, 2], P[:, 1], c=C, s=0.3, marker=".")
    b.view_init(35, -60); b.set_title("oblique"); b.set_box_aspect(np.ptp(P[:, [0, 2, 1]], axis=0))
    fig.tight_layout(); fig.savefig(out / "cloud.png", dpi=110); plt.close(fig)
    print(out, len(F), "frames", len(pc.points), "pts")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        preview(p)

"""Visuals for a real-meal run, from saved outputs (masks/, lift.npz, result.json).

    python pipeline/eval/real_meal_viz.py SCAN_DIR RESULT_DIR --truth "kiwi=95,sweet potato=280"
SCAN_DIR has rgb/poses (data/scans/<scan>); RESULT_DIR has result.json, masks/, lift.npz (rsync'd from the
cluster by scripts/bouchet/real_meal.sbatch). Writes RESULT_DIR/viz/{masks_grid,labels3d,volumes,summary}.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.fuse import fuse
from foodscan.labels import CLUTTER, CONTAINER, load_lift
from foodscan.loader import Scan
from foodscan.segment import load_masks
from foodscan.volume import crop_mesh_by_label, fit_support_plane_info, label_mesh_vertices

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
GRAY = "#9a9a9a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir"); ap.add_argument("result_dir")
    ap.add_argument("--truth", default="", help='"name=grams,..." matched by substring on the predicted name')
    ap.add_argument("--rename", default="")
    a = ap.parse_args()
    scan_dir, rd = Path(a.scan_dir), Path(a.result_dir)
    viz = rd / "viz"; viz.mkdir(exist_ok=True)
    truth = {k.strip(): float(v) for k, v in (p.split("=") for p in a.truth.split(",") if p)}
    res = json.loads((rd / "result.json").read_text())
    items = res["items"]
    if a.rename:   # "id=name,..." (SAM prompt names can collapse, e.g. kiwi -> fruit)
        for p in a.rename.split(","):
            i, n = p.split("="); [it.update(name=n) for it in items if it["id"] == int(i)]
    col = {it["id"]: PAL[i % len(PAL)] for i, it in enumerate(items)}
    gt = lambda name: next((g for k, g in truth.items() if k in name or name in k), None)

    # 1. mask overlay grid
    masks, meta = load_masks(scan_dir, rd / "masks")
    scan = Scan(scan_dir)
    fidx = sorted(masks)
    pick = [fidx[i] for i in np.linspace(0, len(fidx) - 1, min(6, len(fidx))).round().astype(int)]
    by_idx = {f.index: f for f in scan.frames}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, fi in zip(axs.flat, pick):
        img = by_idx[fi].rgb().astype(float) / 255
        m = masks[fi]
        if m.shape[:2] != img.shape[:2]:
            import cv2; m = cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        for lid, c in col.items():
            mm = m == lid
            img[mm] = 0.5 * img[mm] + 0.5 * np.array(matplotlib.colors.to_rgb(c))
        ax.imshow(img); ax.axis("off"); ax.set_title(f"frame {fi}")
    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=col[it["id"]], label=f"{it['id']}: {it['name']}")
                        for it in items], loc="lower center", ncol=len(items))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(viz / "masks_grid.png", dpi=100); plt.close(fig)

    # 2. cloud coloured by label (ARKit world: y up)
    L = load_lift(rd / "lift.npz"); xyz, lab = L["xyz"], L["lift_label"]
    keep = np.random.default_rng(0).permutation(len(xyz))[:150000]
    c = np.array([GRAY] * len(xyz), dtype=object)
    for lid, cc in col.items():
        c[lab == lid] = cc
    fig = plt.figure(figsize=(14, 6.5))
    near = np.linalg.norm(xyz - np.median(xyz[np.isin(lab, list(col))], 0), axis=1) < 0.3 if np.isin(lab, list(col)).any() else slice(None)
    k = keep[near[keep]] if not isinstance(near, slice) else keep
    for j, (el, az, t) in enumerate([(90, -90, "top-down"), (30, -60, "oblique")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.scatter(xyz[k, 0], xyz[k, 2], xyz[k, 1], c=list(c[k]), s=1.5, depthshade=False)
        ax.view_init(el, az); ax.set_title(t); ax.set_box_aspect((1, 1, 0.5)); ax.set_axis_off()
    fig.savefig(viz / "labels3d.png", dpi=100); plt.close(fig)

    # 3. per-item volume geometry (cropped mesh above the support)
    mesh, pcd = fuse(scan, voxel=res.get("voxel", 0.0015), stride=2)
    pts = np.asarray(pcd.points)
    plane, _ = fit_support_plane_info(pts, up_hint=np.array([0.0, 1.0, 0.0]))
    lab_m = lab.copy(); lab_m[(lab_m == CLUTTER) | (lab_m == CONTAINER)] = 0
    vl = label_mesh_vertices(mesh, xyz.astype(np.float64), lab_m, radius=0.003)
    fig = plt.figure(figsize=(6 * len(items), 6))
    for j, it in enumerate(items):
        sub, _ = crop_mesh_by_label(mesh, vl, it["id"], None, min_faces=50)
        V, T = np.asarray(sub.vertices), np.asarray(sub.triangles)
        ax = fig.add_subplot(1, len(items), j + 1, projection="3d")
        if len(T):   # drop stray far faces: keep faces within 8 cm of the vertex median
            T = T[np.linalg.norm(V[T].mean(1) - np.median(V[np.unique(T)], 0), axis=1) < 0.08]
        if len(T):
            ax.add_collection3d(Poly3DCollection(V[T][:, :, [0, 2, 1]], facecolor=col[it["id"]], edgecolor="k", linewidth=0.1, alpha=0.9))
            W = V[np.unique(T)]; lo, hi = W.min(0)[[0, 2, 1]], W.max(0)[[0, 2, 1]]; r = (hi - lo).max() / 2; mid = (hi + lo) / 2
            ax.set_xlim(mid[0] - r, mid[0] + r); ax.set_ylim(mid[1] - r, mid[1] + r); ax.set_zlim(mid[2] - r, mid[2] + r)
        g = gt(it["name"])
        ax.set_title(f"{it['name']}: {it['volume_ml']:.0f} mL\npred {it['mass_g']:.0f} g" + (f" vs true {g:.0f} g" if g else ""), fontsize=11)
        ax.view_init(25, -60)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(viz / "items.png", dpi=100); plt.close(fig)

    # 4. summary: the three images + a table
    fig = plt.figure(figsize=(16, 14))
    for i, n in enumerate(["masks_grid", "labels3d", "items"]):
        ax = fig.add_subplot(4, 1, i + 1); ax.imshow(plt.imread(viz / f"{n}.png")); ax.axis("off")
    rows = []
    for it in items:
        g = gt(it["name"]); m = it["mass_g"]
        rows.append([it["name"], f"{it['volume_ml']:.0f}", f"{it['density_used_g_ml']:.3f}", f"{m:.0f}",
                     f"{g:.0f}" if g else "-", f"{100 * (m - g) / g:+.0f}%" if g else "-"])
    ax = fig.add_subplot(4, 1, 4); ax.axis("off")
    ax.table(cellText=rows, colLabels=["item", "volume mL", "density g/mL", "pred g", "true g", "err"], loc="center").scale(1, 2)
    fig.suptitle(res["scan"]); fig.tight_layout(); fig.savefig(viz / "summary.png", dpi=80); plt.close(fig)
    print("wrote", viz)


if __name__ == "__main__":
    main()

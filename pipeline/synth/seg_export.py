"""Fused, labelled point clouds from synthetic scans: training / test data for 3D segmentation.

    python -m synth.seg_export ../data/synth/hard/scans [--voxel 0.0015] [--tol 0.004] [--crop 0.25]

For each scan with gt/points.npz: TSDF-fuse it exactly as run_meal does (noisy LiDAR,
ARKit poses), then give every fused point the label of the nearest ground-truth
surface sample (item id 1..K, 254 container, 253 clutter) if it lies within `tol`,
else 0 (table / unexplained). Writes SCAN/gt/seg_points.npz:

    xyz (N,3) float32, ARKit metres      rgb (N,3) uint8      normal (N,3) float32
    label (N,) uint8                     dist (N,) float32 to the nearest GT sample

cropped to `crop` m around the dish axis. `tol` should cover fusion error: TSDF
smoothing at the voxel size plus LiDAR noise (a few mm at 30 cm); points further
than that from every surface are fusion artefacts and stay 0. Runs in `foodseg`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from foodscan.fuse import fuse
from foodscan.loader import Scan


def export(scan_dir: Path, voxel: float = 0.0015, tol: float = 0.004, crop: float = 0.25) -> dict:
    gt = json.loads((scan_dir / "gt.json").read_text())
    z = np.load(scan_dir / "gt" / "points.npz")
    _, pcd = fuse(Scan(scan_dir), voxel=voxel, stride=2)
    pcd.estimate_normals()
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    nrm = np.asarray(pcd.normals)

    W2A = np.array(gt["world_to_arkit"])         # dish axis = Blender world origin
    local = (xyz - W2A[:3, 3]) @ W2A[:3, :3]     # back to the Z-up dish frame
    keep = np.hypot(local[:, 0], local[:, 1]) < crop
    xyz, rgb, nrm = xyz[keep], rgb[keep], nrm[keep]

    d, idx = cKDTree(z["xyz"]).query(xyz)
    label = np.where(d <= tol, z["label"][idx], 0).astype(np.uint8)
    np.savez_compressed(scan_dir / "gt" / "seg_points.npz", xyz=xyz.astype(np.float32), rgb=rgb,
                        normal=nrm.astype(np.float32), label=label, dist=d.astype(np.float32))
    food = label[(label > 0) & (label < 253)]
    return {"scan": scan_dir.name, "n_points": int(len(xyz)), "n_food": int(len(food)),
            "n_items": len(gt["items"]), "items_seen": int(len(np.unique(food))),
            "frac_unlabelled": float((label == 0).mean()), "median_dist_mm": float(np.median(d) * 1000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--tol", type=float, default=0.004)
    ap.add_argument("--crop", type=float, default=0.25)
    args = ap.parse_args()
    for scan in sorted(p for p in Path(args.scans).iterdir() if (p / "gt" / "points.npz").exists()):
        r = export(scan, args.voxel, args.tol, args.crop)
        print(f"{r['scan']}: {r['n_points']} pts, {r['n_food']} food, items {r['items_seen']}/{r['n_items']}, "
              f"unlabelled {100 * r['frac_unlabelled']:.1f} %, median dist {r['median_dist_mm']:.2f} mm")


if __name__ == "__main__":
    main()

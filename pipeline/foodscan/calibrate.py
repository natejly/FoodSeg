"""Calibration report: estimated vs true volume for scans of known objects.

    python -m foodscan.calibrate data/calibration.csv data/scans [--voxel 0.0015] [--out report.csv]

calibration.csv columns (see data/calibration.csv): object, true_volume_ml, scan_name, ...
Each scan is fused, the support plane fitted, the largest above-plane cluster
taken as the object, and every estimator in volume.py run on it. Prints a table
and per-estimator MAPE / bias, and writes the rows to --out.

Also runs the single-frame baseline on the middle frame of each scan, so the
calibration set doubles as the first RQ1 data point.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .fuse import fuse, single_frame_cloud
from .loader import Scan
from .volume import (cluster_objects, estimate_all, fit_support_plane, height_map_volume,
                     points_above_plane)

ESTIMATORS = ["mesh_prism_cm3", "height_map_cm3", "poisson_cm3", "convex_hull_cm3"]


def gravity_up(scan: Scan) -> np.ndarray:
    """World 'up' from ARKit's gravity alignment: +Y in ARKit world coordinates.

    Poses are converted per-camera, not per-world, so the world frame is still
    ARKit's (Y up). Returns the up vector in the same frame the point cloud is in.
    """
    return np.array([0.0, 1.0, 0.0])


def process_scan(scan_dir: Path, voxel: float, up: np.ndarray | None) -> dict:
    scan = Scan(scan_dir)
    mesh, pcd = fuse(scan, voxel=voxel, stride=2)
    pts = np.asarray(pcd.points)
    plane = fit_support_plane(pts, up_hint=up)
    clusters = cluster_objects(points_above_plane(pts, plane, min_height=0.003))
    if not clusters:
        return {"error": "no object above plane"}
    obj = clusters[0]
    res = estimate_all(obj, plane, mesh)

    mid = scan.good_frames()[len(scan.good_frames()) // 2]
    p1 = np.asarray(single_frame_cloud(mid).points)
    plane1 = fit_support_plane(p1, up_hint=up)
    c1 = cluster_objects(points_above_plane(p1, plane1, min_height=0.003))
    res["single_frame_height_map_cm3"] = height_map_volume(c1[0], plane1) if c1 else float("nan")
    res["n_frames"] = len(scan.good_frames())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_csv")
    ap.add_argument("scans_dir")
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--no-gravity", action="store_true", help="don't use ARKit +Y as up hint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.calibration_csv)))
    rows = [r for r in rows if r.get("scan_name")]
    out_rows = []
    for r in rows:
        scan_dir = Path(args.scans_dir) / r["scan_name"]
        if not scan_dir.exists():
            print(f"skip {r['scan_name']}: not found")
            continue
        truth = float(r["true_volume_ml"])
        up = None if args.no_gravity else gravity_up(Scan(scan_dir))
        res = process_scan(scan_dir, args.voxel, up)
        row = {"object": r["object"], "scan": r["scan_name"], "true_ml": truth, **res}
        for k in ESTIMATORS + ["single_frame_height_map_cm3"]:
            if k in res and np.isfinite(res[k]):
                row[k + "_err_pct"] = 100 * (res[k] - truth) / truth
        out_rows.append(row)
        print(f"{r['object']:16s} {r['scan_name']:28s} true {truth:7.1f}  "
              + "  ".join(f"{k.split('_cm3')[0]} {res.get(k, float('nan')):7.1f} ({row.get(k + '_err_pct', float('nan')):+5.1f}%)"
                          for k in ESTIMATORS + ["single_frame_height_map_cm3"]))

    if not out_rows:
        return
    print("\nMAPE / mean bias over all scans:")
    for k in ESTIMATORS + ["single_frame_height_map_cm3"]:
        errs = np.array([row[k + "_err_pct"] for row in out_rows if k + "_err_pct" in row])
        if len(errs):
            print(f"  {k:32s} MAPE {np.mean(np.abs(errs)):5.1f}%   bias {np.mean(errs):+5.1f}%   n={len(errs)}")

    if args.out:
        keys = sorted({k for row in out_rows for k in row})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

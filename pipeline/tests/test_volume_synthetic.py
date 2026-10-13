"""End-to-end check on the synthetic box scan: fuse -> plane -> cluster -> volume vs 72 cm^3.

    python tests/test_volume_synthetic.py [SCAN_DIR] [--voxel 0.003] [--noise 0.0005]
If SCAN_DIR is omitted a fresh synthetic scan is generated in a temp dir with --noise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.fuse import fuse, single_frame_cloud  # noqa: E402
from foodscan.loader import Scan  # noqa: E402
from foodscan.volume import (cluster_objects, estimate_all, fit_support_plane,  # noqa: E402
                             points_above_plane)

UP = np.array([0.0, 0.0, 1.0])  # synthetic world: table normal is +Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", nargs="?")
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--noise", type=float, default=0.0005, help="synthetic depth noise sigma (m)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--edge-smear", type=int, default=0, help="synthetic LiDAR edge smear (px)")
    ap.add_argument("--no-clean", action="store_true", help="disable fuse's edge/incidence filters")
    args = ap.parse_args()

    if args.scan_dir:
        scan_dir = Path(args.scan_dir)
    else:
        scan_dir = Path(tempfile.mkdtemp()) / "synth"
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_synthetic_scan.py"),
                        str(scan_dir), "--frames", str(args.frames), "--noise", str(args.noise),
                        "--edge-smear", str(args.edge_smear)],
                       check=True, stdout=subprocess.DEVNULL)

    scan = Scan(scan_dir)
    truth = scan.meta.get("boxVolumeCm3")
    print(f"scan {scan_dir.name}: {len(scan)} frames, truth {truth} cm^3, voxel {args.voxel*1000:.1f} mm")

    print("== full scan (TSDF) ==")
    off = dict(edge_rel=0, max_incidence=0) if args.no_clean else {}
    mesh, pcd = fuse(scan, voxel=args.voxel, stride=1, **off)
    pts = np.asarray(pcd.points)
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 0.2]      # plate-sized region
    plane = fit_support_plane(pts, up_hint=UP)
    obj = cluster_objects(points_above_plane(pts, plane, min_height=0.003))[0]
    report(estimate_all(obj, plane, mesh), truth)

    print("== single RGB-D frame (frame 0) ==")
    p1 = np.asarray(single_frame_cloud(scan.frames[0]).points)
    p1 = p1[np.linalg.norm(p1[:, :2], axis=1) < 0.2]
    plane1 = fit_support_plane(p1, up_hint=UP)
    obj1 = cluster_objects(points_above_plane(p1, plane1, min_height=0.003))[0]
    report(estimate_all(obj1, plane1), truth)


def report(res: dict, truth: float | None) -> None:
    for k, v in res.items():
        line = f"  {k:18s} {v:10.2f}"
        if truth and k.endswith("_cm3"):
            line += f"   err {100 * (v - truth) / truth:+6.1f}%"
        print(line)


if __name__ == "__main__":
    main()

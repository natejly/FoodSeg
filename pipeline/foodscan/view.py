"""Quick viewer: python -m foodscan.view SCAN_DIR/fused.ply [more.ply ...]

Or compare a single frame against the fused scan:
    python -m foodscan.view SCAN_DIR --frame 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from .fuse import single_frame_cloud
from .loader import Scan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help=".ply files, or a scan dir with --frame")
    ap.add_argument("--frame", type=int, default=None, help="overlay this frame's RGB-D cloud (red)")
    args = ap.parse_args()

    geoms = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            scan = Scan(p)
            fused = p / "fused.ply"
            if fused.exists():
                geoms.append(o3d.io.read_triangle_mesh(str(fused)))
            if args.frame is not None:
                pcd = single_frame_cloud(scan.frames[args.frame])
                pcd.paint_uniform_color([1.0, 0.2, 0.2])
                geoms.append(pcd)
        elif p.suffix == ".ply":
            m = o3d.io.read_triangle_mesh(str(p))
            geoms.append(m if len(m.triangles) else o3d.io.read_point_cloud(str(p)))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()

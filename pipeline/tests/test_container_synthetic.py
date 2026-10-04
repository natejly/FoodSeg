"""Bowl test: empty-bowl template + filled-bowl scan -> food volume via the registered
template, versus the naive 'above the table plane' estimate (which cannot see food
below the rim at all).

    python tests/test_container_synthetic.py [EMPTY_DIR FOOD_DIR]
Generates both scans in a temp dir if not given (12 mm fill = 40.7 cm^3).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.container import ContainerTemplate  # noqa: E402
from foodscan.fuse import fuse  # noqa: E402
from foodscan.loader import Scan  # noqa: E402
from foodscan.volume import (cluster_objects, crop_mesh_to_points, fit_support_plane,  # noqa: E402
                             mesh_prism_volume, points_above_plane)

UP = np.array([0.0, 0.0, 1.0])


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def main() -> None:
    if len(sys.argv) >= 3:
        empty_dir, food_dir = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        tmp = Path(tempfile.mkdtemp())
        empty_dir, food_dir = tmp / "bowl_empty", tmp / "bowl_food"
        gen = Path(__file__).parent / "make_synthetic_scan.py"
        subprocess.run([sys.executable, str(gen), str(empty_dir), "--scene", "bowl", "--noise", "0.0005"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(gen), str(food_dir), "--scene", "bowl", "--fill", "0.012",
                        "--noise", "0.0005"], check=True, stdout=subprocess.DEVNULL)

    truth = Scan(food_dir).meta["foodVolumeCm3"]
    print(f"true food volume {truth:.1f} cm^3 (bowl capacity {Scan(food_dir).meta['bowlCapacityCm3']:.1f})")

    # 1. template from the empty scan
    tmpl = ContainerTemplate.from_scan(empty_dir, up_hint=UP)
    print(f"template: {len(tmpl.points)} points")

    # 2. scene: filled bowl; perturb the whole scene by a known rigid motion so
    #    registration has to do real work (the synthetic scans share a frame)
    mesh, pcd = fuse(Scan(food_dir), voxel=0.0015, stride=1)
    pts = np.asarray(pcd.points)
    T_pert = rot_z(17.0)
    T_pert[:3, 3] = [0.012, -0.008, 0.0]
    pts = pts @ T_pert[:3, :3].T + T_pert[:3, 3]
    mesh.transform(T_pert)
    plane = fit_support_plane(pts, up_hint=UP)

    # 3. register template, build support surface
    T = tmpl.register(pts, plane)
    print(f"ICP fitness {tmpl.last_fitness:.3f}, rmse {tmpl.last_rmse*1000:.2f} mm; "
          f"template centre landed at {np.round(T[:2, 3]*1000, 1)} mm (expected ~{np.round(T_pert[:2, 3]*1000, 1)})")
    support = tmpl.support(T, plane)

    # 4. food = points above the support surface
    food = cluster_objects(support.points_above(pts, min_height=0.002))
    food = food[0] if food else np.empty((0, 3))
    sub = crop_mesh_to_points(mesh, food)
    v_tmpl = support.mesh_prism_volume(sub)
    v_tmpl_hm = support.height_map_volume(food)

    # 5. naive: above the table plane
    above = cluster_objects(points_above_plane(pts, plane, min_height=0.002))
    v_plane = mesh_prism_volume(crop_mesh_to_points(mesh, above[0]), plane) if above else 0.0

    print(f"template-relative mesh prism : {v_tmpl:7.1f} cm^3  err {100*(v_tmpl-truth)/truth:+.1f}%")
    print(f"template-relative height map : {v_tmpl_hm:7.1f} cm^3  err {100*(v_tmpl_hm-truth)/truth:+.1f}%")
    print(f"naive above-plane            : {v_plane:7.1f} cm^3  err {100*(v_plane-truth)/truth:+.1f}%")


if __name__ == "__main__":
    main()

"""End-to-end: scan directory -> per-item name, volume, mass, kcal, macros.

    python -m foodscan.run_meal SCAN_DIR [--container data/containers/bowl_A.npz]
                                [--masks SCAN_DIR/masks] [--names "a,b,c"] [--no-nutrition]
                                [--poses refined] [--depth depth_hr] [--out result.json]

Stages (each is a module you can run alone):
  1. fuse            TSDF mesh + point cloud                       fuse.py
  2. support         table plane, or registered container template  volume.py / container.py
  3. names           --names, or masks/instances.json, or a VLM      naming.py
  4. masks -> 3D     per-point label votes                          labels.py  (needs masks/ from segment.py)
                     fallback without masks: the cluster(s) nearest the orbit's look-at point
  5. volume          mesh-prism per item, relative to the support   volume.py / container.py
  6. nutrition       density -> mass -> FNDDS nutrients             nutrition.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .container import ContainerTemplate
from .fuse import fuse
from .labels import decide, lift_masks, split_by_label
from .loader import Scan
from .volume import (Plane, cluster_objects, crop_mesh_to_points, fit_support_plane,
                     mesh_prism_volume, points_above_plane)

ARKIT_UP = np.array([0.0, 1.0, 0.0])


def look_at(scan: Scan) -> np.ndarray:
    """Least-squares point closest to every camera's optical axis (the orbit centre)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for f in scan.good_frames():
        c, d = f.T_cw[:3, 3], f.T_cw[:3, 2]   # OpenCV: +Z forward
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A, b)


def run(scan_dir: Path, container: Path | None = None, masks_dir: Path | None = None,
        names: list[str] | None = None, nutrition: bool = True, poses: str | None = None,
        depth: str = "depth", voxel: float = 0.0015, up_hint: np.ndarray | None = ARKIT_UP) -> dict:
    scan = Scan(scan_dir, poses=poses, depth=depth)
    mesh, pcd = fuse(scan, voxel=voxel, stride=2)
    pts = np.asarray(pcd.points)
    plane = fit_support_plane(pts, up_hint=up_hint)

    # --- support surface --------------------------------------------------
    support = None
    if container:
        tmpl = ContainerTemplate.load(container)
        T = tmpl.register(pts, plane)
        support = tmpl.support(T, plane)
        above = support.points_above(pts, min_height=0.002)
        reg_info = {"container": tmpl.name, "icp_fitness": tmpl.last_fitness, "icp_rmse_mm": tmpl.last_rmse * 1000,
                    "covered_fraction": getattr(tmpl, "covered_fraction", None)}
    else:
        above = points_above_plane(pts, plane, min_height=0.003)
        reg_info = {"container": "plane"}

    def item_volume(item_pts: np.ndarray) -> float:
        sub = crop_mesh_to_points(mesh, item_pts)
        return support.mesh_prism_volume(sub) if support else mesh_prism_volume(sub, plane)

    # --- items --------------------------------------------------------------
    items = []
    masks_dir = masks_dir or (scan_dir / "masks")
    if masks_dir.exists() and any(masks_dir.glob("*.png")):
        from .segment import load_masks
        masks, meta = load_masks(scan_dir if masks_dir == scan_dir / "masks" else masks_dir.parent)
        votes = lift_masks(scan, masks, pts, n_labels=max(meta) if meta else None)
        labels, margin = decide(votes)
        for lid, ipts in split_by_label(pts, labels).items():
            # keep only the part of the item that is food (above the support)
            if support:
                ipts = support.points_above(ipts, min_height=0.001)
            else:
                ipts = points_above_plane(ipts, plane, min_height=0.001)
            if len(ipts) < 50:
                continue
            items.append({"id": lid, "name": meta.get(lid, {}).get("name", f"item_{lid}"),
                          "mask_score": meta.get(lid, {}).get("mean_score"),
                          "label_margin": float(margin[labels == lid].mean()),
                          "n_points": int(len(ipts)), "volume_ml": item_volume(ipts)})
    else:
        # The whole room within max_depth is fused, so keep only what the user orbited:
        # one item without names, else len(names) clusters, nearest the look-at point first.
        X = plane.project_2d(look_at(scan)[None])[0]
        clusters = sorted(cluster_objects(above),
                          key=lambda c: np.linalg.norm(plane.project_2d(c).mean(0) - X))
        names = names or ["item_1"]
        for i, ipts in enumerate(clusters[:len(names)]):
            items.append({"id": i + 1, "name": names[i], "n_points": int(len(ipts)),
                          "volume_ml": item_volume(ipts)})

    for it in items:
        if it["volume_ml"] <= 0:  # mostly downward-facing triangles: overhang/clutter, not food on the support
            it["warning"] = "non-positive volume"

    # --- nutrition ----------------------------------------------------------
    if nutrition and items:
        from .nutrition import Nutrition
        nut = Nutrition(api_key=os.environ.get("FDC_API_KEY", "DEMO_KEY"))
        for it in items:
            try:
                food = nut.lookup(it["name"])
                est = nut.estimate(food, it["volume_ml"])
                it.update({"fdc_id": est.fdc_id, "fdc_description": est.fdc_description,
                           "density_g_per_ml": est.density_g_per_ml, "mass_g": est.mass_g,
                           "kcal": est.kcal, "protein_g": est.protein_g, "carb_g": est.carb_g, "fat_g": est.fat_g})
            except Exception as e:  # noqa: BLE001
                it["nutrition_error"] = str(e)

    totals = {k: float(sum(it.get(k, 0.0) for it in items)) for k in ("volume_ml", "mass_g", "kcal", "protein_g", "carb_g", "fat_g")}
    return {"scan": scan_dir.name, "frames": len(scan.good_frames()), "support": reg_info,
            "plane_normal": plane.normal.tolist(), "items": items, "totals": totals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--container", default=None, help="template .npz from ContainerTemplate.save")
    ap.add_argument("--masks", default=None, help="masks dir (default SCAN_DIR/masks)")
    ap.add_argument("--names", default=None, help="comma-separated names when running without masks")
    ap.add_argument("--no-nutrition", action="store_true")
    ap.add_argument("--poses", default=None, choices=[None, "refined"])
    ap.add_argument("--depth", default="depth", choices=["depth", "depth_hr"])
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--up", default="y", choices=["y", "z"],
                    help="world up axis: y for ARKit gravity-aligned scans (default), z for the synthetic scans. "
                         "Never rely on the plane fitter's own guess: bowls have more points below the table than above.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(Path(args.scan_dir), container=Path(args.container) if args.container else None,
              masks_dir=Path(args.masks) if args.masks else None,
              names=[s.strip() for s in args.names.split(",")] if args.names else None,
              nutrition=not args.no_nutrition, poses=args.poses, depth=args.depth, voxel=args.voxel,
              up_hint=ARKIT_UP if args.up == "y" else np.array([0.0, 0.0, 1.0]))
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()

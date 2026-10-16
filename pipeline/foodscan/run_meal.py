"""End-to-end: scan directory -> per-item name, volume, mass, kcal, macros.

    python -m foodscan.run_meal SCAN_DIR [--container data/containers/bowl_A.npz]
                                [--masks SCAN_DIR/masks] [--names "a,b,c"] [--no-nutrition] [--mass-rule envelope|legacy]
                                [--poses refined] [--depth depth_hr] [--out result.json]

Stages (each is a module you can run alone):
  1. fuse            TSDF mesh + point cloud                       fuse.py
  2. support         table plane, or registered container template  volume.py / container.py
  3. names           --names, or masks/instances.json, or a VLM      naming.py
  4. masks -> 3D     per-point label votes                          labels.py  (needs masks/ from segment.py)
                     writes SCAN_DIR/lift.npz (lift_label/share/nvotes/agree, see labels.py);
                     --seg3d-ckpt CKPT would refine it with pipeline.seg3d.eval.refine
                     --segment runs segment.py first (default --seg-mode image --linker covote)
                     fallback without masks: the cluster(s) nearest the orbit's look-at point
  5. volume          mesh-prism per item, relative to the support   volume.py / container.py
  6. mass+nutrition envelope volume x bulk density (--mass-rule envelope, default; leaf/topping: solid x
                     material or thin-layer) or legacy densities.csv -> FNDDS nutrients   density_table.py, nutrition.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .container import ContainerTemplate
from .fuse import fuse
from .labels import CLUTTER, CONTAINER, DEFAULT_LINKED_WEIGHTING, estimate_normals, lift_for_meal, split_by_label
from .loader import Scan
from .density_table import estimate_mass
from .volume import (Plane, cluster_objects, crop_mesh_by_label, crop_mesh_to_points, fit_support_plane_info,
                     item_volume_result, label_mesh_vertices, mesh_prism_volume, points_above_plane)

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
        depth: str = "depth", voxel: float = 0.0015, up_hint: np.ndarray | None = ARKIT_UP,
        seg3d_ckpt: Path | None = None, repair_ids: bool = True, smooth_radius: float = 0.0,
        crop_radius: float = 0.003, min_label_faces: int = 50, min_item_points: int = 50,
        mass_rule: str = "envelope", weighting: str | None = None,
        segment: bool = False, seg_mode: str = "image", linker: str = "covote") -> dict:
    if mass_rule not in ("envelope", "legacy"):
        raise ValueError(f"mass_rule must be 'envelope' or 'legacy', got {mass_rule!r}")
    scan = Scan(scan_dir, poses=poses, depth=depth)
    mesh, pcd = fuse(scan, voxel=voxel, stride=2)
    pts = np.asarray(pcd.points)
    plane, plane_info = fit_support_plane_info(pts, up_hint=up_hint)
    if plane_info.get("gravity_fallback"):
        print(f"[run_meal] warning: fitted support plane is {plane_info['tilt_deg']:.1f} deg off gravity; "
              "using the gravity-aligned plane through the lowest band")

    # --- support surface --------------------------------------------------
    support = None
    if container:
        tmpl = ContainerTemplate.load(container)
        T = tmpl.register(pts, plane)
        reg_info = {"container": tmpl.name, "icp_fitness": tmpl.last_fitness, "icp_rmse_mm": tmpl.last_rmse * 1000,
                    "covered_fraction": getattr(tmpl, "covered_fraction", None),
                    "container_fit_ok": tmpl.last_ok}
        if tmpl.last_ok:
            support = tmpl.support(T, plane)
            above = support.points_above(pts, min_height=0.002)
        else:
            print(f"[run_meal] warning: container registration failed its gate (fitness {tmpl.last_fitness:.2f}, "
                  f"rmse {tmpl.last_rmse * 1000:.1f} mm); falling back to the fitted plane support")
            reg_info["warning"] = "container fit failed; plane support used"
            above = points_above_plane(pts, plane, min_height=0.003)
    else:
        above = points_above_plane(pts, plane, min_height=0.003)
        reg_info = {"container": "plane"}
    reg_info["plane"] = {k: v for k, v in plane_info.items() if k != "n_inliers"} | {"n_inliers": int(plane_info["n_inliers"])}

    vertex_labels = None      # set once labels exist; None -> bbox crop (no labels available)

    def item_volume(item_pts: np.ndarray, item_id: int | None = None) -> tuple[float, dict]:
        if vertex_labels is not None and item_id is not None:
            sub, crop = crop_mesh_by_label(mesh, vertex_labels, item_id, item_pts, min_faces=min_label_faces)
        else:
            sub = crop_mesh_to_points(mesh, item_pts)
            crop = {"crop": "bbox", "n_faces": len(sub.triangles), "label_faces": None}
        vol = support.mesh_prism_volume(sub) if support else mesh_prism_volume(sub, plane)
        return vol, crop

    # --- items --------------------------------------------------------------
    items, dropped = [], []
    masks_dir = masks_dir or (scan_dir / "masks")
    if segment:    # SAM 3 (GPU) -> masks/; image mode links per-frame instances on this fused cloud
        from .segment import make_linker, segment_scan
        seg_normals = np.asarray(pcd.normals) if pcd.has_normals() else estimate_normals(pts)
        segment_scan(scan_dir, names or [], mode=seg_mode, out_dir=masks_dir,
                     linker=make_linker(scan_dir, pts, seg_normals, method=linker) if seg_mode == "image" else None)
    if masks_dir.exists() and any(masks_dir.glob("*.png")):
        from .segment import load_masks
        masks, meta = load_masks(scan_dir if masks_dir == scan_dir / "masks" else masks_dir.parent)
        # normals: the TSDF cloud has none, and without them the grazing gate and cos weighting are off
        normals = np.asarray(pcd.normals) if pcd.has_normals() else estimate_normals(pts)
        if weighting is None:      # co-vote-linked masks (image mode default) take the quality weight (lift_sweep)
            try:
                link = json.loads((masks_dir / "concepts.json").read_text()).get("link") or {}
            except (OSError, ValueError):
                link = {}
            weighting = DEFAULT_LINKED_WEIGHTING if link.get("linker") == "covote" else "cos"
        lift = lift_for_meal(scan, masks, pts, normals=normals, meta=meta, repair=repair_ids,
                             smooth_radius=smooth_radius, weighting=weighting, save_to=scan_dir / "lift.npz")
        labels = lift.lift_label.copy()
        margin = lift.lift_share
        reg_info["lift"] = {"weighting": weighting, "merged_ids": lift.merge_map, "frames": lift.stats["n_frames"],
                            "agree_fraction": float(lift.lift_agree[lift.lift_nvotes > 0].mean())
                            if (lift.lift_nvotes > 0).any() else None}
        if seg3d_ckpt is not None:
            try:
                import inspect

                try:
                    from pipeline.seg3d.eval import refine   # run from the repo root
                except ModuleNotFoundError:
                    from seg3d.eval import refine            # run from pipeline/ (python -m foodscan.run_meal)
                kw = {"ckpt": seg3d_ckpt} if "ckpt" in inspect.signature(refine).parameters else {}
                refined = np.asarray(refine(str(scan_dir / "lift.npz"), pts, **kw))
                if refined.shape != (len(pts),):
                    raise ValueError(f"refine returned shape {refined.shape}, expected ({len(pts)},)")
                labels = refined.astype(np.int32)
                reg_info["seg3d"] = str(seg3d_ckpt)
            except (ImportError, ValueError) as e:
                print(f"[run_meal] --seg3d-ckpt given but refine is unavailable/unusable ({e}); "
                      "using the lift decision")
        # 253 (clutter) / 254 (container) are classes for the refiner, never food items
        labels[(labels == CLUTTER) | (labels == CONTAINER)] = 0
        vertex_labels = label_mesh_vertices(mesh, pts, labels, radius=crop_radius)
        for lid, ipts in split_by_label(pts, labels).items():
            # keep only the part of the item that is food (above the support)
            if support:
                ipts = support.points_above(ipts, min_height=0.001)
            else:
                ipts = points_above_plane(ipts, plane, min_height=0.001)
            if len(ipts) < min_item_points:
                dropped.append({"id": lid, "name": meta.get(lid, {}).get("name", f"item_{lid}"),
                                "n_points": int(len(ipts))})
                continue
            vol, crop = item_volume(ipts, lid)
            items.append({"id": lid, "name": meta.get(lid, {}).get("name", f"item_{lid}"),
                          "mask_score": meta.get(lid, {}).get("mean_score"),
                          "label_margin": float(margin[labels == lid].mean()),
                          "agree_fraction": float(lift.lift_agree[labels == lid].mean()),
                          "n_points": int(len(ipts)), "crop": crop["crop"],
                          "crop_faces": crop["n_faces"], "crop_label_faces": crop["label_faces"],
                          **item_volume_result(vol, ipts, support or plane,
                                               meta.get(lid, {}).get("name", f"item_{lid}"),
                                               meta.get(lid, {}).get("kind"))})
        if dropped:
            print(f"[run_meal] warning: dropped {len(dropped)} item(s) under {min_item_points} points: "
                  + ", ".join(f"{d['name']} ({d['n_points']})" for d in dropped))
    else:
        # The whole room within max_depth is fused, so keep only what the user orbited:
        # one item without names, else len(names) clusters, nearest the look-at point first.
        X = plane.project_2d(look_at(scan)[None])[0]
        clusters = sorted(cluster_objects(above),
                          key=lambda c: np.linalg.norm(plane.project_2d(c).mean(0) - X))
        names = names or ["item_1"]
        for i, ipts in enumerate(clusters[:len(names)]):
            vol, crop = item_volume(ipts)
            items.append({"id": i + 1, "name": names[i], "n_points": int(len(ipts)),
                          "crop": crop["crop"], "crop_faces": crop["n_faces"], "crop_label_faces": None,
                          **item_volume_result(vol, ipts, support or plane, names[i])})

    for it in items:
        if it["volume_ml"] <= 0:  # mostly downward-facing triangles: overhang/clutter, not food on the support
            it["warning"] = "non-positive volume"

    # --- mass + nutrition -----------------------------------------------------
    # mass_rule 'envelope': mass from volume_envelope_ml x bulk density (leaf / topping: solid estimate x
    # material density, or the thin-layer rule), see density_table.estimate_mass; per-item rule, densities,
    # sources and mass_confidence are recorded. 'legacy': the old Nutrition.estimate(volume_ml) path.
    if mass_rule == "envelope":
        for it in items:
            m = estimate_mass(it["name"], it["kind"], it["volume_envelope_ml"], it["footprint_cm2"], it["height_mm"])
            it.update({k: v for k, v in m.items() if k not in ("kind", "kind_source", "envelope_ratio",
                                                                "volume_solid_est_ml")})
            it["density_g_per_ml"] = m["density_used_g_ml"]
            if it["volume_envelope_ml"] <= 0:
                it["mass_confidence"] = 0.0
    if nutrition and items:
        from .nutrition import Nutrition
        nut = Nutrition(api_key=os.environ.get("FDC_API_KEY", "DEMO_KEY"))
        for it in items:
            try:
                food = nut.lookup(it["name"])     # same name the mass step used
                if mass_rule == "legacy":
                    est = nut.estimate(food, it["volume_ml"])
                else:
                    est = nut.estimate_from_mass(food, it["mass_g"], it["volume_envelope_ml"])
                it.update({"fdc_id": est.fdc_id, "fdc_description": est.fdc_description,
                           "kcal": est.kcal, "protein_g": est.protein_g, "carb_g": est.carb_g, "fat_g": est.fat_g})
                if mass_rule == "legacy":
                    it.update({"density_g_per_ml": est.density_g_per_ml, "mass_g": est.mass_g})
            except Exception as e:  # noqa: BLE001
                it["nutrition_error"] = str(e)

    totals = {k: float(sum(it.get(k, 0.0) for it in items)) for k in ("volume_ml", "volume_envelope_ml", "volume_solid_est_ml", "mass_g", "kcal", "protein_g", "carb_g", "fat_g")}
    return {"scan": scan_dir.name, "mass_rule": mass_rule, "frames": len(scan.good_frames()), "support": reg_info,
            "plane_normal": plane.normal.tolist(), "container_fit_ok": reg_info.get("container_fit_ok"),
            "dropped_items": dropped, "items": items, "totals": totals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--container", default=None, help="template .npz from ContainerTemplate.save")
    ap.add_argument("--masks", default=None, help="masks dir (default SCAN_DIR/masks)")
    ap.add_argument("--names", default=None, help="comma-separated names when running without masks")
    ap.add_argument("--seg3d-ckpt", default=None,
                    help="3D refiner checkpoint; calls pipeline.seg3d.eval.refine(lift.npz, cloud) if that module exists")
    ap.add_argument("--segment", action="store_true",
                    help="run SAM 3 segmentation first (GPU; --names are the prompts) instead of reading existing masks/")
    ap.add_argument("--seg-mode", default="image", choices=["image", "video"],
                    help="with --segment: image (default; per-frame SAM 3 + 3D linking) or video (tracker)")
    ap.add_argument("--linker", default="covote", choices=["covote", "repair"],
                    help="image mode id linking: covote (default, co-vote clustering + cos+qual lift weights) "
                         "or repair (labels.repair_ids, cos weights)")
    ap.add_argument("--weighting", default=None,
                    help="lift vote weighting (default: cos+qual when the masks were co-vote linked, else cos)")
    ap.add_argument("--no-id-repair", action="store_true", help="skip cross-view mask-id merging")
    ap.add_argument("--smooth-radius", type=float, default=0.0, help="neighbour vote smoothing radius in m (0 = off; 0.004 typical)")
    ap.add_argument("--no-nutrition", action="store_true")
    ap.add_argument("--mass-rule", default="envelope", choices=["envelope", "legacy"],
                    help="envelope (default): mass = envelope volume x bulk density, leaf/topping via solid "
                         "estimate x material density or the thin-layer rule; legacy: volume x densities.csv "
                         "(needs nutrition, i.e. no --no-nutrition, to produce a mass)")
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
              up_hint=ARKIT_UP if args.up == "y" else np.array([0.0, 0.0, 1.0]),
              seg3d_ckpt=Path(args.seg3d_ckpt) if args.seg3d_ckpt else None,
              repair_ids=not args.no_id_repair, smooth_radius=args.smooth_radius,
              mass_rule=args.mass_rule, weighting=args.weighting,
              segment=args.segment, seg_mode=args.seg_mode, linker=args.linker)
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()

"""Fused, labelled point clouds from synthetic scans: training / test data for 3D segmentation.

    python -m synth.seg_export ../data/synth/hard/scans [--voxel 0.0015] [--tol 0.004] [--crop 0.25]

For each scan with gt/points.npz: TSDF-fuse it exactly as run_meal does (noisy LiDAR,
ARKit poses), then give every fused point the label of the nearest ground-truth
surface sample (item id 1..K, 254 container, 253 clutter) if it lies within `tol`,
else 0 (unexplained fusion artefact). 252 = table plane. Writes SCAN/gt/seg_points.npz:

    xyz (N,3) float32, ARKit metres      rgb (N,3) uint8      normal (N,3) float32
    label (N,) uint8                     dist (N,) float32 to the nearest GT sample
    lift_label (N,) int32   scene-local id from visibility-gated voting of per-frame masks
                            (pseudo-SAM by default, see synth.votes); 0 = background / no votes
    lift_share (N,) float32 fraction of the point's votes for lift_label (0 if no votes)
    lift_nvotes (N,) int16  frames that voted for the point
    boundary (N,) uint8     1 if GT surfaces of two labels lie within `tol` of the point

It also backfills/updates gt.json (items: mass_g/mass_source, bbox_arkit, n_gt_points,
volume_above_container_ml, volume_above_table_ml, n_fused_points, occluded_fraction; votes params).

The fused cloud is first rigidly ICP-aligned to GT (labelling only; saved xyz stays in the tracker frame, correction in
gt.json seg_export.icp_correction) because exported poses carry drift. Cropped to `crop` m around the dish axis. tol = clip(2.5 x post-ICP median, 4, 8 mm) unless --tol is given; it should cover fusion error: TSDF
smoothing at the voxel size plus LiDAR noise (a few mm at 30 cm); points further
than that from every surface are fusion artefacts and stay 0. Runs in `foodseg`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import votes as V
from .itemstats import TABLE_ID, item_gt_stats, with_table


TOL_MIN, TOL_MAX, TOL_K = 0.004, 0.008, 2.5     # tol = clip(TOL_K * post-ICP median, TOL_MIN, TOL_MAX)
ICP_MAX_CORR = 0.010
ICP_MAX_SHIFT = 0.015      # larger scene displacement (95th pct) / rotation = ICP wandered on a symmetric dish: identity
ICP_MAX_ROT_DEG = 3.0
ICP_SKIP_BELOW = 0.0015    # pre-ICP median already at the fusion noise floor (clean poses): ICP can only slide, skip
ICP_MIN_GAIN = 0.9         # accept only if post median < this * pre


def estimate_correction(xyz: np.ndarray, gxyz: np.ndarray, glab: np.ndarray) -> tuple[np.ndarray, dict]:
    """Robust point-to-plane ICP of the fused cloud onto GT food + container samples (labels < 252).
    Returns (4x4 fused->GT correction, stats with pre/post median distance in metres over the
    fused points within ICP_MAX_CORR of that GT before correction)."""
    import open3d as o3d
    g = gxyz[glab < 252]
    ident = np.eye(4)
    if len(g) < 100 or len(xyz) < 100:
        return ident, {"pre": float("nan"), "post": float("nan"), "n": 0, "fitness": 0.0, "ok": False, "shift": 0.0}
    tree = cKDTree(g)
    d0, _ = tree.query(xyz, distance_upper_bound=ICP_MAX_CORR)
    sel = np.isfinite(d0)
    if sel.sum() < 100:
        return ident, {"pre": float("nan"), "post": float("nan"), "n": int(sel.sum()), "fitness": 0.0, "ok": False, "shift": 0.0}
    if len(g) > 300_000:
        g = g[np.random.default_rng(0).choice(len(g), 300_000, replace=False)]
        tree = cKDTree(g)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(g))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[sel]))
    pre = float(np.median(d0[sel]))
    if pre < ICP_SKIP_BELOW:
        return ident, {"pre": pre, "post": pre, "n": int(sel.sum()), "fitness": 1.0, "ok": True, "shift": 0.0,
                       "skipped": True}
    T = ident
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=0.004))
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
    for mc in (ICP_MAX_CORR, 0.006, 0.004):
        r = o3d.pipelines.registration.registration_icp(src, tgt, mc, T, est, crit)
        T = np.asarray(r.transformation)
    pts = xyz[sel]
    moved = pts @ T[:3, :3].T + T[:3, 3]
    post = float(np.median(tree.query(moved)[0]))
    shift = float(np.percentile(np.linalg.norm(moved - pts, axis=1), 95))   # displacement at the scene, not at the ARKit origin
    rot = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    ok = bool(shift < ICP_MAX_SHIFT and rot < ICP_MAX_ROT_DEG and post < ICP_MIN_GAIN * pre)
    if not ok:
        T, post = ident, pre
    return T, {"pre": pre, "post": post, "n": int(sel.sum()), "fitness": float(r.fitness), "ok": ok, "shift": shift if ok else 0.0}


def align_and_label(xyz: np.ndarray, nrm: np.ndarray, gxyz: np.ndarray, glab: np.ndarray,
                    tol: float | None = None) -> dict:
    """ICP-correct the fused cloud (for labelling only), then label by nearest GT surface within
    tol = clip(TOL_K * post-ICP median, TOL_MIN, TOL_MAX) (or the fixed `tol`). Returns the
    corrected xyz/normals, label, dist, boundary, T, tol and ICP stats."""
    T, st = estimate_correction(xyz, gxyz, glab)
    cx = xyz @ T[:3, :3].T + T[:3, 3]
    cn = nrm @ T[:3, :3].T
    if tol is None:
        post = st["post"] if np.isfinite(st["post"]) else 0.0
        tol = float(np.clip(TOL_K * post, TOL_MIN, TOL_MAX))
    dmin = np.full(len(cx), np.inf)
    label = np.zeros(len(cx), np.uint8)
    n_close = np.zeros(len(cx), np.int8)
    for lid in np.unique(glab):
        d, _ = cKDTree(gxyz[glab == lid]).query(cx, distance_upper_bound=tol)
        n_close += (d <= tol)
        better = d < dmin
        dmin[better], label[better] = d[better], lid
    d_all, _ = cKDTree(gxyz).query(cx)
    label[d_all > tol] = 0                       # fusion artefacts / unexplained stay 0
    return {"xyz": cx, "normal": cn, "label": label, "dist": d_all, "boundary": (n_close >= 2).astype(np.uint8),
            "T": T, "tol": tol, "icp": st}


def export(scan_dir: Path, voxel: float = 0.0015, tol: float | None = None, crop: float = 0.25,
           votes: str = "pseudo") -> dict:
    from foodscan.fuse import fuse
    from foodscan.loader import Scan
    tol_in = tol
    gt = json.loads((scan_dir / "gt.json").read_text())
    z = np.load(scan_dir / "gt" / "points.npz")
    gxyz, glab = with_table(z["xyz"], z["label"], np.array(gt["world_to_arkit"]))   # older scans lack 252
    scan = Scan(scan_dir)
    _, pcd = fuse(scan, voxel=voxel, stride=2)
    pcd.estimate_normals()
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    nrm = np.asarray(pcd.normals)

    W2A = np.array(gt["world_to_arkit"])         # dish axis = Blender world origin
    local = (xyz - W2A[:3, 3]) @ W2A[:3, :3]     # back to the Z-up dish frame
    keep = np.hypot(local[:, 0], local[:, 1]) < crop
    xyz, rgb, nrm = xyz[keep], rgb[keep], nrm[keep]

    al = align_and_label(xyz, nrm, gxyz, glab, tol)     # labelling uses the ICP-corrected cloud; saved xyz does not
    tol, label, d_all, boundary, cxyz = al["tol"], al["label"], al["dist"], al["boundary"], al["xyz"]

    item_ids = [it["id"] for it in gt["items"]]
    masks, vparams = V.get_masks(scan_dir, votes, gt.get("provenance", {}).get("seed", 0) * 100003
                                 + int(gt["scene_id"].lstrip("s") or 0), item_ids)
    ll, ls, lnv = V.lift(scan, xyz, nrm, masks)
    np.savez_compressed(scan_dir / "gt" / "seg_points.npz", xyz=xyz.astype(np.float32), rgb=rgb,
                        normal=nrm.astype(np.float32), label=label, dist=d_all.astype(np.float32),
                        lift_label=ll, lift_share=ls, lift_nvotes=lnv, boundary=boundary)

    # item statistics (backfills older scans) + fusion-dependent ones
    item_gt_stats(gt, gxyz, glab)
    fused = cKDTree(cxyz)
    for it in gt["items"]:
        it["n_fused_points"] = int((label == it["id"]).sum())
        p = gxyz[glab == it["id"]]
        it["occluded_fraction"] = float((fused.query(p, distance_upper_bound=tol)[0] > tol).mean()) if len(p) else 1.0
    gt["votes"] = vparams
    st = al["icp"]
    gt["seg_export"] = {"voxel": voxel, "tol": tol, "crop": crop, "votes": votes,
                        "icp_correction": al["T"].tolist(), "icp_ok": st["ok"], "icp_shift_mm": st["shift"] * 1000, "icp_fitness": st["fitness"],
                        "median_pre_mm": st["pre"] * 1000, "median_post_mm": st["post"] * 1000,
                        "tol_fixed": tol_in is not None}
    tmp = scan_dir / "gt.json.tmp"
    tmp.write_text(json.dumps(gt, indent=1))
    os.replace(tmp, scan_dir / "gt.json")

    food = label[(label > 0) & (label < 252)]
    lab_ok = label > 0
    return {"scan": scan_dir.name, "n_points": int(len(xyz)), "n_food": int(len(food)),
            "n_items": len(gt["items"]), "items_seen": int(len(np.unique(food))),
            "frac_unlabelled": float((label == 0).mean()), "median_dist_mm": float(np.median(d_all) * 1000),
            "mean_lift_share": float(ls[ll > 0].mean()) if (ll > 0).any() else 0.0,
            "n_lift_wrong": int((lab_ok & (ll != label)).sum()), "n_labelled": int(lab_ok.sum()),
            "frac_boundary": float(boundary.mean()), "tol_mm": tol * 1000,
            "pre_mm": st["pre"] * 1000, "post_mm": st["post"] * 1000,
            "shift_mm": st["shift"] * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--voxel", type=float, default=0.0015)
    ap.add_argument("--tol", type=float, default=None,
                    help="fixed label tolerance in m; default: clip(2.5 * post-ICP median, 4 mm, 8 mm)")
    ap.add_argument("--crop", type=float, default=0.25)
    ap.add_argument("--votes", choices=["oracle", "pseudo", "sam3"], default="pseudo",
                    help="per-frame masks lifted into lift_label (pseudo = corrupted oracle; sam3 = <scan>/masks_sam3)")
    args = ap.parse_args()
    for scan in sorted(p for p in Path(args.scans).iterdir() if (p / "gt" / "points.npz").exists()):
        r = export(scan, args.voxel, args.tol, args.crop, args.votes)
        print(f"{r['scan']}: {r['n_points']} pts, {r['n_food']} food, items {r['items_seen']}/{r['n_items']}, "
              f"unlabelled {100 * r['frac_unlabelled']:.1f} %, median dist {r['median_dist_mm']:.2f} mm, "
              f"lift share {r['mean_lift_share']:.2f}, lift!=label {r['n_lift_wrong']}/{r['n_labelled']}")


if __name__ == "__main__":
    main()

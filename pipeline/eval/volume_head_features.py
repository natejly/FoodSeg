"""Per-item feature extraction for the learned hidden-volume head (reuses the synth_volume harness read-only).

    python -m pipeline.eval.volume_head_features [--batches ...] [--out results/volume_head/features.npz]

One row per GT item that has >= MIN_POINTS labelled points in the fused cloud.  Labels = seg_points.npz
`label` (perfect segmentation, as in the volume harness).  Frame: dish frame (z up, z = 0 on the table);
container support = ICP-registered GT-container template (the harness's `foodscan` setup).

INPUT FEATURES (``FEATURES``) - everything is computable from a real scan (cloud + labels + container
template + per-view labels); nothing uses the GT volume, n_gt_points or occluded_fraction.
  Visible volumes (mL, log-transformed as lv_*):
    lv_foodscan   foodscan Support.height_map_volume of the points >= 1 mm above the support
    lv_voxel      2 mm column fill (closed + hole-filled footprint, mean top height)
    lv_hull       convex hull of the above-support points plus their drops onto the support
    lv_table      height-map volume against the flat table plane (no container)
    r_hull_fs, r_voxel_fs   lv_hull - lv_foodscan, lv_voxel - lv_foodscan (log ratios: shape-fill cues)
  Point statistics:
    n_pts, dens_pts_cm2   log labelled point count; points per cm^2 of footprint
    pts_per_ml            log(points / visible mL)
  Footprint (2 mm cells of the above-support points, closed + hole-filled):
    area_cm2        footprint area (log)
    convexity       footprint area / area of its 2D convex hull (1 = convex, low = ragged / L-shaped)
    obs_frac        observed cells / footprint cells (low = holes, unseen interior)
    hole_other_frac fraction of the unobserved footprint cells that are covered by another item's points
                    (an item lying on this one)
  Heights above support (mm):
    h_max (p99), h_mean, h_p10, h_p50, h_p90, edge_h (mean top height on the footprint boundary: high =
    item rests on something), peak_ratio = h_p90 / edge_h
  Box (PCA axes of the xy points, mm): len_mm, wid_mm, hgt_mm (z range above support), aspect = wid/len,
    height_to_width = h_max / wid
  Neighbourhood (ring of 2 cells around the footprint):
    contact_item    fraction of boundary cells with another item's points in the ring
    contact_wall    fraction of boundary cells whose ring contains a steep container surface (smoothed slope > 0.3)
    step_mean/min/max  own top height minus the highest neighbouring item's top height over contact cells
                    (negative = neighbour is taller / stacked on this item); 0 when no contact
    frac_nbr_higher fraction of contact cells where a neighbour is taller than the item
  Views / labelling:
    n_views         gt.json frames_visible (a render-side count; on a real scan = frames in which the
                    item's label projects into the image)
    lift_share      mean seg_points `lift_share` over the item's points (NaN where absent)
    lift_nvotes     mean seg_points `lift_nvotes`
  Scan context:
    n_items, scan_lv_total (log total visible volume of all items), fullness (total visible / container
    capacity), item_share (this item's share of the scan's visible volume), cont_plate, cont_R_cm,
    cont_cap_ml, cont_H_mm
  Item class (coarse kind from gt.json shape.kind): kind_pile, kind_single, kind_piece, kind_topping,
    kind_leaves
DIAGNOSTICS / TARGETS ONLY (kept in the table, never in FEATURES): vol_true, mass_g, rho, occl_frac,
    n_gt_points, vol_above_container.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import synth_volume_methods as M  # noqa: E402

BATCHES = ["hard", "hard_v3", "hard_v9", "hard_v10", "hard_v11", "hard_v12"]
KINDS = ["pile", "single", "piece", "topping", "leaves"]
CELL = M.CELL
MIN_POINTS = M.MIN_POINTS
NAN = float("nan")

FEATURES = [
    "lv_foodscan", "lv_voxel", "lv_hull", "lv_table", "r_hull_fs", "r_voxel_fs",
    "n_pts", "dens_pts_cm2", "pts_per_ml",
    "area_cm2", "convexity", "obs_frac", "hole_other_frac",
    "h_max", "h_mean", "h_p10", "h_p50", "h_p90", "edge_h", "peak_ratio",
    "len_mm", "wid_mm", "hgt_mm", "aspect", "height_to_width",
    "contact_item", "contact_wall", "step_mean", "step_min", "step_max", "frac_nbr_higher",
    "n_views", "lift_share", "lift_nvotes",
    "n_items", "scan_lv_total", "fullness", "item_share", "cont_plate", "cont_R_cm", "cont_cap_ml", "cont_H_mm",
] + [f"kind_{k}" for k in KINDS]

META = ["batch", "scan", "item_id", "category", "kind", "container", "scan_ok", "item_valid", "stacked",
        "vol_true", "mass_g", "rho", "occl_frac", "n_gt_points", "vol_above_container", "vis_fs", "vis_voxel", "vis_hull"]


def norm_cat(c: str | None) -> str:
    return (c or "").lower().replace("_", " ").strip()


def _lg(x: float) -> float:
    return float(np.log(max(x, 1e-3))) if np.isfinite(x) else NAN


def _grid_stats(shape, ij, h):
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    n = shape[0] * shape[1]
    cnt = np.bincount(flat, minlength=n).reshape(shape)
    s = np.bincount(flat, weights=h, minlength=n).reshape(shape)
    return cnt, np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)


def _footprint(obs: np.ndarray) -> np.ndarray:
    pad = 3
    m = np.pad(obs, pad)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=np.ones((3, 3), bool), iterations=2))
    return m[pad:-pad, pad:-pad]


def scan_features(sc: D.Scan, min_points: int = MIN_POINTS) -> list[dict]:
    """Feature rows (one per item with enough points) for one scan."""
    items = sc.items
    if sc.R is None or len(sc.xyz) < 200 or not items:
        return []
    cpts = D.container_points(sc)
    if len(cpts) < 200:
        return []
    sup = M.build_supports(cpts, sc.xyz)["icp"]
    if sup is None:
        return []
    cont = sc.gt.get("container") or {}
    dropped = set(int(i) for i in (sc.manifest.get("dropped_items") or []))
    lab, xyz = sc.label, sc.xyz
    lift_share = lift_nv = None
    try:
        z = np.load(sc.dir / "gt" / "seg_points.npz")
        lift_share = z["lift_share"].astype(np.float64) if "lift_share" in z.files else None
        lift_nv = z["lift_nvotes"].astype(np.float64) if "lift_nvotes" in z.files else None
    except Exception:  # noqa: BLE001
        pass

    food = (lab > 0) & (lab <= D.FOOD_MAX) & np.isfinite(xyz).all(1)
    if food.sum() < min_points:
        return []
    fx = xyz[food]
    base_pt = sup.height_at(np.c_[fx[:, :2], np.zeros(len(fx))])
    o = fx[:, :2].min(0) - 4 * CELL
    ijall = np.floor((fx[:, :2] - o) / CELL).astype(int)
    shape = tuple(ijall.max(0) + 6)
    # support slope grid (container wall detector)
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    cc = np.c_[o[0] + (gi.ravel() + .5) * CELL, o[1] + (gj.ravel() + .5) * CELL, np.zeros(gi.size)]
    base_grid = sup.height_at(cc).reshape(shape)
    gx, gy = np.gradient(ndimage.gaussian_filter(base_grid, 3), CELL)   # smoothed: raw support is mm-noisy
    steep = np.hypot(gx, gy) > 0.3

    per = {}
    lab_f = lab[food]
    for it in items:
        iid = int(it["id"])
        m = lab_f == iid
        if m.sum() < min_points:
            continue
        h = fx[m, 2] - base_pt[m]
        keep = h >= M.MIN_H
        if keep.sum() < min_points:
            continue
        cnt, top = _grid_stats(shape, ijall[m][keep], h[keep])
        per[iid] = dict(idx=np.nonzero(food)[0][m][keep], cnt=cnt, top=top, h=h[keep], xy=fx[m][keep][:, :2])
    if not per:
        return []

    est = {}
    for iid, p in per.items():
        pts = xyz[p["idx"]]
        raw = xyz[lab == iid]
        e = M.estimate_item(raw, xyz, {"icp": sup, "gt": None})
        est[iid] = e
    vis = {i: e["foodscan"] for i, e in est.items()}
    tot_vis = float(np.nansum(list(vis.values())))
    cap = float(cont.get("capacity_ml") or np.nan)

    rows = []
    for it in items:
        iid = int(it["id"])
        if iid not in per:
            continue
        p, e = per[iid], est[iid]
        cnt, top, h = p["cnt"], p["top"], p["h"]
        obs = cnt > 0
        F = _footprint(obs)
        nF = max(int(F.sum()), 1)
        area_m2 = nF * CELL * CELL
        try:
            ii, jj = np.nonzero(F)
            pts2 = np.c_[ii, jj].astype(float)
            ch = ConvexHull(np.vstack([pts2 + d for d in ([0, 0], [1, 0], [0, 1], [1, 1])])).volume
            convexity = float(min(nF / ch, 1.0))
        except Exception:  # noqa: BLE001
            convexity = 1.0
        others = np.zeros(shape, bool)
        other_top = np.full(shape, -np.inf)
        for j2, q in per.items():
            if j2 != iid:
                others |= q["cnt"] > 0
                other_top = np.where(q["cnt"] > 0, np.maximum(other_top, q["top"]), other_top)
        holes = F & ~obs
        hole_other = float((holes & others).sum() / max(holes.sum(), 1)) if holes.any() else 0.0
        bnd = F & ~ndimage.binary_erosion(F, border_value=0)
        nb = max(int(bnd.sum()), 1)
        ring2 = ndimage.binary_dilation(F, iterations=2) & ~F
        near_other = ndimage.binary_dilation(others, iterations=2)
        near_steep = ndimage.binary_dilation(steep, iterations=2)
        c_item = bnd & near_other
        contact_item = float(c_item.sum() / nb)
        contact_wall = float((bnd & near_steep).sum() / nb)
        nbr_top = ndimage.maximum_filter(other_top, size=5)
        cc_mask = c_item & np.isfinite(nbr_top) & obs
        if cc_mask.any():
            dh = (top - nbr_top)[cc_mask]
            step_mean, step_min, step_max = float(dh.mean()), float(dh.min()), float(dh.max())
            frac_hi = float((dh < 0).mean())
        else:
            step_mean = step_min = step_max = frac_hi = 0.0
        bo = bnd & obs
        edge_h = float(top[bo].mean()) if bo.any() else float(top[obs].mean())
        hp = np.percentile(h, [10, 50, 90, 99])
        xy = p["xy"] - p["xy"].mean(0)
        try:
            w, V = np.linalg.eigh(np.cov(xy.T))
            pr = xy @ V
            ext = np.ptp(pr, axis=0)
            len_mm, wid_mm = float(max(ext)) * 1e3, float(min(ext)) * 1e3
        except Exception:  # noqa: BLE001
            len_mm = wid_mm = NAN
        vfs, vvox, vhull, vtab = (e[k] for k in ("foodscan", "voxel", "hull", "table"))
        npts = int((lab == iid).sum())
        lsh = float(np.nanmean(lift_share[lab == iid])) if lift_share is not None and np.isfinite(lift_share[lab == iid]).any() else NAN
        lnv = float(np.nanmean(lift_nv[lab == iid])) if lift_nv is not None else NAN
        kind = (it.get("shape") or {}).get("kind")
        vol = it.get("volume_ml")
        mass = it.get("mass_g")
        r = dict(batch=sc.batch, scan=sc.sid, item_id=iid, category=norm_cat(it.get("category")), kind=kind,
                 container=cont.get("kind"), scan_ok=bool(sc.ok), item_valid=bool(iid not in dropped),
                 stacked=bool(sc.manifest.get("stacked")),
                 vol_true=float(vol) if vol else NAN, mass_g=float(mass) if mass else NAN,
                 rho=(mass / vol) if (mass and vol) else NAN,
                 occl_frac=float(it.get("occluded_fraction", NAN) if it.get("occluded_fraction") is not None else NAN),
                 n_gt_points=float(it.get("n_gt_points") or NAN),
                 vol_above_container=float(it.get("volume_above_container_ml") if it.get("volume_above_container_ml") is not None else NAN),
                 vis_fs=vfs, vis_voxel=vvox, vis_hull=vhull)
        r.update(
            lv_foodscan=_lg(vfs), lv_voxel=_lg(vvox), lv_hull=_lg(vhull), lv_table=_lg(vtab),
            r_hull_fs=_lg(vhull) - _lg(vfs), r_voxel_fs=_lg(vvox) - _lg(vfs),
            n_pts=float(np.log(npts)), dens_pts_cm2=float(npts / (area_m2 * 1e4)),
            pts_per_ml=float(np.log(npts / max(vfs, 1e-3))) if np.isfinite(vfs) else NAN,
            area_cm2=float(np.log(area_m2 * 1e4)), convexity=convexity, obs_frac=float(obs.sum() / nF),
            hole_other_frac=hole_other,
            h_max=float(hp[3] * 1e3), h_mean=float(h.mean() * 1e3), h_p10=float(hp[0] * 1e3), h_p50=float(hp[1] * 1e3),
            h_p90=float(hp[2] * 1e3), edge_h=edge_h * 1e3, peak_ratio=float(hp[2] / max(edge_h, 1e-4)),
            len_mm=len_mm, wid_mm=wid_mm, hgt_mm=float(np.ptp(h) * 1e3),
            aspect=wid_mm / max(len_mm, 1e-3), height_to_width=float(hp[3] * 1e3 / max(wid_mm, 1e-3)),
            contact_item=contact_item, contact_wall=contact_wall, step_mean=step_mean * 1e3, step_min=step_min * 1e3,
            step_max=step_max * 1e3, frac_nbr_higher=frac_hi,
            n_views=float(it.get("frames_visible") if it.get("frames_visible") is not None else NAN),
            lift_share=lsh, lift_nvotes=lnv,
            n_items=float(len(items)), scan_lv_total=_lg(tot_vis),
            fullness=tot_vis / cap if np.isfinite(cap) and cap > 0 else NAN,
            item_share=float(vfs / tot_vis) if tot_vis > 0 and np.isfinite(vfs) else NAN,
            cont_plate=float(cont.get("kind") == "plate"), cont_R_cm=float(cont.get("R", NAN)) * 100,
            cont_cap_ml=cap, cont_H_mm=float(cont.get("H", NAN)) * 1e3,
        )
        for k in KINDS:
            r[f"kind_{k}"] = float(kind == k)
        rows.append(r)
    return rows


def _work(args):
    bd, sid, man = args
    try:
        sc = D.load_scan(Path(bd), sid, man)
        return scan_features(sc)
    except Exception as e:  # noqa: BLE001
        print(f"[err] {bd}/{sid}: {e}", file=sys.stderr)
        return []


def extract(data_root: Path, batches: list[str], procs: int = 12) -> list[dict]:
    jobs = []
    for b in batches:
        bd = data_root / b
        man = D.read_manifest(bd)
        for sid in sorted(p.name for p in (bd / "scans").iterdir() if p.is_dir()):
            jobs.append((str(bd), sid, man))
    with Pool(procs) as pool:
        out = pool.map(_work, jobs, chunksize=1)
    return [r for rs in out for r in rs]


def save(rows: list[dict], path: Path) -> None:
    cols = META + FEATURES
    str_cols = {"batch", "scan", "category", "kind", "container"}
    arrs = {c: np.array([r.get(c, "" if c in str_cols else NAN) for r in rows],
                        dtype=object if c in str_cols else np.float64) for c in cols}
    arrs = {k: (v.astype(str) if k in str_cols else v) for k, v in arrs.items()}
    np.savez_compressed(path, **arrs)


def load(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_ROOT / "data" / "synth"))
    ap.add_argument("--batches", nargs="*", default=BATCHES)
    ap.add_argument("--out", default=str(_ROOT / "results" / "volume_head" / "features.npz"))
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    rows = extract(Path(a.data), a.batches, a.procs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save(rows, Path(a.out))
    print(f"{len(rows)} items -> {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

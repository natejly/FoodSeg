"""End-to-end per-item volume evaluation on the synthetic scans (known per-item volumes).

    cd /home/kyle/Desktop/FoodSeg
    conda run -n foodseg python -m pipeline.eval.synth_volume            # ok scans only in the tables
    conda run -n foodseg python -m pipeline.eval.synth_volume --all      # every scan in the tables

Pipeline under test: fused cloud -> per-item points (labelling source) -> per-item volume against the
container surface -> mass.  Sources: gt, lift, pred:<run>, gtsub:<run>, gt_top (see synth_volume_data).
Estimators: see synth_volume_methods.  Outputs (results/synth_volume/): summary.md, summary.json,
items.csv (one row per source x item), dishes.csv (one row per source x scan), scans.csv.
Always computes every scan; `--all` only chooses which ones enter the tables (counts are reported both ways).
CPU only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import synth_volume_methods as M  # noqa: E402
from pipeline.eval import synth_volume_report as RP  # noqa: E402

BATCHES = ["hard", "hard_v3", "hard_v9", "hard_v10", "hard_v11"]
NAN = float("nan")


def fin_pos(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x) and x > 0


def _tf_from_plane(plane):
    u, v = plane.basis()
    n = plane.normal
    Rm = np.stack([u, v, n])
    d = plane.d
    return lambda p: (np.asarray(p, np.float64) @ Rm.T) + np.array([0.0, 0.0, d])


def evaluate_scan(sc: D.Scan, pred_dirs: list[Path], plane_mode: str = "gt", min_points: int = M.MIN_POINTS):
    """-> (scan_info, item_rows, dish_rows).  Never raises for bad data (problem recorded in scan_info)."""
    items = sc.items
    cont = sc.gt.get("container") or {}
    info = {"batch": sc.batch, "scan": sc.sid, "ok": sc.ok, "n_items": len(items), "n_points": int(len(sc.xyz)),
            "container": cont.get("kind"), "problem": sc.problem, "has_lift": sc.lift is not None,
            "qc": ",".join(sc.manifest.get("qc", []) or []), "plane_tilt_deg": NAN, "plane_off_mm": NAN}
    gt_total = float(sum((it.get("volume_ml") or 0.0) for it in items))
    gt_mass = float(sum((it.get("mass_g") or 0.0) for it in items))
    info["gt_total_ml"] = gt_total
    mass_complete = bool(items) and all(fin_pos(it.get("mass_g")) for it in items)

    tf = lambda p: np.asarray(p, np.float64)   # noqa: E731
    cpts = np.zeros((0, 3))
    supports = {"icp": None, "gt": None, "info": {}}
    sources: list[D.Source] = []
    if sc.R is not None:
        try:
            cpts = D.container_points(sc)
            sources = D.sources_for(sc, pred_dirs)
        except Exception as e:  # noqa: BLE001
            info["problem"] = f"load: {e}"
    if len(sc.xyz) and plane_mode == "fit":
        try:
            from pipeline.foodscan.volume import fit_support_plane
            import open3d as o3d
            o3d.utility.random.seed(0)
            pl = fit_support_plane(sc.xyz, up_hint=np.array([0.0, 0.0, 1.0]))
            info["plane_tilt_deg"] = float(np.degrees(np.arccos(np.clip(pl.normal[2], -1, 1))))
            info["plane_off_mm"] = float(pl.d * 1e3)
            tf = _tf_from_plane(pl)
        except Exception as e:  # noqa: BLE001
            info["plane_error"] = str(e)
    scene_all = tf(sc.xyz) if len(sc.xyz) else np.zeros((0, 3))
    if len(cpts):
        cpts = tf(cpts)
    if len(scene_all) >= 200 and len(cpts) >= 200:
        try:
            cl = tf(sc.xyz[sc.label == 254]) if (sc.label == 254).sum() > 20 else None
            supports = M.build_supports(cpts, scene_all, init_xy=cl[:, :2].mean(0) if cl is not None else None)
        except Exception as e:  # noqa: BLE001
            supports["info"]["support_error"] = f"{e}"
    info.update({k: v for k, v in supports["info"].items()})
    info["support_ok"] = supports["icp"] is not None

    rows, dishes = [], []
    item_by_id = {int(it["id"]): it for it in items}
    for src in sources:
        pts = tf(src.pts)
        inst = src.inst
        # gt_top points live in the GT frame, so they are measured against the true-pose support; every other
        # source lives in the scan's own frame and is measured against the ICP-registered one.
        sup_src = ({"icp": supports["gt"], "icp_c": supports["gt"], "gt": supports["gt"]}
                   if src.name == "gt_top" else supports)
        if src.match:
            m = D.match_instances(src.gt_label, inst)
        else:
            m = {i: (i, 1.0) for i in np.unique(inst) if 0 < i <= D.FOOD_MAX}
        used = {v[0] for v in m.values()}
        est_cache: dict[int, dict] = {}

        def est_for(iid: int) -> dict:
            if iid not in est_cache:
                est_cache[iid] = M.estimate_item(pts[inst == iid], pts, sup_src, min_points=min_points)
            return est_cache[iid]

        base = {"batch": sc.batch, "scan": sc.sid, "ok": sc.ok, "source": src.name, "container": cont.get("kind")}
        for it in items:
            iid_gt = int(it["id"])
            vol, mass = it.get("volume_ml"), it.get("mass_g")
            row = dict(base, item_id=iid_gt, matched=iid_gt in m, category=it.get("category"),
                       kind=(it.get("shape") or {}).get("kind"), asset_source=it.get("source"),
                       gt_volume_ml=vol if vol is not None else NAN,
                       gt_above_rim_ml=it.get("volume_above_container_ml", NAN) if it.get("volume_above_container_ml") is not None else NAN,
                       gt_mass_g=mass if mass is not None else NAN,
                       occluded_fraction=it.get("occluded_fraction", NAN), stacked=sc.manifest.get("stacked"),
                       icp_shift_mm=info.get("icp_shift_mm", NAN),
                       rho_item=(mass / vol) if (mass and vol and vol > 0) else NAN, iou=NAN)
            if iid_gt in m:
                iid, iou = m[iid_gt]
                row["iou"] = iou
                row["n_pts"] = int((inst == iid).sum())
                row.update({f"vol_{k}": v for k, v in est_for(iid).items()})
            else:
                row["n_pts"] = 0
                row.update({f"vol_{k}": NAN for k in M.METHODS})
            rows.append(row)
        for iid in np.unique(inst):
            if 0 < iid <= D.FOOD_MAX and iid not in used and (src.match or int(iid) not in item_by_id):
                row = dict(base, item_id=-int(iid), matched=False, category=None, kind=None, asset_source=None,
                           gt_volume_ml=NAN, gt_above_rim_ml=NAN, gt_mass_g=NAN, rho_item=NAN, iou=NAN, occluded_fraction=NAN, stacked=sc.manifest.get("stacked"),
                           icp_shift_mm=info.get("icp_shift_mm", NAN),
                           n_pts=int((inst == iid).sum()))
                row.update({f"vol_{k}": v for k, v in est_for(int(iid)).items()})
                rows.append(row)
        dishes.append(dict(base, n_gt_items=len(items), gt_total_ml=gt_total, gt_total_g=gt_mass, mass_complete=mass_complete))
    if not sources:   # empty / broken scan: still emit GT-only rows so the items count as missing
        for it in items:
            rows.append(dict(batch=sc.batch, scan=sc.sid, ok=sc.ok, source="gt", container=cont.get("kind"),
                             item_id=int(it["id"]), matched=False, category=it.get("category"),
                             kind=(it.get("shape") or {}).get("kind"), asset_source=it.get("source"),
                             gt_volume_ml=it.get("volume_ml", NAN), gt_above_rim_ml=it.get("volume_above_container_ml") if it.get("volume_above_container_ml") is not None else NAN,
                             gt_mass_g=it.get("mass_g") if it.get("mass_g") is not None else NAN, rho_item=NAN,
                             iou=NAN, occluded_fraction=NAN, stacked=sc.manifest.get("stacked"), icp_shift_mm=NAN, n_pts=0, **{f"vol_{k}": NAN for k in M.METHODS}))
        dishes.append(dict(batch=sc.batch, scan=sc.sid, ok=sc.ok, source="gt", container=cont.get("kind"),
                           n_gt_items=len(items), gt_total_ml=gt_total, gt_total_g=gt_mass, mass_complete=mass_complete))
    return info, rows, dishes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(_ROOT / "data" / "synth"))
    ap.add_argument("--batches", nargs="*", default=BATCHES)
    ap.add_argument("--preds-glob", default="results/seg3d/*/preds",
                    help="glob (relative to repo root) of prediction dirs, files <batch>__<scan>.npz")
    ap.add_argument("--all", action="store_true", help="include scans whose manifest ok flag is false in the tables")
    ap.add_argument("--plane", choices=["gt", "fit"], default="gt",
                    help="table plane: ground truth (default; isolates the volume step) or RANSAC-fitted like run_meal")
    ap.add_argument("--min-points", type=int, default=M.MIN_POINTS)
    ap.add_argument("--out", default=str(_ROOT / "results" / "synth_volume"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N scans per batch")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_dirs = sorted(p for p in _ROOT.glob(a.preds_glob) if p.is_dir())
    infos, rows, dishes = [], [], []
    t0 = time.time()
    for b in a.batches:
        bd = Path(a.data) / b
        if not (bd / "scans").exists():
            print(f"[skip] {b}: no scans dir")
            continue
        man = D.read_manifest(bd)
        sids = sorted(p.name for p in (bd / "scans").iterdir() if p.is_dir())
        if a.limit:
            sids = sids[:a.limit]
        for sid in sids:
            try:
                sc = D.load_scan(bd, sid, man)
                info, r, d = evaluate_scan(sc, pred_dirs, a.plane, a.min_points)
            except Exception as e:  # noqa: BLE001
                info = {"batch": b, "scan": sid, "ok": bool(man.get(sid, {}).get("ok", False)),
                        "problem": f"crash: {e}", "n_items": 0}
                r, d = [], []
                traceback.print_exc()
            infos.append(info)
            rows += r
            dishes += d
            print(f"[{time.time() - t0:6.1f}s] {b}/{sid} ok={info.get('ok')} items={info.get('n_items')} "
                  f"pts={info.get('n_points')} icp={info.get('icp_rmse_mm', NAN):.2f}mm {info.get('problem') or ''}")
    res = RP.build(rows, dishes, infos, include_all=a.all, pred_dirs=[str(p.relative_to(_ROOT)) for p in pred_dirs],
                   plane=a.plane, min_points=a.min_points)
    RP.write(out, res, rows, dishes, infos)
    print(f"wrote {out}/summary.md (+ summary.json, items.csv, dishes.csv, scans.csv)")


if __name__ == "__main__":
    main()

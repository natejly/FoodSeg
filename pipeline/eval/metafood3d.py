"""MetaFood3D: volume estimators on real food scans, single top view vs full 3D, and
volume -> mass -> kcal.

    python -m eval.metafood3d ../data/metafood3d --out ../results/metafood3d

Data: MetaFood3D Point_cloud/4096 (points sampled uniformly on each Revopoint
mesh, metric metres, the object resting on z = 0) and the nutrition sheet
(weight g, energy kcal, macros, ground-truth volume). 637 objects, 108 categories.

Estimators (foodscan.volume, plane z = 0):
    full_hull      convex hull of the whole cloud
    full_poisson   plane-closed Poisson mesh of the whole cloud
    full_hm        height map of the whole cloud (mean height per 2 mm cell; the underside
                   points pull it low - kept as the naive baseline)
    full_thick     per 2 mm column, top minus bottom surface of the whole cloud
    top_hm         height map of the points one overhead camera would see (per-cell
                   topmost points only) - the single-RGB-D-frame case
    top_hull       convex hull of the same top-view points plus their footprint
Volume error is MAPE vs the sheet's volume. Then mass = volume x density with
density fitted per category (median g/mL over the other objects of that
category, else global), leave-one-out, and kcal = mass x the category's kcal/g
(same leave-one-out), i.e. category names are known (oracle naming).
"""
from __future__ import annotations

import argparse
import multiprocessing
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import openpyxl

from foodscan.volume import Plane, convex_hull_volume, height_map_volume, poisson_volume

PLANE = Plane(np.array([0.0, 0.0, 1.0]), 0.0)
CELL = 0.002


def top_view(pts: np.ndarray, cell: float = CELL, tol: float = 0.003) -> np.ndarray:
    """Points within `tol` of the highest point in their (cell x cell) column."""
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] - ij[:, 0].min()) * (ij[:, 1].max() - ij[:, 1].min() + 1) + (ij[:, 1] - ij[:, 1].min())
    top = np.full(key.max() + 1, -np.inf)
    np.maximum.at(top, key, pts[:, 2])
    return pts[pts[:, 2] >= top[key] - tol]


def thickness_volume(pts: np.ndarray, cell: float = CELL) -> float:
    """Closed-surface volume as a sum over cell columns of (top - bottom surface), cm^3.
    A column with no underside points (overhangs aside, the footprint interior seen only
    from above) runs down to the plane."""
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    ij -= ij.min(0)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    hi = np.full(key.max() + 1, -np.inf)
    lo = np.full(key.max() + 1, np.inf)
    np.maximum.at(hi, key, pts[:, 2])
    np.minimum.at(lo, key, pts[:, 2])
    used = np.isfinite(hi)
    lo = np.where(hi[used] - lo[used] < 0.003, 0.0, lo[used])   # only a top surface: down to z = 0
    return float(np.sum(hi[used] - lo) * cell * cell * 1e6)


def load(path: str) -> np.ndarray:
    pts = np.asarray(o3d.io.read_point_cloud(path).points)
    return pts[pts[:, 2] > 0.0005]                      # drop the base disc if sampled


def _poisson(path: str) -> float:
    return float(poisson_volume(load(path), PLANE))


class Isolated:
    """Run fn(arg) in a spawned worker; a segfault (Open3D Poisson on some small clouds)
    or a timeout gives nan and a fresh worker."""

    def __init__(self, fn):
        self.fn, self.ex = fn, None

    def __call__(self, arg, timeout: float = 60.0) -> float:
        from concurrent.futures import ProcessPoolExecutor
        if self.ex is None:
            self.ex = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        try:
            return self.ex.submit(self.fn, arg).result(timeout=timeout)
        except Exception:                               # noqa: BLE001 - BrokenProcessPool, timeout
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = None
            return float("nan")


def volumes(path: str, poisson=None) -> dict:
    pts = load(path)
    top = top_view(pts)
    out = {"n_pts": len(pts), "n_top": len(top),
           "full_poisson": poisson(path) if poisson else float("nan")}
    for name, fn in (("full_hull", lambda: convex_hull_volume(pts, PLANE)),
                     ("full_hm", lambda: height_map_volume(pts, PLANE, cell=CELL)),
                     ("full_thick", lambda: thickness_volume(pts)),
                     ("top_hm", lambda: height_map_volume(top, PLANE, cell=CELL)),
                     ("top_hull", lambda: convex_hull_volume(top, PLANE))):
        try:
            out[name] = float(fn())
        except Exception:                               # noqa: BLE001
            out[name] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/metafood3d"))
    ap.add_argument("--points", default="4096")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = list(openpyxl.load_workbook(args.root / "nutrition.xlsx", read_only=True, data_only=True).worksheets[0]
                .iter_rows(values_only=True))
    objs = []
    for r in rows[1:]:
        if not r[0] or any(not isinstance(r[k], (int, float)) for k in (3, 4, 8)) or r[8] <= 0:
            continue                                    # formula cells without a cached value
        cat, inst = str(r[0]).strip(), str(r[1]).strip()
        plys = sorted((args.root / "Point_cloud" / args.points / cat / inst).glob("*.ply"))
        if plys:
            objs.append({"cat": cat, "inst": inst, "ply": str(plys[0]), "mass": float(r[3]),
                         "kcal": float(r[4]), "vol_gt": float(r[8])})
    print(f"{len(objs)} objects with a point cloud and a volume, {len({o['cat'] for o in objs})} categories")

    # serial (~0.5 s/object): Open3D's thread pool deadlocks in multiprocessing workers here
    poisson = Isolated(_poisson)
    for i, o in enumerate(objs):
        o.update(volumes(o["ply"], poisson))
        if i % 100 == 0:
            print(f"  {i}/{len(objs)}", flush=True)

    ests = ("full_hull", "full_poisson", "full_hm", "full_thick", "top_hm", "top_hull")
    gt = np.array([o["vol_gt"] for o in objs])
    report = {"n_objects": len(objs), "volume_mape_pct": {}, "volume_median_ape_pct": {}, "mass": {}, "kcal": {}}
    for e in ests:
        v = np.array([o[e] for o in objs])
        ok = np.isfinite(v)
        ape = 100 * np.abs(v[ok] - gt[ok]) / gt[ok]
        report["volume_mape_pct"][e] = float(ape.mean())
        report["volume_median_ape_pct"][e] = float(np.median(ape))
        report.setdefault("volume_bias_pct", {})[e] = float(np.median(100 * (v[ok] - gt[ok]) / gt[ok]))

    # leave-one-out per-category density and kcal/g (category name known)
    cats = np.array([o["cat"] for o in objs])
    mass, kcal = np.array([o["mass"] for o in objs]), np.array([o["kcal"] for o in objs])
    for e in ("vol_gt",) + ests:
        v = np.array([o[e] for o in objs])
        mp, kp = np.full(len(objs), np.nan), np.full(len(objs), np.nan)
        for i in range(len(objs)):
            others = np.arange(len(objs)) != i
            same = others & (cats == cats[i]) & np.isfinite(v) & (v > 0)
            pool_ = same if same.sum() >= 1 else others & np.isfinite(v) & (v > 0)
            dens = np.median(mass[pool_] / v[pool_])
            kpg = np.median(kcal[pool_] / mass[pool_])
            mp[i], kp[i] = dens * v[i], dens * v[i] * kpg
        ok = np.isfinite(mp)
        report["mass"][e] = {"mape_pct": float(100 * np.mean(np.abs(mp[ok] - mass[ok]) / mass[ok])),
                             "mae_g": float(np.mean(np.abs(mp[ok] - mass[ok])))}
        report["kcal"][e] = {"mape_pct": float(100 * np.mean(np.abs(kp[ok] - kcal[ok]) / kcal[ok])),
                             "mae_kcal": float(np.mean(np.abs(kp[ok] - kcal[ok])))}
    json.dump(report, open(args.out / "summary.json", "w"), indent=2)
    with open(args.out / "per_object.csv", "w", newline="") as fh:
        keys = ["cat", "inst", "mass", "kcal", "vol_gt", "n_pts", "n_top", *ests]
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(objs)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

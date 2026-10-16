"""Nutrition5k overhead RGB-D benchmark for the single-frame geometry path.

    python -m eval.nutrition5k data/datasets/Nutrition5k --out ../results/n5k

Runs foodscan.volume on every dish of the official depth splits:
back-project the overhead depth, RANSAC the plate plane, DBSCAN the
above-plane points, height-map volume; with food masks, also volume above
a fitted dish surface (dish_volume). Volume -> mass uses one global density
fitted on depth_train_ids (median of mass/volume), or per-ingredient density
(eval.density) given the dish's ingredient names; scores mass and kcal on
depth_test_ids (507 dishes), the split all published N5k depth numbers use.

kcal is reported two ways:
    kcal_oracle_nutr  predicted mass x the dish's true kcal/g, i.e. naming +
                      nutrition lookup assumed perfect; isolates portion error
    kcal_global       predicted volume x one kcal/cm^3 fitted on train

Camera: N5k gives no intrinsics, only the paper's 35.9 cm camera-to-plane
distance and 5.957e-3 cm^2 per pixel there (Thames et al. 2021, sec. 4.4),
which fixes f = 0.359 / sqrt(5.957e-7) = 465 px for a 640x480 pinhole.
Depth is 1e-4 m per unit per the README, but some early (2019-04) dishes are
stored in mm; those are detected by their median and rescaled.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from eval.density import IngredientDensity, choose_lambda
from foodscan.volume import (cluster_objects, convex_hull_volume, fit_support_plane,
                             height_map_volume, points_above_plane)

F_PX = 0.359 / np.sqrt(5.957e-7)
CX, CY = 319.5, 239.5
UP = np.array([0.0, 0.0, -1.0])           # camera looks straight down; up is toward it


def load_depth_m(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    valid = d[d > 0]
    scale = 1e-3 if valid.size and np.median(valid) < 1000 else 1e-4
    return d * scale


def hull_volume(clusters: list[np.ndarray], plane) -> float:
    """Sum of per-cluster convex hulls; Open3D's get_volume throws on the odd
    degenerate (non-watertight) hull, which would otherwise void the whole dish."""
    total = 0.0
    for c in clusters:
        try:
            total += convex_hull_volume(c, plane)
        except RuntimeError:
            return float("nan")
    return total


def backproject(depth: np.ndarray, mask: np.ndarray | None = None, roi: float = 1.0) -> np.ndarray:
    """Points (N,3) in metres for valid pixels inside `mask`; `roi` keeps the central fraction."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    m = depth > 0
    if mask is not None:
        m &= mask
    if roi < 1.0:
        m &= (np.abs(u - CX) < roi * w / 2) & (np.abs(v - CY) < roi * h / 2)
    z = depth[m]
    return np.stack([(u[m] - CX) * z / F_PX, (v[m] - CY) * z / F_PX, z], -1)


COUNTER_BAND = 30       # px; the frame border is mostly bare counter
DISH_MIN_H = 0.025      # m above counter: dishes sit on the scale (~4 cm), the rig's rail is ~8 mm
ON_SCALE_RIM = 0.050    # m; a lower rim means a plate off the scale (a second rig layout), where
                        # the counter is not the plate's support: use the ring plane there
FLOOR_PRIOR = 0.045     # m; bowl floor above the counter (scale top ~40 mm + foot)
FLOOR_PRIOR_WEIGHT = 200.0   # pull of the floor prior, in pixels' worth of observations
RIM_ONLY_SPREAD = 0.010  # m; visible dish heights all within this of the rim -> only the rim shows
BOWL_MIN_DEPTH = 0.015  # m; floor prior is at least this far below the rim
BOWL_RIM = 0.080        # m; on the scale, plate rims sit 55-80 mm above the counter, bowl rims 90-120
MAX_FOOD_H = 0.15
DENSITY_ESTIMATORS = ("vol_container", "vol_dish", "vol_hm_mask", "area_mask")


def dish_volume(depth: np.ndarray, food: np.ndarray, floor_prior: float = FLOOR_PRIOR,
                prior_weight: float = FLOOR_PRIOR_WEIGHT, maps: dict | None = None) -> dict:
    """Food volume above each dish's inner surface, from one overhead frame.

    1. Counter plane: RANSAC on a band around the frame border.
    2. Dishes: connected regions > DISH_MIN_H above the counter that touch the
       food mask, dropout gaps closed and holes filled.
    3. Inner surface: per dish, s(rho) = a + b rho^2 + c rho^4 (rho = radius /
       equivalent dish radius, b, c >= 0) fitted to the dish pixels the food does
       not cover, with a pulled (weight `prior_weight` pixels) toward a floor
       prior. Plates (rim <= BOWL_RIM) expose their floor, so the prior is their
       lowest visible surface and the centre stays flat (the rho^4 term carries
       the rim). Bowls expose only wall above the food line, so the prior is
       `floor_prior` and the surface curves from that floor up to the rim.
       Dishes off the scale (rim < ON_SCALE_RIM) use the ring plane instead.
    4. Volume = sum over food pixels inside a dish of max(h - s, 0) x pixel
       footprint; food pixels with no depth take the mean valid thickness. Food
       outside every dish (mask spill onto the rig) is ignored.

    If `maps` is a dict, it receives per-pixel float32 maps (m, nan = no depth):
    "h_counter" height above the counter, "dish_thick" food height above the
    fitted dish surface (0 off the food).
    """
    from scipy.optimize import lsq_linear

    H, W = depth.shape
    valid = depth > 0
    border = np.zeros((H, W), bool)
    b = COUNTER_BAND
    border[:b] = border[-b:] = border[:, :b] = border[:, -b:] = True
    counter = fit_support_plane(backproject(depth, border), dist_thresh=0.003, up_hint=UP)

    v, u = np.mgrid[0:H, 0:W]
    P = np.stack([(u - CX) * depth / F_PX, (v - CY) * depth / F_PX, depth], -1)
    h = counter.height(P.reshape(-1, 3)).reshape(H, W)
    h[~valid] = np.nan
    px_area = (depth / F_PX) ** 2 * 1e4                       # cm^2 per pixel

    above = (np.nan_to_num(h, nan=0.0) > DISH_MIN_H).astype(np.uint8)
    above = cv2.morphologyEx(above, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # bridge depth dropouts on glossy white plates, which otherwise split a plate into islands
    above = cv2.morphologyEx(above, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    n, lab = cv2.connectedComponents(above)
    uncovered = ~(cv2.dilate(food.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)

    vol, rims, floors = 0.0, [], []
    thick_map = np.zeros((H, W), np.float32)
    for k in range(1, n):
        comp = lab == k
        if (comp & food).sum() < 50:
            continue
        cnt, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros((H, W), np.uint8)
        cv2.drawContours(filled, cnt, -1, 1, -1)
        filled = filled > 0
        fv = filled & valid
        centre = P[fv][:, :2].mean(0)
        radius = np.sqrt(filled.sum() / np.pi) * np.median(depth[fv]) / F_PX
        rho = np.linalg.norm(P[..., :2] - centre, axis=-1) / radius
        rim = float(np.nanpercentile(h[fv], 97))

        vis = fv & uncovered
        r_v, h_v = rho[vis], h[vis]
        if len(r_v) > 5000:
            idx = np.random.default_rng(0).choice(len(r_v), 5000, replace=False)
            r_v, h_v = r_v[idx], h_v[idx]
        # floor prior: a plate shows its floor, so its lowest visible surface;
        # a bowl shows only wall above the food line, so the typical bowl floor
        low = float(np.percentile(h_v, 5)) if len(h_v) > 100 else rim
        is_bowl = rim > BOWL_RIM or rim - low < RIM_ONLY_SPREAD
        a0 = min(floor_prior, rim - BOWL_MIN_DEPTH) if is_bowl else low
        w = np.sqrt(prior_weight)
        A = np.vstack([np.c_[np.ones_like(r_v), r_v ** 2, r_v ** 4],
                       [w, 0.0, 0.0],                  # floor prior
                       [0.0, 0.0, 1.0]])               # weak ridge on c: rim-only data -> paraboloid
        y = np.r_[h_v, w * a0, 0.0]
        lo, hi = [min(a0, floor_prior) - 0.015, 0.0, 0.0], [max(rim, a0) + 1e-4, np.inf, np.inf]
        a_, b_, c_ = lsq_linear(A, y, bounds=(lo, hi)).x
        rims.append(rim)
        floors.append(a_)

        fd = food & filled
        surf = a_ + b_ * rho[fd] ** 2 + c_ * rho[fd] ** 4
        thick = np.clip(h[fd] - surf, 0.0, MAX_FOOD_H)        # nan where no depth
        thick_map[fd] = thick
        ok = np.isfinite(thick)
        if ok.any():
            mean_t = float(thick[ok].mean())
            vol += float(np.sum(thick[ok] * px_area[fd][ok]) * 100)       # m*cm^2 -> cm^3
            vol += float((~ok).sum() * mean_t * 100 * (0.359 / F_PX) ** 2 * 1e4)
    if maps is not None:
        maps["h_counter"], maps["dish_thick"] = h.astype(np.float32), thick_map
    return {"vol_dish": vol, "n_dishes": len(rims),
            "dish_rim_mm": 1000 * max(rims) if rims else float("nan"),
            "dish_floor_mm": 1000 * min(floors) if floors else float("nan")}


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def ring_plane(depth: np.ndarray, food: np.ndarray):
    """Plane through a ring of pixels 5-25 px outside the food (centre of frame if the ring is empty)."""
    f8 = food.astype(np.uint8)
    ring = (cv2.dilate(f8, _disk(25)) > 0) & ~(cv2.dilate(f8, _disk(5)) > 0)
    ring_pts = backproject(depth, ring)
    return fit_support_plane(ring_pts if len(ring_pts) > 200 else backproject(depth, roi=0.5),
                             dist_thresh=0.003, up_hint=UP)


def ring_plane_volume(depth: np.ndarray, food: np.ndarray, hull: bool = True) -> tuple[float, float]:
    """Height map / hull volume above ring_plane."""
    plane = ring_plane(depth, food)
    f8 = food.astype(np.uint8)
    obj = backproject(depth, cv2.erode(f8, _disk(1)) > 0)   # drop flying edge pixels
    obj = obj[(plane.height(obj) > 0) & (plane.height(obj) < 0.15)]
    return height_map_volume(obj, plane), hull_volume(cluster_objects(obj), plane) if hull else float("nan")


def dish_volume_or_ring(depth: np.ndarray, food: np.ndarray, ring_vol: float | None = None) -> dict:
    """dish_volume, falling back to the ring-plane volume when no dish on the scale
    touches the food (plates off the scale are flat, which the ring plane handles)."""
    out = dish_volume(depth, food)
    out["dish_fallback"] = out["n_dishes"] == 0 or out["dish_rim_mm"] < 1000 * ON_SCALE_RIM
    if out["dish_fallback"]:
        out["vol_dish"] = ring_vol if ring_vol is not None else ring_plane_volume(depth, food, hull=False)[0]
    return out


def is_bowl(r: dict) -> bool:
    return not r.get("dish_fallback", True) and r.get("dish_rim_mm", 0.0) > 1000 * BOWL_RIM


def container_volume(r: dict) -> float:
    """Bowls on the scale: fitted dish surface; plates: ring plane (better on train CV)."""
    return r["vol_dish"] if is_bowl(r) else r["vol_hm_mask"]


def load_food(dish_dir: Path, mask_dir: Path | None) -> np.ndarray | None:
    mpath = mask_dir / f"{dish_dir.name}.png" if mask_dir else None
    if mpath and mpath.exists():
        return cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 0
    return None


def process_dish(dish_dir: Path, mask_dir: Path | None = None) -> dict:
    """Only the dish-surface variant (vol_dish), for dishes already cached."""
    out = {"dish_id": dish_dir.name}
    try:
        food = load_food(dish_dir, mask_dir)
        if food is not None:
            out.update(dish_volume_or_ring(load_depth_m(dish_dir / "depth_raw.png"), food)
                       if food.sum() >= 50 else {"vol_dish": 0.0, "n_dishes": 0})
    except Exception as e:                # noqa: BLE001
        out["error_dish"] = repr(e)
    return out


def process(dish_dir: Path, mask_dir: Path | None = None) -> dict:
    """Volumes for one dish under four variants:

    global  no mask, RANSAC over the whole image (calibrate.py's single-frame baseline)
    center  no mask, plane from the central half (avoids locking onto the rig)
    mask    food mask from eval.n5k_masks; plane from a ring of plate pixels
            5-25 px outside the food, so a concave plate's raised rim is not food
    dish    food mask, height above each dish's fitted inner surface (dish_volume)
    (main() adds vol_container: dish for bowls, mask/ring plane for plates)
    area_mask  mask area alone (cm^2), the no-depth control; its fitted
            "density" is g/cm^2
    """
    out = {"dish_id": dish_dir.name}
    try:
        depth = load_depth_m(dish_dir / "depth_raw.png")
        pts = backproject(depth)
        for tag, src in (("global", pts), ("center", backproject(depth, roi=0.5))):
            plane = fit_support_plane(src, dist_thresh=0.002, up_hint=UP)
            clusters = cluster_objects(points_above_plane(pts, plane, min_height=0.004,
                                                          max_height=0.15))
            obj = np.vstack(clusters) if clusters else np.zeros((0, 3))
            out[f"vol_hm_{tag}"] = height_map_volume(obj, plane)
            out[f"vol_hull_{tag}"] = hull_volume(clusters, plane)

        food = load_food(dish_dir, mask_dir)
        if food is not None:
            out["mask_frac"] = float(food.mean())
            # RGB-only control: food area on the plate (pixel footprint at the plate depth)
            out["area_mask"] = float(food.sum() * (0.359 / F_PX) ** 2 * 1e4)   # cm^2
            if food.sum() < 50:
                out["vol_hm_mask"] = out["vol_hull_mask"] = out["vol_dish"] = 0.0
                return out
            out["vol_hm_mask"], out["vol_hull_mask"] = ring_plane_volume(depth, food)
            out.update(dish_volume_or_ring(depth, food, out["vol_hm_mask"]))
    except Exception as e:                # noqa: BLE001 - one bad dish must not stop the run
        out["error"] = repr(e)
    return out


def read_metadata(root: Path) -> dict[str, dict]:
    meta = {}
    for f in sorted((root / "metadata").glob("dish_metadata_cafe*.csv")):
        for row in csv.reader(open(f)):
            meta[row[0]] = {"kcal": float(row[1]), "mass": float(row[2]),
                            "fat": float(row[3]), "carb": float(row[4]), "protein": float(row[5]),
                            "ingredients": [(row[i + 1], float(row[i + 2]))
                                            for i in range(6, len(row) - 6, 7)]}
    return meta


def read_split(root: Path, name: str) -> list[str]:
    return [l.strip() for l in open(root / "dish_ids/splits" / f"{name}.txt") if l.strip()]


def mae_pct(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(pred - true)))
    return mae, 100 * mae / float(np.mean(true))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/n5k"))
    ap.add_argument("--masks", type=Path, default=None, help="eval.n5k_masks output dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redo-dish", action="store_true", help="recompute vol_dish for cached dishes")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(args.root)
    splits = {s: read_split(args.root, s) for s in ("depth_train_ids", "depth_test_ids")}
    img = args.root / "imagery/realsense_overhead"
    ids = [d for s in splits.values() for d in s if (img / d / "depth_raw.png").exists() and d in meta]
    print(f"{len(ids)} dishes with depth + metadata")

    cache = args.out / "per_dish.jsonl"
    done = {}
    if cache.exists():
        done = {r["dish_id"]: r for r in map(json.loads, open(cache))}
    todo = [img / d for d in ids if d not in done]
    # cached dishes from before vol_dish existed: compute only that variant
    todo_dish = [img / d for d in ids if args.masks and d in done and "error" not in done[d]
                 and (args.redo_dish or "vol_dish" not in done[d])]
    with Pool(args.workers) as pool, open(cache, "a") as fh:
        for fn, items in ((process, todo), (process_dish, todo_dish)):
            print(f"{fn.__name__}: {len(items)} dishes", flush=True)
            for i, r in enumerate(pool.imap_unordered(partial(fn, mask_dir=args.masks), items, chunksize=4)):
                done[r["dish_id"]] = {**done.get(r["dish_id"], {}), **r}
                fh.write(json.dumps(done[r["dish_id"]]) + "\n")
                if i % 200 == 0:
                    print(f"  {i}/{len(items)}", flush=True)
    with open(cache, "w") as fh:                              # drop superseded lines
        for r in done.values():
            fh.write(json.dumps(r) + "\n")

    rows = {d: {**done[d], **meta[d]} for d in ids if "error" not in done[d]}
    for r in rows.values():
        if "vol_dish" in r and "vol_hm_mask" in r:
            r["vol_container"] = container_volume(r)
    n_err = sum("error" in done[d] for d in ids)
    report = {"n_errors": n_err, "f_px": F_PX, "estimators": {}}

    train = [d for d in splits["depth_train_ids"] if d in rows]
    test = [d for d in splits["depth_test_ids"] if d in rows]
    col = lambda ds, k: np.array([rows[d][k] for d in ds])   # noqa: E731
    m_tr, m_te = col(train, "mass"), col(test, "mass")
    k_tr, k_te = col(train, "kcal"), col(test, "kcal")

    report["baseline_train_mean"] = {"mass": mae_pct(np.full_like(m_te, m_tr.mean()), m_te),
                                     "kcal": mae_pct(np.full_like(k_te, k_tr.mean()), k_te)}

    def score(name, te, v_te, mass_pred, extra):
        m_te, k_te = col(te, "mass"), col(te, "kcal")
        kcal_density_true = k_te / np.maximum(m_te, 1e-6)
        report["estimators"][name] = {
            "n_test": len(te),
            **extra,
            "pearson_r_volume_vs_mass": float(np.corrcoef(v_te, m_te)[0, 1]),
            "frac_zero_volume_test": float(np.mean(v_te < 1)),
            "mass_mae_g_pct": mae_pct(mass_pred, m_te),
            "kcal_oracle_nutr_mae_pct": mae_pct(mass_pred * kcal_density_true, k_te),
            "mass_mape_pct": float(100 * np.mean(np.abs(mass_pred - m_te) / np.maximum(m_te, 1))),
        }
        with open(args.out / f"test_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dish_id", "volume_cm3", "mass_true", "mass_pred", "kcal_true",
                        "kcal_pred_oracle_nutr"])
            for d, v, mt, mp, kt, kd in zip(te, v_te, m_te, mass_pred, k_te, kcal_density_true):
                w.writerow([d, f"{v:.2f}", f"{mt:.1f}", f"{mp:.1f}", f"{kt:.1f}", f"{mp * kd:.1f}"])

    for est in ("vol_container", "vol_dish", "vol_hm_mask", "area_mask", "vol_hull_mask", "vol_hm_center",
                "vol_hm_global", "vol_hull_center", "vol_hull_global"):
        if any(est not in rows[d] for d in train + test):
            continue
        tr = [d for d in train if np.isfinite(rows[d][est])]
        te = [d for d in test if np.isfinite(rows[d][est])]
        v_tr, v_te = col(tr, est), col(te, est)
        m_tr, k_tr, k_te = col(tr, "mass"), col(tr, "kcal"), col(te, "kcal")
        ok = v_tr > 1
        dens = float(np.median(m_tr[ok] / v_tr[ok]))           # g / cm^3
        kcal_per_cm3 = float(np.median(k_tr[ok] / v_tr[ok]))
        score(est, te, v_te, dens * v_te,
              {"fitted_density_g_per_cm3": dens,
               "kcal_global_mae_pct": mae_pct(kcal_per_cm3 * v_te, k_te)})

        if est not in DENSITY_ESTIMATORS:
            continue
        ing_tr, ing_te = [rows[d]["ingredients"] for d in tr], [rows[d]["ingredients"] for d in te]
        for how in ("names", "frac"):
            lam = choose_lambda(v_tr, m_tr, ing_tr, how)
            model = IngredientDensity(lam).fit(v_tr, m_tr, ing_tr)
            score(f"{est}+ingr_{how}", te, v_te, model.predict_mass(v_te, ing_te, how),
                  {"density": f"per-ingredient, oracle {how}", "ridge_lambda": lam,
                   "n_ingredients_fitted": len(model.vocab)})

    report["n_train"], report["n_test"] = len(train), len(test)
    json.dump(report, open(args.out / "summary.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

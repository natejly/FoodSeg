"""3D instance segmentation of touching foods, on scenes built from real MetaFood3D scans.

    python -m eval.mf3d_composite build ../data/metafood3d --out ../data/metafood3d/composite
    python -m eval.mf3d_composite dbscan ../data/metafood3d/composite --out ../results/metafood3d/seg

MetaFood3D has one food per scan, so segmentation needs scenes: each scene drops
2-5 objects (Point_cloud/4096, metric, resting on z = 0) at random yaw on a plane,
packed along a random walk with gaps from -5 mm (slight interpenetration, i.e.
touching after scan noise) to +20 mm. Objects are split 80/20 train/test by
object, never shared. Every scene is stored twice:
    full  all points of every object (a fused multi-view scan)
    top   only points visible from straight above (one overhead depth frame)
with 0.5 mm Gaussian noise, 8192 points per scene, per-point instance labels.

Metric: instances matched to ground truth by Hungarian on IoU; mean IoU over GT
instances (unmatched = 0) and AP50-style recall (IoU > 0.5); plus per-item
volume error (convex hull of each predicted instance vs. of its GT points), so
segmentation is scored by what it does to portion size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from eval.metafood3d import top_view

N_PTS = 8192


def load_objects(root: Path) -> list[tuple[str, np.ndarray]]:
    objs = []
    for p in sorted((root / "Point_cloud" / "4096").glob("*/*/*.ply")):
        pts = np.asarray(o3d.io.read_point_cloud(str(p)).points)
        pts = pts[pts[:, 2] > 0.0005]
        if len(pts) > 500:
            pts[:, :2] -= pts[:, :2].mean(0)
            objs.append((f"{p.parent.parent.name}/{p.parent.name}", pts))
    return objs


def place(objs: list[np.ndarray], rng) -> list[np.ndarray]:
    """Random yaw; each next object placed beside a random previous one (the anchor), along a
    random direction, at a gap in [-5, 20] mm between their footprints' extents along that
    direction; it is then pushed outward until > 1 mm clear of every other object."""
    placed = []
    for k, o in enumerate(objs):
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        o = o @ R.T
        if k:
            anchor = placed[rng.integers(len(placed))]
            d = rng.normal(size=2)
            d /= np.linalg.norm(d)
            # support distance of each footprint along d
            ra = (anchor[:, :2] - anchor[:, :2].mean(0)) @ d
            rb = -(o[:, :2] @ d)
            gap = rng.uniform(-0.005, 0.020)
            o = o + np.r_[anchor[:, :2].mean(0) + d * (ra.max() + rb.max() + gap), 0.0]
            # only the anchor may touch/overlap; push clear of all others
            for _ in range(100):
                if all(cKDTree(q[::4, :2]).query(o[::4, :2])[0].min() > 0.001
                       for q in placed if q is not anchor):
                    break
                o[:, :2] += d * 0.005
        placed.append(o)
    return placed


def build(root: Path, out: Path, n_train: int, n_test: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    objs = load_objects(root)
    idx = rng.permutation(len(objs))
    n_te = len(objs) // 5
    split = {"test": idx[:n_te], "train": idx[n_te:]}
    out.mkdir(parents=True, exist_ok=True)
    meta = {"objects": [n for n, _ in objs], "split": {k: v.tolist() for k, v in split.items()}}
    for name, n_scenes in (("train", n_train), ("test", n_test)):
        X = {"full": np.zeros((n_scenes, N_PTS, 3), np.float32), "top": np.zeros((n_scenes, N_PTS, 3), np.float32)}
        Y = {"full": np.zeros((n_scenes, N_PTS), np.int8), "top": np.zeros((n_scenes, N_PTS), np.int8)}
        members = []
        for s in range(n_scenes):
            k = rng.integers(2, 6)
            pick = rng.choice(split[name], k, replace=False)
            placed = place([objs[i][1] for i in pick], rng)
            pts = np.concatenate(placed)
            lab = np.concatenate([np.full(len(p), j) for j, p in enumerate(placed)])
            for view in ("full", "top"):
                if view == "top":
                    keep = np.isin(np.arange(len(pts)), _top_index(pts))
                    p, l = pts[keep], lab[keep]
                else:
                    p, l = pts, lab
                sel = rng.choice(len(p), N_PTS, replace=len(p) < N_PTS)
                X[view][s] = p[sel] + rng.normal(0, 0.0005, (N_PTS, 3))
                Y[view][s] = l[sel]
            members.append([int(i) for i in pick])
        np.savez_compressed(out / f"{name}.npz", full=X["full"], top=X["top"],
                            full_lab=Y["full"], top_lab=Y["top"])
        meta[f"{name}_scenes"] = members
        print(f"{name}: {n_scenes} scenes")
    (out / "meta.json").write_text(json.dumps(meta))


def _top_index(pts: np.ndarray) -> np.ndarray:
    top = top_view(pts)
    # indices of rows of pts that survive top_view (exact float match)
    view = pts.view([("", pts.dtype)] * 3).ravel()
    return np.nonzero(np.isin(view, top.astype(pts.dtype).view([("", pts.dtype)] * 3).ravel()))[0]


def hull_vol(p: np.ndarray) -> float:
    if len(p) < 10:
        return 0.0
    try:
        base = p.copy()
        base[:, 2] = 0
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([p, base])))
        return float(pc.compute_convex_hull()[0].get_volume() * 1e6)
    except Exception:                                    # noqa: BLE001
        return 0.0


def score_scene(pts: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> dict:
    g_ids = np.unique(gt)
    p_ids = [i for i in np.unique(pred) if i >= 0]
    iou = np.zeros((len(g_ids), max(len(p_ids), 1)))
    for a, g in enumerate(g_ids):
        for b, p in enumerate(p_ids):
            inter = np.sum((gt == g) & (pred == p))
            iou[a, b] = inter / np.sum((gt == g) | (pred == p))
    r, c = linear_sum_assignment(-iou)
    best = np.zeros(len(g_ids))
    best[r] = iou[r, c]
    vol_err = []
    for a, g in enumerate(g_ids):
        vg = hull_vol(pts[gt == g])
        j = c[list(r).index(a)] if a in r else None
        vp = hull_vol(pts[pred == p_ids[j]]) if j is not None and p_ids else 0.0
        if vg > 1:
            vol_err.append(abs(vp - vg) / vg)
    return {"miou": float(best.mean()), "recall50": float(np.mean(best > 0.5)),
            "vol_ape": vol_err, "n_gt": len(g_ids), "n_pred": len(p_ids)}


def summarize(scores: list[dict]) -> dict:
    ape = np.concatenate([s["vol_ape"] for s in scores])
    return {"mIoU": float(np.mean([s["miou"] for s in scores])),
            "recall@IoU0.5": float(np.mean([s["recall50"] for s in scores])),
            "item_volume_mape_pct": float(100 * ape.mean()),
            "item_volume_median_ape_pct": float(100 * np.median(ape)),
            "count_exact_pct": float(100 * np.mean([s["n_gt"] == s["n_pred"] for s in scores]))}


def _sub(X: np.ndarray, Y: np.ndarray, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Same per-scene subsample as eval.mf3d_seg_train (rng seed 1), so methods see the same points."""
    sel = np.random.default_rng(1).choice(X.shape[1], n, replace=False)
    return X[:, sel], Y[:, sel]


def dbscan_eval(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tr, te = dict(np.load(data / "train.npz")), dict(np.load(data / "test.npz"))
    for d in (tr, te):
        for view in ("full", "top"):
            d[view], d[f"{view}_lab"] = _sub(d[view], d[f"{view}_lab"])
    report = {}
    for view in ("full", "top"):
        # eps chosen on train scenes
        best = None
        for eps in (0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025):
            sc = [score_scene(tr[view][i], tr[f"{view}_lab"][i],
                              DBSCAN(eps=eps, min_samples=8).fit_predict(tr[view][i])) for i in range(0, 200)]
            m = np.mean([s["miou"] for s in sc])
            if best is None or m > best[1]:
                best = (eps, m)
        eps = best[0]
        sc = [score_scene(te[view][i], te[f"{view}_lab"][i],
                          DBSCAN(eps=eps, min_samples=8).fit_predict(te[view][i])) for i in range(len(te[view]))]
        report[view] = {"eps_m": eps, **summarize(sc)}
        print(view, json.dumps(report[view]), flush=True)
    json.dump(report, open(out / "dbscan.json", "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "dbscan"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.path, args.out, args.n_train, args.n_test)
    else:
        dbscan_eval(args.path, args.out)


if __name__ == "__main__":
    main()

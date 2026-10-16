"""Learned heads on frozen DINOv2 features + single-frame geometry, Nutrition5k overhead.

    python -m eval.n5k_regress ../data/datasets/Nutrition5k --out ../results/n5k/regress

Inputs per dish: eval.n5k_features embeddings (food-pooled + CLS) and the
geometric features cached by eval.nutrition5k (area, container / dish / ring /
global volumes, dish type). No ingredient names: the image has to supply food
identity.

Targets: log1p of total mass, kcal, fat, carb, protein (log for the heavy right
tail; exp of a log-scale fit is a median, which suits MAE). For every feature
set x target the model (ridge, RBF SVR, gradient boosting) is chosen by 5-fold
CV MAE on depth_train_ids, refit on all of train and scored once on
depth_test_ids (507 dishes), the split of the Nutrition5k paper's table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from eval.nutrition5k import container_volume, is_bowl, read_metadata, read_split

TARGETS = ("mass", "kcal", "fat", "carb", "protein")
PUBLISHED = {   # Thames et al. 2021, overhead RGB-D, depth_test_ids: MAE and % of mean
    "Nutrition5k 2021 (RGB-D)": {"kcal": (41.3, 16.5), "mass": (29.4, 13.7)},
}


def geometry(r: dict, container: bool = True) -> list[float]:
    """Log volumes/area plus dish type; `container=False` drops the dish-surface features."""
    l = lambda k: float(np.log1p(max(r.get(k, 0.0), 0.0))) if np.isfinite(r.get(k, np.nan)) else 0.0  # noqa: E731
    f = [l("area_mask"), l("vol_hm_mask"), l("vol_hull_mask"), l("vol_hm_global"), l("vol_hm_center"),
         r.get("mask_frac", 0.0)]
    if container:
        rim = r.get("dish_rim_mm", np.nan)
        cv = container_volume(r) if "vol_dish" in r and "vol_hm_mask" in r else 0.0
        f += [float(np.log1p(max(cv, 0.0))), l("vol_dish"), float(is_bowl(r)),
              float(r.get("dish_fallback", True)), (rim if np.isfinite(rim) else 0.0) / 100,
              float(r.get("n_dishes", 0))]
    return f


def candidates(n_img: int) -> dict:
    """Model zoo; the image block (if any) is PCA'd for boosting only."""
    c = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))),
        "svr_c3": make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.05)),
        "svr_c10": make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
    }
    if n_img == 0:
        c["gbm"] = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, random_state=0)
    return c


class PCAThenGBM:
    """Gradient boosting on [geometry, PCA-64 of the image block]."""

    def __init__(self, n_geo: int):
        self.n_geo = n_geo

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X[:, self.n_geo:])
        self.pca = PCA(64, random_state=0).fit(self.sc.transform(X[:, self.n_geo:]))
        self.gbm = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, random_state=0)
        self.gbm.fit(self._z(X), y)
        return self

    def _z(self, X):
        return np.c_[X[:, :self.n_geo], self.pca.transform(self.sc.transform(X[:, self.n_geo:]))]

    def predict(self, X):
        return self.gbm.predict(self._z(X))


def cv_mae(make, X, y_log, folds=5) -> float:
    err = 0.0
    for tr, va in KFold(folds, shuffle=True, random_state=0).split(X):
        m = make().fit(X[tr], y_log[tr])
        err += np.abs(np.expm1(m.predict(X[va])) - np.expm1(y_log[va])).sum()
    return err / len(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n5k", type=Path, default=Path("../results/n5k"))
    ap.add_argument("--features", type=Path, default=Path("../results/n5k/features_dinov2l.npz"))
    ap.add_argument("--out", type=Path, default=Path("../results/n5k/regress"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(args.root)
    rows = {r["dish_id"]: r for r in map(json.loads, open(args.n5k / "per_dish.jsonl"))}
    fz = np.load(args.features)
    fidx = {d: i for i, d in enumerate(fz["ids"])}
    img = np.c_[fz["food"], fz["cls"]].astype(np.float32)

    split = {s: [d for d in read_split(args.root, s) if d in fidx and d in meta]
             for s in ("depth_train_ids", "depth_test_ids")}
    tr, te = split["depth_train_ids"], split["depth_test_ids"]
    print(f"train {len(tr)}, test {len(te)}")
    Y = {s: {t: np.log1p(np.array([meta[d][t] for d in ids])) for t in TARGETS} for s, ids in
         (("tr", tr), ("te", te))}

    def block(ids, kind):
        r = lambda d: rows.get(d, {})  # noqa: E731
        if kind == "geo":
            return np.array([geometry(r(d)) for d in ids])
        if kind == "geo_nocontainer":
            return np.array([geometry(r(d), container=False) for d in ids])
        if kind == "img":
            return img[[fidx[d] for d in ids]]
        raise ValueError(kind)

    feature_sets = {
        "geometry only": ["geo"],
        "image only": ["img"],
        "image + geometry (no container)": ["geo_nocontainer", "img"],
        "image + geometry": ["geo", "img"],
    }
    report = {"n_train": len(tr), "n_test": len(te), "published": PUBLISHED, "results": {}}
    preds = {}
    for name, kinds in feature_sets.items():
        Xtr = np.hstack([block(tr, k) for k in kinds])
        Xte = np.hstack([block(te, k) for k in kinds])
        n_geo = sum(block(tr[:1], k).shape[1] for k in kinds if k != "img")
        n_img = Xtr.shape[1] - n_geo
        res = {}
        for t in TARGETS:
            zoo = {k: (lambda m=m: __import__("sklearn").base.clone(m)) for k, m in candidates(n_img).items()}
            if n_img and n_geo:
                zoo["pca_gbm"] = lambda: PCAThenGBM(n_geo)
            cv = {k: cv_mae(mk, Xtr, Y["tr"][t]) for k, mk in zoo.items()}
            best = min(cv, key=cv.get)
            p = np.expm1(zoo[best]().fit(Xtr, Y["tr"][t]).predict(Xte)).clip(min=0)
            true = np.expm1(Y["te"][t])
            mae = float(np.abs(p - true).mean())
            res[t] = {"model": best, "cv_mae_train": {k: round(v, 2) for k, v in cv.items()},
                      "test_mae": mae, "test_pct_of_mean": 100 * mae / float(true.mean())}
            preds[(name, t)] = p
            print(f"{name:34s} {t:8s} {best:8s} test MAE {mae:7.1f}  {res[t]['test_pct_of_mean']:5.1f}%", flush=True)
        report["results"][name] = res

    json.dump(report, open(args.out / "summary.json", "w"), indent=2)
    with open(args.out / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = [(n, t) for n in feature_sets for t in TARGETS]
        w.writerow(["dish_id"] + [f"true_{t}" for t in TARGETS] + [f"{n}|{t}" for n, t in cols])
        for i, d in enumerate(te):
            w.writerow([d] + [meta[d][t] for t in TARGETS] + [f"{preds[c][i]:.1f}" for c in cols])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

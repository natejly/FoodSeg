"""Learned per-item hidden-volume correction ("volume head") on synthetic food scans.

    python -m pipeline.eval.volume_head                 # features (cached) -> all protocols -> results/volume_head/
    python -m pipeline.eval.volume_head --reextract     # recompute features.npz first
    python -m pipeline.eval.volume_head --quick         # fewer repeats / bootstrap draws (smoke test)

Question: how much of the hidden volume of an item can be recovered statistically from per-item cues
(features in volume_head_features.py) instead of shape completion?  Everything is on log scale:

    target "ratio":   r = log(V_true / V_vis)   (V_vis = foodscan container-relative visible volume)
    target "direct":  log V_true
    prediction        V_hat = V_vis * exp(r_hat)   or   exp(direct)

Models: none (V_vis), global (median r), kind (per-kind median r), ridge, hgb (HistGradientBoosting,
squared error), hgb_l1 (absolute error), hgb_direct, mlp.  Mass = V_hat * rho_hat, rho_hat = median
mass/volume of the training items of the same category (fallback global); the baseline uses V_vis * rho_hat.

Protocols: lobo_v12 (train 5 batches -> test hard_v12), rev_v12 (train hard_v12 -> test 5 batches),
lobo_all (leave-one-batch-out over the 6 batches, pooled), catcv (category-disjoint 5-fold, 5 random
group shuffles, per-item APE averaged over the repeats).  Populations: `valid` (manifest dropped_items
removed) and `all` (every item with >= 20 points and a positive visible volume).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import volume_head_features as F  # noqa: E402

OUT = _ROOT / "results" / "volume_head"
warnings.filterwarnings("ignore")

FEATURE_GROUPS = {
    "visible volumes": ["lv_foodscan", "lv_voxel", "lv_hull", "lv_table", "r_hull_fs", "r_voxel_fs"],
    "point stats": ["n_pts", "dens_pts_cm2", "pts_per_ml"],
    "footprint": ["area_cm2", "convexity", "obs_frac", "hole_other_frac"],
    "heights": ["h_max", "h_mean", "h_p10", "h_p50", "h_p90", "edge_h", "peak_ratio"],
    "box": ["len_mm", "wid_mm", "hgt_mm", "aspect", "height_to_width"],
    "contact / stacking": ["contact_item", "contact_wall", "step_mean", "step_min", "step_max", "frac_nbr_higher"],
    "views / lift": ["n_views", "lift_share", "lift_nvotes"],
    "scan context": ["n_items", "scan_lv_total", "fullness", "item_share"],
    "container": ["cont_plate", "cont_R_cm", "cont_cap_ml", "cont_H_mm"],
    "kind": [f"kind_{k}" for k in F.KINDS],
}
SIZE_BINS = [("<10 mL", 0, 10), ("10-50 mL", 10, 50), (">50 mL", 50, 1e9)]


# ------------------------------------------------------------------------------------------ models

class HeadModel:
    """fit(X, lv_vis, lv_true, kind) / predict(X, lv_vis, kind) -> predicted log V_true."""

    def __init__(self, name: str = "hgb", seed: int = 0):
        self.name, self.seed = name, seed

    # -- helpers
    def _prep(self, X, fit=False):
        X = np.asarray(X, np.float64)
        if fit:
            self.med_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            self.med_ = np.where(np.isfinite(self.med_), self.med_, 0.0)
        Xi = np.where(np.isfinite(X), X, self.med_)
        if self.name in ("ridge", "mlp"):
            if fit:
                self.mu_, self.sd_ = Xi.mean(0), Xi.std(0) + 1e-9
            Xi = np.clip((Xi - self.mu_) / self.sd_, -5, 5)
            return Xi
        return X     # HGB handles NaN natively

    def fit(self, X, lv_vis, lv_true, kind=None):
        r = np.asarray(lv_true) - np.asarray(lv_vis)
        n = self.name
        self.kind_med_ = {}
        if n == "none":
            return self
        self.gmed_ = float(np.median(r))
        if n == "global":
            return self
        if n == "kind":
            for k in set(kind):
                self.kind_med_[k] = float(np.median(r[np.asarray(kind) == k]))
            return self
        Xp = self._prep(X, fit=True)
        if n == "ridge":
            self.m_ = Ridge(alpha=10.0).fit(Xp, r - self.gmed_)
        elif n in ("hgb", "hgb_l1", "hgb_direct"):
            loss = "absolute_error" if n == "hgb_l1" else "squared_error"
            y = np.asarray(lv_true) if n == "hgb_direct" else r
            self.m_ = HistGradientBoostingRegressor(
                loss=loss, max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=8,
                l2_regularization=1.0, random_state=self.seed).fit(Xp, y)
        elif n == "mlp":
            self.m_ = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.3, max_iter=1500, random_state=self.seed + s,
                                    learning_rate_init=3e-3).fit(Xp, r - self.gmed_) for s in range(5)]
        else:
            raise ValueError(n)
        return self

    def predict(self, X, lv_vis, kind=None):
        lv_vis = np.asarray(lv_vis, np.float64)
        n = self.name
        if n == "none":
            return lv_vis
        if n == "global":
            return lv_vis + self.gmed_
        if n == "kind":
            return lv_vis + np.array([self.kind_med_.get(k, self.gmed_) for k in kind])
        Xp = self._prep(X)
        if n == "ridge":
            return lv_vis + self.gmed_ + self.m_.predict(Xp)
        if n in ("hgb", "hgb_l1"):
            return lv_vis + self.m_.predict(Xp)
        if n == "hgb_direct":
            return self.m_.predict(Xp)
        if n == "mlp":
            return lv_vis + self.gmed_ + np.mean([m.predict(Xp) for m in self.m_], 0)
        raise ValueError(n)


MODELS = ["none", "global", "kind", "ridge", "hgb", "hgb_l1", "hgb_direct", "mlp"]


# ------------------------------------------------------------------------------------------ data

ENV_CSV = _ROOT / "results" / "rq1" / "envelope_items.csv"


def _env_lookup(d):
    """GT envelope volume (mL; results/rq1/envelope_items.csv) joined by (batch, scan, item_id); NaN if absent."""
    import csv
    env = {}
    for r in csv.DictReader(open(ENV_CSV)):
        try:
            env[(r["batch"], r["scan"], int(r["item_id"]))] = float(r["volume_envelope_gt"])
        except (ValueError, TypeError):
            pass
    return np.array([env.get((b, s, int(i)), np.nan) for b, s, i in zip(d["batch"], d["scan"], d["item_id"])])


def _scan_nviews(d):
    """Number of captured frames of each scan (frames.jsonl lines; 8 for v9-v11, 24 elsewhere)."""
    cache, out = {}, []
    for b, s in zip(d["batch"], d["scan"]):
        if (b, s) not in cache:
            fp = _ROOT / "data" / "synth" / str(b) / "scans" / str(s) / "frames.jsonl"
            cache[(b, s)] = float(sum(1 for _ in open(fp))) if fp.exists() else np.nan
        out.append(cache[(b, s)])
    return np.array(out)


class Data:
    """target: "envelope" (log V_env_gt / V_vis, default) or "solid" (log volume_ml / V_vis).  Both use the SAME
    item population (items with an envelope value), so the two targets are directly comparable."""

    def __init__(self, path: Path, population: str, target: str = "envelope", mass_source: str = "synth"):
        d = F.load(path)
        if mass_source != "synth":      # USDA side-label masses (pipeline/eval/synth_mass_labels.py); NaN where unlabeled
            from pipeline.eval import synth_mass_labels as ML
            lk = ML.load_mass_lookup(mass_source)
            d["mass_g"] = np.array([lk.get((b, s, int(i)), np.nan) for b, s, i in zip(d["batch"], d["scan"], d["item_id"])])
        d["vol_solid"] = d["vol_true"].copy()
        d["vol_env"] = _env_lookup(d)
        d["nv_scan"] = _scan_nviews(d)
        d["vol_true"] = d["vol_env"] if target == "envelope" else d["vol_solid"]
        d["rho"] = d["mass_g"] / d["vol_true"]
        self.target = target
        vis = d["vis_fs"]
        ok = (np.isfinite(vis) & (vis > 0) & np.isfinite(d["vol_solid"]) & (d["vol_solid"] > 0)
              & np.isfinite(d["vol_env"]) & (d["vol_env"] > 0))
        if population == "valid":
            ok &= d["item_valid"] > 0.5
        self.d = {k: v[ok] for k, v in d.items()}
        d = self.d
        self.n = int(ok.sum())
        self.X = np.stack([d[f] for f in F.FEATURES], 1)
        self.lv_vis = np.log(d["vis_fs"])
        self.lv_true = np.log(d["vol_true"])
        self.vis, self.true = d["vis_fs"], d["vol_true"]
        self.kind, self.cat, self.batch = d["kind"], d["category"], d["batch"]
        self.scan = np.array([f"{b}/{s}" for b, s in zip(d["batch"], d["scan"])])
        self.mass, self.rho = d["mass_g"], d["rho"]
        self.population = population

    def subset(self, mask):
        """Copy restricted to the items where `mask` is True."""
        import copy
        o = copy.copy(self)
        o.d = {k: v[mask] for k, v in self.d.items()}
        for k in ("X", "lv_vis", "lv_true", "vis", "true", "kind", "cat", "batch", "scan", "mass", "rho"):
            setattr(o, k, getattr(self, k)[mask])
        o.n = int(mask.sum())
        return o

    def cols(self, names):
        return [F.FEATURES.index(n) for n in names]


def make_models(name, seed=0):
    return HeadModel(name, seed)


def fit_predict(name, D: Data, tr, te, cols=None, seed=0):
    X = D.X if cols is None else D.X[:, cols]
    m = make_models(name, seed).fit(X[tr], D.lv_vis[tr], D.lv_true[tr], D.kind[tr])
    return m.predict(X[te], D.lv_vis[te], D.kind[te]), m


def rho_hat(D: Data, tr, te):
    """Category-median density of the training items (fallback global median)."""
    cat_tr, rho_tr = D.cat[tr], D.rho[tr]
    g = np.nanmedian(rho_tr) if np.isfinite(rho_tr).any() else 1.0
    med = {c: np.nanmedian(rho_tr[cat_tr == c]) for c in set(cat_tr) if np.isfinite(rho_tr[cat_tr == c]).any()}
    return np.array([med.get(c, g) for c in D.cat[te]])


# ------------------------------------------------------------------------------------------ splits

def split_lobo_v12(D):
    te = np.nonzero(D.batch == "hard_v12")[0]
    return [(np.nonzero(D.batch != "hard_v12")[0], te)]


def split_rev_v12(D):
    return [(np.nonzero(D.batch == "hard_v12")[0], np.nonzero(D.batch != "hard_v12")[0])]


def split_lobo_all(D):
    return [(np.nonzero(D.batch != b)[0], np.nonzero(D.batch == b)[0]) for b in sorted(set(D.batch))]


def split_catcv(D, seed, k=5):
    cats = np.array(sorted(set(D.cat)))
    rng = np.random.default_rng(seed)
    rng.shuffle(cats)
    # greedy balance by item count
    counts = {c: int((D.cat == c).sum()) for c in cats}
    order = sorted(cats, key=lambda c: -counts[c] + rng.random() * 0.5)
    folds = [[] for _ in range(k)]
    load = [0] * k
    for c in order:
        i = int(np.argmin(load))
        folds[i].append(c)
        load[i] += counts[c]
    out = []
    for f in folds:
        te = np.nonzero(np.isin(D.cat, f))[0]
        out.append((np.setdiff1d(np.arange(D.n), te), te))
    return out


# ------------------------------------------------------------------------------------------ metrics

def ape_of(lv_pred, D, idx):
    return np.abs(np.exp(lv_pred - D.lv_true[idx]) - 1.0)


def summarize(lv_pred, D: Data, idx, mass_pred=None):
    """Per-item median/mean APE, median signed bias (%), dish-total median APE / bias (%)."""
    e = np.exp(lv_pred - D.lv_true[idx]) - 1.0
    ape = np.abs(e)
    out = dict(n=int(len(idx)), med_ape=100 * float(np.median(ape)), mean_ape=100 * float(ape.mean()),
               med_bias=100 * float(np.median(e)), mean_bias=100 * float(e.mean()))
    sc = D.scan[idx]
    dp, dt = [], []
    for s in np.unique(sc):
        m = sc == s
        dp.append(np.exp(lv_pred[m]).sum())
        dt.append(D.true[idx][m].sum())
    de = np.array(dp) / np.array(dt) - 1.0
    out.update(dish_med_ape=100 * float(np.median(np.abs(de))), dish_bias=100 * float(np.median(de)), n_dish=len(dp))
    if mass_pred is not None:
        mt = D.mass[idx]
        ok = np.isfinite(mt) & (mt > 0)
        if ok.any():
            me = mass_pred[ok] / mt[ok] - 1.0
            out.update(mass_med_ape=100 * float(np.median(np.abs(me))), mass_mean_ape=100 * float(np.abs(me).mean()),
                       mass_bias=100 * float(np.median(me)))
    return out


def boot_diff(ape_m, ape_b, scans, B=1000, seed=0):
    """Cluster (by scan) paired bootstrap of model - baseline APE: median and mean difference, 95% CI, P(diff<0)."""
    rng = np.random.default_rng(seed)
    us, inv = np.unique(scans, return_inverse=True)
    groups = [np.nonzero(inv == i)[0] for i in range(len(us))]
    dm, da = [], []
    for _ in range(B):
        pick = rng.integers(0, len(us), len(us))
        idx = np.concatenate([groups[i] for i in pick])
        dm.append(np.median(ape_m[idx]) - np.median(ape_b[idx]))
        da.append(np.mean(ape_m[idx]) - np.mean(ape_b[idx]))
    dm, da = 100 * np.array(dm), 100 * np.array(da)
    pt_m = 100 * (np.median(ape_m) - np.median(ape_b))
    pt_a = 100 * (np.mean(ape_m) - np.mean(ape_b))
    return dict(d_med=pt_m, d_med_lo=float(np.percentile(dm, 2.5)), d_med_hi=float(np.percentile(dm, 97.5)),
                d_mean=pt_a, d_mean_lo=float(np.percentile(da, 2.5)), d_mean_hi=float(np.percentile(da, 97.5)),
                p_better=float((dm < 0).mean()))


# ------------------------------------------------------------------------------------------ protocols

def run_protocol(D: Data, name: str, models=MODELS, repeats=5, cols=None):
    """-> dict model -> {idx (test indices, ordered), pred (n_rep x n) log V, rho}.  For catcv repeats>1."""
    res = {}
    if name == "catcv":
        reps = [split_catcv(D, 100 + r) for r in range(repeats)]
    else:
        reps = [{"lobo_v12": split_lobo_v12, "rev_v12": split_rev_v12, "lobo_all": split_lobo_all}[name](D)]
    for m in models:
        preds, rhos = [], []
        for splits in reps:
            p = np.full(D.n, np.nan)
            rh = np.full(D.n, np.nan)
            for tr, te in splits:
                p[te], _ = fit_predict(m, D, tr, te, cols)
                rh[te] = rho_hat(D, tr, te)
            preds.append(p)
            rhos.append(rh)
        preds, rhos = np.array(preds), np.array(rhos)
        idx = np.nonzero(np.isfinite(preds[0]))[0]
        res[m] = dict(idx=idx, pred=preds[:, idx], rho=rhos[:, idx])
    return res


def table_rows(D, res, models=MODELS):
    rows = {}
    for m in models:
        r = res[m]
        idx = r["idx"]
        per = [summarize(r["pred"][i], D, idx, np.exp(r["pred"][i]) * r["rho"][i]) for i in range(len(r["pred"]))]
        rows[m] = {k: float(np.mean([p[k] for p in per])) if k in per[0] else float("nan") for k in per[0]}
        rows[m]["n"] = per[0]["n"]
    return rows


def mean_ape(res, m, D):
    r = res[m]
    return np.mean([ape_of(r["pred"][i], D, r["idx"]) for i in range(len(r["pred"]))], 0)


def breakdown(D, res, models, key):
    """APE / bias per group (size bins, kind, stacked) for the listed models (rep-averaged)."""
    idx = res[models[0]]["idx"]
    if key == "size":
        groups = [(n, (D.true[idx] >= lo) & (D.true[idx] < hi)) for n, lo, hi in SIZE_BINS]
    elif key == "kind":
        groups = [(k, D.kind[idx] == k) for k in F.KINDS]
    elif key == "stacked":
        st = D.d["stacked"][idx] > 0.5
        groups = [("stacked scans", st), ("unstacked scans", ~st)]
    elif key == "occl":
        oc = D.d["occl_frac"][idx]
        groups = [("occluded <50%", oc < 0.5), ("50-80%", (oc >= 0.5) & (oc < 0.8)), (">=80%", oc >= 0.8)]
    out = []
    for g, mask in groups:
        if mask.sum() == 0:
            continue
        row = {"group": g, "n": int(mask.sum())}
        for m in models:
            r = res[m]
            a = np.mean([ape_of(r["pred"][i], D, idx) for i in range(len(r["pred"]))], 0)[mask]
            b = np.mean([np.exp(r["pred"][i] - D.lv_true[idx]) - 1 for i in range(len(r["pred"]))], 0)[mask]
            row[m] = (100 * float(np.median(a)), 100 * float(np.median(b)))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------ importance

def perm_importance(D: Data, splits_list, groups: dict, model="hgb", n_perm=5, seed=0):
    """Held-out permutation importance: increase in mean |log error| when a feature group is shuffled
    within the test fold (averaged over folds / repeats / permutations)."""
    rng = np.random.default_rng(seed)
    gcols = {g: [F.FEATURES.index(f) for f in fs] for g, fs in groups.items()}
    inc = {g: [] for g in gcols}
    base_err = []
    for splits in splits_list:
        for tr, te in splits:
            p0, mdl = fit_predict(model, D, tr, te, None)
            e0 = np.mean(np.abs(p0 - D.lv_true[te]))
            base_err.append(e0)
            for g, cs in gcols.items():
                es = []
                for _ in range(n_perm):
                    Xp = D.X[te].copy()
                    perm = rng.permutation(len(te))
                    Xp[:, cs] = Xp[perm][:, cs]
                    p = mdl.predict(Xp, D.lv_vis[te], D.kind[te])
                    es.append(np.mean(np.abs(p - D.lv_true[te])))
                inc[g].append(np.mean(es) - e0)
    return {g: float(np.mean(v)) for g, v in inc.items()}, float(np.mean(base_err))


# ------------------------------------------------------------------------------------------ learning curve

def learning_curve(D: Data, protocol: str, models=("global", "hgb"), fracs=(0.25, 0.5, 1.0), reps=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for fr in fracs:
        accs = {m: [] for m in models}
        base = []
        for r in range(reps if fr < 1 else 1):
            if protocol == "lobo_v12":
                splits = split_lobo_v12(D)
            else:
                splits = split_catcv(D, 200 + r)
            P = {m: np.full(D.n, np.nan) for m in models}
            for tr, te in splits:
                # subsample whole scans so the fraction means "fraction of synthetic scans"
                sc = np.unique(D.scan[tr])
                keep = rng.choice(sc, max(2, int(round(fr * len(sc)))), replace=False) if fr < 1 else sc
                trs = tr[np.isin(D.scan[tr], keep)]
                for m in models:
                    P[m][te], _ = fit_predict(m, D, trs, te)
            idx = np.nonzero(np.isfinite(P[models[0]]))[0]
            base.append(100 * float(np.median(np.abs(np.exp(D.lv_vis[idx] - D.lv_true[idx]) - 1))))
            for m in models:
                accs[m].append(100 * float(np.median(ape_of(P[m][idx], D, idx))))
        row = {"frac": fr, "n_train_items": int(np.mean([len(tr) for tr, _ in splits]) * fr), "baseline": float(np.mean(base))}
        for m in models:
            row[m] = float(np.mean(accs[m]))
            row[m + "_sd"] = float(np.std(accs[m]))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------ report

def fmt(x, d=0):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", choices=["envelope", "solid"], default="envelope",
                    help="regress log(V_envelope_gt / V_vis) (default) or the original log(volume_ml / V_vis)")
    ap.add_argument("--mass-source", choices=["synth", "usda_bulk", "usda_material"], default="synth",
                    help="mass labels behind the mass columns: generator/asset (default) or USDA side labels; "
                         "non-default outputs go to results/mass_labels/")
    ap.add_argument("--pops", nargs="+", default=["valid"], choices=["valid", "all"])
    a = ap.parse_args()
    feat = OUT / "features.npz"
    if a.reextract or not feat.exists():
        rows = F.extract(_ROOT / "data" / "synth", F.BATCHES)
        OUT.mkdir(parents=True, exist_ok=True)
        F.save(rows, feat)
    reps = 2 if a.quick else 5
    B = 200 if a.quick else 1000
    results: dict = {}
    md = []
    LABEL = {"none": "baseline (V_vis)", "global": "global median ratio", "kind": "per-kind median ratio",
             "ridge": "ridge", "hgb": "HGB (L2)", "hgb_l1": "HGB (L1)", "hgb_direct": "HGB direct log V", "mlp": "MLP"}

    for pop in a.pops:
        D = Data(feat, pop, a.target, a.mass_source)
        results[pop] = {"n_items": D.n, "n_scans": int(len(set(D.scan))), "per_batch": {b: int((D.batch == b).sum()) for b in sorted(set(D.batch))}}
        md.append(f"\n## Population: {pop} (n = {D.n} items, {len(set(D.scan))} scans)\n")
        prot_res = {}
        for prot in ("lobo_v12", "rev_v12", "lobo_all", "catcv"):
            res = run_protocol(D, prot, MODELS, repeats=reps)
            prot_res[prot] = res
            tr = table_rows(D, res)
            results[pop][prot] = tr
            md.append(f"\n### {prot}\n")
            hdr = ["model", "n", "med APE %", "mean APE %", "med bias %", "dish med APE %", "dish bias %", "mass med APE %", "mass bias %"]
            md.append(md_table(hdr, [[LABEL[m], int(tr[m]["n"]), fmt(tr[m]["med_ape"]), fmt(tr[m]["mean_ape"]), fmt(tr[m]["med_bias"]),
                                       fmt(tr[m]["dish_med_ape"]), fmt(tr[m]["dish_bias"]), fmt(tr[m].get("mass_med_ape")),
                                       fmt(tr[m].get("mass_bias"))] for m in MODELS]))
        # best model by catcv median APE (different protocol from the lobo tests)
        learned = ["ridge", "hgb", "hgb_l1", "hgb_direct", "mlp"]
        best = min(learned, key=lambda m: results[pop]["catcv"][m]["med_ape"])
        results[pop]["best"] = best
        md.append(f"\nBest learned model by category-CV median APE: **{LABEL[best]}**.\n")

        # paired bootstrap vs baseline and vs global median
        md.append("\n### Paired cluster bootstrap (by scan), model - baseline APE, percentage points (95% CI)\n")
        brow = []
        results[pop]["boot"] = {}
        for prot in ("lobo_v12", "rev_v12", "lobo_all", "catcv"):
            res = prot_res[prot]
            sc = D.scan[res["none"]["idx"]]
            ab = mean_ape(res, "none", D)
            ag = mean_ape(res, "global", D)
            for m, ref, an in ((best, "baseline", ab), (best, "global", ag), ("global", "baseline", ab)):
                am = mean_ape(res, m, D)
                bd = boot_diff(am, an, sc, B)
                results[pop]["boot"][f"{prot}:{m}-{ref}"] = bd
                brow.append([prot, f"{LABEL[m]} - {ref}", f"{bd['d_med']:+.1f} [{bd['d_med_lo']:+.1f}, {bd['d_med_hi']:+.1f}]",
                             f"{bd['d_mean']:+.1f} [{bd['d_mean_lo']:+.1f}, {bd['d_mean_hi']:+.1f}]", fmt(100 * bd["p_better"])])
        md.append(md_table(["protocol", "comparison", "d median APE", "d mean APE", "P(better) %"], brow))

        # breakdowns for best model
        for prot in ("lobo_v12", "catcv", "rev_v12"):
            for key in ("size", "kind", "stacked", "occl"):
                bd = breakdown(D, prot_res[prot], ["none", "global", best], key)
                results[pop].setdefault("breakdown", {})[f"{prot}:{key}"] = bd
                md.append(f"\n#### {prot} by {key}: median APE % / median bias %\n")
                md.append(md_table(["group", "n", "baseline", "global ratio", LABEL[best]],
                                   [[r["group"], r["n"]] + [f"{r[m][0]:.0f} / {r[m][1]:+.0f}" for m in ("none", "global", best)] for r in bd]))

        if pop == "valid":
            # batch structure + diagnostics that separate batch offsets from per-item cues
            r_all = D.lv_true - D.lv_vis
            brow = []
            for b in sorted(set(D.batch)):
                m = D.batch == b
                brow.append([b, int(m.sum()), fmt(np.median(D.d["n_views"][m])), fmt(np.exp(np.median(r_all[m])), 2),
                             fmt(100 * np.median(np.abs(np.exp(-r_all[m]) - 1)))])
            md.append("\n### Per-batch structure (valid items)\n")
            md.append(md_table(["batch", "n items", "median views seen", "median V_true/V_vis", "baseline med APE %"], brow))
            pb = []
            for rep in range(reps):
                p = np.full(D.n, np.nan)
                for tr, te in split_catcv(D, 100 + rep):
                    for b in set(D.batch[te]):
                        t2 = te[D.batch[te] == b]
                        trb = tr[D.batch[tr] == b]
                        p[t2] = D.lv_vis[t2] + np.median(r_all[trb] if len(trb) else r_all[tr])
                pb.append(summarize(p, D, np.arange(D.n)))
            results[pop]["batch_median_catcv"] = {k: float(np.mean([x[k] for x in pb])) for k in pb[0]}
            md.append(f"\nDiagnostic, category CV with a per-batch median ratio (uses the batch id, i.e. the number of views / difficulty preset): "
                      f"median APE {results[pop]['batch_median_catcv']['med_ape']:.0f}%, bias {results[pop]['batch_median_catcv']['med_bias']:+.0f}%, "
                      f"dish median APE {results[pop]['batch_median_catcv']['dish_med_ape']:.0f}%.\n")
            Dv = D.subset(D.batch == "hard_v12")
            rv = run_protocol(Dv, "catcv", MODELS, repeats=reps)
            tv = table_rows(Dv, rv)
            results[pop]["catcv_v12only"] = tv
            md.append(f"\n### catcv_v12only: category-disjoint 5-fold inside hard_v12 only (n = {Dv.n}; removes batch offsets)\n")
            md.append(md_table(["model", "n", "med APE %", "mean APE %", "med bias %", "dish med APE %", "mass med APE %"],
                               [[LABEL[m], int(tv[m]["n"]), fmt(tv[m]["med_ape"]), fmt(tv[m]["mean_ape"]), fmt(tv[m]["med_bias"]),
                                 fmt(tv[m]["dish_med_ape"]), fmt(tv[m].get("mass_med_ape"))] for m in MODELS]))
            sc = Dv.scan[rv["none"]["idx"]]
            bb = boot_diff(mean_ape(rv, best, Dv), mean_ape(rv, "none", Dv), sc, B)
            results[pop]["boot_v12only"] = bb
            md.append(f"\n{LABEL[best]} - baseline in hard_v12 only: d median APE {bb['d_med']:+.1f} [{bb['d_med_lo']:+.1f}, {bb['d_med_hi']:+.1f}], "
                      f"d mean APE {bb['d_mean']:+.1f} [{bb['d_mean_lo']:+.1f}, {bb['d_mean_hi']:+.1f}].\n")
            # feature-set ablation (category CV)
            sets = {
                "kind + lv_* only": FEATURE_GROUPS["visible volumes"] + FEATURE_GROUPS["kind"],
                "+ geometry (point stats, footprint, heights, box)": FEATURE_GROUPS["visible volumes"] + FEATURE_GROUPS["kind"] + FEATURE_GROUPS["point stats"] + FEATURE_GROUPS["footprint"] + FEATURE_GROUPS["heights"] + FEATURE_GROUPS["box"],
                "all features": F.FEATURES,
                "all minus views/lift (needs no SAM masks)": [f for f in F.FEATURES if f not in FEATURE_GROUPS["views / lift"]],
                "all minus scan context + container": [f for f in F.FEATURES if f not in FEATURE_GROUPS["scan context"] + FEATURE_GROUPS["container"]],
            }
            arow = []
            results[pop]["ablation"] = {}
            for nm, fs in sets.items():
                for prot in ("catcv", "lobo_v12"):
                    r = run_protocol(D, prot, [best], repeats=reps, cols=D.cols(fs))
                    t = table_rows(D, r, [best])[best]
                    results[pop]["ablation"][f"{nm}|{prot}"] = t
                    arow.append([nm, prot, len(fs), fmt(t["med_ape"]), fmt(t["mean_ape"]), fmt(t["med_bias"]), fmt(t["dish_med_ape"])])
            # diagnostic oracle: add the hidden fraction (GT-only) as an input
            D.X = np.c_[D.X, D.d["occl_frac"]]
            F.FEATURES.append("occl_frac_ORACLE")
            for prot in ("catcv", "lobo_v12"):
                r = run_protocol(D, prot, [best], repeats=reps)
                t = table_rows(D, r, [best])[best]
                results[pop]["ablation"][f"oracle occluded_fraction|{prot}"] = t
                arow.append(["all + occluded_fraction (GT-only ORACLE, diagnostic)", prot, len(F.FEATURES), fmt(t["med_ape"]), fmt(t["mean_ape"]), fmt(t["med_bias"]), fmt(t["dish_med_ape"])])
            D.X = D.X[:, :-1]
            F.FEATURES.pop()
            md.append(f"\n### Feature-set ablation ({LABEL[best]})\n")
            md.append(md_table(["feature set", "protocol", "#feat", "med APE %", "mean APE %", "med bias %", "dish med APE %"], arow))

            # permutation importance
            md.append(f"\n### Held-out permutation importance ({LABEL[best]}; increase in mean |log error| when shuffled)\n")
            imp_cat, e0 = perm_importance(D, [split_catcv(D, 300 + r) for r in range(2 if a.quick else 3)], FEATURE_GROUPS, model=best)
            imp_lo, e1 = perm_importance(D, [split_lobo_v12(D)], FEATURE_GROUPS, model=best)
            single = {f: [f] for f in F.FEATURES}
            imp_single, _ = perm_importance(D, [split_catcv(D, 300)], single, model=best, n_perm=3)
            results[pop]["importance_group"] = {"catcv": imp_cat, "lobo_v12": imp_lo, "base_err_catcv": e0, "base_err_lobo": e1}
            results[pop]["importance_single"] = imp_single
            md.append(f"Baseline held-out mean |log error|: category-CV {e0:.3f}, lobo_v12 {e1:.3f}.\n\n")
            md.append(md_table(["feature group", "category CV", "lobo_v12"],
                               [[g, f"{imp_cat[g]:+.3f}", f"{imp_lo[g]:+.3f}"] for g in sorted(imp_cat, key=lambda g: -imp_cat[g])]))
            top = sorted(imp_single, key=lambda f: -imp_single[f])[:12]
            md.append("\nTop single features (category CV, fold 0 split seed 300):\n\n" +
                      md_table(["feature", "increase"], [[f, f"{imp_single[f]:+.3f}"] for f in top]))

            # learning curves
            md.append("\n### Learning curve: median item APE % vs fraction of training scans\n")
            for prot in ("lobo_v12", "catcv"):
                lc = learning_curve(D, prot, reps=3 if a.quick else 8)
                results[pop].setdefault("learning_curve", {})[prot] = lc
                md.append(f"\n{prot}:\n\n" + md_table(["train fraction", "~train items", "baseline", "global ratio", "HGB (L2)", "HGB sd"],
                          [[f"{r['frac']:.2f}", r["n_train_items"], fmt(r["baseline"]), fmt(r["global"]), fmt(r["hgb"]), fmt(r["hgb_sd"], 1)] for r in lc]))
    out_dir, sfx = (OUT, "") if a.mass_source == "synth" else (_ROOT / "results" / "mass_labels", f"_{a.mass_source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # NB: the original solid-target run (results.json / tables.md / summary.md) is kept; this writes per-target files.
    (out_dir / f"results_{a.target}{sfx}.json").write_text(json.dumps(results, indent=1, default=float))
    (out_dir / f"tables_{a.target}{sfx}.md").write_text(f"# target = {a.target}, mass source = {a.mass_source}\n" + "\n".join(md))
    print("wrote", out_dir / f"tables_{a.target}{sfx}.md")


if __name__ == "__main__":
    main()

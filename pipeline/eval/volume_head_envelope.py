"""Envelope-target follow-ups of the volume head: two-stage mass (part 2) and views-aware / mixed-density transfer (part 3).

    python -m pipeline.eval.volume_head_envelope [--quick] [--model hgb_l1]

Writes results/volume_head/mass_two_stage.md|json and results/volume_head/views_aware.md|json.

Part 2.  V_env_hat (corrected visible volume, target = GT envelope) -> mass, two ways, densities = per-kind medians
fitted on hard_v10 ONLY (as rq1_envelope.FIT_BATCH):
    (a) mass = V_env_hat * bulk_density(kind)                     bulk = mass / envelope
    (b) mass = V_env_hat / envelope_ratio(kind) * material(kind)  ratio = envelope / solid, material = mass / solid
Scored on hard_v12 for the baseline (V_vis, no correction) and the corrected volume, under two honest protocols:
    lobo_v12      the head is trained on the other 5 batches (batch-disjoint)
    catcv_v12     the head is trained on hard_v12 items of other food categories (category-disjoint, 5-fold x repeats)
Item APE with scan-cluster bootstrap CI of the paired difference; dish = per-scan sum over the evaluated items.

Part 3.  Views-aware features.  Extra features on top of volume_head_features.FEATURES (scan-level n_views = frames.jsonl
lines, 8 for hard_v9..v11, 24 for hard/hard_v3/hard_v12; log of it; points per view; lift votes / n_views; item
frames_visible / n_views).  Question: if the TRAINING set contains both 8-view and 24-view scans, does the head work on
held-out scans of each density (scan-disjoint, and the stricter category-disjoint), and how does that compare with the
batch-disjoint failure and with density-specific training?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import rq1_envelope as E  # noqa: E402
from pipeline.eval import synth_mass_labels as ML  # noqa: E402
from pipeline.eval import volume_head as VH  # noqa: E402
from pipeline.eval import volume_head_features as F  # noqa: E402

OUT = VH.OUT
MASS_SOURCE = "synth"        # set by --mass-source: mass labels for the density fit and the mass targets
EXTRA = ["nv_scan", "log_nv", "pts_per_view", "nvotes_frac", "vis_frac"]


# ------------------------------------------------------------------------------------------ helpers

def cluster_ci(vals_fn, scans, B=1000, seed=0):
    """Scan-cluster bootstrap of a statistic computed on the resampled index set -> (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    us, inv = np.unique(scans, return_inverse=True)
    groups = [np.nonzero(inv == i)[0] for i in range(len(us))]
    pt = vals_fn(np.arange(len(scans)))
    bs = [vals_fn(np.concatenate([groups[i] for i in rng.integers(0, len(us), len(us))])) for _ in range(B)]
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def dish_apes(scans, est, true):
    us = np.unique(scans)
    return np.array([abs(est[scans == s].sum() / true[scans == s].sum() - 1.0) for s in us]), us


def fmt_ci(t, d=0, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


def with_extras(D: VH.Data):
    """Copy of D whose X has the views-aware columns appended; returns (D2, names)."""
    d = D.d
    nv = d["nv_scan"]
    npts = np.exp(d["n_pts"])
    ex = np.c_[nv, np.log(nv), np.log(npts / nv), d["lift_nvotes"] / nv, d["n_views"] / nv]
    import copy
    o = copy.copy(D)
    o.X = np.c_[D.X, ex]
    return o, list(F.FEATURES) + EXTRA


# ------------------------------------------------------------------------------------------ part 2

def densities(items):
    items = ML.apply_mass_source(items, MASS_SOURCE)      # no-op for "synth"
    d = E.fit_densities(items, E.FIT_BATCH)
    rs = [r for r in items if r["batch"] == E.FIT_BATCH and r["volume_solid"] > 0 and np.isfinite(r["volume_envelope_gt"])
          and r["volume_envelope_gt"] > 0]
    pooled = float(np.median([r["volume_envelope_gt"] / r["volume_solid"] for r in rs]))
    ratio = {}
    for k in {str(r["kind"]) for r in rs}:
        sub = [r["volume_envelope_gt"] / r["volume_solid"] for r in rs if str(r["kind"]) == k]
        ratio[k] = float(np.median(sub)) if len(sub) >= 3 else pooled
    ratio["_all"] = pooled
    return d, ratio


def mass_two_stage(feat, model, quick):
    B = 200 if quick else 1000
    reps = 2 if quick else 5
    D = VH.Data(feat, "valid", "envelope", MASS_SOURCE)
    items = E.read_items(VH.ENV_CSV)
    dens, ratio = densities(items)
    Dv = D.subset(D.batch == "hard_v12")
    out, md = {"densities": dens, "ratio": ratio}, []

    def dget(table, k, key=None):
        e = table.get(k, table["_all"])
        return e[key] if key else e

    protos = {}
    r = VH.run_protocol(D, "lobo_v12", ["none", model], repeats=1)          # train on other 5 batches
    protos["lobo_v12 (batch-disjoint)"] = (D, r)
    r = VH.run_protocol(Dv, "catcv", ["none", model], repeats=reps)         # category-disjoint inside v12
    protos["catcv_v12 (category-disjoint, v12 only)"] = (Dv, r)

    md.append("### Densities fitted on hard_v10 only (per kind)\n")
    md.append(VH.md_table(["kind", "n", "bulk g/mL (mass/envelope)", "material g/mL (mass/solid)", "envelope/solid"],
                          [[k, v["n"], f"{v['bulk']:.2f}", f"{v['material']:.2f}", f"{ratio.get(k, ratio['_all']):.2f}"]
                           for k, v in dens.items()]))
    for pname, (DD, res) in protos.items():
        idx = res["none"]["idx"]
        sc, kind, mtrue = DD.scan[idx], DD.kind[idx], DD.mass[idx]
        ok = np.isfinite(mtrue) & (mtrue > 0)
        idx, sc, kind, mtrue = idx[ok], sc[ok], kind[ok], mtrue[ok]
        bulk = np.array([dget(dens, k, "bulk") for k in kind])
        mat = np.array([dget(dens, k, "material") for k in kind])
        rat = np.array([dget(ratio, k) for k in kind])
        arms = {}
        for arm in ("none", model):
            r_ = res[arm]
            keep = np.isin(r_["idx"], idx)
            V = np.exp(r_["pred"][:, keep]).mean(0) if False else np.exp(r_["pred"][:, keep])    # (reps, n)
            arms[arm] = V
        arms["oracle: V_envelope_gt"] = np.tile(DD.true[idx], (1, 1))
        rows = []
        mb = {}
        for path, fnc in (("(a) V x bulk", lambda V: V * bulk), ("(b) V / ratio x material", lambda V: V / rat * mat)):
            for arm, V in arms.items():
                M = fnc(V)                                   # (reps, n)
                ape = np.abs(M / mtrue - 1.0)                # (reps, n)
                a1 = ape.mean(0)                             # rep-averaged item APE
                bias = (M / mtrue - 1.0).mean(0)
                dape = np.mean([dish_apes(sc, M[i], mtrue)[0] for i in range(len(M))], 0)
                us = dish_apes(sc, M[0], mtrue)[1]
                item = cluster_ci(lambda j: 100 * np.median(a1[j]), sc, B)
                bi = cluster_ci(lambda j: 100 * np.median(bias[j]), sc, B)
                # dish: cluster over scans -> resample dish_ape values
                dsh = cluster_ci(lambda j: 100 * np.median(dape[j]), us, B)
                mb[(path, arm)] = a1
                rows.append([path, arm, len(idx), len(us), fmt_ci(item), fmt_ci(bi, sign=True), fmt_ci(dsh)])
                out.setdefault(pname, {})[f"{path}|{arm}"] = dict(item_med_ape=item, item_bias=bi, dish_med_ape=dsh, dish_ape_values=dape.tolist())
        md.append(f"\n### Mass on hard_v12, {pname}: {model} head, median APE % [95% scan-cluster CI]\n")
        md.append(VH.md_table(["density path", "volume", "n items", "n dishes", "item APE %", "item bias %", "dish APE %"], rows))
        pr = []
        for path in ("(a) V x bulk", "(b) V / ratio x material"):
            bd = VH.boot_diff(mb[(path, model)], mb[(path, "none")], sc, B)
            pr.append([path, f"{bd['d_med']:+.1f} [{bd['d_med_lo']:+.1f}, {bd['d_med_hi']:+.1f}]",
                       f"{bd['d_mean']:+.1f} [{bd['d_mean_lo']:+.1f}, {bd['d_mean_hi']:+.1f}]"])
            out[pname][f"paired|{path}"] = bd
        bd = VH.boot_diff(mb[("(b) V / ratio x material", "none")], mb[("(a) V x bulk", "none")], sc, B)
        bd2 = VH.boot_diff(mb[("(b) V / ratio x material", model)], mb[("(a) V x bulk", model)], sc, B)
        pr.append(["path (b) - path (a), baseline volume", f"{bd['d_med']:+.1f} [{bd['d_med_lo']:+.1f}, {bd['d_med_hi']:+.1f}]",
                   f"{bd['d_mean']:+.1f} [{bd['d_mean_lo']:+.1f}, {bd['d_mean_hi']:+.1f}]"])
        pr.append(["path (b) - path (a), corrected volume", f"{bd2['d_med']:+.1f} [{bd2['d_med_lo']:+.1f}, {bd2['d_med_hi']:+.1f}]",
                   f"{bd2['d_mean']:+.1f} [{bd2['d_mean_lo']:+.1f}, {bd2['d_mean_hi']:+.1f}]"])
        md.append(f"\nPaired item-APE differences (pp): corrected - baseline per path, and path b - path a\n\n")
        md.append(VH.md_table(["comparison", "d median APE", "d mean APE"], pr))
    return out, "\n".join(md)


# ------------------------------------------------------------------------------------------ part 3

def scan_folds(D, seed, k=5):
    """Scan-disjoint folds, round-robin within each view density so each fold has both."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for dens in (8, 24):
        sc = np.unique(D.scan[D.d["nv_scan"] == dens])
        rng.shuffle(sc)
        for i, s in enumerate(sc):
            folds[i % k].append(s)
    return [(np.nonzero(~np.isin(D.scan, f))[0], np.nonzero(np.isin(D.scan, f))[0]) for f in folds]


def run_splits(D, reps_splits, model, cols):
    """-> rep-averaged APE (n,) (NaN where never tested), signed error, predictions (reps, n)."""
    P = np.full((len(reps_splits), D.n), np.nan)
    for r, splits in enumerate(reps_splits):
        for tr, te in splits:
            if model == "dens_median":     # per-density median ratio (knows n_views; a reference, not a model)
                rr = D.lv_true - D.lv_vis
                nv = D.d["nv_scan"]
                P[r, te] = D.lv_vis[te] + np.array([np.median(rr[tr][nv[tr] == v]) if (nv[tr] == v).any() else np.median(rr[tr]) for v in nv[te]])
            else:
                P[r, te], _ = VH.fit_predict(model, D, tr, te, cols)
    return P


def views_aware(feat, model, quick):
    B = 200 if quick else 1000
    reps = 2 if quick else 5
    D0 = VH.Data(feat, "valid", "envelope")
    D, names = with_extras(D0)
    nv = D.d["nv_scan"]
    sets = {
        "no views features": [i for i, f in enumerate(names) if f not in VH.FEATURE_GROUPS["views / lift"] + EXTRA],
        "base (n_views, lift)": [i for i, f in enumerate(names) if f not in EXTRA],
        "views-aware (+ scan n_views, per-view proxies)": list(range(len(names))),
    }
    protos = {
        "scan-disjoint 5-fold (mixed train)": [scan_folds(D, 500 + r) for r in range(reps)],
        "category-disjoint 5-fold (mixed train)": [VH.split_catcv(D, 100 + r) for r in range(reps)],
    }
    # batch-disjoint reference: leave-one-batch-out over all six batches
    protos["batch-disjoint (leave-one-batch-out)"] = [VH.split_lobo_all(D)]
    # density-matched: train on one density, test on same density (scan-disjoint) and on the other density
    def dens_split(tr_d, te_d, seed):
        s = []
        sc_te = np.unique(D.scan[nv == te_d])
        rng = np.random.default_rng(seed)
        rng.shuffle(sc_te)
        if tr_d == te_d:
            for f in np.array_split(sc_te, 5):
                s.append((np.nonzero((nv == tr_d) & ~np.isin(D.scan, f))[0], np.nonzero(np.isin(D.scan, f))[0]))
        else:
            s.append((np.nonzero(nv == tr_d)[0], np.nonzero(nv == te_d)[0]))
        return s
    protos["train 8-view only -> test 8-view (scan-disjoint)"] = [dens_split(8, 8, 600 + r) for r in range(reps)]
    protos["train 24-view only -> test 24-view (scan-disjoint)"] = [dens_split(24, 24, 700 + r) for r in range(reps)]
    protos["train 8-view only -> test 24-view"] = [dens_split(8, 24, 0)]
    protos["train 24-view only -> test 8-view"] = [dens_split(24, 8, 0)]

    arms = [("global median ratio", "global", None), ("per-density median ratio (reference)", "dens_median", None)]
    arms += [(f"{model} / {nm}", model, cols) for nm, cols in sets.items()]
    res, md = {}, []
    md.append(f"Items: {D.n} valid ({int((nv == 8).sum())} in 8-view scans / {int((nv == 24).sum())} in 24-view scans; "
              f"{len(np.unique(D.scan[nv == 8]))} / {len(np.unique(D.scan[nv == 24]))} scans). Target = envelope. Median item APE %, "
              "[paired scan-cluster bootstrap CI of (arm - baseline), pp] / median signed bias %.\n")
    for pname, rs in protos.items():
        P = {}
        for lab, mdl, cols in arms:
            P[lab] = run_splits(D, rs, mdl, cols)
        tested = np.isfinite(P[arms[0][0]][0])
        rows = []
        for dn in (8, 24):
            m = tested & (nv == dn)
            if m.sum() == 0:
                continue
            idx = np.nonzero(m)[0]
            base = np.abs(np.exp(D.lv_vis[idx] - D.lv_true[idx]) - 1)
            row = [f"{dn}-view", len(idx), f"{100 * np.median(base):.0f} / {100 * np.median(np.exp(D.lv_vis[idx] - D.lv_true[idx]) - 1):+.0f}"]
            for lab, _, _ in arms:
                pr = P[lab][:, idx]
                ape = np.mean(np.abs(np.exp(pr - D.lv_true[idx]) - 1), 0)
                bias = np.median(np.mean(np.exp(pr - D.lv_true[idx]) - 1, 0))
                bd = VH.boot_diff(ape, base, D.scan[idx], B)
                res.setdefault(pname, {}).setdefault(f"{dn}", {})[lab] = dict(med_ape=100 * float(np.median(ape)), bias=100 * float(bias), **bd)
                row.append(f"{100 * np.median(ape):.0f} [{bd['d_med_lo']:+.0f},{bd['d_med_hi']:+.0f}] / {100 * bias:+.0f}")
            rows.append(row)
        md.append(f"\n#### {pname}\n")
        md.append(VH.md_table(["test density", "n", "baseline"] + [a[0] for a in arms], rows))
        if pname.startswith("batch-disjoint"):
            brow = []
            for b in sorted(set(D.batch)):
                idx = np.nonzero(D.batch == b)[0]
                base = np.abs(np.exp(D.lv_vis[idx] - D.lv_true[idx]) - 1)
                row = [b, f"{int(nv[idx[0]])}", len(idx), f"{100 * np.median(base):.0f}"]
                for lab, _, _ in arms:
                    ape = np.mean(np.abs(np.exp(P[lab][:, idx] - D.lv_true[idx]) - 1), 0)
                    row.append(f"{100 * np.median(ape):.0f}")
                brow.append(row)
            md.append("\nPer held-out batch (median APE %):\n\n")
            md.append(VH.md_table(["held-out batch", "views", "n", "baseline"] + [a[0] for a in arms], brow))
    # batch-disjoint per-batch detail for the views-aware model
    return res, "\n".join(md), names


def dish_table(feat, model, quick):
    """Headline table, both targets: item median/mean APE, bias, dish APE of baseline vs `model` with paired scan-cluster CIs."""
    B = 200 if quick else 1000
    reps = 2 if quick else 5
    rows, out = [], {}
    for target in ("envelope", "solid"):
        D = VH.Data(feat, "valid", target)
        Dv = D.subset(D.batch == "hard_v12")
        for pname, DD, prot in (("lobo_v12 (train 5 batches -> v12)", D, "lobo_v12"), ("rev_v12 (train v12 -> 5 batches)", D, "rev_v12"),
                                ("lobo_all (leave-one-batch-out)", D, "lobo_all"), ("catcv all batches", D, "catcv"),
                                ("catcv hard_v12 only", Dv, "catcv")):
            res = VH.run_protocol(DD, prot, ["none", model], repeats=1 if prot != "catcv" else reps)
            idx = res["none"]["idx"]
            sc = DD.scan[idx]
            ab, am = VH.mean_ape(res, "none", DD), VH.mean_ape(res, model, DD)
            bd = VH.boot_diff(am, ab, sc, B)
            eb = np.mean([np.exp(res["none"]["pred"][i] - DD.lv_true[idx]) - 1 for i in range(len(res["none"]["pred"]))], 0)
            em = np.mean([np.exp(res[model]["pred"][i] - DD.lv_true[idx]) - 1 for i in range(len(res[model]["pred"]))], 0)
            us = np.unique(sc)

            def dish(res_, m):
                return np.array([np.mean([abs(np.exp(res_[m]["pred"][i][sc == u]).sum() / DD.true[idx][sc == u].sum() - 1)
                                           for i in range(len(res_[m]["pred"]))]) for u in us])
            db, dm = dish(res, "none"), dish(res, model)
            rng = np.random.default_rng(0)
            dd = [100 * (np.median(dm[j]) - np.median(db[j])) for j in (rng.integers(0, len(us), len(us)) for _ in range(B))]
            r_ = dict(n=len(idx), n_dish=len(us), base_med=100 * np.median(ab), mod_med=100 * np.median(am), d_med=bd,
                      base_mean=100 * ab.mean(), mod_mean=100 * am.mean(), base_bias=100 * np.median(eb), mod_bias=100 * np.median(em),
                      base_dish=100 * np.median(db), mod_dish=100 * np.median(dm),
                      d_dish=[100 * (np.median(dm) - np.median(db)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
            out[f"{target}|{pname}"] = r_
            rows.append([target, pname, r_["n"], f"{r_['base_med']:.0f} -> {r_['mod_med']:.0f} ({bd['d_med']:+.0f} [{bd['d_med_lo']:+.0f}, {bd['d_med_hi']:+.0f}])",
                         f"{r_['base_mean']:.0f} -> {r_['mod_mean']:.0f} ({bd['d_mean']:+.0f} [{bd['d_mean_lo']:+.0f}, {bd['d_mean_hi']:+.0f}])",
                         f"{r_['base_bias']:+.0f} -> {r_['mod_bias']:+.0f}",
                         f"{r_['base_dish']:.0f} -> {r_['mod_dish']:.0f} ({r_['d_dish'][0]:+.0f} [{r_['d_dish'][1]:+.0f}, {r_['d_dish'][2]:+.0f}])"])
    md = VH.md_table(["target", "protocol", "n items", "item median APE %: base -> head (diff pp [95% CI])",
                      "item mean APE %", "median bias %", "dish median APE %: base -> head (diff pp [95% CI])"], rows)
    return out, md


def importance_single(feat, model):
    D = VH.Data(feat, "valid", "envelope")
    imp_cat, e0 = VH.perm_importance(D, [VH.split_catcv(D, 300 + r) for r in range(3)], VH.FEATURE_GROUPS, model=model)
    return imp_cat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="hgb_l1")
    ap.add_argument("--mass-source", choices=ML.SOURCES, default="synth",
                    help="mass labels for part 'mass' (density fit on hard_v10 and targets); non-default writes "
                         "results/mass_labels/mass_two_stage_<source>.md|json")
    ap.add_argument("--parts", nargs="+", default=["mass", "views", "dish"], choices=["mass", "views", "dish"])
    a = ap.parse_args()
    feat = OUT / "features.npz"
    global MASS_SOURCE
    MASS_SOURCE = a.mass_source
    if "mass" in a.parts:
        out, md = mass_two_stage(feat, a.model, a.quick)
        od, sfx = (OUT, "") if a.mass_source == "synth" else (OUT.parent / "mass_labels", f"_{a.mass_source}")
        od.mkdir(parents=True, exist_ok=True)
        (od / f"mass_two_stage{sfx}.json").write_text(json.dumps(out, indent=1, default=float))
        (od / f"mass_two_stage{sfx}.md").write_text(f"# Two-stage mass (envelope target, mass source = {a.mass_source})\n\n" + md)
        print("wrote", od / f"mass_two_stage{sfx}.md")
    if "dish" in a.parts:
        out, md = dish_table(feat, a.model, a.quick)
        (OUT / "headline_targets.json").write_text(json.dumps(out, indent=1, default=float))
        (OUT / "headline_targets.md").write_text("# Envelope vs solid target, baseline vs head (paired scan-cluster bootstrap)\n\n" + md)
        print("wrote headline_targets.md")
    if "views" in a.parts:
        res, md, names = views_aware(feat, a.model, a.quick)
        (OUT / "views_aware.json").write_text(json.dumps(res, indent=1, default=float))
        (OUT / "views_aware.md").write_text("# Views-aware / mixed-density transfer (envelope target)\n\n" + md)
        print("wrote views_aware.md")


if __name__ == "__main__":
    main()

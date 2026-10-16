"""Envelope vs solid volume targets for the synthetic RQ1 / synth_volume harnesses.

    cd /home/kyle/Desktop/FoodSeg
    conda run -n foodseg python -m pipeline.eval.rq1_envelope            # GT envelopes + re-scoring + mass
    conda run -n foodseg python -m pipeline.eval.rq1_envelope --score-only   # reuse results/rq1/envelope_items.csv

`volume_ml` in gt.json is the *solid* volume (MetaFood3D mesh / procedural solid).  A scan only sees the visible top
surface and closes it to the container, i.e. it measures an *envelope* (voids inside piles, leaves and pieces count as
volume).  This module computes the GT envelope of every item from the complete noise-free GT surface samples,
closed to the GT container support exactly like the scan method (top envelope on a 2 mm grid -> foodscan
Support.height_map_volume), plus a convex hull (+ drops onto the support) and an alpha-shape, then re-scores the RQ1
arms against solid / envelope targets and in mass (bulk density x envelope volume vs material density x solid volume).
Outputs: results/rq1/envelope_items.csv, results/rq1/envelope.json, results/rq1/envelope.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import rq1_frame_vs_scan as R1  # noqa: E402
from pipeline.eval import rq1_geometry as G  # noqa: E402
from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import synth_volume_methods as M  # noqa: E402

DATA = _ROOT / "data" / "synth"
OUT = _ROOT / "results" / "rq1"
NAN = float("nan")
FIT_BATCH = "hard_v10"
SCORE_BATCHES = ["hard_v12", "hard_v10"]
ARMS = ["overhead", "K2", "K4", "K8", "K12", "K24", "full"]
ALPHA = 0.012


# ------------------------------------------------------------------------------------------- GT envelopes

def column_fill_envelope(pts: np.ndarray, support=None, tol: float = 0.0015, sub: float = 0.001) -> float:
    """Envelope volume (mL) of a complete GT surface sample set closed to `support` (foodscan Support, or the flat
    table plane z = 0 when None): the volume between the support and the item's top surface (highest sample per
    xy position, i.e. what a top-down view of the whole item sees) over its complete plan-view footprint.

    The footprint is the morphological closing (radius 1.5 x the median 3-D sample spacing, 1.5-5 mm) of ALL the
    item's samples (underside and walls included) on a `sub` = 1 mm grid, holes filled; so gaps narrower than
    ~2 r_close inside a pile count as filled, exactly the kind of void a scan's column fill closes.  Height = mean
    of the upper half of the 4 nearest top-envelope samples; cells under 1 mm above the support are dropped (as the
    scan method's `points_above(1 mm)` does).  Deliberately NOT foodscan's Support.height_map_volume: that one has
    a +6.7 % footprint bias on a 3 cm cylinder (see tests), which belongs to the estimator, not to the target."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    pts = pts[np.isfinite(pts).all(1)] if len(pts) else pts
    if len(pts) < 8:
        return 0.0
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    nn = nn[nn > 0]
    r_close = float(np.clip(1.5 * np.median(nn), 0.0015, 0.005)) if len(nn) else 0.003
    env = M.top_envelope(pts, tol=tol)
    ext = pts[:, :2].max(0) - pts[:, :2].min(0)
    while (ext / (sub / 2.0)).prod() > 1.2e7:    # items spread over a huge area (stray samples): coarsen the grid
        sub *= 2.0
    fine = sub / 2.0                       # footprint at 0.5 mm (no touched-cell bias), heights on the 1 mm grid
    pad = r_close + 4 * sub
    lo = pts[:, :2].min(0) - pad
    shape = 2 * (np.ceil((pts[:, :2].max(0) + pad - lo) / sub).astype(int) + 1)
    has = np.zeros(shape, bool)
    c = np.floor((pts[:, :2] - lo) / fine).astype(int)
    has[c[:, 0], c[:, 1]] = True
    grown = ndimage.distance_transform_edt(~has) * fine <= r_close
    occ = ndimage.distance_transform_edt(grown) * fine > r_close
    occ = ndimage.binary_fill_holes(occ | has)
    frac = occ.reshape(shape[0] // 2, 2, shape[1] // 2, 2).mean(axis=(1, 3))      # covered fraction of each 1 mm cell
    gi, gj = np.nonzero(frac > 0)
    w = frac[gi, gj]
    centres = lo + (np.stack([gi, gj], 1) + 0.5) * sub
    k = min(4, len(env))
    _, idx = cKDTree(env[:, :2]).query(centres, k=k)
    zs = np.sort(env[:, 2][idx].reshape(len(centres), -1), axis=1)
    top = zs[:, k // 2:].mean(1)
    base = support.height_at(np.c_[centres, np.zeros(len(centres))]) if support is not None else np.zeros(len(centres))
    h = top - base
    return float((np.where(h > M.MIN_H, h, 0.0) * w).sum() * sub * sub * 1e6)


def hull_envelope(pts: np.ndarray, support=None) -> float:
    """Convex hull of the points plus their drops onto the support (mL)."""
    if support is None:
        a = pts[pts[:, 2] > M.MIN_H]
    else:
        a = support.points_above(pts, min_height=M.MIN_H)
    return M.hull_support_volume(a, support) if len(a) >= 4 else NAN


def alpha_envelope(pts: np.ndarray, support=None, alpha: float = ALPHA, spacing: float = 0.004) -> float:
    """Alpha-shape volume (mL) of the points plus drops onto the support; NaN if the mesh is not watertight."""
    a = pts[pts[:, 2] > M.MIN_H] if support is None else support.points_above(pts, min_height=M.MIN_H)
    if len(a) < 4:
        return NAN
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a.astype(np.float64))).voxel_down_sample(spacing)
    a = np.asarray(pc.points)
    base_z = support.height_at(np.c_[a[:, :2], np.zeros(len(a))]) if support is not None else np.zeros(len(a))
    allp = np.vstack([a, np.c_[a[:, :2], np.minimum(base_z, a[:, 2])]])
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(allp)), alpha)
        if not mesh.is_watertight():
            return NAN
        return float(mesh.get_volume() * 1e6)
    except Exception:  # noqa: BLE001
        return NAN


def envelope_rows(batch: str) -> list[dict]:
    bd = DATA / batch
    man = D.read_manifest(bd)
    rows = []
    for sid in sorted(p.name for p in (bd / "scans").iterdir() if p.is_dir()):
        sc = D.load_scan(bd, sid, man)
        if sc.R is None or not sc.items:
            continue
        try:
            z = np.load(sc.dir / "gt" / "points.npz")
        except Exception as e:  # noqa: BLE001
            print(f"{batch}/{sid}: no gt points ({e})", flush=True)
            continue
        gxyz, glab = D.to_dish(z["xyz"], sc.R, sc.t), z["label"]
        sup = G.gt_support(D.container_points(sc))
        for it in sc.items:
            iid = int(it["id"])
            p = gxyz[glab == iid]
            sh = it.get("shape") or {}
            solid = float(it["volume_ml"])
            row = dict(batch=batch, scan=sid, ok=int(sc.ok), item_id=iid, kind=sh.get("kind"), category=it.get("category"),
                       mass_source=it.get("mass_source"), mass_g=it.get("mass_g"), density_g_ml=it.get("density_g_ml"),
                       volume_solid=solid, n_gt=len(p), sup=int(sup is not None),
                       volume_envelope_gt=column_fill_envelope(p, sup) if len(p) else NAN,
                       volume_hull_gt=hull_envelope(p, sup) if len(p) else NAN,
                       volume_alpha_gt=alpha_envelope(p, sup) if len(p) else NAN)
            rows.append(row)
        print(f"{batch}/{sid}: {len(sc.items)} items", flush=True)
    return rows


FIELDS = ["batch", "scan", "ok", "item_id", "kind", "category", "mass_source", "mass_g", "density_g_ml", "volume_solid",
          "n_gt", "sup", "volume_envelope_gt", "volume_hull_gt", "volume_alpha_gt"]


def read_items(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        for k in ("mass_g", "density_g_ml", "volume_solid", "volume_envelope_gt", "volume_hull_gt", "volume_alpha_gt"):
            r[k] = R1._f(r[k])
        r["item_id"] = int(r["item_id"])
        r["ok"] = int(r["ok"])
        out.append(r)
    return out


def ratio_tables(items: list[dict]) -> dict:
    """envelope/solid (and hull/solid) ratio per kind and per size, per batch."""
    out: dict = {}
    for b in sorted({r["batch"] for r in items}):
        rs = [r for r in items if r["batch"] == b and r["volume_solid"] > 0 and np.isfinite(r["volume_envelope_gt"])]

        def summ(sub):
            if not sub:
                return None
            e = np.array([r["volume_envelope_gt"] / r["volume_solid"] for r in sub])
            h = np.array([r["volume_hull_gt"] / r["volume_solid"] for r in sub if np.isfinite(r["volume_hull_gt"])])
            al = np.array([r["volume_alpha_gt"] / r["volume_solid"] for r in sub if np.isfinite(r["volume_alpha_gt"])])
            tot_s = sum(r["volume_solid"] for r in sub)
            tot_e = sum(r["volume_envelope_gt"] for r in sub)
            return dict(n=len(sub), env_med=float(np.median(e)), env_q25=float(np.percentile(e, 25)),
                        env_q75=float(np.percentile(e, 75)), env_pooled=float(tot_e / tot_s),
                        hull_med=float(np.median(h)) if len(h) else NAN, alpha_med=float(np.median(al)) if len(al) else NAN,
                        alpha_n=len(al))
        d = {"all": summ(rs), "kind": {}, "size": {}}
        for k in sorted({str(r["kind"]) for r in rs}):
            d["kind"][k] = summ([r for r in rs if str(r["kind"]) == k])
        for name, lo, hi in R1.SIZE_BINS:
            d["size"][name] = summ([r for r in rs if lo <= r["volume_solid"] < hi])
        out[b] = d
    return out


# ------------------------------------------------------------------------------------------- re-scoring

def retarget(rows: list[dict], items: list[dict], target: str) -> list[dict]:
    """Copy of the RQ1 rows with true_ml replaced by items[target]; rows with no target are dropped."""
    key = {(r["batch"], r["scan"], r["item_id"]): r[target] for r in items}
    out = []
    for r in rows:
        v = key.get((r["batch"].split("+")[0], r["scan"], int(r["item_id"])))
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        out.append(dict(r, true_ml=float(v)))
    return out


def cluster_ci(agg: dict, batch: str, arm: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Scan-cluster bootstrap CI of median APE / median bias (percent) of one arm."""
    by: dict[str, list] = {}
    for (b, a, s, _), r in agg.items():
        if b == batch and a == arm and np.isfinite(r["ape"]):
            by.setdefault(s, []).append((r["ape"], r["bias"]))
    sc = sorted(by)
    if not sc:
        return {}
    arr = {s: np.array(by[s]) for s in sc}
    allv = np.vstack([arr[s] for s in sc])
    rng = np.random.default_rng(seed)
    ba, bb = [], []
    for _ in range(n_boot):
        x = np.vstack([arr[sc[j]] for j in rng.choice(len(sc), len(sc))])
        ba.append(np.median(x[:, 0]))
        bb.append(np.median(x[:, 1]))
    return dict(ape=float(np.median(allv[:, 0]) * 100), ape_ci=[float(v) for v in np.percentile(ba, [2.5, 97.5]) * 100],
                bias=float(np.median(allv[:, 1]) * 100), bias_ci=[float(v) for v in np.percentile(bb, [2.5, 97.5]) * 100])


def dish_ci(agg: dict, batch: str, arm: str, n_boot: int = 2000, seed: int = 0) -> dict:
    d = R1.dish_totals(agg, batch, arm)
    v = [(abs(e - t) / t, (e - t) / t) for t, e in d.values() if t > 0]
    if not v:
        return {}
    v = np.array(v)
    rng = np.random.default_rng(seed)
    ba, bb = [], []
    for _ in range(n_boot):
        x = v[rng.choice(len(v), len(v))]
        ba.append(np.median(x[:, 0]))
        bb.append(np.median(x[:, 1]))
    return dict(n=len(v), ape=float(np.median(v[:, 0]) * 100), ape_ci=[float(q) for q in np.percentile(ba, [2.5, 97.5]) * 100],
                bias=float(np.median(v[:, 1]) * 100), bias_ci=[float(q) for q in np.percentile(bb, [2.5, 97.5]) * 100])


def fit_densities(items: list[dict], batch: str = FIT_BATCH, min_n: int = 3) -> dict:
    """Per-kind median bulk (mass/envelope) and material (mass/solid) density from `batch`; pooled fallback."""
    rs = [r for r in items if r["batch"] == batch and r["mass_g"] and r["mass_g"] > 0 and r["volume_solid"] > 0
          and np.isfinite(r["volume_envelope_gt"]) and r["volume_envelope_gt"] > 0]

    def med(sub):
        b = np.array([r["mass_g"] / r["volume_envelope_gt"] for r in sub])
        m = np.array([r["mass_g"] / r["volume_solid"] for r in sub])
        return dict(n=len(sub), bulk=float(np.median(b)), bulk_q25=float(np.percentile(b, 25)), bulk_q75=float(np.percentile(b, 75)),
                    material=float(np.median(m)), material_q25=float(np.percentile(m, 25)), material_q75=float(np.percentile(m, 75)))
    pooled = med(rs)
    out = {"_all": pooled}
    for k in sorted({str(r["kind"]) for r in rs}):
        sub = [r for r in rs if str(r["kind"]) == k]
        out[k] = med(sub) if len(sub) >= min_n else dict(pooled, n=len(sub), fallback=True)
    return out


def density_spread(items: list[dict]) -> dict:
    """Same medians/spread on every batch (descriptive, per kind)."""
    out = {}
    for b in sorted({r["batch"] for r in items}):
        rs = [r for r in items if r["batch"] == b and r["mass_g"] and r["mass_g"] > 0 and r["volume_solid"] > 0
              and np.isfinite(r["volume_envelope_gt"]) and r["volume_envelope_gt"] > 0]
        out[b] = {}
        for k in sorted({str(r["kind"]) for r in rs}) + ["_all"]:
            sub = rs if k == "_all" else [r for r in rs if str(r["kind"]) == k]
            bk = np.array([r["mass_g"] / r["volume_envelope_gt"] for r in sub])
            mt = np.array([r["mass_g"] / r["volume_solid"] for r in sub])
            out[b][k] = dict(n=len(sub), bulk_med=float(np.median(bk)), bulk_iqr=[float(x) for x in np.percentile(bk, [25, 75])],
                             mat_med=float(np.median(mt)), mat_iqr=[float(x) for x in np.percentile(mt, [25, 75])],
                             mat_cv=float(np.std(mt) / np.mean(mt)), bulk_cv=float(np.std(bk) / np.mean(bk)))
    return out


def mass_agg(agg_vol: dict, items: dict, dens: dict, which: str) -> dict:
    """Volume agg -> mass agg (same structure, true = mass_g, est = vol x per-kind median density)."""
    out = {}
    for key, r in agg_vol.items():
        it = items.get((key[0].split("+")[0], key[2], key[3]))
        if it is None or not it["mass_g"] or it["mass_g"] <= 0:
            continue
        d = dens.get(str(it["kind"]), dens["_all"])[which]
        true = float(it["mass_g"])
        est = r["est"] * d if np.isfinite(r["est"]) else NAN
        out[key] = dict(true=true, est=est, ape=abs(est - true) / true if np.isfinite(est) else NAN,
                        bias=(est - true) / true if np.isfinite(est) else NAN, vis=r["vis"], kind=r["kind"], ok=r["ok"],
                        n_draws=r["n_draws"])
    return out


def complete_mass_scans(agg: dict, batch: str) -> set[str]:
    """Scans whose items all have a mass label (dish mass total is then comparable)."""
    return {s for s in {k[2] for k in agg if k[0] == batch}}


def score(rows_all: list[dict], items: list[dict]) -> dict:
    res: dict = {"dens": fit_densities(items), "targets": {}, "mass": {}}
    itm = {(r["batch"], r["scan"], r["item_id"]): r for r in items}
    # dishes with any unlabeled item are dropped from dish mass
    bad = {(r["batch"], r["scan"]) for r in items if not (r["mass_g"] and r["mass_g"] > 0)}
    dens = res["dens"]
    for batch in SCORE_BATCHES:
        for tgt in ("volume_solid", "volume_envelope_gt"):
            rr = [r for r in retarget(rows_all, items, tgt) if r["batch"] == batch]
            for meth in ("container", "plane", "hull"):
                agg = R1.aggregate(rr, meth)
                for arm in [a for a in ARMS + ["gt_top", "oblique"] if any(k[1] == a for k in agg)]:
                    st = R1.stats([r for (b, a, _, _), r in agg.items() if b == batch and a == arm])
                    ci = cluster_ci(agg, batch, arm)
                    dc = dish_ci(agg, batch, arm)
                    res["targets"].setdefault(batch, {}).setdefault(tgt, {}).setdefault(meth, {})[arm] = dict(item=st, ci=ci, dish=dc)
                if meth == "container":
                    boots = {}
                    for A, B in (("overhead", "K24"), ("overhead", "K4"), ("K4", "K24"), ("overhead", "full"), ("K2", "K24")):
                        if any(k[1] == A for k in agg) and any(k[1] == B for k in agg):
                            boots[f"{A}-{B}"] = R1.paired_bootstrap(agg, batch, A, B)
                    res["targets"][batch][tgt]["boot_container"] = boots
        # mass
        rr = [r for r in rows_all if r["batch"] == batch]
        agg = R1.aggregate(rr, "container")
        aggp = R1.aggregate(rr, "plane")
        fit_here = fit_densities(items, batch)           # diagnostic only: fitted on the scored batch itself
        for label, a_, which, dn in (("envelope_x_bulk", agg, "bulk", dens), ("solid_x_material", agg, "material", dens),
                                     ("plane_x_material", aggp, "material", dens),
                                     ("envelope_x_bulk_insample", agg, "bulk", fit_here),
                                     ("solid_x_material_insample", agg, "material", fit_here)):
            mag_i = mass_agg(a_, itm, dn, which)                                   # item level: every item with a mass label
            mag = {k: v for k, v in mag_i.items() if (k[0].split("+")[0], k[2]) not in bad}   # dishes: complete mass labels only
            for arm in [a for a in ARMS + ["gt_top"] if any(k[1] == a for k in mag_i)]:
                recs = [r for (b, a, _, _), r in mag_i.items() if b == batch and a == arm]
                res["mass"].setdefault(batch, {}).setdefault(label, {})[arm] = dict(
                    item=R1.stats(recs), ci=cluster_ci(mag_i, batch, arm), dish=dish_ci(mag, batch, arm))
            if label in ("envelope_x_bulk", "solid_x_material"):
                boots = {}
                for A, B in (("overhead", "K24"), ("overhead", "K4"), ("K4", "K24"), ("overhead", "full")):
                    if any(k[1] == A for k in mag) and any(k[1] == B for k in mag):
                        boots[f"{A}-{B}"] = R1.paired_bootstrap(mag, batch, A, B)
                res["mass"][batch].setdefault("boot", {})[label] = boots
        # the volume-envelope oracle floor: GT envelope x bulk (what a perfect envelope scan would give)
    return res


def _fm(x, d=0):
    return R1.fmt(x, d)


def _ci(c, key):
    if not c or key not in c:
        return "-"
    return f"{c[key]:.0f} [{c[key + '_ci'][0]:.0f}, {c[key + '_ci'][1]:.0f}]"


def write_md(items: list[dict], ratios: dict, res: dict, spread: dict, sv: dict | None = None) -> str:
    s = []
    for b, d in ratios.items():
        s.append(f"\n#### GT envelope / solid ratio, {b} (envelope = 2 mm column fill of the full GT top surface closed to the GT container)\n")
        body = []
        for name, sub in [("all", d["all"])] + [(f"kind={k}", v) for k, v in d["kind"].items()] + [(f"size {k}", v) for k, v in d["size"].items()]:
            if not sub:
                continue
            body.append([name, str(sub["n"]), f"{sub['env_med']:.2f}", f"{sub['env_q25']:.2f}-{sub['env_q75']:.2f}",
                         f"{sub['env_pooled']:.2f}", f"{sub['hull_med']:.2f}", f"{sub['alpha_med']:.2f} (n={sub['alpha_n']})"])
        s.append(R1.md_table(["group", "n", "envelope/solid median", "IQR", "pooled (sum/sum)", "hull/solid median", "alpha/solid median"], body))
    for b in SCORE_BATCHES:
        if b not in res["targets"]:
            continue
        s.append(f"\n#### Re-scored RQ1 arms, {b}, container method: median APE % [95% scan-cluster CI] / median bias %; dish APE %\n")
        body = []
        for a in ARMS + ["gt_top"]:
            t = res["targets"][b]
            if a not in t["volume_solid"]["container"]:
                continue
            row = [a]
            for tg in ("volume_solid", "volume_envelope_gt"):
                e = t[tg]["container"][a]
                row += [f"{_ci(e['ci'], 'ape')} / {_fm(e['item']['med_bias'])}", _fm(e["dish"].get("ape")) if e["dish"] else "-"]
            body.append(row)
        s.append(R1.md_table(["arm", "item APE vs solid", "dish APE vs solid", "item APE vs envelope", "dish APE vs envelope"], body))
        s.append("\nPaired bootstrap (container method), difference in median item APE, pp [95% CI]; dish APE difference\n")
        body = []
        for tg in ("volume_solid", "volume_envelope_gt"):
            for k, v in res["targets"][b][tg]["boot_container"].items():
                if v.get("n"):
                    body.append([tg.replace("volume_", ""), k, str(v["n"]), f"{v['med_a']:.0f} / {v['med_b']:.0f}",
                                 f"{v['d_med_ape']:+.0f} [{v['lo']:+.0f}, {v['hi']:+.0f}]",
                                 f"{v['d_dish']:+.0f} [{v.get('dish_lo', NAN):+.0f}, {v.get('dish_hi', NAN):+.0f}]"])
        s.append(R1.md_table(["target", "A - B", "paired items", "med APE A / B", "d med APE", "d dish APE"], body))
    s.append("\n#### Densities (g/mL): bulk = mass / envelope, material = mass / solid (per kind; median and IQR)\n")
    body = []
    for b, dd in spread.items():
        for k, v in dd.items():
            body.append([b, k, str(v["n"]), f"{v['bulk_med']:.2f} ({v['bulk_iqr'][0]:.2f}-{v['bulk_iqr'][1]:.2f}), CV {v['bulk_cv']:.2f}",
                         f"{v['mat_med']:.2f} ({v['mat_iqr'][0]:.2f}-{v['mat_iqr'][1]:.2f}), CV {v['mat_cv']:.2f}"])
    s.append(R1.md_table(["batch", "kind", "n", "bulk density", "material density"], body))
    for b in SCORE_BATCHES:
        if b not in res["mass"]:
            continue
        s.append(f"\n#### Mass, {b} (container-method volume x per-kind median density fitted on {FIT_BATCH}"
                 f"{'; IN-SAMPLE' if b == FIT_BATCH else ''}): median APE % [95% CI] / median bias % [CI]; dish APE % [CI] / dish bias %\n")
        body = []
        for a in ARMS + ["gt_top"]:
            row = [a]
            for lab in ("envelope_x_bulk", "solid_x_material"):
                e = res["mass"][b].get(lab, {}).get(a)
                if not e:
                    row += ["-", "-"]
                    continue
                row += [f"{_ci(e['ci'], 'ape')} / {_ci(e['ci'], 'bias')}", f"{_ci(e['dish'], 'ape')} / {_fm(e['dish'].get('bias')) if e['dish'] else '-'}"]
            body.append(row)
        s.append(R1.md_table(["arm", "item: envelope x bulk", "dish: envelope x bulk", "item: solid x material", "dish: solid x material"], body))
        body = []
        for lab, bt in res["mass"][b].get("boot", {}).items():
            for k, v in bt.items():
                if v.get("n"):
                    body.append([lab, k, str(v["n"]), f"{v['med_a']:.0f} / {v['med_b']:.0f}", f"{v['d_med_ape']:+.0f} [{v['lo']:+.0f}, {v['hi']:+.0f}]",
                                 f"{v['d_dish']:+.0f} [{v.get('dish_lo', NAN):+.0f}, {v.get('dish_hi', NAN):+.0f}]"])
        s.append("\nPaired bootstrap in mass, pp [95% CI]\n")
        s.append(R1.md_table(["density", "A - B", "paired items", "med APE A / B", "d med item APE", "d dish APE"], body))
    for name, d in (sv or {}).items():
        s.append(f"\n#### synth_volume headline re-scored, {name} (ok scans, GT labels): median APE % [95% CI] / median bias %\n")
        body = []
        for k, v in d.items():
            if v.get("solid"):
                body.append([k, str(v["n"]), f"{v['solid']['med_ape']:.0f} [{v['solid']['ape_ci'][0]:.0f}, {v['solid']['ape_ci'][1]:.0f}] / {v['solid']['med_bias']:+.0f}",
                             f"{v['envelope']['med_ape']:.0f} [{v['envelope']['ape_ci'][0]:.0f}, {v['envelope']['ape_ci'][1]:.0f}] / {v['envelope']['med_bias']:+.0f}"])
        s.append(R1.md_table(["source/method", "n items", "vs solid", "vs envelope"], body))
    # in-sample density diagnostic (v12 scored with densities fitted on v12: separates transfer error from the target choice)
    ins = res["mass"].get("hard_v12", {})
    if ins.get("envelope_x_bulk_insample"):
        s.append("\n#### Mass, hard_v12, densities fitted on hard_v12 itself (IN-SAMPLE diagnostic): item APE % / bias %; dish APE %\n")
        body = []
        for a in ARMS:
            row = [a]
            for lab in ("envelope_x_bulk_insample", "solid_x_material_insample"):
                e = ins[lab].get(a)
                row += [f"{_ci(e['ci'], 'ape')} / {_fm(e['ci'].get('bias'))}", _ci(e["dish"], "ape")] if e else ["-", "-"]
            body.append(row)
        s.append(R1.md_table(["arm", "item: envelope x bulk", "dish", "item: solid x material", "dish"], body))
    return "\n".join(s)


# ------------------------------------------------------------------------------------------- synth_volume headline

def synth_rescore(items: list[dict], sv_dir: Path, source: str = "gt", methods=("foodscan", "voxel", "hull"),
                  ok_only: bool = True) -> dict:
    """Re-score the synth_volume harness (items.csv: one row per source x item) against solid and envelope targets."""
    env = {(r["batch"], r["scan"], r["item_id"]): r for r in items}
    rows = [r for r in csv.DictReader(open(sv_dir / "items.csv")) if r["source"] in (source, "gt_top")
            and (not ok_only or r["ok"] == "True")]
    out: dict = {}
    for src in (source, "gt_top"):
        for m in methods:
            rec = []
            for r in rows:
                if r["source"] != src:
                    continue
                e = env.get((r["batch"], r["scan"], int(r["item_id"])))
                est = R1._f(r["vol_" + m])
                if e is None or not np.isfinite(e["volume_envelope_gt"]) or e["volume_envelope_gt"] <= 0:
                    continue
                rec.append((est, e["volume_solid"], e["volume_envelope_gt"], r["scan"], r["batch"]))
            res = {"n": len(rec)}
            for tn, ix in (("solid", 1), ("envelope", 2)):
                ok = [x for x in rec if np.isfinite(x[0])]
                ape = np.array([abs(x[0] - x[ix]) / x[ix] for x in ok])
                bias = np.array([(x[0] - x[ix]) / x[ix] for x in ok])
                if len(ok):
                    # cluster bootstrap over scans (batch/scan)
                    by: dict = {}
                    for x, a_, b_ in zip(ok, ape, bias):
                        by.setdefault((x[4], x[3]), []).append((a_, b_))
                    ks = list(by)
                    arr = [np.array(by[k]) for k in ks]
                    rng = np.random.default_rng(0)
                    bs = [np.median(np.vstack([arr[j] for j in rng.choice(len(ks), len(ks))])[:, 0]) for _ in range(1000)]
                    res[tn] = dict(n_valid=len(ok), med_ape=float(np.median(ape) * 100),
                                   ape_ci=[float(v) for v in np.percentile(bs, [2.5, 97.5]) * 100],
                                   med_bias=float(np.median(bias) * 100), mean_bias=float(bias.mean() * 100))
            out[f"{src}/{m}"] = res
        # dish totals (sum over items of the scan, missing as 0)
    return out


SECTION = "## Envelope vs solid targets, and mass"


def update_summaries(md: str) -> None:
    """Put the envelope section into results/rq1/summary.md (replacing a previous one) and a short note into the
    synth_volume summaries.  rq1_frame_vs_scan.main re-appends results/rq1/envelope_section.md when it rewrites summary.md."""
    reading = OUT / "envelope_reading.md"
    head = (f"{SECTION}\n\nReproduce: `conda run -n foodseg python -m pipeline.eval.rq1_envelope` (`--score-only` reuses "
            "results/rq1/envelope_items.csv). Per-item numbers: results/rq1/envelope_items.csv, all results: results/rq1/envelope.json. "
            "Envelope target = `volume_envelope_gt`; solid target = `volume_ml`; RQ1 container method, GT labels, ok+not-ok scans.\n\n")
    sec = head + (reading.read_text() + "\n" if reading.exists() else "") + md.replace("\n#### ", "\n### ")
    (OUT / "envelope_section.md").write_text(sec)
    sp = OUT / "summary.md"
    if sp.exists():
        t = sp.read_text()
        if SECTION in t:
            t = t[:t.index(SECTION)].rstrip() + "\n"
        sp.write_text(t + "\n" + sec)
    note = ("\n## Note: solid vs envelope target\n\n"
            "`gt_volume_ml` is the solid mesh volume; a scan measures the envelope (visible top surface closed to the container, voids "
            "included). GT envelope = 2 mm column fill of the full GT top surface, results/rq1/envelope_items.csv. Re-scored headline (ok scans, "
            "GT labels, median APE / bias vs solid -> vs envelope): see the 'synth_volume headline re-scored' tables in results/rq1/envelope.md "
            "and results/rq1/summary.md. Short version: `gt_top` 28% -> 1% (v12 subset 15% -> 1%), foodscan on the sparse fused cloud 54% -> 76% "
            "(v12 subset 43% -> 38%), i.e. occlusion, not the target definition, dominates here; per-kind envelope/solid ratios: results/rq1/summary.md.\n")
    for d in (_ROOT / "results" / "synth_volume", _ROOT / "results" / "synth_volume" / "v12"):
        (d / "envelope_note.md").write_text(note)
        sm = d / "summary.md"
        if sm.exists():
            t = sm.read_text()
            if "## Note: solid vs envelope target" in t:
                t = t[:t.index("## Note: solid vs envelope target")].rstrip() + "\n"
            sm.write_text(t + note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", default=["hard", "hard_v3", "hard_v9", "hard_v10", "hard_v11", "hard_v12"])
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ip = OUT / "envelope_items.csv"
    if not a.score_only:
        items = []
        for b in a.batches:
            items += envelope_rows(b)
        with open(ip, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(items)
    items = read_items(ip)
    rows = R1._norm(R1.read_rows(OUT / "rows.csv"))
    ratios = ratio_tables(items)
    res = score(rows, items)
    spread = density_spread(items)
    sv = {"results/synth_volume": synth_rescore(items, _ROOT / "results" / "synth_volume"),
          "results/synth_volume/v12": synth_rescore(items, _ROOT / "results" / "synth_volume" / "v12")}
    (OUT / "envelope.json").write_text(json.dumps(dict(ratios=ratios, scores=res, density_spread=spread, synth_volume=sv),
                                                   indent=1, default=float))
    md = write_md(items, ratios, res, spread, sv)
    (OUT / "envelope.md").write_text(md)
    update_summaries(md)
    print(md)


if __name__ == "__main__":
    main()

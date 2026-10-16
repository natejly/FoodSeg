"""Aggregation + markdown/json/csv output for pipeline.eval.synth_volume."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.eval.synth_volume_methods import METHODS, PRIMARY

NAN = float("nan")
BINS = [("<10 mL", 0, 10), ("10-50 mL", 10, 50), (">50 mL", 50, 1e9)]
CORE_SOURCES = ["gt_top", "gt", "lift"]


# --------------------------------------------------------------------------- stats

def fin(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and np.isfinite(x)


def stats(rows: list[dict], method: str, target: str = "gt_volume_ml") -> dict:
    """APE / bias of `vol_<method>` against `target` over rows with a positive finite target."""
    rows = [r for r in rows if fin(r.get(target)) and r[target] > 0]
    n = len(rows)
    est = np.array([r[f"vol_{method}"] for r in rows], float)
    gt = np.array([r[target] for r in rows], float)
    ok = np.isfinite(est)
    out = {"n": n, "missing": float((~ok).mean()) if n else NAN,
           "nonpos": float((est[ok] <= 0).sum() / n) if n else NAN}
    out["bad"] = (out["missing"] + out["nonpos"]) if n else NAN
    if ok.any():
        rel = (est[ok] - gt[ok]) / gt[ok]
        ape = np.abs(rel)
        out.update(ape_med=float(np.median(ape)) * 100, ape_mean=float(ape.mean()) * 100, ape_p90=float(np.percentile(ape, 90)) * 100,
                   bias_med=float(np.median(rel)) * 100, bias_mean=float(rel.mean()) * 100, n_valid=int(ok.sum()))
    else:
        out.update(ape_med=NAN, ape_mean=NAN, ape_p90=NAN, bias_med=NAN, bias_mean=NAN, n_valid=0)
    return out


def fmt(x, nd=1, pct=False) -> str:
    if not fin(x):
        return "-"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def md(headers: list[str], body: list[list]) -> str:
    s = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in body:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def key(r) -> tuple:
    return (r["batch"], r["scan"], r["item_id"])


# --------------------------------------------------------------------------- densities / mass

def add_mass(rows: list[dict]) -> dict:
    """Oracle densities.  rho_item = mass_g / volume_ml of that item (tautological for a matched item: mass APE ==
    volume APE); rho_cat = median rho over all GT items of the category (still oracle: needs the category)."""
    seen, by_cat = set(), defaultdict(list)
    for r in rows:
        if r["source"] == "gt" and fin(r["rho_item"]) and key(r) not in seen:
            seen.add(key(r))
            by_cat[r["category"]].append(r["rho_item"])
    allr = [v for vs in by_cat.values() for v in vs]
    glob = float(np.median(allr)) if allr else NAN
    rho_cat = {c: (float(np.median(v)) if len(v) >= 3 else glob) for c, v in by_cat.items()}
    for r in rows:
        rc = rho_cat.get(r["category"], glob)
        r["rho_cat"] = rc
        for m in METHODS:
            v = r[f"vol_{m}"]
            r[f"mass_item_{m}"] = v * r["rho_item"] if fin(v) and fin(r["rho_item"]) else NAN
            r[f"mass_cat_{m}"] = v * rc if fin(v) and fin(rc) else NAN
    return {"global": glob, "by_category": rho_cat}


def mass_dishes(rows: list[dict], dishes: list[dict]) -> None:
    """Per (scan, source): estimated total volume and mass (items with no estimate contribute 0; unmatched
    instances use the dish's GT mean density), vs GT totals."""
    grp = defaultdict(list)
    for r in rows:
        grp[(r["batch"], r["scan"], r["source"])].append(r)
    for d in dishes:
        rs = grp.get((d["batch"], d["scan"], d["source"]), [])
        gm = d["gt_total_g"] / d["gt_total_ml"] if d["gt_total_ml"] > 0 else NAN
        for m in METHODS:
            d[f"vol_{m}"] = float(sum(r[f"vol_{m}"] for r in rs if fin(r[f"vol_{m}"])))
            d[f"mass_item_{m}"] = float(sum(r[f"vol_{m}"] * (r["rho_item"] if r["item_id"] > 0 and fin(r["rho_item"]) else gm)
                                            for r in rs if fin(r[f"vol_{m}"]) and (fin(gm) or fin(r["rho_item"]))))
            d[f"mass_cat_{m}"] = float(sum(r[f"vol_{m}"] * (r["rho_cat"] if r["item_id"] > 0 and fin(r["rho_cat"]) else gm)
                                           for r in rs if fin(r[f"vol_{m}"]) and (fin(gm) or fin(r["rho_cat"]))))


def dish_stats(dishes: list[dict], method: str, what: str = "vol", complete_only: bool = False) -> dict:
    gtk = "gt_total_ml" if what == "vol" else "gt_total_g"
    ds = [d for d in dishes if d[gtk] > 0 and ((what == "vol" and not complete_only) or d.get("mass_complete"))]   # mass: only dishes with every item's mass_g labelled
    est = np.array([d[f"{what}_{method}" if what == "vol" else f"{what}_{method}"] for d in ds], float)
    gt = np.array([d[gtk] for d in ds], float)
    if not len(ds):
        return dict(n=0, ape_med=NAN, ape_mean=NAN, bias_med=NAN, bias_mean=NAN)
    rel = (est - gt) / gt
    return dict(n=len(ds), ape_med=float(np.median(np.abs(rel))) * 100, ape_mean=float(np.abs(rel).mean()) * 100,
                bias_med=float(np.median(rel)) * 100, bias_mean=float(rel.mean()) * 100)


# --------------------------------------------------------------------------- build

def build(rows, dishes, infos, include_all: bool, pred_dirs, plane: str, min_points: int) -> dict:
    dens = add_mass(rows)
    mass_dishes(rows, dishes)
    sel = (lambda r: True) if include_all else (lambda r: r["ok"])
    R = [r for r in rows if sel(r)]
    Dd = [d for d in dishes if sel(d)]
    sources = sorted({r["source"] for r in R}, key=lambda s: (s.split(":")[0] not in ("gt_top", "gt", "lift"), s))
    res: dict = {"config": dict(include_all=include_all, plane=plane, min_points=min_points, pred_dirs=pred_dirs,
                                methods=METHODS, primary=PRIMARY, densities=dens)}

    # ---- counts both ways
    cnt = {}
    for label, f in (("ok", lambda i: i["ok"]), ("all", lambda i: True)):
        sub = [i for i in infos if f(i)]
        cnt[label] = dict(scans=len(sub), items=int(sum(i.get("n_items", 0) for i in sub)),
                          empty_cloud=int(sum(1 for i in sub if i.get("n_points", 0) == 0)),
                          with_lift=int(sum(1 for i in sub if i.get("has_lift"))),
                          support_failed=int(sum(1 for i in sub if not i.get("support_ok", False) and i.get("n_points", 0) > 0)))
    per_batch = defaultdict(lambda: Counter())
    for i in infos:
        pb = per_batch[i["batch"]]
        pb["scans"] += 1
        pb["ok"] += int(i["ok"])
        pb["items_all"] += i.get("n_items", 0)
        pb["items_ok"] += i.get("n_items", 0) if i["ok"] else 0
        pb["empty"] += int(i.get("n_points", 0) == 0)
        pb["lift"] += int(bool(i.get("has_lift")))
    res["counts"] = dict(cnt, per_batch={k: dict(v) for k, v in per_batch.items()})

    # ---- 1. per-item error, source x method
    t1 = []
    for s in sources:
        rs = [r for r in R if r["source"] == s]
        for m in METHODS:
            t1.append(dict(source=s, method=m, **stats(rs, m)))
    res["per_item"] = t1

    # ---- 2. size bins / kind / container (primary methods)
    def sub_by(rs, pred):
        return [r for r in rs if pred(r)]
    t2 = []
    for s in sources:
        rs = [r for r in R if r["source"] == s]
        for name, lo, hi in BINS:
            for m in PRIMARY + ["table"]:
                t2.append(dict(source=s, method=m, group="size", value=name,
                               **stats(sub_by(rs, lambda r: lo <= r["gt_volume_ml"] < hi if fin(r["gt_volume_ml"]) else False), m)))
        def occ(r):
            o = r.get("occluded_fraction")
            return "n/a" if not fin(o) else ("<0.3" if o < 0.3 else "0.3-0.7" if o < 0.7 else ">=0.7")

        def icp(r):
            o = r.get("icp_shift_mm")
            return "n/a" if not fin(o) else ("<10 mm" if o < 10 else ">=10 mm (failed)")
        for gname, fn in (("kind", lambda r: str(r["kind"]) if r["kind"] else None),
                          ("container", lambda r: str(r["container"]) if r["container"] else None),
                          ("asset_source", lambda r: str(r["asset_source"]) if r["asset_source"] else None),
                          ("batch", lambda r: str(r["batch"])), ("occluded_fraction", occ), ("stacked_scan", lambda r: str(r.get("stacked"))),
                          ("icp_pose_error", icp)):
            for v in sorted({fn(r) for r in rs} - {None}):
                for m in PRIMARY + (["foodscan_gtreg", "foodscan_cinit"] if gname == "icp_pose_error" else []):
                    t2.append(dict(source=s, method=m, group=gname, value=v, **stats(sub_by(rs, lambda r: fn(r) == v), m)))
    res["groups"] = t2

    # ---- 3. vs volume_above_container_ml (rim-plane definition)
    t3 = []
    for s in ("gt", "gt_top"):
        rs = [r for r in R if r["source"] == s and fin(r["gt_above_rim_ml"]) and r["gt_above_rim_ml"] >= 1.0]
        for m in ("foodscan", "table", "hull_table"):
            t3.append(dict(source=s, method=m, **stats(rs, m, "gt_above_rim_ml")))
        t3.append(dict(source=s, method="(foodscan vs total volume_ml, same items)", **stats(
            [dict(r, gt_above_rim_ml=r["gt_volume_ml"]) for r in rs], "foodscan", "gt_above_rim_ml")))
    res["above_rim"] = t3

    # ---- 4. ladders (paired items valid in every listed source)
    def ladder(srcs: list[str], scans_filter, label: str):
        by = {s: {key(r): r for r in R if r["source"] == s} for s in srcs}
        common_scans = None
        for s in srcs:
            ss = {(r["batch"], r["scan"]) for r in by[s].values()}
            common_scans = ss if common_scans is None else common_scans & ss
        common_scans = {x for x in (common_scans or set()) if scans_filter(x)}
        out = []
        for m in PRIMARY:
            keys = None
            for s in srcs:
                ks = {k for k, r in by[s].items() if (k[0], k[1]) in common_scans and k[2] > 0 and fin(r["gt_volume_ml"])
                      and r["gt_volume_ml"] > 0 and fin(r[f"vol_{m}"])}
                keys = ks if keys is None else keys & ks
            keys = keys or set()
            for s in srcs:
                st = stats([by[s][k] for k in keys], m)
                dd = [d for d in Dd if d["source"] == s and (d["batch"], d["scan"]) in common_scans]
                out.append(dict(ladder=label, source=s, method=m, n_items=len(keys), n_scans=len(common_scans),
                                dish_ape_med=dish_stats(dd, m)["ape_med"], dish_ape_mean=dish_stats(dd, m)["ape_mean"],
                                dish_bias_med=dish_stats(dd, m)["bias_med"], **{k: v for k, v in st.items() if k not in ("n",)}))
        return out
    lad = ladder(CORE_SOURCES, lambda x: True, "lift")
    for s in sources:
        if s.startswith("pred:"):
            run = s.split(":", 1)[1]
            lad += ladder(["gt_top", "gt", f"gtsub:{run}", s], lambda x: True, s)
    res["ladders"] = lad

    # ---- 5. dish totals (volume and oracle-density mass)
    t5 = []
    for s in sources:
        ds = [d for d in Dd if d["source"] == s]
        for m in METHODS:
            v = dish_stats(ds, m, "vol")
            mi = dish_stats(ds, m, "mass_item")
            mc = dish_stats(ds, m, "mass_cat")
            vm = dish_stats(ds, m, "vol", complete_only=True)
            t5.append(dict(source=s, method=m, n_dishes=v["n"], vol_ape_med=v["ape_med"], vol_ape_mean=v["ape_mean"],
                           vol_bias_med=v["bias_med"], vol_bias_mean=v["bias_mean"],
                           massitem_ape_med=mi["ape_med"], massitem_ape_mean=mi["ape_mean"], massitem_bias_med=mi["bias_med"],
                           masscat_ape_med=mc["ape_med"], masscat_ape_mean=mc["ape_mean"], masscat_bias_med=mc["bias_med"],
                           n_mass_dishes=mc["n"], vol_ape_med_massdishes=vm["ape_med"]))
    res["dish"] = t5

    # ---- 6. per-item mass (oracle rho_cat), plus tautology check for rho_item
    t6 = []
    for s in sources:
        rs = [r for r in R if r["source"] == s and fin(r["gt_mass_g"]) and r["gt_mass_g"] > 0]
        for m in PRIMARY:
            for tag in ("item", "cat"):
                est = np.array([r[f"mass_{tag}_{m}"] for r in rs], float)
                gt = np.array([r["gt_mass_g"] for r in rs], float)
                ok = np.isfinite(est)
                rel = (est[ok] - gt[ok]) / gt[ok]
                t6.append(dict(source=s, method=m, density=("oracle per-item" if tag == "item" else "oracle category median"),
                               n=len(rs), missing=float((~ok).mean()) if len(rs) else NAN,
                               ape_med=float(np.median(np.abs(rel))) * 100 if len(rel) else NAN,
                               ape_mean=float(np.abs(rel).mean()) * 100 if len(rel) else NAN,
                               bias_med=float(np.median(rel)) * 100 if len(rel) else NAN))
    res["mass_item"] = t6

    # ---- 7. ok vs all headline
    t7 = []
    for label, f in (("ok", lambda r: r["ok"]), ("all", lambda r: True)):
        for s in ("gt_top", "gt", "lift"):
            rs = [r for r in rows if r["source"] == s and f(r)]
            for m in PRIMARY:
                t7.append(dict(subset=label, source=s, method=m, **stats(rs, m)))
    res["ok_vs_all"] = t7

    # ---- 8. scans: support quality
    ic = [i for i in infos if "icp_rmse_mm" in i]
    res["support"] = dict(
        n=len(ic), icp_shift_mm_med=float(np.median([i["icp_shift_mm"] for i in ic])) if ic else NAN,
        icp_shift_mm_max=float(np.max([i["icp_shift_mm"] for i in ic])) if ic else NAN,
        support_rmse_mm_med=float(np.median([i["support_rmse_mm"] for i in ic])) if ic else NAN,
        support_rmse_mm_max=float(np.max([i["support_rmse_mm"] for i in ic])) if ic else NAN,
        icp_failed=[f'{i["batch"]}/{i["scan"]}: {i.get("icp_error") or i.get("template_error") or i.get("support_error")}'
                    for i in infos if i.get("n_points", 0) > 0 and not i.get("support_ok", False)])
    res["reading"] = reading(res)
    return res


# --------------------------------------------------------------------------- reading

def _get(tab, **kw):
    for r in tab:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return {}


def reading(res: dict) -> list[str]:
    from pipeline.eval.synth_volume_report_text import reading_lines
    return reading_lines(res, _get)


# --------------------------------------------------------------------------- output

def _t1_md(t1):
    b = [[r["source"], r["method"], r["n"], fmt(r["missing"], pct=True), fmt(r["nonpos"], pct=True), fmt(r["ape_med"], 0),
          fmt(r["ape_mean"], 0), fmt(r["bias_med"], 0), fmt(r["bias_mean"], 0)] for r in t1]
    return md(["source", "method", "n items", "missing", "<=0", "med APE %", "mean APE %", "med bias %", "mean bias %"], b)


def render(res: dict) -> str:
    c, cfg = res["counts"], res["config"]
    s = ["# Synthetic per-item volume evaluation\n"]
    s.append(f"Tables use **{'all' if cfg['include_all'] else 'ok-flagged'}** scans; table plane = **{cfg['plane']}**; items with fewer than "
             f"{cfg['min_points']} points are 'missing'. Reproduce: `conda run -n foodseg python -m pipeline.eval.synth_volume"
             f"{' --all' if cfg['include_all'] else ''}` from the repo root.\n")
    s.append("## Reading\n")
    s += [f"{i + 1}. {l}\n" for i, l in enumerate(res["reading"])]
    s.append("\n## Method definitions\n")
    s.append("- **foodscan**: GT container samples -> top-down envelope, densified to 2 mm (stand-in for an empty-container scan) -> "
             "`ContainerTemplate.from_points` -> `register` (ICP on the scene cloud) -> `Support` height field -> `points_above(1 mm)` -> "
             "`Support.height_map_volume` (point analogue of run_meal's mesh-prism integral; no TSDF mesh in the harness).\n"
             "- **foodscan_gtreg**: same with the template at its true pose (isolates ICP). **foodscan_bbox**: integrates all above-support "
             "scene points inside the item's padded bbox, emulating `crop_mesh_to_points` (a bbox crop of the whole mesh, not a label crop).\n"
             "- **hull**: convex hull of the above-support item points plus their drops onto the container surface. **voxel**: 2 mm column "
             "fill (closed + hole-filled footprint, mean top height, bottom = container surface). **table / hull_table**: same idea against the flat table plane "
             "(the no-container branch of run_meal / `volume.convex_hull_volume`).\n"
             "- **Sources**: `gt_top` = noise-free GT surface samples, top surface per item (estimator floor); `gt` = GT labels on the fused "
             "scan cloud (sparse, noisy, occluded); `lift` = `lift_label` (SAM-mask votes), IoU-matched; `pred:<run>` = refiner "
             "predictions, `gtsub:<run>` = GT labels on the same voxel subset. Targets: `volume_ml` (whole item, including hidden parts); "
             "`volume_above_container_ml` is the volume above the container *rim plane* (itemstats.py), not above the container surface.\n"
             "- APE = |est - gt| / gt per item; bias = signed (est - gt) / gt. 'missing' = no instance matched or fewer than the point minimum; "
             "'<=0' = estimate not positive. Mass uses **oracle density** (label/category known), never a predicted one.\n")
    s.append("\n## Data and counts (scans / items; ok-flag vs all)\n")
    pb = c["per_batch"]
    s.append(md(["batch", "scans", "ok", "items (all)", "items (ok)", "empty cloud", "with lift"],
                [[k, v["scans"], v["ok"], v["items_all"], v["items_ok"], v["empty"], v["lift"]] for k, v in pb.items()]
                + [["**total ok**", c["ok"]["scans"], "", "", c["ok"]["items"], c["ok"]["empty_cloud"], c["ok"]["with_lift"]],
                   ["**total all**", c["all"]["scans"], "", c["all"]["items"], "", c["all"]["empty_cloud"], c["all"]["with_lift"]]]))
    sp = res["support"]
    s.append(f"\nContainer registration ({sp['n']} scans): ICP pose error vs truth median {fmt(sp['icp_shift_mm_med'], 2)} mm "
             f"(max {fmt(sp['icp_shift_mm_max'], 2)}); support height-field RMS vs oracle-pose support median {fmt(sp['support_rmse_mm_med'], 2)} mm "
             f"(max {fmt(sp['support_rmse_mm_max'], 2)}). Scans with no support: {len(sp['icp_failed'])}.\n")
    s.append("\n## 1. Per-item volume error vs `volume_ml`\n")
    s.append(_t1_md(res["per_item"]))
    s.append("\n## 2. By item size, kind, container (primary methods)\n")
    for grp in ("size", "kind", "container", "asset_source", "batch", "occluded_fraction", "stacked_scan", "icp_pose_error"):
        rows = [r for r in res["groups"] if r["group"] == grp and (r["source"] in ("gt", "lift", "gt_top") or r["source"].startswith("pred:"))]
        if not rows:
            continue
        s.append(f"\n**by {grp}**\n\n")
        s.append(md(["source", grp, "method", "n", "missing+<=0", "med APE %", "mean APE %", "med bias %"],
                    [[r["source"], r["value"], r["method"], r["n"], fmt(r["bad"], pct=True), fmt(r["ape_med"], 0), fmt(r["ape_mean"], 0),
                      fmt(r["bias_med"], 0)] for r in rows]))
    s.append("\n## 3. Against `volume_above_container_ml` (rim-plane definition; items with >= 1 mL above the rim)\n")
    s.append(md(["source", "method", "n", "missing", "med APE %", "mean APE %", "med bias %"],
                [[r["source"], r["method"], r["n"], fmt(r["missing"], pct=True), fmt(r["ape_med"], 0), fmt(r["ape_mean"], 0),
                  fmt(r["bias_med"], 0)] for r in res["above_rim"]]))
    s.append("\n## 4. Error decomposition (paired: same items valid in every listed source)\n")
    s.append("Ladder gt_top -> gt (+ sensor sparsity / noise / occlusion) -> lift or pred (+ segmentation). `gtsub` = GT labels on the prediction subset.\n\n")
    s.append(md(["ladder", "source", "method", "items", "scans", "item med APE %", "item mean APE %", "item med bias %", "dish med APE %", "dish mean APE %", "dish med bias %"],
                [[r["ladder"], r["source"], r["method"], r["n_items"], r["n_scans"], fmt(r["ape_med"], 0), fmt(r["ape_mean"], 0),
                  fmt(r["bias_med"], 0), fmt(r["dish_ape_med"], 0), fmt(r["dish_ape_mean"], 0), fmt(r["dish_bias_med"], 0)] for r in res["ladders"]]))
    s.append("\n## 5. Dish totals (sum over all instances; items without an estimate contribute 0)\n")
    s.append("Mass columns use only dishes whose every item has a positive `mass_g` (the first presets left it null on ~40% of items).\n\n")
    s.append(md(["source", "method", "dishes", "vol med APE %", "vol mean APE %", "vol med bias %", "mass dishes", "vol med APE % (mass dishes)", "mass APE med % (cat density)", "mass APE mean %", "mass bias med %",
                 "mass APE med % (item density)"],
                [[r["source"], r["method"], r["n_dishes"], fmt(r["vol_ape_med"], 0), fmt(r["vol_ape_mean"], 0), fmt(r["vol_bias_med"], 0),
                  r["n_mass_dishes"], fmt(r["vol_ape_med_massdishes"], 0), fmt(r["masscat_ape_med"], 0), fmt(r["masscat_ape_mean"], 0), fmt(r["masscat_bias_med"], 0), fmt(r["massitem_ape_med"], 0)]
                 for r in res["dish"] if r["method"] in PRIMARY + ["table"]]))
    s.append("\n## 6. Per-item mass with oracle density\n")
    s.append("`oracle per-item` density = gt mass / gt volume of the same item, so its mass APE equals the volume APE (shown as a check); "
             "`oracle category median` is the non-tautological one. Neither is a deployable density.\n\n")
    s.append(md(["source", "method", "density", "n", "missing", "med APE %", "mean APE %", "med bias %"],
                [[r["source"], r["method"], r["density"], r["n"], fmt(r["missing"], pct=True), fmt(r["ape_med"], 0), fmt(r["ape_mean"], 0),
                  fmt(r["bias_med"], 0)] for r in res["mass_item"]]))
    s.append("\n## 7. ok-flag vs all scans (item APE)\n")
    s.append(md(["subset", "source", "method", "n", "missing", "med APE %", "mean APE %", "med bias %"],
                [[r["subset"], r["source"], r["method"], r["n"], fmt(r["missing"], pct=True), fmt(r["ape_med"], 0), fmt(r["ape_mean"], 0),
                  fmt(r["bias_med"], 0)] for r in res["ok_vs_all"]]))
    if sp["icp_failed"]:
        s.append("\n## Scans without a container support\n" + "\n".join(f"- {x}" for x in sp["icp_failed"]) + "\n")
    return "".join(s)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write(out: Path, res: dict, rows, dishes, infos) -> None:
    note = out / "envelope_note.md"                # written by pipeline.eval.rq1_envelope
    (out / "summary.md").write_text(render(res) + (note.read_text() if note.exists() else ""))
    (out / "summary.json").write_text(json.dumps(_clean(res), indent=1))

    def dump(name, recs):
        keys: list[str] = []
        for r in recs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)
    dump("items.csv", rows)
    dump("dishes.csv", dishes)
    dump("scans.csv", infos)

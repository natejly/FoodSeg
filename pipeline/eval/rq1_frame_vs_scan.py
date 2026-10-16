"""RQ1 on synthetic data: does a full orbit scan estimate food volume better than one overhead RGB-D frame?

    cd /home/kyle/Desktop/FoodSeg
    conda run -n foodseg python -m pipeline.eval.rq1_frame_vs_scan                 # hard_v12 (+ hard_v10, hard_v9)
    conda run -n foodseg python -m pipeline.eval.rq1_frame_vs_scan --batches hard_v12 --report-only

GT labels (oracle per-frame instance ids / fused GT labels) are used for segmentation in every arm, so the
comparison isolates geometry: visibility, noise, support construction.  Arms (see rq1_geometry):
    overhead   the frame whose camera axis is closest to the table normal
    oblique    random non-overhead frame with tilt >= 20 deg (3 draws)
    K<k>       k frames evenly spaced over the orbit, fused by back-projection + 2 mm voxel (no TSDF),
               up to 3 phase offsets each
    full       the existing fused gt/seg_points.npz (GT labels)
Volume methods per arm: plane (table plane closing), container (support fitted from the arm's visible container
points, GT container if fewer than 300), hull (convex hull + drops onto that support), gtcont (diagnostic: GT
container surface).  No depth-upsampled variant: smooth_depth/ has the same 256x192 resolution as depth/.
Outputs: results/rq1/{rows.csv, summary.md, summary.json}.
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

from pipeline.eval import rq1_geometry as G  # noqa: E402
from pipeline.eval import synth_volume_data as D  # noqa: E402

DATA = _ROOT / "data" / "synth"
OUT = _ROOT / "results" / "rq1"
KS = [1, 2, 4, 8, 12, 24]
N_OBL = 3
MAX_PHASE = 3
OBL_MIN_TILT = 20.0
NAN = float("nan")
FIELDS = ["batch", "scan", "ok", "arm", "draw", "item_id", "kind", "category", "container", "true_ml", "n_pts",
          "vis", "sup_src", "tilt", "plane", "container_m", "hull", "gtcont"]


# ------------------------------------------------------------------------------------------- per scan

def _arm_rows(meta: dict, arm: str, draw: int, tilt: float, pts: np.ndarray, lab: np.ndarray, sc: D.Scan,
              sup_gt, gt_surf: dict[int, np.ndarray]) -> list[dict]:
    sup_fit = G.fitted_support(pts[lab == G.CONTAINER_ID])
    src = "fit"
    if sup_fit is None:
        sup_fit, src = sup_gt, "gt_fallback"
    rows = []
    for it in sc.items:
        iid = int(it["id"])
        p = pts[lab == iid]
        v = G.item_volumes(p, sup_fit, sup_gt)
        rows.append(dict(meta, arm=arm, draw=draw, item_id=iid, kind=(it.get("shape") or {}).get("kind"),
                         category=it.get("category"), true_ml=float(it["volume_ml"]), n_pts=int(len(p)),
                         vis=G.visibility(gt_surf.get(iid, np.zeros((0, 3))), p), sup_src=src, tilt=tilt,
                         plane=v["plane"], container_m=v["container"], hull=v["hull"], gtcont=v["gtcont"]))
    return rows


def run_scan(batch_dir: Path, sid: str, manifest: dict, seed: int = 0, clean: bool = True,
             align: bool = False) -> list[dict]:
    sc = D.load_scan(batch_dir, sid, manifest)
    if sc.R is None or not sc.items:
        return []
    meta = dict(batch=sc.batch + ("+icp" if align else ""), scan=sid, ok=int(sc.ok), container=(sc.gt.get("container") or {}).get("kind"))
    frames = G.load_frames(sc.dir)
    n = len(frames)
    z = np.load(sc.dir / "gt" / "points.npz")
    gxyz, glab = D.to_dish(z["xyz"], sc.R, sc.t), z["label"]
    target = G.gt_target(gxyz) if align else None
    clouds = [G.frame_cloud(sc.dir, f, sc.R, sc.t, clean=clean, target=target) for f in frames]
    sup_gt = G.gt_support(D.container_points(sc))
    gt_surf: dict[int, np.ndarray] = {}
    for it in sc.items:
        gt_surf[int(it["id"])] = gxyz[glab == int(it["id"])]

    score = G.topdown_scores(frames, sc.R)
    top = int(np.argmax(score))
    rng = np.random.default_rng(seed + int(sid[1:]) if sid[1:].isdigit() else seed)
    obl_pool = [i for i in range(n) if i != top and G.tilt_deg(score[i]) >= OBL_MIN_TILT] or [i for i in range(n) if i != top]
    obl = [int(i) for i in rng.choice(obl_pool, size=min(N_OBL, len(obl_pool)), replace=False)]

    def fuse(idx: list[int]):
        p = np.vstack([clouds[i][0] for i in idx])
        l = np.concatenate([clouds[i][1] for i in idx])
        return G.voxel_fuse(p, l) if len(idx) > 1 else (p, l)

    rows: list[dict] = []
    rows += _arm_rows(meta, "overhead", 0, G.tilt_deg(score[top]), *clouds[top], sc, sup_gt, gt_surf)
    for d, i in enumerate(obl):
        rows += _arm_rows(meta, "oblique", d, G.tilt_deg(score[i]), *clouds[i], sc, sup_gt, gt_surf)
    for k in KS:
        if k > n:
            continue
        n_ph = min(MAX_PHASE, max(1, n // k))
        offs = sorted({int(round(j * (n / k) / n_ph)) for j in range(n_ph)}) if k < n else [0]
        for d, off in enumerate(offs):
            idx = G.even_indices(n, k, off)
            rows += _arm_rows(meta, f"K{k}", d, NAN, *fuse(idx), sc, sup_gt, gt_surf)
    rows += _arm_rows(meta, "full", 0, NAN, sc.xyz, sc.label, sc, sup_gt, gt_surf)
    # estimator floor: noise-free GT surface, top (visible-from-above) envelope per item + container
    tp, tl = [], []
    for lid in [int(i["id"]) for i in sc.items] + [G.CONTAINER_ID]:
        m = glab == lid
        if m.any():
            e = G.M.top_envelope(gxyz[m], tol=0.0015)
            tp.append(e)
            tl.append(np.full(len(e), lid, np.int32))
    if tp:
        rows += _arm_rows(meta, "gt_top", 0, NAN, np.vstack(tp), np.concatenate(tl), sc, sup_gt, gt_surf)
    return rows


def run_batch(batch: str, scans: list[str] | None = None, clean: bool = True, align: bool = False) -> list[dict]:
    bd = DATA / batch
    man = D.read_manifest(bd)
    sids = scans or sorted(p.name for p in (bd / "scans").iterdir() if p.is_dir())
    rows: list[dict] = []
    for sid in sids:
        t0 = time.time()
        try:
            r = run_scan(bd, sid, man, clean=clean, align=align)
            rows += r
            print(f"{batch}/{sid}: {len(r)} rows in {time.time() - t0:.1f}s", flush=True)
        except Exception:  # noqa: BLE001
            print(f"{batch}/{sid}: FAILED\n{traceback.format_exc()}", flush=True)
    return rows


# ------------------------------------------------------------------------------------------- aggregation

METHOD_COLS = {"plane": "plane", "container": "container_m", "hull": "hull", "gtcont": "gtcont"}
SIZE_BINS = [("<10 mL", 0, 10), ("10-50 mL", 10, 50), (">50 mL", 50, 1e9)]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def aggregate(rows: list[dict], method: str) -> dict[tuple, dict]:
    """(batch, arm, scan, item) -> {ape, bias, est, true, vis, kind, ok, n_draws}: mean over draws of APE and
    signed bias; items with no estimate in any draw are missing (ape = nan)."""
    col = METHOD_COLS[method]
    g: dict[tuple, list] = {}
    for r in rows:
        g.setdefault((r["batch"], r["arm"], r["scan"], int(r["item_id"])), []).append(r)
    out = {}
    for key, rs in g.items():
        true = _f(rs[0]["true_ml"])
        est = np.array([_f(r[col]) for r in rs])
        okm = np.isfinite(est)
        e = est[okm]
        out[key] = dict(
            true=true, est=float(e.mean()) if len(e) else NAN,
            ape=float(np.mean(np.abs(e - true) / true)) if len(e) and true > 0 else NAN,
            bias=float(np.mean((e - true) / true)) if len(e) and true > 0 else NAN,
            vis=float(np.nanmean([_f(r["vis"]) for r in rs])), kind=rs[0]["kind"], ok=int(rs[0]["ok"]),
            n_draws=len(rs), raw=[_f(r[col]) for r in rs])
    return out


def size_bin(v: float) -> str:
    for name, lo, hi in SIZE_BINS:
        if lo <= v < hi:
            return name
    return SIZE_BINS[-1][0]


def stats(recs: list[dict]) -> dict:
    n = len(recs)
    ape = np.array([r["ape"] for r in recs])
    bias = np.array([r["bias"] for r in recs])
    vis = np.array([r["vis"] for r in recs])
    v = np.isfinite(ape)
    return dict(n=n, missing=float(1 - v.mean()) if n else NAN, med_ape=float(np.median(ape[v]) * 100) if v.any() else NAN,
                mean_ape=float(ape[v].mean() * 100) if v.any() else NAN,
                med_bias=float(np.median(bias[v]) * 100) if v.any() else NAN,
                mean_bias=float(bias[v].mean() * 100) if v.any() else NAN,
                vis=float(np.nanmean(vis) * 100) if np.isfinite(vis).any() else NAN)


def dish_totals(agg: dict, batch: str, arm: str) -> dict[str, tuple[float, float]]:
    """scan -> (true total, estimated total with missing items = 0), mean over draws."""
    tot: dict[str, list[float]] = {}
    for (b, a, s, _), r in agg.items():
        if b != batch or a != arm:
            continue
        t = tot.setdefault(s, [0.0, 0.0])
        t[0] += r["true"]
        t[1] += 0.0 if not np.isfinite(r["est"]) else r["est"]
    return {s: (a, b) for s, (a, b) in tot.items()}


def dish_ape(agg: dict, batch: str, arm: str) -> tuple[float, float, int]:
    d = dish_totals(agg, batch, arm)
    if not d:
        return NAN, NAN, 0
    ape = np.array([abs(e - t) / t for t, e in d.values() if t > 0])
    bias = np.array([(e - t) / t for t, e in d.values() if t > 0])
    return float(np.median(ape) * 100), float(np.median(bias) * 100), len(ape)


def paired_bootstrap(agg: dict, batch: str, arm_a: str, arm_b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Cluster (scan) bootstrap of median APE(A) - median APE(B) on items valid in both arms, and of
    the dish-total APE difference (mean over scans)."""
    pairs = []
    for (b, a, s, i), r in agg.items():
        if b == batch and a == arm_a:
            o = agg.get((b, arm_b, s, i))
            if o is not None and np.isfinite(r["ape"]) and np.isfinite(o["ape"]):
                pairs.append((s, r["ape"], o["ape"]))
    if not pairs:
        return dict(n=0)
    scans = sorted({p[0] for p in pairs})
    by = {s: np.array([(p[1], p[2]) for p in pairs if p[0] == s]) for s in scans}
    da, db = dish_totals(agg, batch, arm_a), dish_totals(agg, batch, arm_b)
    dsc = [s for s in scans if s in da and s in db and da[s][0] > 0]
    dd = {s: abs(da[s][1] - da[s][0]) / da[s][0] - abs(db[s][1] - db[s][0]) / db[s][0] for s in dsc}
    rng = np.random.default_rng(seed)
    allp = np.vstack([by[s] for s in scans])
    point = float(np.median(allp[:, 0]) - np.median(allp[:, 1])) * 100
    dpoint = float(np.mean(list(dd.values()))) * 100 if dd else NAN
    bs, bd = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(scans), len(scans))
        x = np.vstack([by[scans[j]] for j in pick])
        bs.append(np.median(x[:, 0]) - np.median(x[:, 1]))
        if dd:
            bd.append(np.mean([dd[dsc[j % len(dsc)]] for j in rng.choice(len(dsc), len(dsc))]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    out = dict(n=len(allp), n_scans=len(scans), d_med_ape=point, lo=float(lo), hi=float(hi),
               med_a=float(np.median(allp[:, 0]) * 100), med_b=float(np.median(allp[:, 1]) * 100),
               d_dish=dpoint)
    if bd:
        out["dish_lo"], out["dish_hi"] = [float(x) for x in np.percentile(bd, [2.5, 97.5]) * 100]
    return out


# ------------------------------------------------------------------------------------------- report

PRIMARY = ["plane", "container", "hull"]


def arms_for(rows: list[dict], batch: str) -> list[str]:
    have = {r["arm"] for r in rows if r["batch"] == batch}
    order = ["overhead", "oblique"] + [f"K{k}" for k in KS] + ["full", "gt_top"]
    return [a for a in order if a in have]


def fmt(x, d=0) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def md_table(head: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out) + "\n"


def report(rows: list[dict], batches: list[str]) -> tuple[str, dict]:
    s: list[str] = []
    js: dict = {}
    aggs = {m: aggregate(rows, m) for m in METHOD_COLS}
    for batch in batches:
        arms = arms_for(rows, batch)
        if not arms:
            continue
        br = [r for r in rows if r["batch"] == batch]
        scans = sorted({r["scan"] for r in br})
        ok_scans = sorted({r["scan"] for r in br if int(r["ok"])})
        items = {(r["scan"], int(r["item_id"])) for r in br}
        tilt = [float(r["tilt"]) for r in br if r["arm"] == "overhead" and np.isfinite(_f(r["tilt"]))]
        fb = np.mean([r["sup_src"] == "gt_fallback" for r in br if r["arm"] in ("overhead", "oblique")])
        s.append(f"\n# {batch}\n")
        s.append(f"{len(scans)} scans ({len(ok_scans)} ok-flagged), {len(items)} items, "
                 f"{len({r['draw'] for r in br})} max draws per arm; overhead frame tilt from vertical: "
                 f"median {np.median(tilt):.0f} deg (max {max(tilt):.0f}); single-frame container fit fell back to "
                 f"the GT container in {fb * 100:.0f}% of item rows.\n")
        js[batch] = {"scans": len(scans), "ok_scans": len(ok_scans), "items": len(items), "per_item": {}, "dish": {},
                     "boot": {}}

        compact = "+" in batch
        for subset, label in ((None, "all scans"), (set(ok_scans), "ok-flagged scans only")):
            if subset is not None and (not subset or compact):
                continue
            s.append(f"\n## Per-item volume error vs `volume_ml` ({label}); items with no estimate are 'missing'\n")
            body = []
            for a in arms:
                for m in PRIMARY + ["gtcont"]:
                    recs = [r for (b, aa, sc, _), r in aggs[m].items() if b == batch and aa == a and
                            (subset is None or sc in subset)]
                    st = stats(recs)
                    if subset is None:
                        js[batch]["per_item"][f"{a}/{m}"] = st
                    body.append([a, m, str(st["n"]), f"{st['missing'] * 100:.0f}%", fmt(st["med_ape"]), fmt(st["mean_ape"]),
                                 fmt(st["med_bias"]), fmt(st["mean_bias"]), fmt(st["vis"])])
            s.append(md_table(["arm", "method", "n items", "missing", "med APE %", "mean APE %", "med bias %",
                               "mean bias %", "GT-surface visibility %"], body))

        shown = [a for a in ["overhead", "oblique", "K4", "K8", "K24", "full", "gt_top"] if a in arms]
        for grp, keyfn, cats in (() if compact else (("size", lambda r: size_bin(r["true"]), [b[0] for b in SIZE_BINS]),
                                 ("kind", lambda r: str(r["kind"]), None))):
            if cats is None:
                cnt: dict = {}
                for r in aggs["container"].values():
                    cnt[str(r["kind"])] = cnt.get(str(r["kind"]), 0) + 1
                cats = sorted(cnt, key=lambda c: -cnt[c])[:6]
            s.append(f"\n## By {grp}: median APE % / median bias % (n items in the overhead arm)\n")
            for m in PRIMARY:
                head = ["arm"] + [c for c in cats]
                body = []
                for a in shown:
                    row = [a]
                    for c in cats:
                        recs = [r for (b, aa, sc, _), r in aggs[m].items() if b == batch and aa == a and keyfn(r) == c]
                        st = stats(recs)
                        row.append(f"{fmt(st['med_ape'])} / {fmt(st['med_bias'])} (n={st['n']})")
                    body.append(row)
                s.append(f"\n{m}\n")
                s.append(md_table(head, body))

        s.append("\n## Dish totals: median APE % / median bias % over scans (missing items count as 0)\n")
        body = []
        for a in arms:
            row = [a]
            for m in PRIMARY + ["gtcont"]:
                da, db, nn = dish_ape(aggs[m], batch, a)
                js[batch]["dish"][f"{a}/{m}"] = dict(med_ape=da, med_bias=db, n=nn)
                row.append(f"{fmt(da)} / {fmt(db)}")
            body.append(row)
        s.append(md_table(["arm"] + PRIMARY + ["gtcont"], body))

        s.append("\n## APE vs number of frames K (evenly spaced, fused by 2 mm voxel; averaged over phases)\n")
        kar = [a for a in arms if a.startswith("K")]
        body = []
        for a in ["overhead", "oblique"] + kar + ["full", "gt_top"]:
            if a not in arms:
                continue
            row = [a]
            for m in PRIMARY:
                st = stats([r for (b, aa, _, _), r in aggs[m].items() if b == batch and aa == a])
                row.append(f"{fmt(st['med_ape'])} / {fmt(st['med_bias'])}")
            row.append(fmt(stats([r for (b, aa, _, _), r in aggs['container'].items() if b == batch and aa == a])["vis"]))
            row.append(fmt(dish_ape(aggs["container"], batch, a)[0]))
            body.append(row)
        s.append("per-item median APE % / median bias %\n")
        s.append(md_table(["arm"] + PRIMARY + ["visibility % (container rows)", "dish APE % (container)"], body))

        s.append("\n## Paired cluster bootstrap (scans resampled, 2000x): median APE(A) - median APE(B), percentage points\n")
        body = []
        for A, B in (("overhead", "K24"), ("overhead", "K4"), ("overhead", "K8"), ("overhead", "full"), ("oblique", "K24"),
                     ("K4", "K24"), ("K8", "K24")):
            if A not in arms or B not in arms:
                continue
            for m in PRIMARY:
                b = paired_bootstrap(aggs[m], batch, A, B)
                js[batch]["boot"][f"{A}-{B}/{m}"] = b
                if not b.get("n"):
                    continue
                body.append([f"{A} - {B}", m, str(b["n"]), f"{b['med_a']:.0f} / {b['med_b']:.0f}",
                             f"{b['d_med_ape']:+.0f} [{b['lo']:+.0f}, {b['hi']:+.0f}]",
                             f"{b['d_dish']:+.0f} [{b.get('dish_lo', NAN):+.0f}, {b.get('dish_hi', NAN):+.0f}]"])
        s.append(md_table(["A - B", "method", "paired items", "med APE A / B", "d med APE [95% CI]",
                           "d dish APE (mean over scans) [95% CI]"], body))

        s.append("\n## Visibility explains the error (container method; median signed bias % by GT-surface visibility)\n")
        vb = [("<20%", 0, .2), ("20-50%", .2, .5), (">=50%", .5, 1.01)]
        body = []
        for a in [x for x in ["overhead", "K4", "K24", "full"] if x in arms]:
            row = [a]
            for name, lo, hi in vb:
                recs = [r for (b, aa, _, _), r in aggs["container"].items() if b == batch and aa == a and lo <= r["vis"] < hi]
                st = stats(recs)
                row.append(f"{fmt(st['med_bias'])} (APE {fmt(st['med_ape'])}, n={st['n']})")
            body.append(row)
        s.append(md_table(["arm"] + [v[0] for v in vb], body))
    return "".join(x if x.endswith("\n") else x + "\n" for x in s), js


# ------------------------------------------------------------------------------------------- main

def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _norm(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["item_id"] = int(r["item_id"])
        r["draw"] = int(r["draw"])
        for k in ("plane", "container_m", "hull", "gtcont", "vis", "tilt", "true_ml"):
            r[k] = _f(r[k])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", default=["hard_v12", "hard_v10", "hard_v9"])
    ap.add_argument("--scans", nargs="+", default=None, help="restrict to these scan ids (debug)")
    ap.add_argument("--icp-batches", nargs="*", default=["hard_v12"],
                    help="also run these batches with oracle ICP pose refinement (reported as <batch>+icp)")
    ap.add_argument("--report-only", action="store_true", help="reuse results/rq1/rows.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "rows.csv"
    if a.report_only:
        rows = _norm(read_rows(rp))
    else:
        rows = []
        for b in a.batches:
            rows += run_batch(b, a.scans)
        for b in a.icp_batches:
            rows += run_batch(b, a.scans, align=True)
        with open(rp, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
    batches = [b + v for b in a.batches for v in ("", "+icp") if any(r["batch"] == b + v for r in rows)]
    text, js = report(rows, batches)
    head = ("# RQ1 (synthetic): single overhead RGB-D frame vs multi-frame / full orbit scan\n\n"
            "Reproduce: `conda run -n foodseg python -m pipeline.eval.rq1_frame_vs_scan` from the repo root. "
            "Segmentation = GT labels in every arm (oracle per-frame ids for frame arms, GT labels on the fused cloud for `full`). "
            "Methods: `plane` = height map vs the GT table plane; `container` = vs a support fitted from the arm's visible "
            "container points (GT container if < 300 points); `hull` = convex hull + drops onto that support; `gtcont` = "
            "diagnostic, vs the GT container surface. APE = |est - true| / true per item (mean over draws/phases of an arm); "
            "target is `volume_ml` (whole item incl. hidden parts).\n")
    reading = OUT / "reading.md"
    body = (reading.read_text() + "\n") if reading.exists() else ""
    env_sec = OUT / "envelope_section.md"          # written by pipeline.eval.rq1_envelope
    tail = ("\n" + env_sec.read_text()) if env_sec.exists() else ""
    (OUT / "summary.md").write_text(head + "\n" + body + text + tail)
    (OUT / "summary.json").write_text(json.dumps(js, indent=1, default=float))
    print(f"wrote {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()

"""Validate the envelope x bulk-density mass step on synthetic scans (stand-in for real ones).

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pipeline.tools.mass_step_synth \
        [--batch data/synth/hard_v12] [--n 5] [--out results/mass_step]

Per scan: GT instance labels (seg_points.npz `label`) -> per-item points -> ICP-registered container Support
(synth_volume_methods.build_supports, as run_meal) -> envelope = Support.height_map_volume (the point-cloud
stand-in for mesh_prism_volume) -> foodscan.volume.item_volume_result -> density_table.estimate_mass.
Variants (mass per item, compared with gt.json mass_g):
  new          rule table: bulk rule or solid x material / thin layer, USDA-style table densities, name-inferred kind
  new_skind    same, kind taken from the synthetic shape kind (leaves/topping/piece/single) instead of the name
  legacy       old formula on the table: volume_envelope x table bulk density for EVERY kind (no densities.csv
               exists in this repo, so the real legacy path would produce no mass at all)
  oracle       solid-volume estimate x the item's true synthetic density (mass/volume_ml): isolates volume error
  gtenv        new rule fed the noise-free GT envelope volume (results/rq1/envelope_items.csv): isolates density error
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.eval import synth_volume_data as D  # noqa: E402
from pipeline.eval import synth_volume_methods as M  # noqa: E402
from pipeline.foodscan import density_table as DT  # noqa: E402
from pipeline.foodscan.volume import item_volume_result  # noqa: E402

ENV_ITEMS = ROOT / "results" / "rq1" / "envelope_items.csv"


def gt_envelopes() -> dict:
    out = {}
    if ENV_ITEMS.exists():
        for r in csv.DictReader(open(ENV_ITEMS, newline="")):
            try:
                out[(r["batch"], r["scan"], int(r["item_id"]))] = float(r["volume_envelope_gt"])
            except ValueError:
                pass
    return out


def scan_rows(sc: D.Scan, env_gt: dict) -> list[dict]:
    cpts = D.container_points(sc)
    cl = sc.xyz[sc.label == 254]
    sup = M.build_supports(cpts, sc.xyz, init_xy=cl[:, :2].mean(0) if len(cl) > 20 else None)["icp"]
    if sup is None:
        return []
    rows = []
    for it in sc.items:
        iid, m_gt, v_gt = int(it["id"]), it.get("mass_g"), it.get("volume_ml")
        pts = sc.xyz[sc.label == iid]
        a = sup.points_above(pts, min_height=M.MIN_H) if len(pts) else pts
        base = {"scan": sc.key, "item_id": iid, "name": it.get("category") or it.get("name"),
                "skind": (it.get("shape") or {}).get("kind"), "gt_mass": m_gt, "gt_vol": v_gt}
        if len(a) < M.MIN_POINTS or not m_gt:
            rows.append(dict(base, ok=False))
            continue
        env = sup.height_map_volume(a)
        rec = item_volume_result(env, a, sup, base["name"])
        new = DT.estimate_mass(base["name"], None, env, rec["footprint_cm2"], rec["height_mm"])
        sk = DT.estimate_mass(base["name"], base["skind"], env, rec["footprint_cm2"], rec["height_mm"])
        rho = m_gt / v_gt if v_gt else None
        ge = env_gt.get((sc.batch, sc.sid, iid))
        rows.append(dict(base, ok=True, kind=rec["kind"], env=env, solid=rec["volume_solid_est_ml"],
                         foot=rec["footprint_cm2"], h=rec["height_mm"], rule=new["mass_rule"],
                         conf=new["mass_confidence"], rho_used=new["density_used_g_ml"],
                         m_new=new["mass_g"], m_new_skind=sk["mass_g"], skind_rule=sk["mass_rule"],
                         m_legacy=env * DT.lookup(base["name"], rec["kind"])[0],
                         m_oracle=rec["volume_solid_est_ml"] * rho if rho else np.nan,
                         m_gtenv=(DT.estimate_mass(base["name"], None, ge, rec["footprint_cm2"], rec["height_mm"])["mass_g"]
                                  if ge else np.nan)))
    return rows


VARIANTS = [("new", "m_new"), ("new_skind", "m_new_skind"), ("legacy", "m_legacy"), ("oracle", "m_oracle"), ("gtenv", "m_gtenv")]


def ape_bias(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    k = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if not k.any():
        return np.nan, np.nan, 0
    e = (pred[k] - gt[k]) / gt[k]
    return 100 * float(np.median(np.abs(e))), 100 * float(np.median(e)), int(k.sum())


def tables(rows: list[dict]) -> str:
    ok = [r for r in rows if r.get("ok")]
    out = [f"{len(ok)} items scored of {len(rows)} ({len({r['scan'] for r in rows})} scans). Cells: median APE % / median bias % (n).\n"]
    for title, key in (("by name-inferred kind", "kind"), ("by synthetic shape kind", "skind")):
        out.append(f"\n#### Item level, {title}\n")
        out.append("| group | n | " + " | ".join(v for v, _ in VARIANTS) + " |")
        out.append("|---|---|" + "---|" * len(VARIANTS))
        for g in ["ALL"] + sorted({str(r[key]) for r in ok}):
            sel = [r for r in ok if g == "ALL" or str(r[key]) == g]
            cells = []
            for _, col in VARIANTS:
                a, b, n = ape_bias([r[col] for r in sel], [r["gt_mass"] for r in sel])
                cells.append(f"{a:.0f} / {b:+.0f}")
            out.append(f"| {g} | {len(sel)} | " + " | ".join(cells) + " |")
    rules = {}
    for r in ok:
        rules.setdefault(r["rule"], []).append(r)
    out.append("\n#### Rule fired (new)\n")
    out.append("| rule | n | median conf |\n|---|---|---|")
    for k, v in rules.items():
        out.append(f"| {k} | {len(v)} | {np.median([r['conf'] for r in v]):.2f} |")
    out.append("\n#### Dish level (sum over items of scans where every item was scored)\n")
    out.append("| scans | " + " | ".join(v for v, _ in VARIANTS) + " |\n|---|" + "---|" * len(VARIANTS))
    by = {}
    for r in rows:
        by.setdefault(r["scan"], []).append(r)
    full = {s: v for s, v in by.items() if all(r.get("ok") for r in v)}
    cells = []
    for _, col in VARIANTS:
        p = [sum(r[col] for r in v) for v in full.values()]
        g = [sum(r["gt_mass"] for r in v) for v in full.values()]
        a, b, n = ape_bias(p, g)
        cells.append(f"{a:.0f} / {b:+.0f}")
    out.append(f"| {len(full)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(ROOT / "data" / "synth" / "hard_v12"))
    ap.add_argument("--n", type=int, default=5, help="number of ok scans (first n by id)")
    ap.add_argument("--all-scans", action="store_true", help="also use scans the manifest flags not-ok (more kinds)")
    ap.add_argument("--out", default=str(ROOT / "results" / "mass_step"))
    a = ap.parse_args()
    bd = Path(a.batch)
    man = D.read_manifest(bd)
    env_gt = gt_envelopes()
    rows, used = [], 0
    for sid in sorted(man):
        if used >= a.n:
            break
        if not man[sid].get("ok") and not a.all_scans:
            continue
        sc = D.load_scan(bd, sid, man)
        if sc.problem or not sc.items:
            continue
        r = scan_rows(sc, env_gt)
        if r:
            rows += r
            used += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out / f"items_{bd.name}_n{a.n}{"_all" if a.all_scans else ""}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    txt = tables(rows)
    (out / f"summary_{bd.name}_n{a.n}{"_all" if a.all_scans else ""}.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

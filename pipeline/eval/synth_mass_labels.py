"""USDA-consistent mass labels for the synthetic scans (side files; gt.json is never modified).

    python -m pipeline.eval.synth_mass_labels            # writes <scan>/gt/mass_labels.json + results/mass_labels/summary.md

Per item: synthetic mass (generator density table or asset weight), the USDA bulk / material densities from
data/density/bulk_density.csv (pipeline/foodscan/density_table.py), and
    mass_usda_bulk_g     = volume_envelope_gt_ml (results/rq1/envelope_items.csv) * usda_bulk
    mass_usda_material_g = volume_ml * usda_material
    mass_asset_g         = real weight of the scanned MetaFood3D asset (only where mass_source == "asset")
The asset weights are the only real-mass reference; comparing USDA masses with them is the accuracy check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline.foodscan.density_table import DensityTable  # noqa: E402

SYNTH = ROOT / "data" / "synth"
BATCHES = ["hard", "hard_v3", "hard_v9", "hard_v10", "hard_v11", "hard_v12"]
ENV_CSV = ROOT / "results" / "rq1" / "envelope_items.csv"
OUT = ROOT / "results" / "mass_labels"
KEYS = ["id", "category", "kind", "mass_synth_g", "mass_source_synth", "density_synth_g_ml", "usda_name_matched",
        "usda_match_method", "usda_bulk_g_ml", "usda_material_g_ml", "usda_confidence", "volume_ml",
        "volume_envelope_gt_ml", "mass_usda_bulk_g", "mass_usda_material_g", "mass_asset_g"]
SOURCES = ("synth", "usda_bulk", "usda_material")
_FIELD = {"synth": "mass_synth_g", "usda_bulk": "mass_usda_bulk_g", "usda_material": "mass_usda_material_g"}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def read_envelopes(path: Path = ENV_CSV) -> dict:
    env = {}
    for r in csv.DictReader(open(path)):
        v = _num(r["volume_envelope_gt"])
        if v is not None and v > 0:
            env[(r["batch"], r["scan"], int(r["item_id"]))] = v
    return env


def item_label(it: dict, env_ml: float | None, table: DensityTable) -> dict:
    """Mass-label record of one gt.json item."""
    kind = (it.get("shape") or {}).get("kind")
    vol = _num(it.get("volume_ml"))
    mass = _num(it.get("mass_g"))
    src = it.get("mass_source") or ("asset" if it.get("source") == "metafood3d" and mass else "unknown")
    dens = _num(it.get("density_g_ml"))
    if dens is None and mass and vol:
        dens = mass / vol                      # asset items: implied density of the real weight
    cat = it.get("category") or it.get("name") or ""
    row, method, conf = table.match(cat, kind)
    bulk, _, _ = table.lookup(cat, kind)
    mat = row.material if row is not None else None
    return dict(
        id=it["id"], category=cat, kind=kind, mass_synth_g=mass, mass_source_synth=src, density_synth_g_ml=dens,
        usda_name_matched=row.food if row is not None else None, usda_match_method=method,
        usda_bulk_g_ml=bulk, usda_material_g_ml=mat, usda_confidence=conf, volume_ml=vol,
        volume_envelope_gt_ml=env_ml,
        mass_usda_bulk_g=bulk * env_ml if env_ml else None,
        mass_usda_material_g=mat * vol if (mat and vol) else None,
        mass_asset_g=mass if src == "asset" else None)


def write_scan(batch: str, scan_dir: Path, env: dict, table: DensityTable) -> list[dict]:
    gt = json.loads((scan_dir / "gt.json").read_text())
    items = [item_label(it, env.get((batch, scan_dir.name, it["id"])), table) for it in gt["items"]]
    (scan_dir / "gt").mkdir(exist_ok=True)
    (scan_dir / "gt" / "mass_labels.json").write_text(json.dumps(
        dict(scan=scan_dir.name, batch=batch, items=items), indent=1))
    return items


def load_mass_lookup(source: str, synth_root: Path = SYNTH, batches=BATCHES) -> dict:
    """{(batch, scan, item_id): mass_g} read from the side files for `source` in synth|usda_bulk|usda_material."""
    f = _FIELD[source]
    out = {}
    for b in batches:
        for p in sorted((synth_root / b / "scans").glob("*/gt/mass_labels.json")):
            d = json.loads(p.read_text())
            for it in d["items"]:
                v = _num(it.get(f))
                if v is not None and v > 0:
                    out[(b, d["scan"], int(it["id"]))] = v
    return out


def apply_mass_source(items: list[dict], source: str) -> list[dict]:
    """Replace mass_g of rq1 envelope items (read_items rows) by the chosen label; unlabeled items get NaN."""
    if source == "synth":
        return items
    lk = load_mass_lookup(source)
    for r in items:
        r["mass_g"] = lk.get((r["batch"], r["scan"], int(r["item_id"])), float("nan"))
    return items


# ------------------------------------------------------------------------------------------ analysis

def log_ratio_stats(a, b):
    r = np.log(np.asarray(a, float) / np.asarray(b, float))
    return dict(n=len(r), med_ratio=float(np.exp(np.median(r))), med_abs_log=float(np.median(np.abs(r))),
                iqr_lo=float(np.exp(np.percentile(r, 25))), iqr_hi=float(np.exp(np.percentile(r, 75))))


def md_table(header, rows):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "".join(
        "| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def summarize(rows: list[dict]) -> tuple[str, list[dict]]:
    ok = [r for r in rows if r["mass_synth_g"] and r["mass_usda_bulk_g"]]
    cats = sorted({r["category"] for r in rows})
    md = ["# USDA vs synthetic mass labels\n",
          f"{len(rows)} items in {len({(r['batch'], r['scan']) for r in rows})} scans; {len(ok)} with a GT envelope volume "
          "(mass_usda_bulk defined). Reproduce: `python -m pipeline.eval.synth_mass_labels`.\n"]
    # 1. real asset weights
    asset = [r for r in ok if r["mass_asset_g"]]
    md.append("\n## 1. Accuracy check: USDA-based mass vs real asset weights (MetaFood3D scanned assets)\n")
    md.append("Ratio = USDA mass / real weight (1.0 is perfect); the synthetic label equals the real weight here by construction. "
              "Material path uses volume_ml x USDA material density; bulk path uses GT envelope x USDA bulk density "
              "(the envelope includes air gaps and the column fill, so it is expected to overshoot solid mass for single/piece items).\n")
    if asset:
        hdr = ["group", "n", "bulk: median ratio", "bulk IQR", "bulk median abs log", "material: median ratio", "material IQR",
               "material median abs log"]
        def rowf(name, sub):
            b = log_ratio_stats([r["mass_usda_bulk_g"] for r in sub], [r["mass_asset_g"] for r in sub])
            m = [r for r in sub if r["mass_usda_material_g"]]
            ms = log_ratio_stats([r["mass_usda_material_g"] for r in m], [r["mass_asset_g"] for r in m]) if m else None
            return [name, len(sub), f"{b['med_ratio']:.2f}", f"{b['iqr_lo']:.2f}-{b['iqr_hi']:.2f}", f"{b['med_abs_log']:.2f}",
                    f"{ms['med_ratio']:.2f}" if ms else "-", f"{ms['iqr_lo']:.2f}-{ms['iqr_hi']:.2f}" if ms else "-",
                    f"{ms['med_abs_log']:.2f}" if ms else "-"]
        body = [rowf("all asset items", asset)]
        for k in sorted({r["kind"] for r in asset}):
            body.append(rowf(f"kind {k}", [r for r in asset if r["kind"] == k]))
        md.append(md_table(hdr, body))
        md.append("\nPer category (n >= 2):\n\n")
        body = [rowf(c, [r for r in asset if r["category"] == c]) + [
            next(r["usda_name_matched"] or "median" for r in asset if r["category"] == c)]
            for c in sorted({r["category"] for r in asset}) if sum(r["category"] == c for r in asset) >= 2]
        md.append(md_table(hdr + ["USDA row"], body))
    else:
        md.append("No asset items.\n")
    # 2. per category
    md.append("\n## 2. Per-category density: generator vs USDA\n")
    md.append("`synth dens` = generator density (median over procedural items; for asset items the implied real weight/solid volume). "
              "`synth bulk` = pooled synthetic mass / GT envelope. `mass ratio` = median mass_usda_bulk / mass_synth. "
              "`dens ratio` = synth dens / USDA bulk (the table-to-table comparison of docs/density.md; the generator applies its density to the SOLID "
              "volume, so for leaves and toppings the pooled bulk column is far below it). Sorted by |log(dens ratio)|.\n")
    body = []
    for c in cats:
        sub = [r for r in ok if r["category"] == c]
        if not sub:
            continue
        sd = [r["density_synth_g_ml"] for r in sub if r["density_synth_g_ml"]]
        sb = sum(r["mass_synth_g"] for r in sub) / sum(r["volume_envelope_gt_ml"] for r in sub)
        ub = sub[0]["usda_bulk_g_ml"]
        mr = float(np.median([r["mass_usda_bulk_g"] / r["mass_synth_g"] for r in sub]))
        srcs = "/".join(sorted({r["mass_source_synth"] for r in sub}))
        dr = np.median(sd) / ub if sd else 1.0
        body.append((abs(np.log(dr)), [c, srcs, len(sub), f"{np.median(sd):.2f}" if sd else "-", f"{sb:.2f}", f"{ub:.2f}",
                                           f"{dr:.2f}", f"{sb / ub:.2f}", f"{mr:.2f}", sub[0]["usda_name_matched"] or f"({sub[0]['usda_match_method']})"]))
    body.sort(key=lambda t: -t[0])
    md.append(md_table(["category", "synth source", "n", "synth dens", "synth bulk", "USDA bulk", "dens ratio", "pooled bulk ratio", "mass ratio",
                        "USDA row"], [b[1] for b in body]))
    # 3. per kind
    md.append("\n## 3. Per-kind median |log ratio| of USDA mass to synthetic mass\n")
    body = []
    for k in sorted({r["kind"] for r in ok}):
        sub = [r for r in ok if r["kind"] == k]
        b = log_ratio_stats([r["mass_usda_bulk_g"] for r in sub], [r["mass_synth_g"] for r in sub])
        m = [r for r in sub if r["mass_usda_material_g"]]
        ms = log_ratio_stats([r["mass_usda_material_g"] for r in m], [r["mass_synth_g"] for r in m]) if m else None
        body.append([k, len(sub), f"{b['med_abs_log']:.2f}", f"{b['med_ratio']:.2f}", f"{ms['med_abs_log']:.2f}" if ms else "-",
                     f"{ms['med_ratio']:.2f}" if ms else "-"])
    md.append(md_table(["kind", "n", "bulk |log|", "bulk median ratio", "material |log|", "material median ratio"], body))
    # 4. changed items
    ch = [r for r in ok if abs(r["mass_usda_bulk_g"] / r["mass_synth_g"] - 1) > 0.30]
    chm = [r for r in rows if r["mass_synth_g"] and r["mass_usda_material_g"] and abs(r["mass_usda_material_g"] / r["mass_synth_g"] - 1) > 0.30]
    nm = sum(1 for r in rows if r["mass_synth_g"] and r["mass_usda_material_g"])
    md.append(f"\n## 4. Items whose mass changes by more than 30 %\n\nBulk path: {len(ch)} of {len(ok)} items "
              f"({100 * len(ch) / max(len(ok), 1):.0f} %); material path: {len(chm)} of {nm}. Full list: `changed_items.csv`. "
              "By category (bulk path):\n\n")
    byc = {}
    for r in ch:
        byc.setdefault(r["category"], []).append(r["mass_usda_bulk_g"] / r["mass_synth_g"])
    ntot = {c: sum(r["category"] == c for r in ok) for c in byc}
    md.append(md_table(["category", "changed / total", "median ratio"],
                       [[c, f"{len(v)} / {ntot[c]}", f"{np.median(v):.2f}"] for c, v in sorted(byc.items(), key=lambda t: -len(t[1]))]))
    return "\n".join(md), ch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", nargs="+", default=BATCHES)
    ap.add_argument("--synth-root", default=str(SYNTH))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    table, env = DensityTable(), read_envelopes()
    rows = []
    for b in a.batches:
        for sd in sorted((Path(a.synth_root) / b / "scans").iterdir()):
            if (sd / "gt.json").exists():
                for it in write_scan(b, sd, env, table):
                    rows.append(dict(it, batch=b, scan=sd.name))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md, ch = summarize(rows)
    (out / "summary.md").write_text(md)
    with open(out / "changed_items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "scan", "id", "category", "kind", "mass_synth_g", "mass_usda_bulk_g", "ratio"])
        for r in ch:
            w.writerow([r["batch"], r["scan"], r["id"], r["category"], r["kind"], f"{r['mass_synth_g']:.3f}",
                        f"{r['mass_usda_bulk_g']:.3f}", f"{r['mass_usda_bulk_g'] / r['mass_synth_g']:.3f}"])
    print(f"{len(rows)} items; wrote side files and {out / 'summary.md'}")


if __name__ == "__main__":
    main()

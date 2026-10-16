"""Sourced per-food *bulk* density table (g/mL) for the envelope-volume -> mass step.

    from foodscan.density_table import lookup, bulk_vs_material
    d, source, conf = lookup("steamed rice")          # (0.668, "USDA ...", 0.9)
    mass_g = envelope_volume_ml * d

Bulk (as-served) density = mass / volume of the pile *including the air between pieces*,
which is what an envelope volume (visible top surface closed to the container) measures.
Material density = void-free solid. Table: data/density/bulk_density.csv (provenance in
data/density/sources.md, method in docs/density.md).

Lookup order: exact name -> synonym -> token-containment / difflib fuzzy match -> median of
the requested kind -> global median. `confidence` is 1.0 exact, 0.9 synonym, 0.5-0.8 fuzzy,
0.35 kind median, 0.2 global median.

CLI (from pipeline/):  python -m foodscan.density_table [--n5k PATH] [--compare]
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "density" / "bulk_density.csv"
N5K_INGR = ROOT / "data" / "datasets" / "Nutrition5k" / "metadata" / "ingredients_metadata.csv"
SYNTH_DENSITY = ROOT / ".claude" / "worktrees" / "synth-hard" / "pipeline" / "synth" / "density.py"
ENVELOPE_ITEMS = ROOT / "results" / "rq1" / "envelope_items.csv"

KINDS = ("pile", "pieces", "leaf", "puree", "liquid", "solid", "scan")
# synthetic generator shape kinds -> table kinds (for the kind fallback)
KIND_ALIASES = {"leaves": "leaf", "piece": "pieces", "single": "pieces", "topping": "pile"}


@dataclass(frozen=True)
class Row:
    food: str
    kind: str
    bulk: float
    material: float | None
    synonyms: tuple[str, ...]
    source: str
    basis: str
    is_estimate: bool
    uncertainty: str


def norm(s: str) -> str:
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(_sing(w) for w in s.split())


def _sing(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("oes", "ches", "shes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w


class DensityTable:
    def __init__(self, path: Path | str = DEFAULT_CSV):
        self.path = Path(path)
        self.rows: list[Row] = []
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                self.rows.append(Row(
                    food=r["food"], kind=r["kind"], bulk=float(r["bulk_density_g_ml"]),
                    material=float(r["material_density_g_ml"]) if r.get("material_density_g_ml") else None,
                    synonyms=tuple(x for x in r.get("synonyms", "").split(";") if x),
                    source=r["source"], basis=r["basis"], is_estimate=r["is_estimate"] == "1",
                    uncertainty=r.get("uncertainty", "")))
        self.exact: dict[str, Row] = {}
        self.syn: dict[str, Row] = {}
        for r in self.rows:
            self.exact[norm(r.food)] = r
        for r in self.rows:
            for s in r.synonyms:
                self.syn.setdefault(norm(s), r)
        self.kind_median = {k: statistics.median(r.bulk for r in self.rows if r.kind == k)
                            for k in {r.kind for r in self.rows}}
        self.global_median = statistics.median(r.bulk for r in self.rows)

    def match(self, name: str, kind: str | None = None) -> tuple[Row | None, str, float]:
        """(row, method, confidence); row is None for the median fallbacks."""
        q = norm(name)
        if q in self.exact:
            return self.exact[q], "exact", 1.0
        if q in self.syn:
            return self.syn[q], "synonym", 0.9
        qt = set(q.split())
        best, best_score = None, 0.0
        for key, r in [*self.exact.items(), *self.syn.items()]:
            kt = set(key.split())
            if not kt:
                continue
            klast, qlast = key.split()[-1], q.split()[-1]
            if kt <= qt:                       # every key word is in the query ...
                if klast != qlast:             # ... and the head noun (last word) agrees
                    continue
                score = 0.6 + 0.2 * len(key) / max(len(q), 1)
            elif qt <= kt and len(q) >= 4 and qlast in kt:  # query is a more generic form of the key
                score = 0.5 + 0.1 * len(q) / len(key)
            else:
                ratio = difflib.SequenceMatcher(None, q, key).ratio()
                score = ratio * 0.8 if ratio >= 0.88 else 0.0
            if kind and score and r.kind != KIND_ALIASES.get(kind, kind):
                score -= 0.1                   # prefer a same-kind row
            if score > best_score:
                best, best_score = r, score
        if best is not None and best_score >= 0.5:
            return best, "fuzzy", round(min(best_score, 0.8), 2)
        k = KIND_ALIASES.get(kind or "", kind)
        if k in self.kind_median:
            return None, "kind", 0.35
        return None, "global", 0.2

    def lookup(self, name: str, kind: str | None = None) -> tuple[float, str, float]:
        row, method, conf = self.match(name, kind)
        if row is not None:
            return row.bulk, row.source + (" [estimate]" if row.is_estimate else ""), conf
        k = KIND_ALIASES.get(kind or "", kind)
        if method == "kind":
            return self.kind_median[k], f"median of kind '{k}'", conf
        return self.global_median, "global median of table", conf

    def bulk_vs_material(self, name: str) -> tuple[float, float | None, float | None]:
        """(bulk, material, bulk/material); material and ratio are None when unknown."""
        row, _, _ = self.match(name)
        if row is None:
            return self.global_median, None, None
        ratio = row.bulk / row.material if row.material else None
        return row.bulk, row.material, ratio


@lru_cache(maxsize=1)
def default_table() -> DensityTable:
    return DensityTable()


def lookup(name: str, kind: str | None = None) -> tuple[float, str, float]:
    return default_table().lookup(name, kind)


def bulk_vs_material(name: str):
    return default_table().bulk_vs_material(name)


# ---- mass step (envelope volume -> mass) ---------------------------------------------

# Envelope / solid volume ratio per kind. The scan measures the ENVELOPE (visible top surface
# closed down to the container, air gaps included), so solid volume ~ envelope / ratio.
# Source: medians of GT envelope / GT solid `volume_ml` on hard_v12 (results/rq1/envelope.md,
# results/rq1/envelope_reading.md item 1): piece 1.08, pile 1.18, single 1.28, topping 3.5,
# leaves 15 (v10 point: 1.06 / 1.93 / 2.08 / 6.0 / 13). pieces and solid reuse the piece / single
# values; puree and liquid fill their container flat, so envelope = solid (1.0, an assumption).
ENVELOPE_RATIO = {"pieces": 1.08, "pile": 1.18, "single": 1.28, "solid": 1.08,
                  "puree": 1.0, "liquid": 1.0, "topping": 3.5, "leaf": 15.0}

# Thin-layer rule: used when an item's height above the support is below THIN_HEIGHT_MM.
# mass = footprint area x THIN_THICKNESS_MM[kind] x material density. Thicknesses are
# assumptions (a sauce film ~1.5 mm, a bed of leaves ~2 mm of solid leaf), not measurements.
THIN_HEIGHT_MM = 4.0
THIN_THICKNESS_MM = {"topping": 1.5, "leaf": 2.0}

# Mass rule per kind: "bulk" = envelope x bulk density; "solid" = envelope / ratio x material
# density (thin-layer rule overrides it below THIN_HEIGHT_MM). Edit here to change a kind.
RULE_BY_KIND = {"pile": "bulk", "pieces": "bulk", "single": "bulk", "solid": "bulk",
                "puree": "bulk", "liquid": "bulk", "leaf": "solid", "topping": "solid"}

# Lookup-confidence multiplier per rule: how far the rule itself can be trusted.
RULE_CONFIDENCE = {"envelope_bulk": 1.0, "solid_material": 0.6, "thin_layer": 0.4}

MASS_KINDS = tuple(ENVELOPE_RATIO)
# synthetic-generator / table spellings -> mass-step kinds
KIND_NORMALISE = {"leaves": "leaf", "piece": "pieces", "scan": "pile"}
SAUCE_HEAD_WORDS = {"sauce", "gravy", "dressing", "salsa", "pesto", "syrup", "ketchup", "mustard",
                    "mayo", "mayonnaise", "aioli", "vinaigrette", "drizzle", "topping", "glaze", "dip",
                    "cheese melted", "melted cheese"}


def normalise_kind(kind: str | None) -> str | None:
    if not kind:
        return None
    k = KIND_NORMALISE.get(kind.lower(), kind.lower())
    return k if k in ENVELOPE_RATIO else None


def infer_kind(name: str, hint: str | None = None, table: "DensityTable | None" = None) -> tuple[str, str]:
    """(kind, how) from the item name. A valid `hint` (e.g. from naming/instances.json) wins;
    sauce-like names (head noun sauce / gravy / dressing / salsa / pesto ...) are 'topping';
    otherwise the kind of the matched density row (exact / synonym / fuzzy); else 'pile'."""
    k = normalise_kind(hint)
    if k:
        return k, "hint"
    t = table or default_table()
    q = norm(name)
    if q.split() and (q.split()[-1] in SAUCE_HEAD_WORDS or q in SAUCE_HEAD_WORDS or q.startswith("melted ")):
        return "topping", "name"
    row, method, _ = t.match(name)
    if row is not None:
        return normalise_kind(row.kind) or "pile", f"table:{method}"
    return "pile", "default"


def volume_solid_est(volume_envelope_ml: float, kind: str) -> float:
    return volume_envelope_ml / ENVELOPE_RATIO.get(normalise_kind(kind) or "pile", ENVELOPE_RATIO["pile"])


def material_lookup(name: str, kind: str | None = None,
                    table: "DensityTable | None" = None) -> tuple[float, str, float]:
    """(material density g/mL, source, confidence). Matched row's material column, else the median
    material density of the matched row's kind / the requested kind, else 1.0 (water-like)."""
    t = table or default_table()
    row, method, conf = t.match(name, kind)
    if row is not None and row.material:
        return row.material, row.source + (" [estimate]" if row.is_estimate else ""), conf
    k = normalise_kind(row.kind if row is not None else kind)
    mats = [r.material for r in t.rows if r.material and (k is None or normalise_kind(r.kind) == k)]
    if mats:
        return statistics.median(mats), f"median material density of kind '{k}'", min(conf, 0.35)
    return 1.0, "assumed 1.0 g/mL (no material density known)", 0.2


def estimate_mass(name: str, kind: str | None, volume_envelope_ml: float, footprint_cm2: float | None = None,
                  height_mm: float | None = None, rule: str = "envelope",
                  bulk_oracle: float | None = None, material_oracle: float | None = None,
                  table: "DensityTable | None" = None) -> dict:
    """Envelope volume -> mass. rule 'envelope' (default) picks per kind:
        envelope_bulk   mass = volume_envelope_ml x bulk density          (pile/pieces/single/solid/puree/liquid)
        solid_material  mass = volume_envelope_ml / ENVELOPE_RATIO[kind] x material density     (leaf, topping)
        thin_layer      mass = footprint_cm2 x THIN_THICKNESS_MM[kind]/10 x material density
                        (leaf / topping only, when height_mm < THIN_HEIGHT_MM)
    `bulk_oracle` / `material_oracle` replace the table density (validation against synthetic densities).
    Returns mass_g, mass_rule, density_used_g_ml, density_kind, density_source, density_conf, kind,
    kind_source, envelope_ratio, mass_confidence, volume_solid_est_ml, and `notes`."""
    t = table or default_table()
    kind_n, kind_src = infer_kind(name, kind, t)
    ratio = ENVELOPE_RATIO[kind_n]
    solid = volume_envelope_ml / ratio
    out = {"kind": kind_n, "kind_source": kind_src, "envelope_ratio": ratio, "volume_solid_est_ml": solid}
    thin = (RULE_BY_KIND[kind_n] == "solid" and kind_n in THIN_THICKNESS_MM and height_mm is not None
            and footprint_cm2 is not None and height_mm < THIN_HEIGHT_MM)
    if RULE_BY_KIND[kind_n] == "bulk":
        if bulk_oracle is not None:
            d, src, conf = bulk_oracle, "oracle", 1.0
        else:
            d, src, conf = t.lookup(name, kind_n)
        mass, label, dk = volume_envelope_ml * d, "envelope_bulk", "bulk"
    else:
        if material_oracle is not None:
            d, src, conf = material_oracle, "oracle", 1.0
        else:
            d, src, conf = material_lookup(name, kind_n, t)
        if thin:
            mass, label = footprint_cm2 * THIN_THICKNESS_MM[kind_n] / 10.0 * d, "thin_layer"
        else:
            mass, label = solid * d, "solid_material"
        dk = "material"
    out.update(mass_g=float(max(mass, 0.0)), mass_rule=label, density_used_g_ml=float(d), density_kind=dk,
               density_source=src, density_conf=float(conf),
               mass_confidence=round(float(conf) * RULE_CONFIDENCE[label], 3))
    if label == "thin_layer":
        out["thickness_mm"] = THIN_THICKNESS_MM[kind_n]
    return out


# ---- CLI ---------------------------------------------------------------------------

def _synth_categories(path: Path = SYNTH_DENSITY) -> dict[str, float]:
    """DENSITY dict of the synthetic generator, parsed without importing it."""
    if not path.exists():
        return {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "DENSITY":
            return ast.literal_eval(node.value)
    return {}


def _coverage(t: DensityTable, names: list[str], label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in names:
        counts[t.match(n)[1]] = counts.get(t.match(n)[1], 0) + 1
    tot = max(len(names), 1)
    hit = sum(counts.get(m, 0) for m in ("exact", "synonym", "fuzzy"))
    print(f"{label}: {len(names)} names, matched to a named row {hit} ({100*hit/tot:.1f}%)  "
          + ", ".join(f"{m}={counts.get(m, 0)}" for m in ("exact", "synonym", "fuzzy", "kind", "global")))
    return counts


def compare_envelope(t: DensityTable, path: Path = ENVELOPE_ITEMS, thresh: float = 0.30) -> None:
    """Pooled synthetic bulk (sum mass / sum GT envelope volume) per category vs this table.

    The synthetic masses come from the synthetic density table or scanned-asset masses, not
    from reality: this only shows where the two disagree, not which is right.
    """
    if not path.exists():
        return
    agg: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                m, v = float(r["mass_g"]), float(r["volume_envelope_gt"])
            except ValueError:
                continue
            if m > 0 and v > 0:
                a = agg.setdefault(r["category"], [0.0, 0.0, 0])
                a[0] += m; a[1] += v; a[2] += 1
    print(f"\nsynthetic envelope bulk (pooled mass/envelope) vs table, |diff| > {thresh:.0%}:")
    bad = 0
    for cat, (m, v, n) in sorted(agg.items()):
        row, method, _ = t.match(cat)
        tb = row.bulk if row else t.global_median
        if abs(m / v / tb - 1) > thresh:
            bad += 1
            print(f"  {cat:22s} n={n:3d} synth {m/v:.2f}  table {tb:.2f} ({row.food if row else 'median'})  {100*(m/v/tb-1):+.0f} %")
    print(f"  {bad}/{len(agg)} categories disagree")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--n5k", default=str(N5K_INGR))
    ap.add_argument("--list-misses", action="store_true")
    ap.add_argument("--envelope", action="store_true", help="synthetic mass/envelope bulk vs table")
    ap.add_argument("--compare", action="store_true", help="table vs synthetic per-category densities")
    a = ap.parse_args()
    t = DensityTable(a.csv)
    print(f"{len(t.rows)} rows ({sum(r.is_estimate for r in t.rows)} estimates); "
          f"kinds " + ", ".join(f"{k}={sum(r.kind == k for r in t.rows)}" for k in sorted(t.kind_median))
          + f"; global median {t.global_median:.3f}")
    n5k: list[str] = []
    if Path(a.n5k).exists():
        with open(a.n5k, newline="") as f:
            n5k = [r["ingr"] for r in csv.DictReader(f) if r["ingr"].strip() and r["ingr"].strip() != "deprecated"]
        _coverage(t, n5k, "Nutrition5k ingredients")
        if a.list_misses:
            print("  not matched:", ", ".join(n for n in n5k if t.match(n)[1] in ("kind", "global")))
    synth = _synth_categories()
    if synth:
        _coverage(t, list(synth), "synthetic DENSITY categories")
        if a.list_misses:
            print("  not matched:", ", ".join(n for n in synth if t.match(n)[1] in ("kind", "global")))
    if ENVELOPE_ITEMS.exists():
        with open(ENVELOPE_ITEMS, newline="") as f:
            cats = sorted({r["category"] for r in csv.DictReader(f)})
        _coverage(t, cats, "synthetic scene categories (envelope_items.csv)")
    if a.compare and synth:
        print("\nsynthetic table vs this table (|diff| > 30 % of this table's bulk):")
        for cat, sd in sorted(synth.items()):
            row, method, _ = t.match(cat)
            if row is None:
                continue
            rel = sd / row.bulk - 1
            if abs(rel) > 0.30:
                print(f"  {cat:18s} synth {sd:.2f}  table {row.bulk:.3f} ({row.food}, {method})  {100*rel:+.0f} %")
    if a.envelope:
        compare_envelope(t)


if __name__ == "__main__":
    main()

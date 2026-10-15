"""How diverse is a batch? Counts over its scene specs, to steer sampling toward the targets in
docs/dataset_curation.md (section 8).

    python -m synth.diversity ../data/synth/<batch> [--json out.json]

Axes: meal source (template / n5k / random), cuisine, dish (plate / bowl, procedural / scanned),
heap layout, foods (procedural by consistency, scans by category), items per scene, scenes
with soft food / piles / leaves / clutter, table and lighting variety. Each axis prints its
distribution and an effective number of categories (exp of the entropy): how many equally
common categories it is worth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .looks import kind_of


def effective(c: Counter) -> float:
    p = np.array(list(c.values()), float)
    p = p[p > 0] / p.sum()
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.0


def report(root: Path) -> dict:
    specs = [json.loads(p.read_text()) for p in sorted((root / "specs").glob("*/spec.json"))]
    ax: dict[str, Counter] = {k: Counter() for k in
                              ("source", "cuisine", "dish", "layout", "food", "consistency", "n_items", "flags",
                               "table", "hdri")}
    for s in specs:
        m = s.get("meal") or {"source": "random", "cuisine": None}
        ax["source"][m["source"]] += 1
        ax["cuisine"][m.get("cuisine") or "random"] += 1
        c = s["container"]
        ax["dish"][f"{c['kind']} ({'scanned' if c.get('asset') else 'procedural'})"] += 1
        ax["layout"][(s.get("heap") or {}).get("layout", "single food" if any(
            it.get("physics") for it in s["items"]) else "no heap")] += 1
        ax["n_items"][len(s["items"])] += 1
        for it in s["items"]:
            name = it["category"].lower().replace("_", " ")
            ax["food"][name] += 1
            ax["consistency"]["scan" if it["dynamic"] or it.get("matrix_world") else
                              "leaves" if it.get("cloth") else kind_of(it["name"])] += 1
        shapes = [(it.get("shape") or {}).get("kind") for it in s["items"]]
        for flag, on in (("soft food", any(it.get("physics") for it in s["items"])), ("pile", "pile" in shapes),
                         ("leaves", "leaves" in shapes), ("topping", "topping" in shapes),
                         ("clutter", bool(s.get("distractors")))):
            ax["flags"][flag] += on
        t = s.get("table", {})
        ax["table"][(t.get("pbr") or {}).get("kind", "procedural")] += 1
        ax["hdri"][Path(s["lighting"]["hdri"]).parent.name if s["lighting"].get("hdri") else "none"] += 1
    return {"n_scenes": len(specs), **{k: dict(v.most_common()) for k, v in ax.items()},
            "effective": {k: round(effective(v), 1) for k, v in ax.items() if k not in ("flags", "n_items")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    r = report(Path(args.root))
    print(f"{r['n_scenes']} scenes")
    for k in ("source", "cuisine", "dish", "layout", "consistency", "table", "flags", "n_items"):
        tot = r["n_scenes"] if k == "flags" else sum(r[k].values())
        print(f"  {k:<12}" + ", ".join(f"{a} {100 * b / max(tot, 1):.0f}%" for a, b in r[k].items())
              + (f"   (effective {r['effective'][k]})" if k in r["effective"] else ""))
    print(f"  foods: {len(r['food'])} distinct (effective {r['effective']['food']}); "
          f"top: {', '.join(f'{a} {b}' for a, b in list(r['food'].items())[:12])}")
    print(f"  rooms: {len(r['hdri'])} distinct HDRIs (effective {r['effective']['hdri']})")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()

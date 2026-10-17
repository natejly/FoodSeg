"""Volume error of run_meal on synthetic scans with known per-item volume.

    python -m synth.evaluate ../data/synth/v0/scans --out ../results/synth/v0 \
        [--support plane container both] [--masks gt]

For each scan: fuse, then measure every item against (a) the table plane and (b)
the exact container template in gt/container.npz, with oracle masks from gt/masks.
Items are matched to ground truth by mask id. Reports per-container-kind errors,
the comparison the N5k single-frame run could not make (no bowl floors there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foodscan.run_meal import run


def evaluate(scan: Path, support: str) -> list[dict]:
    gt = json.loads((scan / "gt.json").read_text())
    res = run(scan, container=(scan / "gt" / "container.npz") if support == "container" else None,
              masks_dir=scan / "gt" / "masks", nutrition=False)
    pred = {it["id"]: it for it in res["items"]}
    rows = []
    for it in gt["items"]:
        p = pred.get(it["id"])
        v = p["volume_ml"] if p else 0.0
        rows.append({"scan": scan.name, "support": support, "kind": gt["container"]["kind"],
                     "id": it["id"], "name": it["name"], "fill": it["fill"] is not None,
                     "off_container": it["off_container"], "frames_visible": it["frames_visible"],
                     "gt_ml": it["volume_ml"], "pred_ml": v, "err_ml": v - it["volume_ml"],
                     "err_pct": 100 * (v - it["volume_ml"]) / it["volume_ml"], "found": p is not None,
                     # fell off the dish during the physics drop, or never in view: not a fair test
                     "excluded": it["off_container"] or it["frames_visible"] < 3})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"n_excluded": sum(r["excluded"] for r in rows)}
    rows = [r for r in rows if not r["excluded"]]
    for sup in sorted({r["support"] for r in rows}):
        for kind in ("bowl", "plate", "all"):
            for fill in (True, False, None):
                rs = [r for r in rows if r["support"] == sup and (kind == "all" or r["kind"] == kind)
                      and (fill is None or r["fill"] == fill)]
                if not rs:
                    continue
                e = np.array([r["err_pct"] for r in rs])
                key = f"{sup}/{kind}/" + {True: "fill", False: "discrete", None: "all"}[fill]
                out[key] = {"n": len(rs), "mape": float(np.abs(e).mean()), "median_signed_pct": float(np.median(e)),
                            "mae_ml": float(np.mean([abs(r["err_ml"]) for r in rs]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--out", required=True)
    ap.add_argument("--support", nargs="+", default=["plane", "container"], choices=["plane", "container"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scan in sorted(p for p in Path(args.scans).iterdir() if (p / "gt.json").exists()):
        for sup in args.support:
            try:
                new = evaluate(scan, sup)
            except Exception as e:  # noqa: BLE001
                print(f"{scan.name} [{sup}] failed: {e}")
                continue
            rows += new
            for r in new:
                print(f"{r['scan']} {r['support']:<9} {r['kind']:<5} {r['name']:<16} "
                      f"gt {r['gt_ml']:6.1f}  pred {r['pred_ml']:6.1f}  ({r['err_pct']:+.0f} %)")
    with open(out / "per_item.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"excluded (off the dish or < 3 frames visible): {summ.pop('n_excluded')} item-rows")
    for k, v in summ.items():
        print(f"{k:<28} n={v['n']:<4} MAPE {v['mape']:6.1f} %  median {v['median_signed_pct']:+6.1f} %")


if __name__ == "__main__":
    main()

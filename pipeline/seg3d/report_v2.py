"""Round-2 report: per-arm table + paired per-scan bootstrap vs lift_label.  python -m seg3d.report_v2"""
import json
import numpy as np
from .ablate import OUT, DECODERS

B = json.loads((OUT / "v2_baseline.json").read_text())
runs = {}
for f in sorted(OUT.glob("v2_*_s*/final.json")):
    info = json.loads((f.parent / "run_info.json").read_text())
    runs.setdefault(info["name"], []).append((info["seed"], json.loads(f.read_text()), info))
rng = np.random.default_rng(0)


def boot(d, n=1000):
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


rows, out = [], {}
for reg in ("r1", "r2"):
    b = B[reg]["lift_baseline"]
    rows.append(f"\n#### Eval regime {reg.upper()}\n")
    rows.append("| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift (95% CI) | corrected | broken |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|")
    rows.append(f"| lift_label | - | - | {b['sem_miou']:.3f} | {b['inst_miou']:.3f} | {b['pq']:.3f} | {b['exact_count']:.2f} | - | 0 | 0 |")
    bs = np.array(b["scan_inst_miou"])
    for name, lst in runs.items():
        is_koc = "decoder=koc" in lst[0][2].get("sets", [])
        for dec in ("plain", "anchored", "koc", "koc2"):
            if dec.startswith("koc") and not is_koc:
                continue
            key = "model" if (dec == "plain" and not is_koc) or dec == "koc" else "model_" + dec
            if any(key not in x[1][reg] or x[1][reg][key] is None for x in lst):
                continue
            rs = [x[1][reg][key] for x in lst]
            per = np.mean([r["scan_inst_miou"] for r in rs], 0) - bs      # seeds averaged per scan, then paired
            m, lo, hi = boot(per)
            g = lambda k: np.mean([r[k] for r in rs])
            rows.append(f"| {name} ({lst[0][2]['train_regime']}-trained) | {len(lst)} | {dec} | {g('sem_miou'):.3f} | {g('inst_miou'):.3f} | {g('pq'):.3f} | "
                        f"{g('exact_count'):.2f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {g('corrected_food'):.3f} | {g('broken_food'):.3f} |"
                        + (f" keep {g('keep_rate'):.3f} unrec {g('unrecoverable'):.3f} (of wrong {g('unrecoverable_of_wrong'):.3f})" if dec.startswith("koc") else ""))
            out[f"{reg}/{name}/{dec}"] = dict(diff=m, lo=lo, hi=hi, seeds=len(lst), iters=[x[1]["train"]["iters"] for x in lst])
(OUT / "round2.json").write_text(json.dumps(out, indent=1))
print("\n".join(rows))
print({n: [(x[0], x[1]["train"]["iters"], round(x[2]["wall_s"])) for x in l] for n, l in runs.items()})

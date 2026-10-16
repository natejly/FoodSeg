"""Round-3 report: per-arm tables + paired bootstraps vs lift_label (scan-level and item-level, 95 % CI).
python -m seg3d.report_v3   (writes results/seg3d/round3.json, prints markdown)"""
import json
import numpy as np
from .ablate import OUT

B = json.loads((OUT / "v3_baseline.json").read_text())
SP = json.loads((OUT / "split_v3.json").read_text())
SP2 = json.loads((OUT / "split_v2.json").read_text())
v2_train = set(SP2["train"])
unseen = np.array([n not in v2_train for n in SP["val"]])        # val scans a round-2 checkpoint never trained on
rng = np.random.default_rng(0)


def load(prefix):
    runs = {}
    for f in sorted(OUT.glob(f"{prefix}_*_s*/final.json")) if prefix == "v3" else sorted(OUT.glob("v3x_*/final.json")):
        info = json.loads((f.parent / "run_info.json").read_text())
        key = info["name"] if prefix == "v3" else f"{info['name']}"
        runs.setdefault(key, []).append((info["seed"], json.loads(f.read_text()), info))
    return runs


def boot(per, n=2000):
    per = np.asarray(per, float)
    m = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(n)])
    return per.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_items(d_item, n_per_scan, n=2000):
    """cluster bootstrap: resample scans, pool their item differences."""
    off = np.r_[0, np.cumsum(n_per_scan)]
    chunks = [d_item[off[i]:off[i + 1]] for i in range(len(n_per_scan))]
    m = []
    for _ in range(n):
        idx = rng.integers(0, len(chunks), len(chunks))
        m.append(np.concatenate([chunks[i] for i in idx]).mean())
    return d_item.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"            # noqa: E731


def table(runs, title, out, subset=False):
    rows = []
    for reg in ("r1", "r2"):
        b = B[reg]["lift_baseline"]
        rows.append(f"\n#### {title}, eval regime {reg.upper()}\n")
        rows.append("| arm | seeds | decode | sem mIoU | inst mIoU | PQ | count | d vs lift per scan (95% CI) | d vs lift per item (95% CI) | corrected | broken | keep |")
        rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows.append(f"| lift_label | - | - | {b['sem_miou']:.3f} | {b['inst_miou']:.3f} | {b['pq']:.3f} | {b['exact_count']:.2f} | - | - | 0 | 0 | 1 |")
        bs, bi, bn = np.array(b["scan_inst_miou"]), np.array(b["item_inst_iou"]), np.array(b["scan_n_items"])
        for name, lst in runs.items():
            is_koc = "decoder=koc" in lst[0][2].get("sets", [])
            for dec in ("plain", "anchored", "koc", "koc2"):
                if dec.startswith("koc") and not is_koc:
                    continue
                key = "model" if (dec == "plain" and not is_koc) or dec == "koc" else "model_" + dec
                if any(x[1][reg].get(key) is None for x in lst):
                    continue
                rs = [x[1][reg][key] for x in lst]
                per = np.mean([r["scan_inst_miou"] for r in rs], 0) - bs
                it = np.mean([r["item_inst_iou"] for r in rs], 0) - bi
                s, i = boot(per), boot_items(it, bn)
                g = lambda k: np.mean([r[k] for r in rs])                        # noqa: E731
                extra = f"{g('keep_rate'):.3f}" if dec.startswith("koc") else "-"
                rows.append(f"| {name} | {len(lst)} | {dec} | {g('sem_miou'):.3f} | {g('inst_miou'):.3f} | {g('pq'):.3f} | {g('exact_count'):.2f} | {fmt(s)} | {fmt(i)} | "
                            f"{g('corrected_food'):.3f} | {g('broken_food'):.3f} | {extra} |")
                out[f"{title}/{reg}/{name}/{dec}"] = dict(scan=s, item=i, seeds=len(lst), inst_miou=g("inst_miou"), sem=g("sem_miou"),
                                                          corrected=g("corrected_food"), broken=g("broken_food"), count=g("exact_count"))
                if subset:                                                      # same, restricted to val scans unseen by round 2
                    su = boot(per[unseen])
                    rows[-1] += f" unseen-by-v2 ({unseen.sum()} scans): {fmt(su)}"
    return rows


out, rows = {}, []
rows += table(load("v3"), "Round-3 runs (trained on split_v3)", out)
rows += table(load("x"), "Round-2 checkpoints evaluated on split_v3 val (no retraining)", out, subset=True)
for reg in ("r1", "r2"):
    b = B[reg]["lift_baseline"]
    out[f"lift/{reg}"] = {k: b[k] for k in ("sem_miou", "inst_miou", "pq", "exact_count", "n_items", "n_scans")}
(OUT / "round3.json").write_text(json.dumps(out, indent=1))
print("\n".join(rows))
print({n: [(x[0], x[1]["train"]["iters"], round(x[2]["wall_s"])) for x in l] for n, l in load("v3").items() if "train" in l[0][1]})

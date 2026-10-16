"""Sequential ablation driver for the seg3d refiner.  Run from /home/kyle/Desktop/FoodSeg/pipeline:

  python -m seg3d.ablate --split                 # (re)write results/seg3d/split.json + print counts
  python -m seg3d.ablate --run  [--budget-min N] # run the grid in priority order, one training job at a time
  python -m seg3d.ablate --aggregate             # results/seg3d/<run>/final.json -> ablations.json + ablations.md tables

Each job = `python -m seg3d.train` subprocess (strictly sequential, max_minutes cap) followed by `--final` eval of
last.pt (NOT best.pt: best.pt is picked on the same tiny val set, which would be optimistic) on both lift regimes
(r1 stored lift as-is, r2 hard lift) x decoders (plain, anchored, anchored_split, anchored_nob).  Jobs whose
final.json exists are skipped, so the driver is resumable.  Hand-written text in ablations.md outside the
AUTO markers is preserved by --aggregate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "seg3d"
ROOTS = ["data/synth/hard", "data/synth/hard_v3", "data/synth/hard_v9", "data/synth/hard_v10"]
SPLIT = OUT / "split.json"
DECODERS = ["plain", "anchored", "anchored_split", "anchored_nob"]

# schedule shared by every run (set from the timing pilot): epochs x (n_train_scans * epoch_repeat) batches
SCHED = dict(epochs=8, freeze_epochs=8, patch_size=512, **{"data.epoch_repeat": 4}, val_every=4, max_minutes=6, grad_accum=2, num_workers=2)
FULL_SETS = {}                                  # defaults = Sonata frozen, all inputs, all augmentations

NOLIFT = ['data.zero_groups=["lift"]']
GRID = [   # (name, extra --set overrides, train_regime, eval_regime (headline), seeds-priority)
    ("full_r2", [], "r2", "r2"),
    ("full_r1", [], "r1", "r1"),
    ("scratch_r2", ["pretrained=false", "freeze_epochs=0", "unfreeze=all"], "r2", "r2"),
    ("nolift_r2", NOLIFT, "r2", "r2"),
    ("norgb_r2", ['data.zero_groups=["rgb"]'], "r2", "r2"),
    ("nodropout_r2", ["aug.drop_rgb_p=0.0", "aug.drop_normal_p=0.0", "aug.drop_lift_p=0.0"], "r2", "r2"),
    ("nocorrupt_r2", ["aug.lift_corrupt=false"], "r2", "r2"),
    ("dgcnn_xyz_r2", ["model=dgcnn", 'data.zero_groups=["rgb","normal","lift"]'], "r2", "r2"),
    ("dgcnn_xyzrgb_r2", ["model=dgcnn", 'data.zero_groups=["normal","lift"]'], "r2", "r2"),
    ("unfreeze_top_r2", ["freeze_epochs=2", "unfreeze=top"], "r2", "r2"),
]
# order of execution: every config at seed 0 first, then second seeds in this order
SEED1_PRIORITY = ["full_r2", "nolift_r2", "full_r1", "norgb_r2", "nodropout_r2", "nocorrupt_r2", "dgcnn_xyzrgb_r2", "scratch_r2", "dgcnn_xyz_r2"]


# ----------------------------------------------------------------------------- split
def make_split(seed=0):
    import numpy as np
    from .data import load_scan  # noqa: F401  (only to fail early if the env is broken)
    scans = {}
    for r in ROOTS:
        man = {}
        mf = REPO / r / "manifest.jsonl"
        if mf.exists():
            man = {j["scan"]: j for j in map(json.loads, mf.read_text().splitlines() if mf.exists() else [])}
        for p in sorted((REPO / r / "scans").glob("*/gt/seg_points.npz")):
            sd = p.parent.parent
            ok = man.get(sd.name, {}).get("ok", True) if r in ROOTS[:2] else man.get(sd.name, {}).get("ok", False)
            n_items = len(json.loads((sd / "gt.json").read_text())["items"]) if (sd / "gt.json").exists() else None
            lab = np.load(p)["label"]
            n_vis = int(len(np.unique(lab[(lab > 0) & (lab < 252)])))
            scans[str(sd.relative_to(REPO))] = dict(batch=Path(r).name, ok=bool(ok), n_items=n_items, n_items_in_cloud=n_vis,
                                                    has_lift="lift_label" in np.load(p).files)
    rng = np.random.default_rng(seed)
    val, train = [], []
    for b in [Path(r).name for r in ROOTS]:
        names = [n for n, v in scans.items() if v["batch"] == b]
        elig = [n for n in names if scans[n]["ok"] or b in ("hard", "hard_v3")]
        n_val = min(int(round(0.25 * len(names))), max(1, int(0.6 * len(elig))))
        pick = set(rng.permutation(elig)[:n_val].tolist()) if elig else set()
        val += sorted(pick)
        train += [n for n in names if n not in pick]
    stats = {}
    for nm, lst in (("train", train), ("val", val)):
        stats[nm] = dict(scans=len(lst), items_gt_json=sum(scans[n]["n_items"] or 0 for n in lst),
                         items_in_cloud=sum(scans[n]["n_items_in_cloud"] for n in lst),
                         by_batch={b: sum(scans[n]["batch"] == b for n in lst) for b in [Path(r).name for r in ROOTS]},
                         ok_scans=sum(scans[n]["ok"] for n in lst))
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(dict(seed=seed, train=train, val=val, stats=stats, scans=scans), indent=1))
    return stats


# ----------------------------------------------------------------------------- running
def base_sets(extra, seed, train_regime, eval_regime):
    s = [f"train_regime={train_regime}", f"eval_regime={eval_regime}", f"seed={seed}",
         "data.roots=" + json.dumps(ROOTS), "data.split_file=results/seg3d/split.json"]
    s += [f"{k}={v}" for k, v in SCHED.items()]
    return s + list(extra)


def run_job(name, extra, tr, ev, seed, extra_env=None):
    run = f"abl_{name}_s{seed}"
    d = OUT / run
    if (d / "final.json").exists():
        print("skip (done)", run)
        return
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "-m", "seg3d.train", "--run", run, "--set"] + base_sets(extra, seed, tr, ev)
    t0 = time.time()
    with open(d / "train.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=REPO / "pipeline", env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=60 * 14).returncode
    wall = time.time() - t0
    (d / "run_info.json").write_text(json.dumps(dict(name=name, seed=seed, rc=rc, wall_s=wall, train_regime=tr, eval_regime=ev, sets=extra)))
    print(f"{run}: rc={rc} wall {wall:.0f}s")
    if rc == 0:
        rc2 = subprocess.run([sys.executable, "-m", "seg3d.ablate", "--final", run], cwd=REPO / "pipeline", env=env).returncode
        if rc2 == 0 and not name.startswith("full"):
            (d / "last.pt").unlink(missing_ok=True)
        (d / "best.pt").unlink(missing_ok=True)


def final_eval(run):
    import torch
    from . import eval as E
    from .train import Config, build_model
    d = OUT / run
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    res = {}
    for reg in ("r1", "r2"):
        res[reg] = E.evaluate(model, E.make_sets(cfg, "val", reg), cfg, baseline=False, decoders=tuple(DECODERS) + (("koc", "koc2") if cfg.decoder == "koc" else ()), volumes=False)
    hist = json.loads((d / "metrics.json").read_text())
    info = json.loads((d / "run_info.json").read_text())
    res["train"] = dict(peak_gib=max(h["peak_gib"] for h in hist), s_per_iter=float(np.mean([h["s_per_iter"] for h in hist])),
                        iters=hist[-1]["iters"], wall_s=info["wall_s"], epochs_done=len(hist),
                        train_loss=[h["train_loss"] for h in hist],
                        val_curve=[(h["epoch"], h["val"]["model"]["inst_miou"]) for h in hist if "val" in h and h["val"]["model"]],
                        phases=[h["phase"] for h in hist])
    (d / "final.json").write_text(json.dumps(res, indent=1))


def run_baseline():
    from . import eval as E
    from .train import Config
    d = OUT / "abl_baseline"
    d.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.data.roots, cfg.data.split_file = ROOTS, "results/seg3d/split.json"
    res = {reg: E.evaluate(None, E.make_sets(cfg, "val", reg), cfg, baseline=True, volumes=False) for reg in ("r1", "r2")}
    (d / "final.json").write_text(json.dumps(res, indent=1))
    for reg in res:
        b = res[reg]["lift_baseline"]
        print(f"lift baseline {reg}: inst_miou {b['inst_miou']:.3f} sem_miou {b['sem_miou']:.3f} pq {b['pq']:.3f} count {b['exact_count']:.2f}")


def run_grid(budget_min):
    t0 = time.time()
    jobs = [(n, e, tr, ev, 0) for n, e, tr, ev in GRID]
    byname = {n: (n, e, tr, ev) for n, e, tr, ev in GRID}
    jobs += [(*byname[n], 1) for n in SEED1_PRIORITY]
    skipped = []
    for n, e, tr, ev, seed in jobs:
        if n == "unfreeze_top_r2" and gpu_other_mib() > 1024:
            skipped.append((n, seed, "GPU not free (> 1 GB used by others)"))
            continue
        el = (time.time() - t0) / 60
        if el > budget_min - 8:
            skipped.append((n, seed, f"budget ({el:.0f}/{budget_min} min)"))
            continue
        run_job(n, e, tr, ev, seed)
    (OUT / "ablate_skipped.json").write_text(json.dumps(skipped))
    print("skipped:", skipped)


def gpu_other_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return int(o.strip().splitlines()[0])
    except Exception:                                       # noqa: BLE001
        return 99999


# ----------------------------------------------------------------------------- aggregation
KEYS = ["sem_miou", "inst_miou", "pq", "boundary_iou", "exact_count", "corrected_food", "broken_food"]


def _ms(vals):
    v = np.array(vals, float)
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} ({v.min():.2f}-{v.max():.2f})"


def collect():
    runs = {}
    for f in sorted(OUT.glob("abl_*_s*/final.json")):
        info = json.loads((f.parent / "run_info.json").read_text())
        res = json.loads(f.read_text())
        hist = json.loads((f.parent / "metrics.json").read_text())
        # train-only peak: epochs before the first validation pass (eval runs fp32 and has its own, higher peak)
        n_pre = SCHED["val_every"] - 1
        res["train"]["peak_train_gib"] = max(h["peak_gib"] for h in hist[:max(n_pre, 1)])
        runs.setdefault(info["name"], []).append((info["seed"], res, info))
    base = json.loads((OUT / "abl_baseline" / "final.json").read_text()) if (OUT / "abl_baseline" / "final.json").exists() else None
    return runs, base


def scan_se(results):
    """SE of the mean per-scan instance mIoU, averaging per-scan values over seeds."""
    arr = np.mean([r["scan_inst_miou"] for r in results], 0)
    return float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")


def paired(rs, b):
    """Mean per-scan instance-mIoU difference to the lift baseline (+- SE over scans; seeds averaged first)."""
    d = np.mean([r["scan_inst_miou"] for r in rs], 0) - np.array(b["scan_inst_miou"])
    return f"{d.mean():+.3f} +-{d.std(ddof=1) / math.sqrt(len(d)):.3f}" if len(d) > 1 else "-"


def aggregate():
    runs, base = collect()
    sp = json.loads(SPLIT.read_text())
    skipped = json.loads((OUT / "ablate_skipped.json").read_text()) if (OUT / "ablate_skipped.json").exists() else []
    agg = dict(split=sp["stats"], skipped=skipped, runs={})
    md = []
    st = sp["stats"]
    md.append("### Data and split (auto)\n")
    md.append(f"Fixed split (`results/seg3d/split.json`, seed {sp['seed']}, stratified by batch, by scan dir, used for every run): "
              f"**train {st['train']['scans']} scans / {st['train']['items_gt_json']} items**, "
              f"**val {st['val']['scans']} scans / {st['val']['items_gt_json']} items** "
              f"({st['val']['items_in_cloud']} items present in the val clouds).  Per batch (train / val): "
              + ", ".join(f"{b} {st['train']['by_batch'][b]}/{st['val']['by_batch'][b]}" for b in st["train"]["by_batch"])
              + f".  `ok` scans: train {st['train']['ok_scans']}, val {st['val']['ok_scans']} (val is restricted to `ok` scans of hard_v9/v10 + all of hard/hard_v3).\n")
    for reg, title in (("r1", "R1 - stored lift as-is (old files: GT-derived lift with the default mild corruption)"),
                       ("r2", "R2 - hard lift (merge 0.6, blob 0.5 @15-40 mm, drop 0.15, morph 2-4 voxels; fixed seed per scan)")):
        md.append(f"\n### {title}\n")
        md.append("Mean (min-max over seeds) on the val split; `corr` = fraction of lift-wrong points fixed, `broke` = fraction of lift-right "
                  "points broken (food domain); `SE` = standard error of the per-scan instance mIoU (scan-to-scan noise); "
                  "peak GiB / s per iter / wall minutes are for the training run.\n")
        md.append("| config | train | decode | seeds | sem mIoU | inst mIoU | SE | d vs lift (+-SE) | PQ | bIoU | count | corr | broke | iters | GiB | s/it | wall min |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        if base:
            for key, nm in (("lift_baseline", "lift_label (no training)"), ("lift_baseline_anchored", "lift_label, trivially anchored")):
                r = base[reg][key]
                md.append(f"| {nm} | - | - | - | {r['sem_miou']:.3f} | {r['inst_miou']:.3f} | {scan_se([r]):.3f} | - | {r['pq']:.3f} | {r['boundary_iou']:.3f} | "
                          f"{r['exact_count']:.2f} | 0 | 0 | - | - | - | - |")
                agg["runs"].setdefault("lift_baseline", {})[f"{reg}/{key}"] = r
        for name, lst in runs.items():
            for dec in DECODERS:
                if dec == "anchored_nob" and name not in ("full_r2", "full_r1"):
                    continue
                key = "model" if dec == "plain" else "model_" + dec
                rs = [x[1][reg][key] for x in lst]
                tr = [x[1]["train"] for x in lst]
                if dec == "plain" or dec.startswith("anchored"):
                    md.append(f"| {name} | {lst[0][2]['train_regime']} | {dec} | {len(lst)} | " + " | ".join(
                        [_ms([r[k] for r in rs]) if k != "inst_miou" else _ms([r[k] for r in rs]) for k in ["sem_miou", "inst_miou"]])
                        + f" | {scan_se(rs):.3f} | {paired(rs, base[reg]['lift_baseline'])} | " + " | ".join(_ms([r[k] for r in rs]) for k in ["pq", "boundary_iou", "exact_count", "corrected_food", "broken_food"])
                        + f" | {np.mean([t['iters'] for t in tr]):.0f} | {max(t['peak_train_gib'] for t in tr):.2f} | {np.mean([t['s_per_iter'] for t in tr]):.2f} | {np.mean([t['wall_s'] for t in tr]) / 60:.1f} |")
                agg["runs"].setdefault(name, {})[f"{reg}/{dec}"] = dict(seeds=[x[0] for x in lst], per_seed=rs)
        md.append("")
    for name, lst in runs.items():
        agg["runs"].setdefault(name, {})["train"] = [dict(seed=x[0], **x[1]["train"]) for x in lst]
    if skipped:
        md.append("\nSkipped by the driver: " + "; ".join(f"{n} seed {s} ({why})" for n, s, why in skipped) + "\n")
    (OUT / "ablations.json").write_text(json.dumps(agg, indent=1))
    auto = "<!-- AUTO:BEGIN -->\n" + "\n".join(md) + "\n<!-- AUTO:END -->"
    p = OUT / "ablations.md"
    old = p.read_text() if p.exists() else ""
    if "<!-- AUTO:BEGIN -->" in old:
        a, b = old.index("<!-- AUTO:BEGIN -->"), old.index("<!-- AUTO:END -->") + len("<!-- AUTO:END -->")
        p.write_text(old[:a] + auto + old[b:])
    else:
        p.write_text("# seg3d ablations\n\n" + auto + "\n")
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", help="comma list name:seed to run just those")
    ap.add_argument("--budget-min", type=float, default=100)
    ap.add_argument("--final")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.split:
        print(json.dumps(make_split(), indent=1))
    if a.baseline:
        run_baseline()
    if a.final:
        final_eval(a.final)
    if a.only:
        byname = {n: (n, e, tr, ev) for n, e, tr, ev in GRID}
        for t in a.only.split(","):
            n, s = t.split(":")
            run_job(*byname[n], int(s))
    if a.run:
        run_grid(a.budget_min)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()

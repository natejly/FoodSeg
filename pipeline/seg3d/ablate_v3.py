"""Round 3 (hard_v12, item-level filtering, split_v3).

  python -m seg3d.ablate_v3 --split | --baseline | --run [--budget-min N] | --xeval   (run from .../FoodSeg/pipeline)
split_v3: every scan with >= 1 valid item (item_filter=manifest) of hard, hard_v3, hard_v9..v12; val = 35 % of usable scans stratified by
batch, greedy to minimise food-category overlap with train.  Runs: results/seg3d/v3_<name>_s<seed>/ ; cross-evals of round-2
checkpoints on the new val set: results/seg3d/v3x_<run>/final.json.  Report: python -m seg3d.report_v3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ablate as A
from .data import item_status

REPO, OUT = A.REPO, A.OUT
ROOTS = A.ROOTS + ["data/synth/hard_v11", "data/synth/hard_v12"]
BATCHES = [Path(r).name for r in ROOTS]
SPLIT = OUT / "split_v3.json"
SPLIT_REL = "results/seg3d/split_v3.json"
VAL_FRAC = 0.35
FILTER = "manifest"


def scan_table(mode=FILTER):
    sc = {}
    for r in ROOTS:
        for p in sorted((REPO / r / "scans").glob("*/gt/seg_points.npz")):
            sd = p.parent.parent
            z = np.load(p)
            lab = z["label"]
            if len(lab) == 0:
                continue                                     # empty fused cloud
            valid, ign = item_status(sd, lab, mode)
            gt = json.loads((sd / "gt.json").read_text())
            cat = {it["id"]: it["category"].strip().lower() for it in gt["items"]}
            sc[str(sd.relative_to(REPO))] = dict(batch=Path(r).name, n_items_gt_json=len(gt["items"]), n_valid=len(valid),
                                                 n_ignored=len(ign), has_lift="lift_label" in z.files,
                                                 cats=sorted({cat[i] for i in valid if i in cat}))
    return sc


def make_split(seed=0):
    sc = scan_table()
    usable = sorted(n for n, v in sc.items() if v["n_valid"] >= 1)
    quota = {}
    for b in BATCHES:
        nb = [n for n in usable if sc[n]["batch"] == b]
        quota[b] = int(round(VAL_FRAC * len(nb)))
    rng = np.random.default_rng(seed)
    allc = lambda names: {c for n in names for c in sc[n]["cats"]}               # noqa: E731
    val = []
    while any(quota.values()):
        pool = [n for n in usable if n not in val and quota[sc[n]["batch"]] > 0]
        best, bs = None, None
        for n in rng.permutation(pool).tolist():
            v2 = val + [n]
            leak = len(allc(v2) & allc([m for m in usable if m not in v2]))
            if bs is None or leak < bs:
                best, bs = n, leak
        val.append(best)
        quota[sc[best]["batch"]] -= 1
    val = sorted(val)
    train = [n for n in usable if n not in val]
    stats = {"unusable_scans": sorted(n for n in sc if n not in usable)}
    for nm, lst in (("train", train), ("val", val)):
        stats[nm] = dict(scans=len(lst), valid_items=sum(sc[n]["n_valid"] for n in lst), ignored_items=sum(sc[n]["n_ignored"] for n in lst),
                         categories=len(allc(lst)),
                         by_batch={b: dict(scans=sum(sc[n]["batch"] == b for n in lst), valid_items=sum(sc[n]["n_valid"] for n in lst if sc[n]["batch"] == b))
                                   for b in BATCHES})
    stats["leaked_categories"] = sorted(allc(val) & allc(train))
    stats["usable_by_batch"] = {b: sum(sc[n]["batch"] == b for n in usable) for b in BATCHES}
    SPLIT.write_text(json.dumps(dict(seed=seed, filter=FILTER, train=train, val=val, stats=stats, scans=sc), indent=1))
    return stats


def sets(extra, seed, tr, patch=512):
    s = [f"train_regime={tr}", f"eval_regime={tr}", f"seed={seed}", "data.roots=" + json.dumps(ROOTS),
         f"data.split_file={SPLIT_REL}", f"data.item_filter={FILTER}"]
    sched = dict(A.SCHED, patch_size=patch)
    return s + [f"{k}={v}" for k, v in sched.items()] + list(extra)


def gpu_used():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    u, t = map(int, o.strip().split(","))
    return u, t


def run_job(name, extra, tr, seed):
    run = f"v3_{name}_s{seed}"
    d = OUT / run
    if (d / "final.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "-m", "seg3d.train", "--run", run, "--set"] + sets(extra, seed, tr)
    t0 = time.time()
    with open(d / "train.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=REPO / "pipeline", env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=60 * 16).returncode
    (d / "run_info.json").write_text(json.dumps(dict(name=name, seed=seed, rc=rc, wall_s=time.time() - t0, train_regime=tr, eval_regime=tr, sets=extra)))
    print(f"{run}: rc={rc} wall {time.time() - t0:.0f}s", flush=True)
    if rc == 0:
        subprocess.run([sys.executable, "-m", "seg3d.ablate", "--final", run], cwd=REPO / "pipeline", env=env)
        (d / "best.pt").unlink(missing_ok=True)


def baseline():
    from . import eval as E
    from .train import Config
    cfg = Config()
    cfg.data.roots, cfg.data.split_file, cfg.data.item_filter = ROOTS, SPLIT_REL, FILTER
    res = {reg: E.evaluate(None, E.make_sets(cfg, "val", reg), cfg, baseline=True, volumes=False) for reg in ("r1", "r2")}
    (OUT / "v3_baseline.json").write_text(json.dumps(res, indent=1))
    for reg in res:
        b = res[reg]["lift_baseline"]
        print(reg, {k: round(b[k], 3) for k in ("sem_miou", "inst_miou", "pq", "exact_count", "n_items", "n_scans")})


def xeval(runs):
    """Evaluate round-2 checkpoints (last.pt) on the split_v3 val set without retraining."""
    import torch
    from . import eval as E
    from .train import Config, build_model
    for run in runs:
        d = OUT / f"v3x_{run}"
        if (d / "final.json").exists():
            continue
        d.mkdir(exist_ok=True)
        ck = torch.load(OUT / run / "last.pt", map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ck["config"])
        model = build_model(cfg, pretrained=False).cuda()
        model.load_state_dict(ck["model"])
        cfg.data.roots, cfg.data.split_file, cfg.data.item_filter = ROOTS, SPLIT_REL, FILTER
        res = {reg: E.evaluate(model, E.make_sets(cfg, "val", reg), cfg, baseline=False,
                               decoders=tuple(A.DECODERS) + (("koc", "koc2") if cfg.decoder == "koc" else ()), volumes=False) for reg in ("r1", "r2")}
        info = json.loads((OUT / run / "run_info.json").read_text())
        (d / "run_info.json").write_text(json.dumps(dict(info, name=info["name"], src=run)))
        (d / "final.json").write_text(json.dumps(res, indent=1))
        print("xeval done", run, flush=True)
        del model
        torch.cuda.empty_cache()


JOBS = [("full_r2", [], "r2", 0), ("koc_r2", ["decoder=koc"], "r2", 0), ("nolift_r2", A.NOLIFT, "r2", 0),
        ("full_r2", [], "r2", 1), ("koc_r2", ["decoder=koc"], "r2", 1), ("nolift_r2", A.NOLIFT, "r2", 1)]


def run(budget):
    t0, skipped = time.time(), []
    for name, extra, tr, seed in JOBS:
        el = (time.time() - t0) / 60
        used, tot = gpu_used()
        if el > budget - 9:
            skipped.append((name, seed, f"budget {el:.0f}/{budget} min")); continue
        if used > 4096:
            skipped.append((name, seed, f"GPU {used} MiB used by others")); continue
        run_job(name, extra, tr, seed)
        (OUT / "v3_skipped.json").write_text(json.dumps(skipped))
    print("skipped", skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true"); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--xeval", action="store_true")
    ap.add_argument("--budget-min", type=float, default=40)
    a = ap.parse_args()
    if a.split:
        print(json.dumps(make_split(), indent=1))
    if a.baseline:
        baseline()
    if a.run:
        run(a.budget_min)
    if a.xeval:
        xeval(["v2_full_r2_s0", "v2_full_r2_s1", "v2_koc_r2_s0", "v2_nolift_r2_s0", "v2_nolift_r2_s1"])


if __name__ == "__main__":
    main()

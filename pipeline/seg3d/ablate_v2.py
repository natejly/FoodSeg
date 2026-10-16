"""Round 2 of the seg3d ablations: hard_v11 added, split_v2 (category-disjoint where feasible, val = ok scans only).

  python -m seg3d.ablate_v2 --split | --baseline | --run [--budget-min N] | --report
Reuses seg3d.ablate (final_eval via `python -m seg3d.ablate --final`, which reads the split from the run's config).
Run from /home/kyle/Desktop/FoodSeg/pipeline.  Outputs results/seg3d/v2_<name>_s<seed>/ and results/seg3d/round2.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import ablate as A

REPO, OUT = A.REPO, A.OUT
ROOTS = A.ROOTS + ["data/synth/hard_v11"]
SPLIT = OUT / "split_v2.json"
SPLIT_REL = "results/seg3d/split_v2.json"
VAL_FRAC = 0.4


def scan_table():
    scans = {}
    for r in ROOTS:
        mf = REPO / r / "manifest.jsonl"
        man = {j["scan"]: j for j in map(json.loads, mf.read_text().splitlines())} if mf.exists() else {}
        for p in sorted((REPO / r / "scans").glob("*/gt/seg_points.npz")):
            sd = p.parent.parent
            b = os.path.basename(r)
            ok = True if b in ("hard", "hard_v3") else man.get(sd.name, {}).get("ok", False)
            z = np.load(p)
            lab = z["label"]
            if len(lab) == 0:
                continue                                  # empty fused cloud (hard_v11 s000001): unloadable
            vis = set(np.unique(lab[(lab > 0) & (lab < 252)]).tolist())
            gt = json.loads((sd / "gt.json").read_text())
            cats = sorted({it["category"].strip().lower() for it in gt["items"] if it["id"] in vis})
            scans[str(sd.relative_to(REPO))] = dict(batch=b, ok=bool(ok), n_items=len(gt["items"]), n_items_in_cloud=len(vis),
                                                    has_lift="lift_label" in z.files, cats=cats)
    return scans


def make_split(seed=0):
    sc = scan_table()
    ok = sorted(n for n, v in sc.items() if v["ok"])
    n_val = int(round(VAL_FRAC * len(ok)))
    rng = np.random.default_rng(seed)
    allc = lambda names: {c for n in names for c in sc[n]["cats"]}
    val, rest = [], list(ok)
    while len(val) < n_val:
        best, bs = None, None
        for n in rng.permutation(rest).tolist():
            v2 = val + [n]
            train = [m for m in sc if m not in v2]
            leak = len(allc(v2) & allc(train))
            score = (leak, -len(sc[n]["cats"]) * 0)   # fewest leaked categories; ties by random order
            if bs is None or score < bs:
                best, bs = n, score
        val.append(best); rest.remove(best)
    val = sorted(val)
    train = [n for n in sc if n not in val]
    leaked = sorted(allc(val) & allc(train))
    stats = {}
    for nm, lst in (("train", train), ("val", val)):
        stats[nm] = dict(scans=len(lst), items_gt_json=sum(sc[n]["n_items"] for n in lst),
                         items_in_cloud=sum(sc[n]["n_items_in_cloud"] for n in lst),
                         by_batch={os.path.basename(r): sum(sc[n]["batch"] == os.path.basename(r) for n in lst) for r in ROOTS},
                         ok_scans=sum(sc[n]["ok"] for n in lst), categories=len(allc(lst)))
    stats["leaked_categories"] = leaked
    stats["n_ok_total"] = len(ok)
    SPLIT.write_text(json.dumps(dict(seed=seed, train=train, val=val, stats=stats, scans=sc), indent=1))
    return stats


def base_sets(extra, seed, tr, ev, patch):
    s = [f"train_regime={tr}", f"eval_regime={ev}", f"seed={seed}", "data.roots=" + json.dumps(ROOTS),
         f"data.split_file={SPLIT_REL}"]
    sched = dict(A.SCHED, patch_size=patch)
    return s + [f"{k}={v}" for k, v in sched.items()] + list(extra)


def run_job(name, extra, tr, seed, patch):
    run = f"v2_{name}_s{seed}"
    d = OUT / run
    if (d / "final.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "-m", "seg3d.train", "--run", run, "--set"] + base_sets(extra, seed, tr, tr, patch)
    t0 = time.time()
    with open(d / "train.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=REPO / "pipeline", env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=60 * 14).returncode
    (d / "run_info.json").write_text(json.dumps(dict(name=name, seed=seed, rc=rc, wall_s=time.time() - t0, train_regime=tr, eval_regime=tr, sets=extra)))
    print(f"{run}: rc={rc} wall {time.time() - t0:.0f}s", flush=True)
    if rc == 0:
        subprocess.run([sys.executable, "-m", "seg3d.ablate", "--final", run], cwd=REPO / "pipeline", env=env)
        (d / "best.pt").unlink(missing_ok=True)


def baseline():
    from . import eval as E
    from .train import Config
    cfg = Config()
    cfg.data.roots, cfg.data.split_file = ROOTS, SPLIT_REL
    res = {reg: E.evaluate(None, E.make_sets(cfg, "val", reg), cfg, baseline=True, volumes=False) for reg in ("r1", "r2")}
    (OUT / "v2_baseline.json").write_text(json.dumps(res, indent=1))
    for reg in res:
        b = res[reg]["lift_baseline"]
        print(reg, {k: round(b[k], 3) for k in ("sem_miou", "inst_miou", "pq", "exact_count")})


def gpu_used():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    u, t = map(int, o.strip().split(","))
    return u, t


JOBS = [("full_r2", [], "r2", 0), ("nolift_r2", A.NOLIFT, "r2", 0), ("full_r2", [], "r2", 1), ("nolift_r2", A.NOLIFT, "r2", 1),
        ("koc_r2", ["decoder=koc"], "r2", 0),
        ("unfreeze_top_r2", ["freeze_epochs=2", "unfreeze=top"], "r2", 0),
        ("full_r1", [], "r1", 0), ("nolift_r1", A.NOLIFT, "r1", 0), ("full_r1", [], "r1", 1), ("nolift_r1", A.NOLIFT, "r1", 1)]


def run(budget):
    t0 = time.time()
    skipped = []
    for name, extra, tr, seed in JOBS:
        used, tot = gpu_used()
        patch = 512 if tot - used < 3072 else 1024
        if name.startswith("unfreeze") and used >= 1536:
            skipped.append((name, seed, f"GPU {used} MiB used by others (>=1.5 GB)")); continue
        el = (time.time() - t0) / 60
        if el > budget - 7:
            skipped.append((name, seed, f"budget {el:.0f}/{budget} min")); continue
        run_job(name, extra, tr, seed, patch)
        (OUT / "v2_skipped.json").write_text(json.dumps(skipped))
    print("skipped", skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true"); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--budget-min", type=float, default=70)
    a = ap.parse_args()
    if a.split:
        print(json.dumps(make_split(), indent=1))
    if a.baseline:
        baseline()
    if a.run:
        run(a.budget_min)


if __name__ == "__main__":
    main()

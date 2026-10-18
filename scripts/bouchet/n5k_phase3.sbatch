#!/bin/bash
#SBATCH --job-name=n5k_phase3
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu_b200,gpu_h100,gpu_rtx6000,gpu_h200
#SBATCH --gpus=1
#SBATCH --exclude=a1127u05n01
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%j.out
#
# Nutrition5k Phase 3: the single, pre-registered test evaluation (docs/n5k_preregistration.md).
# PRIMARY: equal-weight mean of all 15 full-data Phase 2 models (A1 cnx_log + A2 cnl_raw + B1 cnx80_log, seeds 0-4)
# on the 507 depth_test_ids dishes, 4-flip TTA (eval.n5k_predict.predict_run, unchanged). SECONDARY (labelled): each
# model alone and the A1/A2/B1 group ensembles. One job, one pass over test: every model is run once, all numbers derived from those predictions.
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
module load miniconda CUDA/12.8.0
source activate $PROJ/envs/thesis
cd $PROJ/pipeline
export HF_HOME=$PROJ/checkpoints/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=$PROJ/results/n5k/phase3
mkdir -p "$OUT"
[ -e "$OUT/metrics.json" ] && { echo "ABORT: $OUT/metrics.json exists; test already evaluated"; exit 4; }

LOCAL=/tmp/n5k_cache_$SLURM_JOB_ID
cp -r $PROJ/data/cache/n5k_448_sam3b "$LOCAL"; trap 'rm -rf "$LOCAL"' EXIT
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
if [ "$used" -gt 2000 ]; then echo "ABORT: assigned GPU already has ${used} MiB in use on $(hostname) (foreign process?)"; exit 3; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PROJ=$PROJ LOCAL=$LOCAL OUT=$OUT python - <<'PY'
import csv, json, os
from pathlib import Path
import numpy as np
from eval.n5k_predict import predict_run, run_ids, pct, fmt
from eval.n5k_train import TARGETS, Data
from eval.nutrition5k import read_metadata

proj = Path(os.environ["PROJ"]); out = Path(os.environ["OUT"])
root = proj / "datasets/Nutrition5k"
groups = {"A1_cnx_log": "cnx_log_full", "A2_cnl_raw": "cnl_raw_full", "B1_cnx80_log": "cnx80_log_full"}
runs = {g: [proj / f"results/n5k/phase2/{p}_s{s}" for s in range(5)] for g, p in groups.items()}
for r in sum(runs.values(), []):
    assert (r / "best.pt").exists() and (r / "summary.json").exists(), r
    a = json.load(open(r / "summary.json"))["args"]
    assert a["full"] == "True" and a["channels"] == "rgbd_thick", r

meta = read_metadata(root)
data = Data(Path(os.environ["LOCAL"]), "rgbd_thick")
a0 = json.load(open(runs["A1_cnx_log"][0] / "summary.json"))["args"]
ids = run_ids(a0, root, data, meta, "test")
assert len(ids) == 507, len(ids)
true = np.array([[meta[d][t] for t in TARGETS] for d in ids])

preds = {}
for g, rs in runs.items():
    for r in rs:
        preds[r.name] = predict_run(r, root, data, meta, ids, 16)
        print(r.name, fmt(pct(preds[r.name], true)), flush=True)

def write(path, p):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dish_id"] + [f"true_{t}" for t in TARGETS] + [f"pred_{t}" for t in TARGETS])
        for d, t, q in zip(ids, true, p):
            w.writerow([d] + list(t) + [f"{x:.2f}" for x in q])

sets = {"PRIMARY_ensemble_15": list(preds)}
for g, rs in runs.items():
    sets[f"SECONDARY_group_{g}"] = [r.name for r in rs]
res = {}
for name, members in sets.items():
    p = np.mean([preds[m] for m in members], 0)
    m = pct(p, true)
    res[name] = dict(n_models=len(members), n_dishes=len(ids), mae_pct=dict(zip(TARGETS, m)), mass_kcal=float(np.mean(m[:2])), mean5=float(np.mean(m)),
                     mae_abs=dict(zip(TARGETS, np.abs(p - true).mean(0).tolist())), test_mean=dict(zip(TARGETS, true.mean(0).tolist())))
    print(f"{name:<28}{fmt(m)}", flush=True)
    write(out / f"predictions_{name}.csv", p)
for n, p in preds.items():
    m = pct(p, true)
    res[f"SECONDARY_single_{n}"] = dict(n_models=1, mae_pct=dict(zip(TARGETS, m)), mass_kcal=float(np.mean(m[:2])), mean5=float(np.mean(m)))
sing = [res[f"SECONDARY_single_{n}"]["mean5"] for n in preds]
print(f"SECONDARY single-model mean5: mean {np.mean(sing):.2f} sd {np.std(sing, ddof=1):.2f}")
json.dump(res, open(out / "metrics.json", "w"), indent=1)
PY

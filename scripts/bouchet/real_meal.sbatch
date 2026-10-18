#!/bin/bash
#SBATCH --job-name=real_meal
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu_devel,gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%j.out
#
# FoodScan run_meal --segment (SAM 3) on real iPad scans in $SCR/scans (same copies real_n5k used).
# Output per scan: $SCR/meal/<scan>/{result.json,run.log,masks/,lift.npz}; visuals are made locally
# (pipeline/eval/real_meal_viz.py).
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
SCR=/nfs/roberts/scratch/cpsc4900/cpsc4900_njl36/real_alt
module load miniconda CUDA/12.8.0
source activate $PROJ/envs/thesis
cd $PROJ/pipeline
export HF_HOME=$PROJ/checkpoints/hf_cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SCANS=("$@"); [ ${#SCANS[@]} -gt 0 ] || SCANS=(20261017_213105_scan 20261017_212836_scan)
for s in "${SCANS[@]}"; do
  OUT=$SCR/meal/$s; mkdir -p $OUT
  python -u -m foodscan.run_meal $SCR/scans/$s --segment --names "kiwi,sweet potato" --no-nutrition \
      --out $OUT/result.json 2>&1 | tee $OUT/run.log
  rm -rf $OUT/masks; rsync -a --exclude raw $SCR/scans/$s/masks $SCR/scans/$s/lift.npz $OUT/   # raw/ (replay maps) is large; skip
done

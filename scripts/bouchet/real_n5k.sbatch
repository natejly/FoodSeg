#!/bin/bash
#SBATCH --job-name=real_n5k
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu_devel,gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%j.out
#
# Nutrition5k Phase-2 ensemble (15 models) on real iPad scans: eval.real_n5k predict. Scans are rsynced to
# $SCR/real_alt/scans; outputs in $SCR/real_alt/out/<scan>[_camh359].
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
SCR=/nfs/roberts/scratch/cpsc4900/cpsc4900_njl36/real_alt
module load miniconda CUDA/12.8.0
source activate $PROJ/envs/thesis
cd $PROJ/pipeline
export HF_HOME=$PROJ/checkpoints/hf_cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for s in 20261017_213105_scan 20261017_212836_scan; do
  python -u -m eval.real_n5k predict $SCR/scans/$s $SCR/out/$s --proj $PROJ --cam-h 0.319
  python -u -m eval.real_n5k predict $SCR/scans/$s $SCR/out/${s}_camh359 --proj $PROJ --cam-h 0.359   # sensitivity
done

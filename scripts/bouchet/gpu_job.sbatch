#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%j.out
#
# Generic single-GPU job. Pass the command after the script:
#   sbatch scripts/bouchet/gpu_job.sbatch python -m foodscan.segment data/scans/m001_orbit --names "rice,chicken"
# Pick a GPU type with:  sbatch --gpus=l40s:1 ...   (a40 / l40s / rtx_5000_ada on 'gpu';
# h100 on --partition=gpu_h100). SAM 3D Objects needs >= 32 GB, so a40 or l40s.
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
module load miniconda CUDA/12.8.0
source activate $PROJ/envs/thesis
cd $PROJ/pipeline
export HF_HOME=$PROJ/checkpoints/hf_cache
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "running: $*"
exec "$@"

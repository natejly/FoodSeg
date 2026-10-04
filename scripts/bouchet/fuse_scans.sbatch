#!/bin/bash
#SBATCH --job-name=fuse
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-0%8
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%A_%a.out
#
# TSDF-fuse every scan under $SCANS as a job array (one scan per task).
#   N=$(ls -d data/scans/*/ | wc -l); sbatch --array=0-$((N-1))%8 scripts/bouchet/fuse_scans.sbatch
# Env: VOXEL (default 0.0015), STRIDE (default 2), SCANS (default project/data/scans)
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
SCANS=${SCANS:-$PROJ/data/scans}
module load miniconda
source activate $PROJ/envs/thesis
cd $PROJ/pipeline

mapfile -t DIRS < <(ls -d "$SCANS"/*/ | sort)
SCAN=${DIRS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> $SCAN"
python -m foodscan.fuse "$SCAN" --voxel "${VOXEL:-0.0015}" --stride "${STRIDE:-2}"

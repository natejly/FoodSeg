#!/bin/bash
#SBATCH --job-name=synth
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-49
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%A_%a.out
#
# Synthetic meal scans with BlenderProc, one array task = a block of scenes.
#   sbatch scripts/bouchet/synth_render.sbatch v0 100     # 50 tasks x 100 scenes = 5k scenes
#   sbatch --gpus=l40s:1 --array=0-9 scripts/bouchet/synth_render.sbatch v0 100
# Reruns skip finished scenes. Needs the synth env and Blender once:
#   conda create -y -p $PROJ/envs/synth python=3.11 && $PROJ/envs/synth/bin/pip install -r pipeline/synth/requirements.txt
#   Blender 4.2.1 from a mirror into $PROJ/blender/ (download.blender.org may 403):
#   curl -L https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz | tar xJ -C $PROJ/blender
#   python -m synth.assets --out $PROJ/data/synth/assets.jsonl --metafood3d $PROJ/datasets/MetaFood3D ...
set -euo pipefail
PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
VERSION=${1:-v0}
PER_TASK=${2:-100}
module load miniconda CUDA/12.8.0
source activate $PROJ/envs/synth
cd $PROJ/pipeline
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# BlenderProc writes temp files per process; keep them node-local
export TMPDIR=${TMPDIR:-/tmp}/synth_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"
START=$((SLURM_ARRAY_TASK_ID * PER_TASK))
python -m synth.batch --assets $PROJ/data/synth/assets.jsonl --root $PROJ/data/synth/$VERSION \
  --n "$PER_TASK" --start "$START" --spp 64 --blender $PROJ/blender/blender-4.2.1-linux-x64 \
  ${HDRI_DIR:+--hdri-dir $HDRI_DIR}

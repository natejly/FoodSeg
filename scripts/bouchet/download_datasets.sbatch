#!/bin/bash
#SBATCH --job-name=dl_datasets
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=23:00:00
#SBATCH --output=/nfs/roberts/project/cpsc4900/cpsc4900_njl36/logs/%x-%j.out
#
# Downloads the freely available datasets into project storage.
# Gated ones (MetaFood3D, DishSeg24k, BenchSeg) need a human on a form; see docs/model_survey.md.
#
#   sbatch scripts/bouchet/download_datasets.sbatch            # everything below
#   sbatch scripts/bouchet/download_datasets.sbatch n5k_full   # also the 181 GB side-angle videos
set -euo pipefail

PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
DS=$PROJ/datasets
module load miniconda
source activate $PROJ/envs/thesis
mkdir -p "$DS"
cd "$DS"

echo "== FoodSeg103 (HF, Apache-2.0) =="
hf download EduardoPacheco/FoodSeg103 --repo-type dataset --local-dir "$DS/FoodSeg103"

echo "== Nutrition5k metadata + splits + overhead RGB-D (CC BY 4.0) =="
# Public GCS bucket. Overhead imagery is ~20 GB; side-angle videos are the other ~160 GB.
N5K=gs://nutrition5k_dataset/nutrition5k_dataset
mkdir -p "$DS/Nutrition5k"
gsutil -m cp -r "$N5K/metadata" "$N5K/dish_ids" "$DS/Nutrition5k/"
gsutil -m cp -r "$N5K/imagery/realsense_overhead" "$DS/Nutrition5k/imagery/"
if [ "${1:-}" = "n5k_full" ]; then
  gsutil -m cp -r "$N5K/imagery/side_angles" "$DS/Nutrition5k/imagery/"
fi

echo "== MobileBrick (iPhone LiDAR small-object benchmark) =="
# Download links are on the repo README; they change, so fetch the README and print them.
mkdir -p "$DS/MobileBrick"
curl -sL https://raw.githubusercontent.com/ActiveVisionLab/MobileBrick/main/README.md \
  -o "$DS/MobileBrick/README.md"
grep -Eo 'https?://[^ )]+' "$DS/MobileBrick/README.md" | grep -iE 'zip|drive|download|dataset' || true
echo "MobileBrick: follow the link(s) above manually (not automated)."

echo "== ungated model weights =="
hf download depth-anything/prompt-depth-anything-vitl --local-dir "$PROJ/checkpoints/promptda-vitl"
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir "$PROJ/checkpoints/Qwen3-VL-8B-Instruct"

echo "== gated weights (only work after approval + 'hf auth login') =="
hf download facebook/sam3 --local-dir "$PROJ/checkpoints/sam3" || echo "sam3: not approved yet"
hf download facebook/sam3.1 --local-dir "$PROJ/checkpoints/sam3.1" || echo "sam3.1: not approved yet"
if hf download --repo-type model --local-dir "$PROJ/checkpoints/sam-3d-objects-dl" --max-workers 1 facebook/sam-3d-objects; then
  mkdir -p "$PROJ/checkpoints/sam-3d-objects"
  mv "$PROJ/checkpoints/sam-3d-objects-dl/checkpoints" "$PROJ/checkpoints/sam-3d-objects/hf" 2>/dev/null || true
  rm -rf "$PROJ/checkpoints/sam-3d-objects-dl"
else
  echo "sam-3d-objects: not approved yet"
fi

echo "== done =="
du -sh "$DS"/* "$PROJ/checkpoints"/* 2>/dev/null

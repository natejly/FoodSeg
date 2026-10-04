#!/bin/bash
# One-time: create the thesis conda env on Bouchet (run on the login node).
#   bash scripts/bouchet/setup_env.sh
set -euo pipefail

PROJ=/nfs/roberts/project/cpsc4900/cpsc4900_njl36
ENV=$PROJ/envs/thesis

module load miniconda
if [ ! -d "$ENV" ]; then
  conda create -y -p "$ENV" python=3.12
fi
source activate "$ENV"
pip install --upgrade pip
# download tooling
pip install "huggingface_hub[cli]" gsutil
# pipeline deps (CPU open3d; swap for the CUDA wheel later if needed)
pip install open3d numpy opencv-python-headless pillow tqdm pycolmap

mkdir -p "$PROJ"/{datasets,data,checkpoints,runs,logs}
echo "env: $ENV"
echo "activate with:  module load miniconda && source activate $ENV"
echo "then:           hf auth login   # paste a Hugging Face token to fetch gated weights"

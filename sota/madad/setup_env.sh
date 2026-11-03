#!/bin/bash
# Environment for MAD-AD (github.com/farzad-bz/MAD-AD @ df05cd0): the training/inference subset of its
# requirements.txt (torch 2.3.1, numpy 1.26.4, scipy 1.14.1, einops 0.8.0, albumentations 1.4.21, transformers 4.46.1
# for get_cosine_schedule_with_warmup). Not installed: anomalib, taming-transformers, kornia, diffusers, timm
# (evaluation metrics / VAE training only). Python 3.10 (the repo used 3.11; 3.10 is what uv has locally).
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_madad
$UV venv --python 3.10 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "numpy==1.26.4" "scipy==1.14.1" "einops==0.8.0" \
  "albumentations==1.4.21" "transformers==4.46.1" "pandas" "pyyaml" "pillow" "tqdm"
$V/bin/python -I -c "import torch, albumentations, transformers, scipy; print(torch.__version__, albumentations.__version__, transformers.__version__, scipy.__version__, torch.cuda.is_available())"

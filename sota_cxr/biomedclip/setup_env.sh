#!/bin/bash
# Environment for BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): the model card pins
# open_clip_torch 2.23.0 and transformers 4.35.2; torch 2.8 cu128.
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_cxr_biomedclip
$UV venv --python 3.12 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "open_clip_torch==2.23.0" "transformers==4.35.2" \
  "numpy" "pandas" "Pillow"
$V/bin/python -c "import torch, open_clip, transformers; import importlib.metadata as m; print(torch.__version__, m.version('open_clip_torch'), transformers.__version__, torch.cuda.is_available())"

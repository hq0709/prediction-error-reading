#!/bin/bash
# Environment for MedGemma 1.5 4B (unsloth/medgemma-1.5-4b-it mirror): transformers 4.57.3 (the version recorded in
# the checkpoint's generation_config.json), torch 2.8 cu128, accelerate.
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_cxr_medgemma
$UV venv --python 3.12 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  "transformers==4.57.3" "accelerate==1.10.1" "Pillow" "numpy" "pandas" "sentencepiece" "protobuf"
$V/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

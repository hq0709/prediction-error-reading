#!/bin/bash
# Environment for M3D-LaMed-Phi-3-4B (official requirements.txt, with transformers raised to 4.42.4:
# Phi-3 classes do not exist in the pinned 4.39.1; the HF checkpoint was saved with 4.42.0.dev0).
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_m3d_lamed
$UV venv --python 3.10 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.2.1+cu118" "torchvision==0.17.1+cu118" \
  "transformers==4.42.4" "accelerate==0.31.0" "sentencepiece" "protobuf" \
  "einops==0.8.0" "monai==1.3.0" "nibabel==5.2.1" "numpy==1.26.4" "scipy==1.13.0" \
  "SimpleITK==2.3.1" "safetensors==0.4.3" "Pillow==10.3.0" "tqdm==4.66.2" "pandas==2.2.2"
$V/bin/python -c "import torch, transformers, monai; print(torch.__version__, transformers.__version__, monai.__version__, torch.cuda.is_available())"

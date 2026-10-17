#!/bin/bash
# Environment for Hulu-Med-14B following the README install block (torch 2.4.0 cu118, flash-attn 2.7.3,
# transformers 4.51.2, accelerate 1.7.0, nibabel) plus the inference subset of requirements.txt.
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_hulumed14b
$UV venv --python 3.10 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.4.0+cu118" "torchvision==0.19.0+cu118" \
  "transformers==4.51.2" "accelerate==1.7.0" "tokenizers>=0.21,<0.22" "huggingface_hub>=0.30.0,<1.0" \
  "nibabel==5.3.2" "timm==1.0.3" "numpy==1.24.4" "decord==0.6.0" "imageio==2.34.0" "imageio-ffmpeg==0.4.9" \
  "opencv-python-headless==4.8.1.78" "ffmpeg-python" "einops==0.6.1" "sentencepiece" "safetensors" "Pillow" "scipy" "SimpleITK==2.3.1"
# flash-attn 2.7.3 prebuilt wheel for torch 2.4 / CUDA 11 / cp310 (the README pins 2.7.3)
$UV pip install --python $V/bin/python \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
$V/bin/python -c "import torch, transformers, flash_attn; print(torch.__version__, transformers.__version__, flash_attn.__version__, torch.cuda.is_available())"

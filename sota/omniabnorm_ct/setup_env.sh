#!/bin/bash
# Environment for OmniAbnorm-CT (zhaoziheng/OminiAbnorm-CT). Pins follow the README "packages that need particular
# attention" and requirements.txt (inference subset): torch 2.0.0, transformers 4.49.0, flash-attn 2.3.6, monai 1.3.0,
# scipy 1.12.0, peft 0.10.0, accelerate 1.2.1, qwen-vl-utils 0.0.10, numpy 1.24.4, einops 0.8.0, opencv 4.9.0.80,
# positional-encodings 6.0.3, tokenizers 0.21.1, huggingface-hub 0.30.2, safetensors 0.5.1, torchvision 0.15.1.
# torch 2.0.0 is the CUDA 11.8 build (sm_90 kernels for H100); flash-attn is the matching prebuilt wheel.
# pillow 11.2.1: the pinned 11.2.0 was never published on PyPI (11.2.1 is the first 11.2 release).
# setuptools: triton 2.0.0 (torch 2.0 dependency) imports it; uv venvs do not ship it.
# opencv-python-headless (same version) instead of opencv-python: no GUI/libGL on this server, same cv2 code.
# The customised dynamic-network-architectures shipped in the repo is installed from a copy (clone stays untouched).
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_omniabnorm
SCR=/tmp/claude-653073370/-home-local-PARTNERS-hj880/65df248e-f7ef-4bd0-a6fe-2e7b1e7c373d/scratchpad
$UV venv --allow-existing --python 3.10 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.0.0+cu118" "torchvision==0.15.1+cu118" \
  "transformers==4.49.0" "tokenizers==0.21.1" "huggingface-hub==0.30.2" "safetensors==0.5.1" \
  "accelerate==1.2.1" "peft==0.10.0" "qwen-vl-utils==0.0.10" \
  "numpy==1.24.4" "scipy==1.12.0" "einops==0.8.0" "opencv-python-headless==4.9.0.80" \
  "positional-encodings==6.0.3" "monai==1.3.0" "pillow==11.2.1" "SimpleITK==2.3.1" "nibabel==5.2.0" \
  "packaging" "ninja" "psutil" "setuptools==69.5.1"
$UV pip install --python $V/bin/python --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.6/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
rm -rf $SCR/dna_build_oa && mkdir -p $SCR/dna_build_oa
cp -r /raid/home/CAMCA/hj880/ext_src/OmniAbnorm-CT/model/dynamic-network-architectures-main $SCR/dna_build_oa/
$UV pip install --python $V/bin/python --no-deps $SCR/dna_build_oa/dynamic-network-architectures-main
$V/bin/python -c "import torch, transformers, flash_attn, peft, numpy, cv2; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, flash_attn.__version__, peft.__version__, numpy.__version__, cv2.__version__)"
$UV pip freeze --python $V/bin/python > /raid/home/CAMCA/hj880/perr/sota/omniabnorm_ct/pip_freeze.txt

#!/bin/bash
# Environment for Dinomaly (github.com/guojiajeremy/Dinomaly @ 1f252be). The repo pins torch 1.12.0+cu113, which has no
# H100 (sm_90) kernels, so torch 2.3.1+cu121 is used; the other pins that matter for the code we run: timm 0.9.12,
# scikit-learn (imported by models/uad.py), numpy/scipy/pillow. xformers is NOT installed (the vendored DINOv2 Block
# expects the plain-attention path that returns (x, attn)).
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_dinomaly
$UV venv --python 3.10 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "numpy==1.26.4" "scipy" "scikit-learn" "timm==0.9.12" \
  "pandas" "pillow" "tqdm"
$V/bin/python -I -c "import torch, timm, sklearn; print(torch.__version__, timm.__version__, sklearn.__version__, torch.cuda.is_available())"

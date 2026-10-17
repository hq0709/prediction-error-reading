#!/bin/bash
# Environment for torchxrayvision pretrained DenseNets (official weights, downloaded by the library).
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_cxr_xrv
$UV venv --python 3.12 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "torchxrayvision" "numpy" "pandas" "scikit-learn" "scikit-image" "Pillow"
$V/bin/python -c "import torch, torchxrayvision as xrv; print(torch.__version__, xrv.__version__, torch.cuda.is_available())"

#!/bin/bash
# Environment for the supervised DenseNet-121 baseline: torch 2.8 cu128 + torchvision 0.23 (ImageNet weights),
# pandas / scikit-learn for the tables and AUROC, Pillow (libjpeg-turbo) for JPEG decoding.
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121
$UV venv --python 3.12 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "numpy" "pandas" "scikit-learn" "Pillow"
$V/bin/python -c "import torch, torchvision, PIL, sklearn, pandas; print(torch.__version__, torchvision.__version__, PIL.__version__, sklearn.__version__, pandas.__version__, torch.cuda.is_available())"

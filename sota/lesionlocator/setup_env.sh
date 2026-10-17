#!/bin/bash
# Environment for LesionLocator (github.com/MIC-DKFZ/LesionLocator, commit ef073ba). README recipe:
# python 3.12 + `pip install -e .`; torch wheel chosen for the local driver (CUDA 12.8 build).
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
VENV=/raid/home/CAMCA/hj880/venvs/sota_lesionlocator
SRC=/raid/home/CAMCA/hj880/ext_src/LesionLocator
$UV venv --python 3.12 $VENV
$UV pip install --python $VENV/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
$UV pip install --python $VENV/bin/python -e $SRC
$VENV/bin/python -I -c "import torch, numpy, lesionlocator, unigradicon, icon_registration, cc3d; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
$UV pip freeze --python $VENV/bin/python > /raid/home/CAMCA/hj880/perr/sota/lesionlocator/pip_freeze.txt

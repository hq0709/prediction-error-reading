#!/bin/bash
# Environment for CheXzero (Tiu et al. 2022): official code (rajpurkarlab/CheXzero, cloned to ext_src/CheXzero) on a
# current torch (the repo pins torch 1.10; the model code is plain CLIP and runs unchanged), plus gdown for the
# official Google-Drive checkpoints.  torch >= 2.6 loads the checkpoints with weights_only=True by default.
set -e
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
V=/raid/home/CAMCA/hj880/venvs/sota_cxr_chexzero
$UV venv --python 3.12 $V
$UV pip install --python $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" "numpy" "pandas" "Pillow" "opencv-python-headless" \
  "ftfy" "regex" "tqdm" "h5py" "scikit-learn" "matplotlib" "gdown"
$V/bin/python -c "import torch, cv2, PIL, ftfy, regex; print(torch.__version__, cv2.__version__, PIL.__version__, torch.cuda.is_available())"

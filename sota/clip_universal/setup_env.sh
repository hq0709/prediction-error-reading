#!/bin/bash
# Environment for the CLIP-Driven Universal Model (Liu et al., ICCV 2023), "Direct Inference in Your OWN CT scans"
# path (pred_pseudo.py, Swin UNETR checkpoint). Pins from requirements.txt: monai 0.9.0, h5py 3.6.0,
# connected-components-3d, fastremap, tqdm (torch 1.11.0 / SimpleITK 1.2.4, see below); README: python 3.7, monai[all].
# torch 2.1.2+cu121 instead of 1.11.0+cu113: H100 needs sm_90 kernels (CUDA >= 11.8 build). Python 3.10: torch 2.x has
# no 3.7 build; 3.10 is the interpreter already used by the other baselines. SimpleITK 2.3.1 instead of 1.2.4 (no 3.10
# wheel; SimpleITK is not imported on the inference path). monai core + nibabel instead of monai[all] (the NIfTI reader
# is the only optional dependency used); utils/utils.py additionally imports pandas, matplotlib, scikit-learn, scipy.
# numpy 1.23.5: monai 0.9 predates numpy 1.24's removal of the np.bool/np.float aliases.
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
VENV=/raid/home/CAMCA/hj880/venvs/sota_clip_universal
$UV venv --python 3.10 $VENV
$UV pip install --python $VENV/bin/python torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$UV pip install --python $VENV/bin/python numpy==1.23.5 monai==0.9.0 h5py==3.6.0 connected-components-3d fastremap tqdm \
    nibabel==5.2.1 scipy==1.10.1 pandas==1.5.3 matplotlib scikit-learn==1.2.2 einops==0.6.1 SimpleITK==2.3.1
$VENV/bin/python -c "import torch, numpy, monai, h5py, cc3d, fastremap, nibabel, scipy, sklearn; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, monai.__version__, h5py.__version__, nibabel.__version__, scipy.__version__)"
$UV pip freeze --python $VENV/bin/python > /raid/home/CAMCA/hj880/perr/sota/clip_universal/pip_freeze.txt

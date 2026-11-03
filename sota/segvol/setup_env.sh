#!/bin/bash
# Environment for SegVol (Du et al., NeurIPS 2024), HF quickstart path (BAAI/SegVol, model_segvol_single.py).
# Pins from the HF model card / GitHub README "key requirements": monai 0.9.0, einops 0.6.1, transformers 4.18.0,
# matplotlib; "pytorch v1.11.0 or higher" -> torch 2.1.2+cu121 (H100 needs sm_90 kernels, i.e. a CUDA >= 11.8 build).
# Python 3.10 (the card suggests a 3.8 conda env; 3.10 is the interpreter already used by the other baselines, and all
# pinned packages ship 3.10 wheels). monai core + nibabel instead of monai[all]: the NIfTI reader is the only optional
# dependency on the inference path. numpy 1.23.5: monai 0.9 predates numpy 1.24's removal of the np.bool/np.float aliases.
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
VENV=/raid/home/CAMCA/hj880/venvs/sota_segvol
$UV venv --python 3.10 $VENV
$UV pip install --python $VENV/bin/python torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$UV pip install --python $VENV/bin/python numpy==1.23.5 monai==0.9.0 einops==0.6.1 transformers==4.18.0 matplotlib \
    nibabel==5.2.1 scipy==1.10.1 SimpleITK==2.3.1
$VENV/bin/python -c "import torch, numpy, monai, transformers, einops, nibabel, scipy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, monai.__version__, transformers.__version__, einops.__version__, nibabel.__version__, scipy.__version__)"
$UV pip freeze --python $VENV/bin/python > /raid/home/CAMCA/hj880/perr/sota/segvol/pip_freeze.txt

#!/bin/bash
# Environment for SAT-Pro (zhaoziheng/SAT). Pins follow the SAT README "key requirements";
# torch uses a CUDA 12.1 build (H100 needs sm_90, which the README-era torch 1.x wheels lack).
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
VENV=/raid/home/CAMCA/hj880/venvs/sota_sat
$UV venv --python 3.10 $VENV
export VIRTUAL_ENV=$VENV
$UV pip install --python $VENV/bin/python torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$UV pip install --python $VENV/bin/python numpy==1.21.5 monai==1.1.0 transformers==4.21.3 nibabel==4.0.2 einops==0.6.1 \
    positional_encodings==6.0.1 scipy==1.9.3 pandas==1.5.3 openpyxl SimpleITK==2.3.1
# customised dynamic-network-architectures shipped inside the SAT repo (installed from a copy, not editable,
# so nothing is written into the clone)
rm -rf /tmp/claude-653073370/-home-local-PARTNERS-hj880/65df248e-f7ef-4bd0-a6fe-2e7b1e7c373d/scratchpad/dna_build && mkdir -p /tmp/claude-653073370/-home-local-PARTNERS-hj880/65df248e-f7ef-4bd0-a6fe-2e7b1e7c373d/scratchpad/dna_build
cp -r /raid/home/CAMCA/hj880/ext_src/SAT/model/dynamic-network-architectures-main /tmp/claude-653073370/-home-local-PARTNERS-hj880/65df248e-f7ef-4bd0-a6fe-2e7b1e7c373d/scratchpad/dna_build/
$UV pip install --python $VENV/bin/python /tmp/claude-653073370/-home-local-PARTNERS-hj880/65df248e-f7ef-4bd0-a6fe-2e7b1e7c373d/scratchpad/dna_build/dynamic-network-architectures-main
$VENV/bin/python -c "import torch, numpy, monai, transformers; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, monai.__version__, transformers.__version__); print(torch.from_numpy(numpy.zeros(3)).sum())"
$UV pip freeze --python $VENV/bin/python > /raid/home/CAMCA/hj880/perr/sota/sat_pro/pip_freeze.txt

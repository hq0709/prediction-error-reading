#!/bin/bash
# Environment for VISTA3D (MONAI bundle "vista3d" 0.5.11). Versions follow configs/metadata.json of the bundle
# (monai 1.4.0, pytorch 2.4.0, numpy 1.24.4 and its required_packages_version) plus "monai[fire,ignite]" from the
# bundle README install guide.
set -euxo pipefail
export UV_CACHE_DIR=/raid/home/CAMCA/hj880/venvs/uv_cache UV_PYTHON_INSTALL_DIR=/raid/home/CAMCA/hj880/venvs/uv_pythons
UV=/raid/home/CAMCA/hj880/venvs/uvtool/bin/uv
VENV=/raid/home/CAMCA/hj880/venvs/sota_vista3d
$UV venv --python 3.10 $VENV
$UV pip install --python $VENV/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
$UV pip install --python $VENV/bin/python "monai[fire,ignite]==1.4.0" numpy==1.24.4 matplotlib==3.9.1 einops==0.7.0 \
    scikit-image==0.23.2 nibabel==5.2.1 pytorch-ignite==0.4.11 tensorboard==2.17.0 requests huggingface_hub \
    SimpleITK==2.3.1 scipy
$UV pip install --python $VENV/bin/python cucim-cu12==24.6.0 || echo "cucim install failed (optional)"
$UV pip install --python $VENV/bin/python mlflow==2.17.2 || echo "mlflow install failed (training-only)"
$VENV/bin/python -c "import torch, numpy, monai; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, monai.__version__); from monai.apps.vista3d.transforms import VistaPreTransformd; from monai.networks.nets import vista3d132; print('ok')"
$UV pip freeze --python $VENV/bin/python > /raid/home/CAMCA/hj880/perr/sota/vista3d/pip_freeze.txt

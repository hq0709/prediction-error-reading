#!/bin/bash
# Label efficiency: the DenseNet reader with a given input mode on nested subsets of the 60k training images, with
# epochs scaled so that every run takes about the same number of steps.   bash run_cls_labeleff.sh <mode> <gpu>
cd /raid/home/CAMCA/hj880/perr
mode=$1; gpu=$2
for spec in "2000 60" "6000 20" "20000 7"; do
  read n ep <<< "$spec"
  OMP_NUM_THREADS=8 TORCH_HOME=/raid/home/CAMCA/hj880/hf_home/torch CUDA_VISIBLE_DEVICES=$gpu \
    /raid/home/CAMCA/hj880/venvs/sota_cxr_densenet121/bin/python train_cxr_cls.py --mode $mode \
    --keys results/cxr_train${n}_keys.txt --epochs $ep --name dn${n}_$mode > logs/cls_dn${n}_$mode.log 2>&1
done

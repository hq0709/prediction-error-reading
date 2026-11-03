#!/usr/bin/env python
"""CLIP-Driven Universal Model (Liu et al., ICCV 2023) on the cached LUNA16 / MSD-liver / HCC-TACE-Seg arrays, through
the official "Direct Inference in Your OWN CT scans" path (pred_pseudo.py) with the released Swin UNETR checkpoint.

Per case:
  cached LPS array -> uncompressed NIfTI with the explicit affine diag(-sx, -sy, sz) (common.write_nifti)
  -> official transforms of dataset/dataloader.py:get_loader_without_gt (LoadImaged, AddChanneld, Orientationd RAS,
     Spacingd 1.5 x 1.5 x 1.5 mm bilinear, ScaleIntensityRanged [-175, 250] -> [0, 1] clipped, CropForegroundd,
     ToTensord), pred_pseudo.py defaults
  -> official Universal_model (Swin UNETR backbone, CLIP text embeddings of the 32 template classes as organ_embedding,
     dynamic heads), checkpoint loaded with pred_pseudo.py's key mapping
  -> pred_pseudo.py validation(): monai sliding_window_inference (roi 96^3, sw_batch_size 1, overlap 0.5, gaussian),
     sigmoid, threshold_organ (0.5 for every class), organ_post_process(organ_list = [target]) -- for 27 "Liver Tumor"
     the tumour mask is kept only inside the closed + dilated largest "Liver" component; 30 "Lung Tumor" is unchanged
  -> save_results() without SaveImaged: Invertd (nearest) of the post-processed target mask back to the input NIfTI grid;
     side branch: the target channel's sigmoid probability inverted with the same Invertd, used to score components
  -> inverted arrays transposed (x, y, z) -> (z, y, x) = cached index; fixed mask -> answer rule (common.mask_to_answer),
     voxel volume sz*sy*sx.
Organ list: the official script post-processes all 32 classes; every class's post-processing is independent of the
others' post-processed masks (27 reads the raw thresholded liver channel), so organ_list = [target] gives the same
target mask (the script's own commented per-dataset lists do the same: 'liver' -> [6, 27], 'lung' -> [30]).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True  # nothing is written into the downloaded code directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_cached, load_spacing, mask_to_answer, write_nifti  # noqa: E402

CODE = "/raid/home/CAMCA/hj880/ext_src/clip_universal_code"  # github ljwztc/CLIP-Driven-Universal-Model @ 43c1239
CKPT = "/raid/home/CAMCA/hj880/ext_src/clip_universal_weights/clip_driven_universal_swin_unetr.pth"  # HF @ 76f0304
CACHE = "/raid/home/CAMCA/hj880/perr/cache"
# 1-based template index (utils/utils.py ORGAN_NAME): 30 "Lung Tumor" (MSD Task06) is the only lung-lesion class (no
# nodule class); 27 "Liver Tumor" (LiTS / MSD Task03).
TARGET = {"luna": 30, "liver": 27, "liver_ext": 27}


def official_args(data_root):
    """pred_pseudo.py argparse defaults (the values that reach the inference path)."""
    return argparse.Namespace(dist=False, local_rank=None, device=None, epoch=0, log_name="Nvidia", resume=CKPT,
                              backbone="swinunetr", max_epoch=1000, store_num=10, lr=1e-4, weight_decay=1e-5,
                              data_root_path=data_root, result_save_path=None, batch_size=1, num_workers=8,
                              a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, space_x=1.5, space_y=1.5, space_z=1.5,
                              roi_x=96, roi_y=96, roi_z=96, num_samples=1, phase="test", cache_dataset=False,
                              store_result=False, cache_rate=0.6, threshold_organ="Pancreas Tumor", threshold=0.6)


def import_official():
    sys.path.insert(0, CODE)
    import dataset.dataloader as dl
    import utils.utils as uu
    from model.Universal_model import Universal_model
    return dl, uu, Universal_model


def add_sphere(vol, sp, zyx, diam=10.0, hu=300):
    vol = vol.copy()
    zz, yy, xx = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    d2 = ((zz - zyx[0]) * sp[0]) ** 2 + ((yy - zyx[1]) * sp[1]) ** 2 + ((xx - zyx[2]) * sp[2]) ** 2
    vol[d2 <= (diam / 2) ** 2] = hu
    return vol


def make_loader(ds, cases, ndir, val_transforms, num_workers, sphere=None):
    """monai Dataset/DataLoader exactly as get_loader_without_gt builds them (batch 1, no shuffle, list_data_collate),
    over our file list; the NIfTI is written from the cached array inside the worker just before loading."""
    from monai.data import DataLoader, Dataset, list_data_collate

    class CaseDataset(Dataset):
        def _transform(self, index):
            d = self.data[index]
            if not os.path.exists(d["image"]):
                vol, sp = load_cached(ds, d["name"])
                if sphere is not None:  # coordinate check only
                    vol = add_sphere(vol, sp, sphere)
                write_nifti(vol, sp, d["image"] + ".tmp.nii")
                os.replace(d["image"] + ".tmp.nii", d["image"])
            return super()._transform(index)

    data = [{"image": f"{ndir}/{c}.nii", "name": c} for c in cases]
    return DataLoader(CaseDataset(data=data, transform=val_transforms), batch_size=1, shuffle=False,
                      num_workers=num_workers, collate_fn=list_data_collate)


def build_model(Universal_model, NUM_CLASS):
    import torch

    model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr",
                            encoding="word_embedding")
    # pred_pseudo.py: load checkpoint['net'] with its key mapping (weights_only=True instead of a plain torch.load)
    store_dict = model.state_dict()
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=True)
    load_dict = checkpoint["net"]
    num_count = 0
    for key, value in load_dict.items():
        if "swinViT" in key or "encoder" in key or "decoder" in key:
            name = "backbone." + ".".join(key.split(".")[1:])
        else:
            name = ".".join(key.split(".")[1:])
        store_dict[name] = value
        num_count += 1
    model.load_state_dict(store_dict)
    print("Use pretrained weights. load", num_count, "params into", len(store_dict.keys()), flush=True)
    model.cuda()
    model.eval()
    return model


def invert_to_cache(batch, arrays, val_transforms):
    """save_results() for given single-channel arrays ({key: (1, 1, H, W, D) tensor}) without SaveImaged: the official
    Invertd (nearest) back to the input NIfTI grid, then (x, y, z) -> (z, y, x)."""
    from monai.data import decollate_batch
    from monai.transforms import Compose, Invertd

    b = {k: v for k, v in batch.items() if k not in ("image", "results")}
    b["image"] = batch["image"].cpu()
    b.update(arrays)
    post = Compose([Invertd(keys=list(arrays), transform=val_transforms, orig_keys="image", nearest_interp=True,
                            to_tensor=True)])
    out = [post(i) for i in decollate_batch(b)][0]
    return {k: np.asarray(out[k][0].cpu().numpy()).transpose(2, 1, 0) for k in arrays}


def predict(model, uu, batch, target, args):
    """pred_pseudo.py validation() for one batch; returns the post-processed target mask and its probability,
    both (1, 1, H, W, D) on the processed grid."""
    import torch
    import torch.nn.functional as F
    from monai.inferers import sliding_window_inference

    image, name = batch["image"].cuda(), batch["name"]
    with torch.no_grad():
        pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5,
                                        mode="gaussian")
        pred_sigmoid = F.sigmoid(pred)
    pred_hard = uu.threshold_organ(pred_sigmoid)
    pred_hard = pred_hard.cpu()
    prob = pred_sigmoid[:, target - 1:target].float().cpu()  # side branch (component scores)
    del pred, pred_sigmoid, image
    torch.cuda.empty_cache()
    organ_list = [target]
    pred_hard_post = uu.organ_post_process(pred_hard.numpy(), organ_list,
                                           args.log_name + "/" + name[0].split("/")[0] + "/" + name[0].split("/")[-1], args)
    pred_hard_post = torch.tensor(pred_hard_post)
    mask = pred_hard_post[:, target - 1].unsqueeze(1)
    return mask, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"], required=True)
    ap.add_argument("--cases", required=True, help="whitespace-separated case ids (file)")
    ap.add_argument("--out", required=True, help="output jsonl (appended; finished cases are skipped)")
    ap.add_argument("--work", default=f"{HERE}/_work")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (official: 4)")
    ap.add_argument("--save_npz", default=None, help="dir: keep mask/prob per case (smoke tests only)")
    a = ap.parse_args()
    target = TARGET[a.ds]

    cases = open(a.cases).read().split()
    done = {json.loads(l)["case"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run; target class {target}", flush=True)
    if not todo:
        return
    ndir = f"{a.work}/nii_{a.ds}_{os.getpid()}"
    os.makedirs(ndir, exist_ok=True)

    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")  # as pred_pseudo.py
    dl, uu, Universal_model = import_official()
    organ_name = uu.ORGAN_NAME[target - 1]
    print(f"target {target} = '{organ_name}'", flush=True)
    args = official_args(ndir)
    _, val_transforms = dl.get_loader_without_gt(args)  # official transforms (its glob of *.nii.gz finds nothing here)
    t0 = time.time()
    model = build_model(Universal_model, uu.NUM_CLASS)
    torch.backends.cudnn.benchmark = True  # as pred_pseudo.py
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    loader = make_loader(a.ds, todo, ndir, val_transforms, a.num_workers)
    if a.save_npz:
        os.makedirs(a.save_npz, exist_ok=True)
    fout = open(a.out, "a")
    t_all = time.time()
    t_wait = time.time()
    for batch in loader:
        t_load = time.time() - t_wait
        case = batch["name"][0]
        tm = time.time()
        grid = list(batch["image"].shape[2:])
        mask, prob = predict(model, uu, batch, target, args)
        t_model = time.time() - tm
        ti = time.time()
        inv = invert_to_cache(batch, {organ_name: mask, "prob": prob}, val_transforms)
        m, pr = inv[organ_name], inv["prob"]
        t_inv = time.time() - ti
        vol_shape = np.load(f"{CACHE}/{a.ds}/{case}.vol.npy", mmap_mode="r").shape
        sp = load_spacing(a.ds, case)
        assert m.shape == tuple(vol_shape) and pr.shape == tuple(vol_shape), (m.shape, pr.shape, vol_shape)
        binm = m > 0.5
        ans = mask_to_answer(binm, pr, sp[0] * sp[1] * sp[2], lambda idx: np.asarray(idx, dtype=np.float64), vol_shape)
        rec = {"case": case, **{k: ans[k] for k in ("present", "zyx", "diameter_mm", "score", "candidates")},
               "raw": json.dumps({"target": target, "class": organ_name, "n_components": ans["n_components"],
                                  "mask_vox": int(binm.sum()), "mask_vox_processed": int(mask.sum().item()),
                                  "max_prob": round(float(pr.max()), 4), "clipped": ans.get("clipped", False),
                                  "grid": grid, "t_model_s": round(t_model, 1), "t_invert_s": round(t_inv, 1),
                                  "t_load_s": round(t_load, 1)})}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if a.save_npz:
            np.savez_compressed(f"{a.save_npz}/{case}.npz", prob=pr.astype(np.float16), mask=binm)
        try:
            os.remove(f"{ndir}/{case}.nii")
        except FileNotFoundError:
            pass
        print(f"{case} present={ans['present']} ncomp={ans['n_components']} zyx={ans['zyx']} d={ans['diameter_mm']} "
              f"s={ans['score']} grid={grid} model={t_model:.1f}s invert={t_inv:.1f}s load={t_load:.1f}s", flush=True)
        t_wait = time.time()
    print(f"total {time.time() - t_all:.0f}s for {len(todo)} cases", flush=True)
    try:
        os.rmdir(ndir)
    except OSError:
        pass


if __name__ == "__main__":
    main()

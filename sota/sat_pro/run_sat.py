#!/usr/bin/env python
"""SAT-Pro (Zhao et al., npj Digit. Med. 2025) on the cached LUNA16 / MSD-liver arrays, official code and weights.

Per case:
  cached LPS array -> NIfTI with an explicit affine (common.write_nifti)
  -> official data/inference_dataset.py: Inference_Dataset / load_image (LoadImaged, EnsureChannelFirstd,
     Orientationd RAS, Spacingd (1, 1, 3) bilinear, CropForegroundd, ToTensord, CT clip [-500, 1000] + z-score),
     split_3d into 288x288x96 patches with 50 % overlap
  -> official SAT-Pro (UNET-L Maskformer) + knowledge text encoder, prompt = one class name in SAT's vocabulary
  -> the loop of evaluate/inference_engine.py:inference copied verbatim (autocast, sigmoid, Gaussian-weighted
     accumulation, prediction / accumulation, threshold 0.5)
  -> fixed mask -> answer rule (common.mask_to_answer) on the processed grid; centroids go back to the cached
     index through the processed-grid affine that MONAI tracks (captured just before the official Normalization).

Only change to the official code path: the processed image affine is captured (the official engine writes its masks
with an identity affine in the resampled/cropped space and never maps them back), and the BioLORD-2023-C backbone of
the text encoder is read from a local snapshot because transformers 4.21.3 cannot download from today's hub; its
weights are overwritten by SAT's text_encoder.pth anyway.
"""
import argparse
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import affine_mapper, load_cached, load_spacing, mask_to_answer, write_nifti  # noqa: E402

SAT = "/raid/home/CAMCA/hj880/ext_src/SAT"
SNAP = "/raid/home/CAMCA/hj880/hf_home/hub/models--zzh99--SAT/snapshots/56d1377859292e0cf2e9d0d7ff8ff696763725ca/Pro"
BIOLORD = ("/raid/home/CAMCA/hj880/hf_home/hub/models--FremyCompany--BioLORD-2023-C/snapshots/"
           "4bb5abd13930002136e3a5750b84474763c775c5")
PROMPT = {"luna": "lung nodule", "liver": "liver tumor", "liver_ext": "liver tumor"}
CACHE_DS = {"luna": "luna", "liver": "liver", "liver_ext": "liver_ext"}


def _write(job):
    ds, case, path, sphere = job
    if os.path.exists(path):
        return path
    vol, sp = load_cached(ds, case)
    if sphere is not None:  # coordinate check: 10 mm sphere of +300 HU at a known index
        vol = add_sphere(vol, sp, sphere)
    write_nifti(vol, sp, path + ".tmp.nii")
    os.replace(path + ".tmp.nii", path)
    return path


def add_sphere(vol, sp, zyx, diam=10.0, hu=300):
    vol = vol.copy()
    zz, yy, xx = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    d2 = ((zz - zyx[0]) * sp[0]) ** 2 + ((yy - zyx[1]) * sp[1]) ** 2 + ((xx - zyx[2]) * sp[2]) ** 2
    vol[d2 <= (diam / 2) ** 2] = hu
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=["luna", "liver", "liver_ext"], required=True)
    ap.add_argument("--cases", required=True, help="whitespace-separated case ids (file)")
    ap.add_argument("--out", required=True, help="output jsonl (appended; finished cases are skipped)")
    ap.add_argument("--work", default=f"{HERE}/_work")
    ap.add_argument("--batchsize_3d", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--port", default="29517")
    ap.add_argument("--save_npz", default=None, help="dir: keep prob/mask/affine per case (smoke tests only)")
    ap.add_argument("--sphere", default=None, help="z,y,x: insert a synthetic sphere (coordinate check only)")
    ap.add_argument("--keep_nifti", action="store_true")
    a = ap.parse_args()

    cases = open(a.cases).read().split()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["case"] for l in open(a.out) if l.strip()}
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tag = "sphere" if a.sphere else a.ds
    ndir = f"{a.work}/nii_{tag}"
    os.makedirs(ndir, exist_ok=True)
    sphere = tuple(float(v) for v in a.sphere.split(",")) if a.sphere else None
    paths = {c: f"{ndir}/{c}.nii" for c in todo}

    # official jsonl input format (image, label list, modality, dataset)
    jsonl = f"{a.work}/{tag}_{os.getpid()}.jsonl"
    with open(jsonl, "w") as f:
        for c in todo:
            f.write(json.dumps({"image": paths[c], "label": [PROMPT[a.ds]], "modality": "ct", "dataset": tag}) + "\n")

    # ---- official environment of inference.py (single process, world size 1) ----
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=a.port)
    sys.path.insert(0, SAT)
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader

    import data.inference_dataset as ids
    import model.knowledge_encoder as ke
    from evaluate.inference_engine import compute_gaussian
    from evaluate.params import parse_args
    from model.build_model import build_maskformer, load_checkpoint
    from model.text_encoder import Text_Encoder

    ke.Knowledge_Encoder.__init__.__defaults__ = (BIOLORD,)  # local copy of 'FremyCompany/BioLORD-2023-C'

    _orig_norm = ids.Normalization
    _capt = {}

    def _norm_capture(img, image_type):
        _capt["affine"] = np.asarray(img.affine, dtype=np.float64)  # processed voxel -> RAS world (mm)
        return _orig_norm(img, image_type)

    ids.Normalization = _norm_capture

    class Dataset(ids.Inference_Dataset):
        def __getitem__(self, idx):
            image = self.lines[idx]["image"]
            if not os.path.exists(image):  # written lazily in the worker if the pre-pass has not reached it
                c = os.path.basename(image)[:-4]
                _write((CACHE_DS[a.ds], c, image, sphere))
            out = super().__getitem__(idx)
            out["affine"] = _capt["affine"]
            out["case"] = os.path.basename(image)[:-4]
            return out

    argv = ["inference.py", "--rcd_dir", a.work, "--datasets_jsonl", jsonl, "--vision_backbone", "UNET-L",
            "--checkpoint", f"{SNAP}/SAT_Pro.pth", "--text_encoder", "ours",
            "--text_encoder_checkpoint", f"{SNAP}/text_encoder.pth", "--max_queries", "256",
            "--batchsize_3d", str(a.batchsize_3d), "--num_workers", str(a.num_workers)]
    sys.argv = argv
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200))

    # pre-write the first NIfTI files (workers write any missing ones lazily)
    with Pool(8) as p:
        for _ in p.imap(_write, [(CACHE_DS[a.ds], c, paths[c], sphere) for c in todo[:16]]):
            pass

    testset = Dataset(args.datasets_jsonl, args.max_queries, args.batchsize_3d)
    loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers,
                        collate_fn=ids.collate_fn, prefetch_factor=2 if args.num_workers else None)
    model = build_maskformer(args, device, 0)
    text_encoder = Text_Encoder(text_encoder=args.text_encoder, checkpoint=args.text_encoder_checkpoint,
                                partial_load=args.text_encoder_partial_load, open_bert_layer=12,
                                open_modality_embed=False, gpu_id=0, device=device)
    model, _, _ = load_checkpoint(checkpoint=args.checkpoint, resume=False, partial_load=args.partial_load,
                                  model=model, device=device)
    model.eval()
    text_encoder.eval()
    if a.save_npz:
        os.makedirs(a.save_npz, exist_ok=True)

    fout = open(a.out, "a")
    t_all = time.time()
    with torch.no_grad():
        gaussian = torch.tensor(compute_gaussian((288, 288, 96))).to(device)  # hwd
        end_time = time.time()
        for batch in loader:
            t_load = time.time() - end_time
            t0 = time.time()
            case = batch["case"]
            # ---------------- verbatim from evaluate/inference_engine.py:inference ----------------
            batched_patches = batch['batched_patches']
            batched_y1y2_x1x2_z1z2 = batch['batched_y1y2_x1x2_z1z2']
            split_labels = batch['split_queries']
            split_n1n2 = batch['split_n1n2']
            labels = batch['labels']
            modality = batch['modality']
            _, h, w, d = batch['chwd']
            n = len(labels)
            prediction = torch.zeros((n, h, w, d))
            accumulation = torch.zeros((n, h, w, d))
            with autocast():
                queries_ls = []
                for labels_ls, n1n2 in zip(split_labels, split_n1n2):
                    queries_ls.append(text_encoder(labels_ls, modality))
                torch.cuda.empty_cache()
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patches, batched_y1y2_x1x2_z1z2):
                    patches = patches.to(device=device)
                    prediction_patch = model(queries=queries_ls, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)
                    prediction_patch = prediction_patch.detach()
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]
                        tmp = prediction_patch[b, :, :y2-y1, :x2-x1, :z2-z1] * gaussian[:y2-y1, :x2-x1, :z2-z1]
                        prediction[:, y1:y2, x1:x2, z1:z2] += tmp.cpu()
                        accumulation[:, y1:y2, x1:x2, z1:z2] += gaussian[:y2-y1, :x2-x1, :z2-z1].cpu()
            prediction = prediction / accumulation
            # official: prediction = torch.where(prediction > 0.5, 1.0, 0.0)
            # ------------------------------------------------------------------------------------------
            prob = prediction[0].numpy()
            mask = prob > 0.5
            t_model = time.time() - t0
            vol_shape = np.load(f"/raid/home/CAMCA/hj880/perr/cache/{CACHE_DS[a.ds]}/{case}.vol.npy", mmap_mode="r").shape
            sp = load_spacing(CACHE_DS[a.ds], case)
            A = batch["affine"]
            ans = mask_to_answer(mask, prob, abs(np.linalg.det(A[:3, :3])), affine_mapper(A, sp), vol_shape)
            rec = {"case": case, **{k: ans[k] for k in ("present", "zyx", "diameter_mm", "score", "candidates")},
                   "raw": json.dumps({"prompt": labels[0], "n_components": ans["n_components"],
                                      "mask_vox": int(mask.sum()), "max_prob": round(float(prob.max()), 4),
                                      "clipped": ans.get("clipped", False), "grid": [h, w, d],
                                      "t_model_s": round(t_model, 1), "t_load_s": round(t_load, 1)})}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if a.save_npz:
                np.savez_compressed(f"{a.save_npz}/{case}.npz", prob=prob.astype(np.float16), affine=A,
                                    image=batch["image"][0].numpy().astype(np.float16))
            if not a.keep_nifti:
                try:
                    os.remove(f"{ndir}/{case}.nii")
                except FileNotFoundError:
                    pass
            print(f"{case} present={ans['present']} ncomp={ans['n_components']} zyx={ans['zyx']} "
                  f"d={ans['diameter_mm']} s={ans['score']} grid={h}x{w}x{d} model={t_model:.1f}s load={t_load:.1f}s",
                  flush=True)
            end_time = time.time()
    print(f"total {time.time() - t_all:.0f}s for {len(todo)} cases", flush=True)
    os.remove(jsonl)


if __name__ == "__main__":
    main()

"""LesionLocator (Rokuss et al., CVPR 2025) longitudinal tracking on autoPET IV Longitudinal-CT pairs.

For every pair in a plan (make_plan.py) and every baseline piece q of that pair, this runs the body of the documented
command

    LesionLocator_track -bl <pid>_BL_img_q.nii.gz -fu <pid>_FU_img_NN.nii.gz -p <pid>_BL_mask_q.nii.gz -t box
                        -o <out> -m LesionLocatorCheckpoint            (all 5 folds, device cuda)

in-process, with the models loaded once (lesionlocator/inference/lesionlocator_segment_and_track.py:
segment_and_track, same constructor and call arguments). Stage 1 segments every baseline lesion from a box derived
from the baseline mask (LesionLocatorSeg/bbox_optimized); stage 2 tracks each segmented lesion into the follow-up
(LesionLocatorTrack: UniGradICON registration of the whole scans, warp, U-Net refinement). Inputs are the original
NIfTI files; nothing from the follow-up side except the FU image is read.

Per baseline piece, <out>/done.json records each lesion's baseline segmentation volume, follow-up segmentation volume,
follow-up centroid ([z, y, x] voxel index of the FU image = our native FU grid) and timings.

    CUDA_VISIBLE_DEVICES=1 python run_lesionlocator.py plan_eval.json _work/eval
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi

from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter
from lesionlocator.inference.lesionlocator_segment_and_track import LesionLocatorSegTracker

CKPT = "/raid/home/CAMCA/hj880/ext_src/LesionLocator_weights/LesionLocatorCheckpoint"
FILE_ENDING = ".nii.gz"


class _ClearNegativeSeg:
    """Preprocessor proxy: seg values < 0 -> 0 after run_case.

    Bug fix (the only change to LesionLocator's computation). nnU-Net's crop_to_nonzero marks seg voxels outside the
    image's (hole-filled) nonzero mask with -1. The segmenter clears them before deriving prompts
    (get_prompt_from_inst_or_bin_seg: seg[seg < 0] = 0), but LesionLocatorSegTracker.track_single_lesion feeds the
    preprocessed per-lesion prompt to TrackNet as is. TrackNet then centres its U-Net crop on torch.nonzero(warped
    prompt), which counts the -1 voxels. 64 of 67 evaluation BL scans have a few hundred exterior voxels of exactly
    0 HU, so the crop lands far from small lesions (seen on training pair 02522a2b27_00: crop z 31-127 for a lesion
    warped to z 194)."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def run_case(self, *args, **kwargs):
        data, seg, props = self.preprocessor.run_case(*args, **kwargs)
        if seg is not None:
            seg[seg < 0] = 0
        return data, seg, props


class FixedSegTracker(LesionLocatorSegTracker):
    def track_single_lesion(self, lesion_index, bl, fu, bl_path, fu_path, output_folder, prompts, properties,
                            preprocessor):
        return super().track_single_lesion(lesion_index, bl, fu, bl_path, fu_path, output_folder, prompts,
                                           properties, _ClearNegativeSeg(preprocessor))


def mask_stats(path):
    """Voxel count, ml, centroid [z, y, x] and number of 26-connected components of a binary mask file."""
    img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img) > 0
    vox_ml = float(np.prod(img.GetSpacing())) / 1000.0
    n = int(a.sum())
    out = {"nvox": n, "ml": n * vox_ml, "shape_zyx": list(a.shape), "spacing_zyx": list(img.GetSpacing()[::-1]),
           "direction_identity": bool(np.allclose(img.GetDirection(), np.eye(3).ravel()))}
    if n:
        idx = np.argwhere(a)
        out["centroid_zyx"] = [float(v) for v in idx.mean(0)]
        lab, ncomp = ndi.label(a, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel())[1:]
        out["ncomp"] = int(ncomp)
        out["largest_comp_frac"] = float(sizes.max() / n)
        out["bbox_zyx"] = [int(v) for v in idx.min(0)] + [int(v) for v in idx.max(0)]
    return out


def run_piece(segmenter, tracker, pair, bl, out):
    """One LesionLocator_track call: baseline piece bl -> follow-up pair['fu']. Returns the per-lesion record."""
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    # stage 1: segment_and_track() with -t box (same arguments as the CLI)
    predicted_files = segmenter.predict_from_files(bl["img"], out, bl["mask"], "box", overwrite=True,
                                                   num_processes_preprocessing=1,
                                                   num_processes_segmentation_export=3)
    t1 = time.time()
    # stage 2: tracking of the baseline segmentations (prompt = list of per-lesion files, as in the CLI)
    tracker.track(baseline_image=bl["img"], follow_up_images=[pair["fu"]], output_folder=out, prompt=predicted_files)
    t2 = time.time()
    fu_stem = os.path.basename(pair["fu"]).replace(FILE_ENDING, "")
    lesions = []
    for k, f in enumerate(predicted_files or [], start=1):
        rec = {"track_index": k, "bl_seg_file": f}
        if f is not None:
            m = re.search(r"_lesion_(\d+)\.nii\.gz$", f)
            rec["bl_lesion_id"] = int(m.group(1)) if m else None
            rec["bl_seg"] = mask_stats(f)
        fu_file = os.path.join(out, f"{fu_stem}_lesion_{k}{FILE_ENDING}")
        if os.path.exists(fu_file):
            rec["fu_seg_file"] = fu_file
            rec["fu_seg"] = mask_stats(fu_file)
        lesions.append(rec)
    return {"key": pair["key"], "bl_piece": bl["piece"], "bl_img": bl["img"], "bl_mask": bl["mask"], "fu": pair["fu"],
            "n_bl_lesions_annot": len(bl["lesions"]), "binary_mode": bl["binary_mode"],
            "n_predicted_files": len(predicted_files or []), "lesions": lesions,
            "seconds": {"segment_bl": round(t1 - t0, 1), "track": round(t2 - t1, 1), "total": round(time.time() - t0, 1)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("work")
    ap.add_argument("-m", default=CKPT)
    ap.add_argument("-f", nargs="+", default=(0, 1, 2, 3, 4))
    ap.add_argument("--shard", type=int, default=0, help="this process's shard (several processes share GPU 1)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no_fix", action="store_true", help="official tracker without the negative-prompt fix")
    ap.add_argument("--claim", action="store_true",
                    help="work queue: any number of processes take BL pieces via atomic claim files (largest first)")
    args = ap.parse_args()

    # as in segment_and_track() for device cuda
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda")
    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), flush=True)

    plan = json.load(open(args.plan))
    if args.nshards > 1:   # greedy balance by lesion count, deterministic
        load, owner = [0] * args.nshards, {}
        for p in sorted(plan, key=lambda p: (-sum(len(b["lesions"]) for b in p["bl"]), p["key"])):
            s = min(range(args.nshards), key=lambda j: (load[j], j))
            owner[p["key"]] = s
            load[s] += sum(len(b["lesions"]) for b in p["bl"])
        plan = [p for p in plan if owner[p["key"]] == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(plan)} pairs, {load[args.shard]} lesions", flush=True)
    t_load = time.time()
    segmenter = LesionLocatorSegmenter(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                       perform_everything_on_device=True, device=device, verbose=False,
                                       allow_tqdm=True, verbose_preprocessing=False)
    segmenter.initialize_from_trained_model_folder(os.path.join(args.m, "LesionLocatorSeg", "bbox_optimized"),
                                                   args.f, "checkpoint_final.pth")
    tracker = (LesionLocatorSegTracker if args.no_fix else FixedSegTracker)(
        model_folder=os.path.join(args.m, "LesionLocatorTrack"), folds=args.f, device=device)
    print(f"models loaded in {time.time() - t_load:.0f} s; tracker {type(tracker).__name__}", flush=True)

    t_all = time.time()
    jobs = [(i, pair, bl) for i, pair in enumerate(plan, 1) for bl in pair["bl"]]
    if args.claim:
        jobs.sort(key=lambda j: (-len(j[2]["lesions"]), j[1]["key"], j[2]["piece"]))
    for i, pair, bl in jobs:
        if True:
            out = os.path.join(args.work, pair["key"], f"bl{bl['piece']:02d}")
            done = os.path.join(out, "done.json")
            if os.path.exists(done):
                print(f"[{i}/{len(plan)}] {pair['key']} bl{bl['piece']:02d} already done", flush=True)
                continue
            if args.claim:
                os.makedirs(out, exist_ok=True)
                try:
                    fd = os.open(os.path.join(out, "claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, f"{os.getpid()} {time.strftime('%F %T')}\n".encode())
                os.close(fd)
            if not bl["lesions"]:
                os.makedirs(out, exist_ok=True)
                json.dump({"key": pair["key"], "bl_piece": bl["piece"], "lesions": [], "seconds": {"total": 0.0},
                           "n_bl_lesions_annot": 0}, open(done, "w"), indent=1)
                continue
            print(f"\n##### [{i}/{len(plan)}] {pair['key']} BL piece {bl['piece']} ({len(bl['lesions'])} lesions) "
                  f"elapsed {time.time() - t_all:.0f} s", flush=True)
            rec = run_piece(segmenter, tracker, pair, bl, out)
            json.dump(rec, open(done + ".tmp", "w"), indent=1)
            os.replace(done + ".tmp", done)
            nt = sum(1 for x in rec["lesions"] if x.get("fu_seg", {}).get("nvox", 0) > 0)
            print(f"##### {pair['key']} bl{bl['piece']:02d}: {nt}/{len(rec['lesions'])} tracked, {rec['seconds']}",
                  flush=True)
    print(f"ALL DONE in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()

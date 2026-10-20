"""MSD Task06 Lung (non-small-cell lung cancer, TCIA; CC BY-SA 4.0) as an external lung test for systems trained on
LUNA16 nodules (<= 30 mm, another hospital): 63 labelled CTs, tumours often larger than any LUNA16 nodule. Each case is
cached exactly like a LUNA16 study (cache/luna/T06_<n>.vol / .lung / .meta, LPS, int16 HU) so the world model, the
harness strips, the reader and the 3-D detector run on it unchanged; the lung mask is TotalSegmentator's five lobes
(no label is used for the domain). The tumour mask, grown by 3 mm, is kept apart (.tumour.npy) for scoring only:
a reported point is a hit when it lies inside it.

    TOTALSEG_HOME_DIR=... venvs/totalseg/bin/python prep_task06.py [--shard i n]     # segment and cache
    venvs/perr/bin/python prep_task06.py --iso                                         # 1 mm cache and domain
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/raid/home/CAMCA/hj880/msd/Task06_Lung"
SEG = "/raid/home/CAMCA/hj880/data_ext/t06_totalseg"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/luna"
LIST = "/raid/home/CAMCA/hj880/perr/results/t06_cases.txt"
TS = "/raid/home/CAMCA/hj880/venvs/totalseg/bin/TotalSegmentator"
LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
         "lung_lower_lobe_right"]


def cases():
    return sorted("T06_" + f.split("_")[1].split(".")[0] for f in os.listdir(f"{ROOT}/imagesTr") if f.startswith("lung_"))


def name(case):
    return f"lung_{case.split('_')[1]}.nii.gz"


def prepare(case):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    out = f"{SEG}/{case}"
    if not all(os.path.exists(f"{out}/{l}.nii.gz") for l in LOBES):
        os.makedirs(out, exist_ok=True)
        subprocess.run([TS, "-i", f"{ROOT}/imagesTr/{name(case)}", "-o", out, "--roi_subset", *LOBES, "--quiet"], check=True)
    stem = f"{CACHE}/{case}"
    if os.path.exists(stem + ".meta.npy"):
        return
    img = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/imagesTr/{name(case)}"), "LPS")
    lab = sitk.DICOMOrient(sitk.ReadImage(f"{ROOT}/labelsTr/{name(case)}"), "LPS")
    vol = np.clip(sitk.GetArrayFromImage(img), -1024, 3071).astype(np.int16)
    tum = sitk.GetArrayFromImage(lab) > 0
    lung = np.zeros(vol.shape, bool)
    for l in LOBES:
        lung |= sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(f"{out}/{l}.nii.gz"), "LPS")) > 0
    sx, sy, sz = img.GetSpacing()
    spacing = (sz, sy, sx)
    comp, n = ndi.label(tum, structure=np.ones((3, 3, 3)))
    nods = []
    for i, sl in enumerate(ndi.find_objects(comp), 1):
        m = comp[sl] == i
        d = (6 * float(m.sum()) * sz * sy * sx / np.pi) ** (1 / 3)
        c = np.argwhere(m).mean(0) + np.array([s.start for s in sl])
        nods.append(((float(c[0]), float(c[1]), float(c[2])), d))
    r = [max(1, int(round(3.0 / s))) for s in spacing]
    zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    ball = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2 <= 9.0
    np.save(stem + ".vol.npy", vol)
    np.save(stem + ".lung.npy", lung.astype(np.uint8))
    np.save(stem + ".tumour.npy", ndi.binary_dilation(tum, ball).astype(np.uint8))
    np.save(stem + ".meta.npy", {"spacing": spacing, "nodules": nods, "sha256": hashlib.sha256(vol.tobytes()).hexdigest(),
                                 "lung_ml": float(lung.sum()) * sz * sy * sx / 1000,
                                 "tumour_in_lung": float((tum & ndi.binary_dilation(lung, iterations=3)).sum()) / max(int(tum.sum()), 1)})
    print(case, len(nods), "tumour components, largest %.0f mm" % max(d for _, d in nods), flush=True)


def build(case):
    from perr.context_wm import build_iso
    from perr.study import load_luna
    from perr.worldmodels import lung_domain
    s = load_luna(case)
    lung_domain(s)
    build_iso(case, "lung")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--iso", action="store_true")
    a = ap.parse_args()
    if a.iso:
        from multiprocessing import Pool
        cs = [c for c in cases() if os.path.exists(f"{CACHE}/{c}.meta.npy")]
        with Pool(16) as p:
            p.map(build, cs, chunksize=1)
        open(LIST, "w").write("\n".join(cs) + "\n")
        print(len(cs), "cases on the 1 mm grid")
        return
    for c in cases()[a.shard[0]::a.shard[1]]:
        try:
            prepare(c)
        except Exception as e:
            print(c, "FAILED", repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()

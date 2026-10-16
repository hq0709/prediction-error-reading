"""External liver-tumour test set: HCC-TACE-Seg (TCIA, CC BY 4.0), portal-venous CT, cached like liver.load_liver.

Source. Moawad et al., "Multimodality annotated HCC cases with and without advanced imaging segmentation", TCIA
(doi:10.7937/TCIA.5FNA-0924); data descriptor Sci Data 10:33 (2023). 105 HCC patients treated with TACE at MD
Anderson (2002-2012); pre- and post-procedure CT; one expert DICOM-SEG per patient on the pre-procedure CT
(semi-automatic in Amira, clinically curated, converted with dcmqi). Segments: Liver, Mass, Portal vein, Abdominal
aorta, plus Necrosis in HCC_017 (numbered 3 there, with Portal vein 4 and Aorta 5); they are remapped by label to
1 Liver, 2 Mass, 3 Portal vein, 4 Abdominal aorta, 5 Necrosis. Segments do not overlap: Liver excludes the others.
Per the descriptor the portal-venous phase was segmented (the other phases were registered to it).
Downloaded without credentials from the public IDC bucket (idc-index 0.13.1, IDC v25; all 572 CT series and the 105
expert SEG series; the AI-generated BAMF/AIMI SEG series are not used) into ROOT/dicom/<pid>/<study>/<mod>_<series>/.

Training-data overlap. SAT-Pro: not in SAT-DS (72-dataset list, data/dataset_config/72.json, SAT-DS README).
VISTA3D: not in any training/validation list of stages 1-4 (configs/train/hyper_parameters_stage*.yaml); the
research repo keeps it under data/external/ for zero-shot evaluation (configs/zeroshot_eval/infer_iter_point_hcc.yaml)
and the paper (arXiv 2406.05285, Sec. 5) lists "the Hepatocellular Carcinoma (105 scans) dataset" among 4 external
evaluation sets. (TCIA Colorectal-Liver-Metastases, by contrast, is a VISTA3D training set: "CRLM-CT", tumour -> 26.)

DICOM -> NIfTI (convert(); needs pydicom, run with /raid/home/CAMCA/hj880/venvs/liverext/bin/python). The SEG's
referenced-instance lists cannot be trusted: 23 SEGs reference slices from 2-3 acquisitions that share slice
positions inside one series (dcmqi picked one instance per position), one SEG has no ReferencedSeriesSequence
(HCC_048), one stores a 180-degree in-plane flip in its IOP/IPP (HCC_003), some frames reference nothing, and
acquisition times are removed by anonymisation. The SEG frame positions are therefore taken as the grid the masks
were drawn on, and the CT is chosen by geometry and content:
  1. candidates = every CT acquisition (series, AcquisitionNumber) of the SEG's study with the SEG's pixel spacing,
     an axial orientation and uniform slice spacing; each SEG frame is mapped to the candidate through world
     coordinates (frame IPP/IOP -> nearest CT slice, in-plane by IPP difference, nearest neighbour);
  2. exact = every frame with mask content lies within 0.05 mm of a CT slice and >= 99 % of the mask lands inside;
  3. phase from mean HU of the Portal vein (P) and Aorta (A) segments: pre-contrast if A < 100 and P < 100;
     arterial if P / A < 0.45; portal-venous if P >= 100; else indeterminate (on this data every arterial
     acquisition has P / A <= 0.41 and P <= 137 HU, every selected venous one P / A >= 0.51; the aorta mask is
     unreliable in a few SEGs, hence the P-based rules);
  4. pick the exact portal-venous candidate with the lowest HU SD inside the Liver segment; if there is none, the
     exact candidate with the lowest SD (any phase); if nothing is exact, the best-covered candidate (flagged).
The whole selected acquisition is written (labels are zero outside the SEG frames): NIFTI/imagesTr/<case>.nii.gz
(int16 HU, clipped to [-1024, 3071]), NIFTI/labelsTr/<case>.nii.gz (uint8 segment numbers) and NIFTI/meta/<case>.json
(selection record, per-candidate HU statistics, fit under +-2 slice shifts).

Cache (load_case(); numpy/scipy/SimpleITK only), exactly as liver.load_liver: vol int16 [z, y, x] in LPS
(SimpleITK.DICOMOrient), liver uint8 = Liver | Mass | Necrosis with enclosed axial holes filled (the carved-out
intrahepatic portal-vein branches: median 1.5 % of the liver, max 5.6 %, 83 % of it Portal-vein label; MSD/LiTS
liver labels include intrahepatic vessels, their axial holes are < 1 % of the liver), tumour = Mass | Necrosis,
26-connected components, equivalent diameter (6 V / pi)^(1/3), big / small = components >= / < 10 mm grown by a
3 mm ball, meta = {"spacing": (sz, sy, sx), "nodules": [...]}.

Evaluation list (eval_cases(), written to EVAL): exact portal-venous grid; SD of the Liver segment <= 45 HU, not
improved by more than 2 HU by a +-1/2 slice shift and not beaten by more than 5 HU by another exact acquisition
(the masks were drawn on this acquisition, not on a displaced phase); labels not truncated (the liver does not reach
an end of the segmented range while the CT continues beyond it); at least one tumour >= 10 mm; not in EXCLUDE.
Result: 93 of 105 (excluded: 015, 016, 089 arterial; 005, 006, 009, 021, 027, 030, 039 masks fit a shift or another
phase better; 024 inexact grid; 010 truncated labels). Contact sheets of all 93 were checked by eye.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .liver import MIN_DIAM, TOL_MM, has_target  # noqa: F401  (same target definition as the MSD world)
from .study import Nodule, Study

ROOT = "/raid/home/CAMCA/hj880/data_ext/hcc_tace_seg"
DICOM = f"{ROOT}/dicom"
NIFTI = f"{ROOT}/nifti"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/liver_ext"
EVAL = "/raid/home/CAMCA/hj880/perr/sota/eval_liver_ext.txt"
LIVER, MASS, PORTAL, AORTA, NECROSIS = 1, 2, 3, 4, 5      # label values written to labelsTr
SEGMENT_IDS = {"Liver": LIVER, "Mass": MASS, "Portal vein": PORTAL, "Abdominal aorta": AORTA, "Necrosis": NECROSIS}
Z_TOL = 0.05          # mm, frame position -> CT slice for an exact grid match
SD_MAX, SD_SHIFT, SD_OTHER = 45.0, 2.0, 5.0   # HU; fit criteria of the evaluation list
EXCLUDE: dict[str, str] = {}   # case -> reason; visual QC of all eligible cases found none to add


def download() -> None:
    """Fetch the collection from the public IDC bucket (no credentials): every CT series and the 105 expert SEG
    series of hcc_tace_seg (AI-generated analysis-result SEGs excluded). Needs idc-index (liverext venv)."""
    from idc_index import IDCClient

    c = IDCClient()
    d = c.index
    d = d[(d.collection_id == "hcc_tace_seg") & d.analysis_result_id.isna()]
    os.makedirs(DICOM, exist_ok=True)
    d.to_csv(f"{ROOT}/series_index.csv", index=False)
    c.download_from_selection(downloadDir=DICOM, seriesInstanceUID=list(d.SeriesInstanceUID),
                              dirTemplate="%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID",
                              quiet=False, show_progress_bar=False, use_s5cmd_sync=True)


def pids() -> list[str]:
    return sorted(p for p in os.listdir(DICOM) if p.startswith("HCC_"))


def cases() -> list[str]:
    """Converted cases (NIfTI written)."""
    d = f"{NIFTI}/labelsTr"
    return sorted(f[:-7] for f in os.listdir(d) if f.endswith(".nii.gz")) if os.path.isdir(d) else []


def meta_record(case: str) -> dict:
    with open(f"{NIFTI}/meta/{case}.json") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------- DICOM -> NIfTI

def _phase(hu: dict) -> str:
    a, p = hu.get(str(AORTA), {}).get("mean"), hu.get(str(PORTAL), {}).get("mean")
    if a is None or p is None:
        return "unknown"
    if a < 100 and p < 100:
        return "pre-contrast"
    if p / a < 0.45:
        return "arterial"
    if p >= 100:
        return "portal-venous"
    return "indeterminate"


def _read_seg(pydicom, path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    sh = ds.SharedFunctionalGroupsSequence[0]
    iop = np.array([float(x) for x in sh.PlaneOrientationSequence[0].ImageOrientationPatient])
    ps = np.array([float(x) for x in sh.PixelMeasuresSequence[0].PixelSpacing])  # (row spacing, column spacing)
    # segment numbers differ between SEGs (HCC_017: 3 Necrosis, 4 Portal vein, 5 Abdominal aorta): map by label
    canon = {int(s.SegmentNumber): SEGMENT_IDS[str(s.SegmentLabel)] for s in ds.SegmentSequence}
    frames = []
    for i, g in enumerate(ds.PerFrameFunctionalGroupsSequence):
        sn = canon[int(g.SegmentIdentificationSequence[0].ReferencedSegmentNumber)]
        ipp = np.array([float(x) for x in g.PlanePositionSequence[0].ImagePositionPatient])
        frames.append((sn, ipp, i))
    segs = {canon[int(s.SegmentNumber)]: str(s.SegmentLabel) for s in ds.SegmentSequence}
    ref = [str(r.SeriesInstanceUID) for r in ds.get("ReferencedSeriesSequence", [])]
    return ds, arr, iop, ps, frames, segs, ref


def _acquisitions(pydicom, study_dir, ps):
    """CT acquisitions of a study: (series, AcquisitionNumber, IOP, size) groups with the SEG's pixel spacing."""
    out = []
    for sd in sorted(glob.glob(f"{study_dir}/CT_*")):
        groups = {}
        for f in glob.glob(f"{sd}/*.dcm"):
            h = pydicom.dcmread(f, stop_before_pixels=True)
            if np.abs(np.array([float(x) for x in h.PixelSpacing]) - ps).max() > 1e-4:
                continue
            iop = tuple(round(float(x), 4) for x in h.ImageOrientationPatient)
            key = (str(h.get("AcquisitionNumber", "")), iop, int(h.Rows), int(h.Columns))
            groups.setdefault(key, []).append((np.array([float(x) for x in h.ImagePositionPatient]), f,
                                               str(h.get("SeriesDescription", "")), float(h.get("SliceThickness", 0) or 0)))
        for (acq, iop, R, C), v in groups.items():
            iop = np.array(iop)
            n = np.cross(iop[:3], iop[3:])
            if abs(abs(n[2]) - 1) > 1e-3 or len(v) < 10:
                continue
            v.sort(key=lambda t: float(t[0] @ n))
            z = np.array([float(t[0] @ n) for t in v])
            dz = np.diff(z)
            out.append(dict(series=os.path.basename(sd)[3:], acq=acq, desc=v[0][2], iop=iop, n=n, R=R, C=C,
                            ipp=np.array([t[0] for t in v]), z=z, files=[t[1] for t in v],
                            dz=float(np.median(dz)), uniform=bool(dz.min() > 0 and dz.max() - dz.min() <= 0.01),
                            thickness=v[0][3]))
    return out


def _map_frames(arr, frames, iop_s, ps, a, shift: int = 0):
    """Label volume on acquisition a's grid (world-coordinate mapping, nearest neighbour) + mapping statistics."""
    rs, cs = iop_s[:3], iop_s[3:]
    rc, cc = a["iop"][:3], a["iop"][3:]
    lab = np.zeros((len(a["z"]), a["R"], a["C"]), np.uint8)
    tot = inside = 0
    zres_max = 0.0
    for sn, ipp, i in frames:
        rr, cl = np.nonzero(arr[i])
        if len(rr) == 0:
            continue
        w = ipp[None] + cl[:, None] * ps[1] * rs[None] + rr[:, None] * ps[0] * cs[None]
        zw = float(w[0] @ a["n"])
        tot += len(rr)
        if zw < a["z"][0] - a["dz"] / 2 or zw > a["z"][-1] + a["dz"] / 2:
            continue  # frame beyond the CT's z-range: counts against coverage
        k = int(np.argmin(np.abs(a["z"] - zw)))
        zres_max = max(zres_max, abs(float(a["z"][k]) - zw))
        k += shift
        if not 0 <= k < len(a["z"]):
            continue
        d = w - a["ipp"][k][None]
        col = np.rint(d @ rc / ps[1]).astype(int)
        row = np.rint(d @ cc / ps[0]).astype(int)
        ok = (row >= 0) & (row < a["R"]) & (col >= 0) & (col < a["C"])
        inside += int(ok.sum())
        lab[k, row[ok], col[ok]] = sn
    return lab, (inside / tot if tot else 0.0), zres_max


def _read_ct(pydicom, a):
    vol = np.zeros((len(a["files"]), a["R"], a["C"]), np.int16)
    for k, f in enumerate(a["files"]):
        c = pydicom.dcmread(f)
        hu = np.rint(c.pixel_array * float(c.get("RescaleSlope", 1)) + float(c.get("RescaleIntercept", 0)))
        vol[k] = np.clip(hu, -1024, 3071)
    return vol


def _hu_stats(vol, lab, segs):
    return {str(s): {"mean": round(float(vol[lab == s].mean()), 1), "sd": round(float(vol[lab == s].std()), 1),
                     "vox": int((lab == s).sum())} for s in segs if (lab == s).any()}


def convert(case: str, overwrite: bool = False) -> dict:
    """DICOM CT + SEG -> NIfTI pair and selection record (see module docstring). Needs pydicom."""
    import pydicom

    out_json = f"{NIFTI}/meta/{case}.json"
    if os.path.exists(out_json) and not overwrite:
        return meta_record(case)
    seg_files = glob.glob(f"{DICOM}/{case}/*/SEG_*/*.dcm")
    if len(seg_files) != 1:
        raise RuntimeError(f"{case}: {len(seg_files)} SEG files")
    ds, arr, iop_s, ps, frames, segs, ref = _read_seg(pydicom, seg_files[0])
    study_dir = os.path.dirname(os.path.dirname(seg_files[0]))
    n_s = np.cross(iop_s[:3], iop_s[3:])
    seg_z = sorted({round(float(ipp @ n_s), 3) for _, ipp, _ in frames})
    content_z = sorted({round(float(ipp @ n_s), 3) for sn, ipp, i in frames if arr[i].any()})
    liver_z = sorted({round(float(ipp @ n_s), 3) for sn, ipp, i in frames if sn == LIVER and arr[i].any()})
    rec = {"case": case, "seg_file": os.path.relpath(seg_files[0], ROOT), "segments": segs,
           "seg_referenced_series": ref, "seg_iop": [round(float(x), 4) for x in iop_s],
           "seg_pixel_spacing": [float(x) for x in ps], "seg_frames": len(frames), "seg_positions": len(seg_z),
           "seg_z_range": [seg_z[0], seg_z[-1]], "content_z_range": [content_z[0], content_z[-1]],
           "liver_at_seg_bounds": bool(liver_z[0] <= seg_z[0] + 1e-3 or liver_z[-1] >= seg_z[-1] - 1e-3),
           "candidates": []}
    best = None
    for a in _acquisitions(pydicom, study_dir, ps):
        if not a["uniform"]:
            rec["candidates"].append({"series": a["series"], "acq": a["acq"], "desc": a["desc"], "n": len(a["z"]),
                                      "skipped": "non-uniform slice spacing"})
            continue
        lab, cov, zres = _map_frames(arr, frames, iop_s, ps, a)
        c = {"series": a["series"], "acq": a["acq"], "desc": a["desc"], "n": len(a["z"]), "dz": round(a["dz"], 3),
             "thickness": a["thickness"], "coverage": round(cov, 4), "zres_max": round(zres, 3),
             "exact": bool(zres <= Z_TOL and cov >= 0.99)}
        if cov >= 0.5:
            vol = _read_ct(pydicom, a)
            c["hu"] = _hu_stats(vol, lab, segs)
            c["phase"] = _phase(c["hu"])
            c["liver_sd"] = c["hu"].get(str(LIVER), {}).get("sd")
            sh = {}
            for s in (-2, -1, 1, 2):
                l2, cv2, _ = _map_frames(arr, frames, iop_s, ps, a, shift=s)
                m = l2 == LIVER
                if cv2 >= 0.95 and m.any():
                    sh[str(s)] = round(float(vol[m].std()), 1)
            c["liver_sd_shift"] = sh
            key = (not c["exact"], c["phase"] != "portal-venous", c["liver_sd"] if c["exact"] else -cov,
                   c["liver_sd"])
            if best is None or key < best[0]:
                best = (key, c, a, vol, lab)
        rec["candidates"].append(c)
    if best is None:
        rec["selected"] = None
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=1)
        return rec
    _, c, a, vol, lab = best
    rec["selected"] = {k: c[k] for k in ("series", "acq", "desc", "n", "dz", "thickness", "coverage", "zres_max",
                                         "exact", "phase", "liver_sd", "liver_sd_shift", "hu")}
    # labels are truncated if the liver reaches an end of the segmented range while the CT continues beyond it
    lo, hi = float(a["z"][0]), float(a["z"][-1])
    rec["labels_truncated"] = bool((liver_z[0] <= seg_z[0] + 1e-3 and lo < seg_z[0] - a["dz"] / 2) or
                                   (liver_z[-1] >= seg_z[-1] - 1e-3 and hi > seg_z[-1] + a["dz"] / 2))
    rec["ct_z_range"] = [round(lo, 3), round(hi, 3)]
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((float(ps[1]), float(ps[0]), float(a["dz"])))
    img.SetOrigin(tuple(float(x) for x in a["ipp"][0]))
    r, cdir, n = a["iop"][:3], a["iop"][3:], a["n"]
    img.SetDirection(tuple(float(x) for x in (r[0], cdir[0], n[0], r[1], cdir[1], n[1], r[2], cdir[2], n[2])))
    limg = sitk.GetImageFromArray(lab)
    limg.CopyInformation(img)
    for d in ("imagesTr", "labelsTr", "meta"):
        os.makedirs(f"{NIFTI}/{d}", exist_ok=True)
    sitk.WriteImage(img, f"{NIFTI}/imagesTr/{case}.nii.gz", useCompression=True)
    sitk.WriteImage(limg, f"{NIFTI}/labelsTr/{case}.nii.gz", useCompression=True)
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=1)
    return rec


# ------------------------------------------------------------------------------------------------- cache

def load_case(case: str) -> Study:
    """Study with lung := liver mask (the domain) and nodules := tumour components; caches like liver.load_liver."""
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{CACHE}/{case}"
    if not os.path.exists(stem + ".meta.npy"):
        img = sitk.DICOMOrient(sitk.ReadImage(f"{NIFTI}/imagesTr/{case}.nii.gz"), "LPS")
        lab = sitk.DICOMOrient(sitk.ReadImage(f"{NIFTI}/labelsTr/{case}.nii.gz"), "LPS")
        vol = np.clip(sitk.GetArrayFromImage(img), -1024, 3071).astype(np.int16)
        lab = sitk.GetArrayFromImage(lab).astype(np.uint8)
        sx, sy, sz = img.GetSpacing()
        spacing = (sz, sy, sx)
        tum = (lab == MASS) | (lab == NECROSIS)
        liver = np.stack([ndi.binary_fill_holes(s) for s in ((lab == LIVER) | tum)])
        comp, n = ndi.label(tum, structure=np.ones((3, 3, 3)))
        nods, big, small = [], np.zeros(tum.shape, bool), np.zeros(tum.shape, bool)
        for i, sl in enumerate(ndi.find_objects(comp), 1):
            m = comp[sl] == i
            v = float(m.sum()) * sz * sy * sx
            d = (6 * v / np.pi) ** (1 / 3)
            c = np.argwhere(m).mean(0) + np.array([s.start for s in sl])
            nods.append(((float(c[0]), float(c[1]), float(c[2])), d))
            (big if d >= MIN_DIAM else small)[sl] |= m
        r = [max(1, int(round(TOL_MM / s))) for s in spacing]
        zz, yy, xx = np.ogrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
        ball = (zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2 <= TOL_MM ** 2
        np.save(stem + ".vol.npy", vol)
        np.save(stem + ".liver.npy", liver.astype(np.uint8))
        np.save(stem + ".big.npy", ndi.binary_dilation(big, ball).astype(np.uint8))
        np.save(stem + ".small.npy", ndi.binary_dilation(small, ball).astype(np.uint8))
        np.save(stem + ".meta.npy", {"spacing": spacing, "nodules": nods})
    meta = np.load(stem + ".meta.npy", allow_pickle=True).item()
    return Study(case, np.load(stem + ".vol.npy", mmap_mode="r"), tuple(meta["spacing"]),
                 np.load(stem + ".liver.npy", mmap_mode="r"), "", [Nodule(tuple(z), d) for z, d in meta["nodules"]])


def label_at(case: str, zyx) -> str:
    z, y, x = (int(round(c)) for c in zyx)
    big = np.load(f"{CACHE}/{case}.big.npy", mmap_mode="r")
    if big[z, y, x]:
        return "big"
    return "small" if np.load(f"{CACHE}/{case}.small.npy", mmap_mode="r")[z, y, x] else "none"


# ---------------------------------------------------------------------------------------------- eval list

def eligibility(case: str) -> list[str]:
    """Reasons a converted case is not in the evaluation list (empty list = eligible)."""
    rec = meta_record(case)
    sel = rec.get("selected")
    if not sel:
        return ["no CT acquisition matches the SEG"]
    why = []
    if not sel["exact"]:
        why.append(f"inexact grid (zres {sel['zres_max']} mm, coverage {sel['coverage']})")
    if sel["phase"] != "portal-venous":
        why.append(f"phase {sel['phase']}")
    if sel["liver_sd"] is None or sel["liver_sd"] > SD_MAX:
        why.append(f"liver SD {sel['liver_sd']} HU > {SD_MAX}")
    better = {s: v for s, v in sel["liver_sd_shift"].items() if v < sel["liver_sd"] - SD_SHIFT}
    if better:
        why.append(f"z-shift fits better {better}")
    rivals = [(c["liver_sd"], c["desc"], c["acq"]) for c in rec["candidates"] if c.get("exact") and c.get("liver_sd")
              and (c["series"], c["acq"]) != (sel["series"], sel["acq"])]
    if rivals and min(rivals)[0] < sel["liver_sd"] - SD_OTHER:
        why.append(f"masks fit another acquisition better {min(rivals)}")
    if rec.get("labels_truncated"):
        why.append("liver reaches the first/last SEG frame and the CT continues beyond it")
    if case in EXCLUDE:
        why.append(EXCLUDE[case])
    if not why and not has_target(load_case(case)):
        why.append("no tumour >= 10 mm")
    return why


def eval_cases(write: bool = False) -> list[str]:
    out = [c for c in cases() if not eligibility(c)]
    if write:
        with open(EVAL, "w") as f:
            f.write("\n".join(out) + "\n")
    return out


# --------------------------------------------------------------------------------------------- visual QC

def qc_png(case: str, path: str, half_mm: float = 120.0, size: int = 512) -> str:
    """Axial slice through the largest tumour's centroid: soft-tissue window, liver outline green, tumour red
    (>= 10 mm) / orange (< 10 mm), the centroid marked. Radiological view: patient left on image right."""
    s = load_case(case)
    z, y, x = (int(round(v)) for v in max(s.nodules, key=lambda n: n.diameter_mm).zyx)
    lab = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(f"{NIFTI}/labelsTr/{case}.nii.gz"), "LPS"))[z]
    im = np.clip((np.asarray(s.vol[z], np.float32) - (50 - 175)) / 350 * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([im] * 3, -1)
    liver = np.asarray(s.lung[z]).astype(bool)
    tum = (lab == MASS) | (lab == NECROSIS)

    def edge(m):
        return m & ~ndi.binary_erosion(m)

    rgb[edge(liver)] = (0, 255, 0)
    big = np.load(f"{CACHE}/{case}.big.npy", mmap_mode="r")[z].astype(bool)
    rgb[edge(tum) & big] = (255, 0, 0)
    rgb[edge(tum) & ~big] = (255, 160, 0)
    sy, sx = s.spacing[1], s.spacing[2]
    r0, r1 = int(max(0, y - half_mm / sy)), int(min(im.shape[0], y + half_mm / sy))
    c0, c1 = int(max(0, x - half_mm / sx)), int(min(im.shape[1], x + half_mm / sx))
    crop = Image.fromarray(rgb[r0:r1, c0:c1]).resize((size, int(size * (r1 - r0) * sy / ((c1 - c0) * sx))), Image.NEAREST)
    d = ImageDraw.Draw(crop)
    fy, fx = (y - r0) * crop.height / (r1 - r0), (x - c0) * crop.width / (c1 - c0)
    d.line([(fx - 16, fy), (fx - 6, fy)], fill=(255, 0, 255), width=3)
    d.line([(fx + 6, fy), (fx + 16, fy)], fill=(255, 0, 255), width=3)
    rec = meta_record(case)["selected"]
    dmax = max(n.diameter_mm for n in s.nodules)
    txt = f"{case} z={z}/{s.shape[0]} {rec['desc'][:24]} {rec['phase']} largest {dmax:.0f} mm"
    d.rectangle([0, 0, 7 * len(txt) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), txt, fill=(255, 255, 0))
    d.text((3, crop.height - 14), "R", fill=(255, 255, 0))
    d.text((crop.width - 10, crop.height - 14), "L", fill=(255, 255, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    crop.save(path)
    return path


if __name__ == "__main__":
    # python -m perr.liver_ext download | convert [case ...] (liverext venv) ; eval (any venv; builds the cache)
    import sys
    from multiprocessing import Pool

    cmd = sys.argv[1:2]
    if cmd == ["download"]:
        download()
    elif cmd == ["convert"]:
        todo = sys.argv[2:] or pids()
        with Pool(int(os.environ.get("NPROC", "16"))) as p:
            for r in p.imap_unordered(convert, todo):
                sel = r.get("selected") or {}
                print(r["case"], sel.get("desc"), sel.get("acq"), sel.get("phase"), sel.get("exact"), sel.get("liver_sd"), flush=True)
    elif cmd == ["eval"]:
        for c in cases():
            why = eligibility(c)
            if why:
                print(c, "excluded:", "; ".join(why))
        print(len(eval_cases(write=True)), "cases ->", EVAL)

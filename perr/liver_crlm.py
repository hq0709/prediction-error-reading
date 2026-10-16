"""Additional liver-tumour training data: TCIA Colorectal-Liver-Metastases (CRLM, CC BY 4.0), portal-venous CT,
cached exactly like liver.load_liver / liver_ext.load_case. Train on usable_cases() (LIST), not on cases().

Source. TCIA collection Colorectal-Liver-Metastases (doi:10.7937/QXK2-QG03): 197 patients with colorectal liver
metastases, one pre-operative contrast CT series per patient (one acquisition each; GE LightSpeed in 195; slice
spacing 5 mm in 76, 1.5 mm in 75, 7.5 mm in 27, 2.5 mm in 14, 0.8-1.6 mm in 5) and one DICOM-SEG (dcmqi) with the
segments Liver, Liver Remnant, Hepatic (veins), Portal (vein) and Tumor_1 ... Tumor_n (1 to 35 per SEG), all MANUAL.
Downloaded without credentials from the public IDC bucket (idc-index 0.13.1, IDC v25; the 197 CT and 197 expert SEG
series, 10.9 GB; the AI-generated BAMF/AIMI SEG series are not used) into ROOT/dicom/<pid>/<study>/<mod>_<series>/.
Case names: PatientID CRLM-CT-1001 -> case CRLM_1001 (prefix_number, like HCC_001). A different collection from the
HCC-TACE-Seg test set (sota/eval_liver_ext.txt): no patient overlap.

Segments overlap (unlike HCC-TACE-Seg): Liver contains every tumour and most of the vessel masks (the rest is the
extrahepatic portal trunk and the hepatic-vein / IVC outflow, which some SEGs follow into the right atrium); Liver
Remnant is a part of Liver. labelsTr therefore stores a bit mask: 1 Liver, 2 Liver remnant, 4 Hepatic vein, 8 Portal
vein, 16 Tumour (any Tumor_k). Some SEGs repeat a tumour under two segment numbers (identical masks, e.g. 35 Tumor_k
segments but 17 distinct tumours in CRLM_1070); the union is unaffected.

DICOM -> NIfTI (convert(); needs pydicom, run with /raid/home/CAMCA/hj880/venvs/liverext/bin/python). Following
liver_ext, the masks are placed by 3D position, not by the SEG's references: each SEG frame goes to the CT through
world coordinates (frame IPP/IOP -> nearest CT slice along the slice normal, in-plane by IPP difference, nearest
neighbour). Candidates are the CT acquisitions (series, AcquisitionNumber) of the SEG's study with the SEG's pixel
spacing, an axial orientation and uniform slice spacing (exactly one per patient here); exact = every frame with
mask content lies within Z_TOL of a CT slice and >= 99 % of the mask lands inside (all 197; on this collection the
SEG grid is the CT grid and every frame reference agrees with its position, CRLM_1074 included, whose CT and SEG
both carry a 180-degree in-plane rotation). The fit is then checked on the image, in LPS index terms:
  fit_vessel: mean HU of the vessel masks (hepatic | portal, inside Liver) under every shift dz in -2..2 slices,
     dy, dx in -3..3 pixels; at the true placement the bright vessels are covered best. 190 / 197 peak at 0, the
     other 7 within one pixel in-plane (CRLM_1074 at dy = dx = +1, +5 HU: a one-pixel offset along both rotated
     axes, kept as is: < 1 mm, well inside the 3 mm target tolerance); none prefers another slice.
  fit_sd: SD of the liver parenchyma (Liver minus tumours and vessel masks) under the placement, z +-1/2 slices,
     y/x +-2/4 pixels, left-right / antero-posterior mirroring about the image centre and cranio-caudal mirroring
     about the liver's extent. Gross misplacement raises it far above the 12-45 HU of the placements; on thin slices
     a one-slice caudal shift can lower it by a few HU because contours at the dome include partial-volume lung
     (the vessel test shows these masks are not displaced), so it is used for the mirrors and SD_MAX only.
Phase from the expert vessel masks: P / H = median HU of the portal / hepatic-vein masks' in-plane cores inside the
liver (all voxels of the mask if the core has < 20), L = median HU of the parenchyma; see _phase. In the portal-
venous phase the hepatic veins are opacified above the parenchyma; before it (late arterial / portal inflow) only
the portal vein is. All 197 have P >= 106 HU and P - L >= 20 HU; H - L >= 14 HU in all but CRLM_1072 (-1 HU:
unopacified hepatic veins, aorta about 282 HU against P 144 HU, corticomedullary kidneys: an early acquisition).
Written: NIFTI/imagesTr/<case>.nii.gz (int16 HU, clipped to [-1024, 3071]), NIFTI/labelsTr/<case>.nii.gz (uint8 bit
mask), NIFTI/meta/<case>.json (record: selection, HU statistics, phase, fits, tumour segments and components).

Cache (load_case(); numpy/scipy/SimpleITK only), exactly as liver_ext.load_case: vol int16 [z, y, x] in LPS
(SimpleITK.DICOMOrient), liver uint8 = Liver | Tumour with enclosed axial holes filled (solid like the LiTS masks;
Liver already includes the tumours and intrahepatic vessels, filling adds a median 0.01 %, max 1.1 %), tumour =
Tumour bit, 26-connected components, equivalent diameter (6 V / pi)^(1/3), big / small = components >= / < 10 mm
grown by a 3 mm ball, meta = {"spacing": (sz, sy, sx), "nodules": [...]}.

Usable list (usable_cases(), written to LIST; see eligibility()): 194 of 197. Excluded: CRLM_1072 (early phase,
above), CRLM_1187 (a 40 mm tumour mask in the right lung above the dome, median -868 HU: no lesion there) and
CRLM_1034 (tumour masks in the lung above the dome, -726 HU, and in abdominal-wall fat, -77 HU). Only
these two have tumour voxels in lung beyond a few dome partial-volume voxels (<= 0.05 % elsewhere). Contact sheets
of all 197 (axial through the largest tumour, coronal and sagittal through the liver) were checked by eye and added
nothing to EXCLUDE. Known, kept: CRLM_1027's top (7.5 mm) slice has a comb-shaped liver contour; in 28 cases the
liver reaches the first or last CT slice (scan coverage); on thick slices the dome's last partial-volume slice is
sometimes left unlabelled (of 8 cases with a broad top labelled slice checked by eye, CRLM_1035 and CRLM_1165; the
others have heart or lung above). The cache holds the usable cases only (load_case() builds any other on demand).
Usable set: 514 tumour components, 375 >= 10 mm (median 18.9 mm) and 139 smaller; median 14.3 mm (IQR 9.8-24.0,
1.2-128 mm); 186 cases with a target, 8 without; 1-17 components per case (median 2).
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .liver import MIN_DIAM, TOL_MM, has_target  # noqa: F401  (same target definition as the MSD world)
from .study import Nodule, Study

ROOT = "/raid/home/CAMCA/hj880/data_ext/crlm"
DICOM = f"{ROOT}/dicom"
NIFTI = f"{ROOT}/nifti"
CACHE = "/raid/home/CAMCA/hj880/perr/cache/liver_crlm"
LIST = "/raid/home/CAMCA/hj880/perr/results/crlm_cases.txt"
QC = "/raid/home/CAMCA/hj880/perr/results/crlm_qc"
LIVER, REMNANT, HEPATIC, PORTAL, TUMOUR = 1, 2, 4, 8, 16      # bits of labelsTr
SEGMENT_BITS = {"Liver": LIVER, "Liver Remnant": REMNANT, "Hepatic": HEPATIC, "Portal": PORTAL}
Z_TOL = 0.05          # mm, frame position -> CT slice for an exact grid match
SD_MAX = 45.0         # HU, parenchyma SD of a usable placement (as liver_ext)
VESSEL_HU_GAP = 10.0  # HU, hepatic veins above the parenchyma in the portal-venous phase
MISPLACED_HU = -50.0  # HU, a tumour component whose median is below this lies in air or fat
EXCLUDE: dict[str, str] = {}   # case -> reason; visual QC of all 197 found none beyond the automatic criteria


def pid_to_case(pid: str) -> str:
    return "CRLM_" + pid.rsplit("-", 1)[1]


def case_to_pid(case: str) -> str:
    return "CRLM-CT-" + case.split("_", 1)[1]


def download() -> None:
    """Fetch the collection from the public IDC bucket (no credentials): the 197 CT series and the 197 expert SEG
    series of colorectal_liver_metastases (AI-generated analysis-result SEGs excluded). Needs idc-index."""
    from idc_index import IDCClient

    c = IDCClient()
    d = c.index
    d = d[(d.collection_id == "colorectal_liver_metastases") & d.analysis_result_id.isna()]
    os.makedirs(DICOM, exist_ok=True)
    d.to_csv(f"{ROOT}/series_index.csv", index=False)
    c.download_from_selection(downloadDir=DICOM, seriesInstanceUID=list(d.SeriesInstanceUID),
                              dirTemplate="%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID",
                              quiet=False, show_progress_bar=False, use_s5cmd_sync=True)


def pids() -> list[str]:
    return sorted(p for p in os.listdir(DICOM) if p.startswith("CRLM-CT-"))


def cases() -> list[str]:
    """Converted cases (NIfTI written); usable_cases() is the training list."""
    d = f"{NIFTI}/labelsTr"
    return sorted(f[:-7] for f in os.listdir(d) if f.endswith(".nii.gz")) if os.path.isdir(d) else []


def meta_record(case: str) -> dict:
    with open(f"{NIFTI}/meta/{case}.json") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------- DICOM -> NIfTI

def _vessel_hu(hu: dict, name: str):
    core = hu.get(f"{name}_core_in")
    if core and core["vox"] >= 20:
        return core["median"]
    return (hu.get(name) or {}).get("median")


def _phase(hu: dict) -> str:
    """Phase from the expert vessel masks (P portal vein, H hepatic veins, L parenchyma; medians in HU)."""
    p, h, l = _vessel_hu(hu, "portal"), _vessel_hu(hu, "hepatic"), (hu.get("parenchyma") or {}).get("median")
    if p is None or h is None or l is None:
        return "unknown"
    if p < 100 and h < 100:
        return "pre-contrast"
    if p >= 100 and h - l >= VESSEL_HU_GAP:
        return "portal-venous"
    if p >= 100 and p - l >= 20:
        return "arterial"          # portal vein opacified, hepatic veins not yet (late arterial / portal inflow)
    return "indeterminate"


def _read_seg(pydicom, path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    sh = ds.SharedFunctionalGroupsSequence[0]
    iop = np.array([float(x) for x in sh.PlaneOrientationSequence[0].ImageOrientationPatient])
    ps = np.array([float(x) for x in sh.PixelMeasuresSequence[0].PixelSpacing])  # (row spacing, column spacing)
    seg = {}   # segment number -> (bit, tumour segment index or 0); mapped by label, not by number
    for s in ds.SegmentSequence:
        m = re.fullmatch(r"Tumor_(\d+)", str(s.SegmentLabel))
        seg[int(s.SegmentNumber)] = (TUMOUR, int(m.group(1))) if m else (SEGMENT_BITS[str(s.SegmentLabel)], 0)
    frames = []
    for i, g in enumerate(ds.PerFrameFunctionalGroupsSequence):
        bit, inst = seg[int(g.SegmentIdentificationSequence[0].ReferencedSegmentNumber)]
        ipp = np.array([float(x) for x in g.PlanePositionSequence[0].ImagePositionPatient])
        frames.append((bit, inst, ipp, i))
    labels = {int(s.SegmentNumber): str(s.SegmentLabel) for s in ds.SegmentSequence}
    ref = [str(r.SeriesInstanceUID) for r in ds.get("ReferencedSeriesSequence", [])]
    return arr, iop, ps, frames, labels, ref


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


def _map_frames(arr, frames, iop_s, ps, a):
    """Bit-mask label volume on acquisition a's grid (world-coordinate mapping, nearest neighbour), the flat voxel
    indices of every tumour segment, coverage (fraction of mask pixels landing inside) and the largest
    frame-to-slice distance."""
    rs, cs = iop_s[:3], iop_s[3:]
    rc, cc = a["iop"][:3], a["iop"][3:]
    lab = np.zeros((len(a["z"]), a["R"], a["C"]), np.uint8)
    inst: dict[int, list] = {}
    tot = inside = 0
    zres_max = 0.0
    for bit, k_inst, ipp, i in frames:
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
        d = w - a["ipp"][k][None]
        col = np.rint(d @ rc / ps[1]).astype(int)
        row = np.rint(d @ cc / ps[0]).astype(int)
        ok = (row >= 0) & (row < a["R"]) & (col >= 0) & (col < a["C"])
        inside += int(ok.sum())
        lab[k, row[ok], col[ok]] |= bit
        if k_inst:
            inst.setdefault(k_inst, []).append(np.ravel_multi_index((np.full(int(ok.sum()), k), row[ok], col[ok]), lab.shape))
    inst = {k: np.unique(np.concatenate(v)) for k, v in inst.items()}
    return lab, inst, (inside / tot if tot else 0.0), zres_max


def _read_ct(pydicom, a):
    vol = np.zeros((len(a["files"]), a["R"], a["C"]), np.int16)
    for k, f in enumerate(a["files"]):
        c = pydicom.dcmread(f)
        hu = np.rint(c.pixel_array * float(c.get("RescaleSlope", 1)) + float(c.get("RescaleIntercept", 0)))
        vol[k] = np.clip(hu, -1024, 3071)
    return vol


def _lps(x, a):
    """Acquisition-ordered [slice, row, column] array -> LPS index order (axial IOPs with +-1 cosines only)."""
    r, c, n = a["iop"][:3], a["iop"][3:], a["n"]
    x = x if n[2] > 0 else x[::-1]
    x = x if c[1] > 0 else x[:, ::-1]      # rows -> posterior
    return x if r[0] > 0 else x[:, :, ::-1]  # columns -> patient left


def _stat(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return {"mean": round(float(x.mean()), 1), "median": round(float(np.median(x)), 1), "sd": round(float(x.std()), 1),
            "vox": int(x.size)}


def _core(m):
    """In-plane erosion by one pixel (vessel cores, less partial volume)."""
    st = np.zeros((3, 3, 3), bool)
    st[1] = ndi.generate_binary_structure(2, 1)
    return ndi.binary_erosion(m, st)


def _parenchyma(lab):
    return ((lab & LIVER) > 0) & ((lab & (TUMOUR | HEPATIC | PORTAL)) == 0)


def _shifted(lab, dz=0, dy=0, dx=0):
    out = np.zeros_like(lab)
    src = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip((dz, dy, dx), lab.shape))
    dst = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip((dz, dy, dx), lab.shape))
    out[dst] = lab[src]
    return out


def _fit_sd(vol, lab):
    """Parenchyma HU SD under the placement and under alternative placements (see module docstring)."""
    alts = {"0": lab}
    for s in (-2, -1, 1, 2):
        alts[f"z{s:+d}"] = _shifted(lab, dz=s)
    for s in (-4, -2, 2, 4):
        alts[f"y{s:+d}"] = _shifted(lab, dy=s)
        alts[f"x{s:+d}"] = _shifted(lab, dx=s)
    alts["flip_lr"] = lab[:, :, ::-1]
    alts["flip_ap"] = lab[:, ::-1, :]
    zz = np.nonzero((lab & LIVER).any(axis=(1, 2)))[0]
    fz = lab.copy()
    fz[zz[0]:zz[-1] + 1] = lab[zz[0]:zz[-1] + 1][::-1]
    alts["flip_si"] = fz
    out = {}
    for k, l in alts.items():
        m = _parenchyma(l)
        out[k] = round(float(vol[m].std()), 1) if m.sum() > 1000 else None
    return out


def _fit_vessel(vol, lab):
    """Mean HU of the vessel masks (inside Liver) under shifts dz in -2..2, dy and dx in -3..3 (LPS index terms)."""
    ves = ((lab & (HEPATIC | PORTAL)) > 0) & ((lab & LIVER) > 0)
    if ves.sum() < 100:
        return None
    zz, yy, xx = (np.nonzero(ves.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1)))
    sl = tuple(slice(max(0, a.min() - p), a.max() + p + 1) for a, p in zip((zz, yy, xx), (2, 3, 3)))
    v, m = vol[sl].astype(np.float32), ves[sl].astype(np.uint8)
    hu = {}
    for dz in range(-2, 3):
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ms = _shifted(m, dz, dy, dx).astype(bool)
                if ms.sum() >= 0.9 * m.sum():
                    hu[(dz, dy, dx)] = float(v[ms].mean())
    best = max(hu, key=hu.get)
    return {"at0": round(hu[(0, 0, 0)], 1), "best": list(best), "gain": round(hu[best] - hu[(0, 0, 0)], 1),
            "z-1": round(hu[(-1, 0, 0)] - hu[(0, 0, 0)], 1), "z+1": round(hu[(1, 0, 0)] - hu[(0, 0, 0)], 1)}


def _tumour_record(vol, lab, inst, spacing):
    """Tumour segments (duplicates merged when they share >= 50 % of the smaller), 26-connected components of the
    union (as cached), and components lying in air or fat (median HU < MISPLACED_HU: not a lesion)."""
    keys = sorted(inst)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = np.intersect1d(inst[a], inst[b], assume_unique=True).size
            if n and n >= 0.5 * min(inst[a].size, inst[b].size):
                parent[find(b)] = find(a)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    tum = (lab & TUMOUR) > 0
    liv = (lab & LIVER) > 0
    comp, ncomp = ndi.label(tum, structure=np.ones((3, 3, 3)))
    vv = float(np.prod(spacing))
    comps, misplaced = [], []
    for i, sl in enumerate(ndi.find_objects(comp), 1):
        m = comp[sl] == i
        d = (6 * float(m.sum()) * vv / np.pi) ** (1 / 3)
        c = np.argwhere(m).mean(0) + np.array([s.start for s in sl])
        rec = {"zyx": [round(float(x), 1) for x in c], "diam": round(d, 1), "median_hu": float(np.median(vol[sl][m])),
               "outside_liver": round(float((m & ~liv[sl]).sum() / m.sum()), 3)}
        comps.append(rec)
        if rec["median_hu"] < MISPLACED_HU:
            misplaced.append(rec)
    return {"tumour_segments": len(keys), "tumour_segments_distinct": len(groups),
            "tumour_duplicates": [g for g in groups.values() if len(g) > 1],
            "tumour_components": comps, "tumour_misplaced": misplaced}


def convert(case: str, overwrite: bool = False) -> dict:
    """DICOM CT + SEG -> NIfTI pair and record (see module docstring). Needs pydicom."""
    import pydicom

    out_json = f"{NIFTI}/meta/{case}.json"
    if os.path.exists(out_json) and not overwrite:
        return meta_record(case)
    pid = case_to_pid(case)
    seg_files = glob.glob(f"{DICOM}/{pid}/*/SEG_*/*.dcm")
    if len(seg_files) != 1:
        raise RuntimeError(f"{case}: {len(seg_files)} SEG files")
    arr, iop_s, ps, frames, labels, ref = _read_seg(pydicom, seg_files[0])
    study_dir = os.path.dirname(os.path.dirname(seg_files[0]))
    n_s = np.cross(iop_s[:3], iop_s[3:])
    content_z = sorted({round(float(ipp @ n_s), 3) for _, _, ipp, i in frames if arr[i].any()})
    rec = {"case": case, "pid": pid, "seg_file": os.path.relpath(seg_files[0], ROOT), "segments": labels,
           "seg_referenced_series": ref, "seg_iop": [round(float(x), 4) for x in iop_s],
           "seg_pixel_spacing": [float(x) for x in ps], "seg_frames": len(frames),
           "content_z_range": [content_z[0], content_z[-1]], "candidates": []}
    best = None
    for a in _acquisitions(pydicom, study_dir, ps):
        if not a["uniform"]:
            rec["candidates"].append({"series": a["series"], "acq": a["acq"], "desc": a["desc"], "n": len(a["z"]),
                                      "skipped": "non-uniform slice spacing"})
            continue
        lab, inst, cov, zres = _map_frames(arr, frames, iop_s, ps, a)
        c = {"series": a["series"], "acq": a["acq"], "desc": a["desc"], "n": len(a["z"]), "dz": round(a["dz"], 3),
             "thickness": a["thickness"], "coverage": round(cov, 4), "zres_max": round(zres, 3),
             "exact": bool(zres <= Z_TOL and cov >= 0.99)}
        rec["candidates"].append(c)
        if best is None or (c["exact"], cov) > (best[0]["exact"], best[0]["coverage"]):
            best = (c, a, lab, inst)
    os.makedirs(f"{NIFTI}/meta", exist_ok=True)
    if best is None:
        rec["selected"] = None
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=1)
        return rec
    c, a, lab, inst = best
    vol = _read_ct(pydicom, a)
    # statistics and fits in LPS index order (views, no copies)
    V, Lb = _lps(vol, a), _lps(lab, a)
    liv, tum = (Lb & LIVER) > 0, (Lb & TUMOUR) > 0
    hep, por = (Lb & HEPATIC) > 0, (Lb & PORTAL) > 0
    c["hu"] = {"parenchyma": _stat(V[_parenchyma(Lb)]), "tumour": _stat(V[tum]),
               "portal": _stat(V[por]), "hepatic": _stat(V[hep]),
               "portal_core_in": _stat(V[_core(por) & liv]), "hepatic_core_in": _stat(V[_core(hep) & liv])}
    c["phase"] = _phase(c["hu"])
    c["fit_sd"] = _fit_sd(V, Lb)
    c["fit_vessel"] = _fit_vessel(V, Lb)
    rec["selected"] = c
    # liver extent (LPS z: up = superior): first / last labelled slice, end areas, HU just beyond each end
    area = liv.sum(axis=(1, 2))
    zz = np.nonzero(area)[0]
    rec["liver_slices"] = [int(zz[0]), int(zz[-1]), int(len(a["z"]))]
    rec["liver_at_scan_end"] = bool(zz[0] == 0 or zz[-1] == len(a["z"]) - 1)
    rec["liver_end_area"] = [round(float(area[zz[0]] / area.max()), 3), round(float(area[zz[-1]] / area.max()), 3)]
    rec["liver_beyond_hu"] = [round(float(np.median(V[nb][liv[k]])), 1) if 0 <= nb < len(a["z"]) else None
                              for k, nb in ((zz[0], zz[0] - 1), (zz[-1], zz[-1] + 1))]
    filled = np.stack([ndi.binary_fill_holes(s) for s in (liv | tum)])
    rec["fill_frac"] = round(float((filled & ~(liv | tum)).sum() / filled.sum()), 4)
    inst_lps = {}
    for k, idx in inst.items():
        m = np.zeros(lab.shape, bool)
        m.ravel()[idx] = True
        inst_lps[k] = np.flatnonzero(_lps(m, a))
    rec.update(_tumour_record(V, Lb, inst_lps, (a["dz"], float(ps[0]), float(ps[1]))))
    rec["ct_z_range"] = [round(float(a["z"][0]), 3), round(float(a["z"][-1]), 3)]
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((float(ps[1]), float(ps[0]), float(a["dz"])))
    img.SetOrigin(tuple(float(x) for x in a["ipp"][0]))
    r, cdir, n = a["iop"][:3], a["iop"][3:], a["n"]
    img.SetDirection(tuple(float(x) for x in (r[0], cdir[0], n[0], r[1], cdir[1], n[1], r[2], cdir[2], n[2])))
    limg = sitk.GetImageFromArray(lab)
    limg.CopyInformation(img)
    for d in ("imagesTr", "labelsTr"):
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
        tum = (lab & TUMOUR) > 0
        liver = np.stack([ndi.binary_fill_holes(s) for s in (((lab & LIVER) > 0) | tum)])
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


# --------------------------------------------------------------------------------------------- usable list

def eligibility(case: str) -> list[str]:
    """Reasons a converted case is not usable (empty list = usable): exact grid; portal-venous phase; parenchyma SD
    <= SD_MAX and every mirrored placement worse; vessel masks brightest at the placement (same slice, within one
    pixel in-plane); no tumour component in air or fat; at least one tumour; not in EXCLUDE (visual QC)."""
    rec = meta_record(case)
    sel = rec.get("selected")
    if not sel:
        return ["no CT acquisition matches the SEG"]
    why = []
    if not sel["exact"]:
        why.append(f"inexact grid (zres {sel['zres_max']} mm, coverage {sel['coverage']})")
    if sel["phase"] != "portal-venous":
        hu = sel["hu"]
        why.append(f"phase {sel['phase']} (P {_vessel_hu(hu, 'portal')}, H {_vessel_hu(hu, 'hepatic')}, "
                   f"L {hu['parenchyma']['median']} HU)")
    sd = sel["hu"]["parenchyma"]["sd"]
    if sd > SD_MAX:
        why.append(f"parenchyma SD {sd} HU > {SD_MAX}")
    mirrors = [k for k in ("flip_lr", "flip_ap", "flip_si") if (sel["fit_sd"].get(k) or 1e9) <= sd]
    if mirrors:
        why.append(f"a mirrored placement fits as well {mirrors}")
    fv = sel.get("fit_vessel")
    if fv is None:
        why.append("no vessel masks to check the fit")
    elif fv["best"][0] != 0 or max(abs(fv["best"][1]), abs(fv["best"][2])) > 1:
        why.append(f"vessel masks fit shifted by {fv['best']} (dz, dy, dx) better (+{fv['gain']} HU)")
    if rec["tumour_misplaced"]:
        why.append("tumour mask in air or fat: " + ", ".join(f"{t['diam']} mm at {t['zyx']} median {t['median_hu']} HU"
                                                          for t in rec["tumour_misplaced"]))
    if not rec["tumour_components"]:
        why.append("no tumour")
    if case in EXCLUDE:
        why.append(EXCLUDE[case])
    return why


def usable_cases(write: bool = False) -> list[str]:
    out = [c for c in cases() if not eligibility(c)]
    if write:
        os.makedirs(os.path.dirname(LIST), exist_ok=True)
        with open(LIST, "w") as f:
            f.write("\n".join(out) + "\n")
    return out


# --------------------------------------------------------------------------------------------- visual QC

def _labels_lps(case: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(f"{NIFTI}/labelsTr/{case}.nii.gz"), "LPS"))


def qc_png(case: str, path: str, zyx=None, half_mm: float = 110.0, tile: int = 360, vessels: bool = True) -> str:
    """Axial, coronal and sagittal slices through zyx (default: the largest tumour's centroid), soft-tissue window,
    from the cache (LPS): liver outline green, tumour red (>= 10 mm) / orange (< 10 mm), portal vein blue and hepatic
    veins cyan (labelsTr), the point marked magenta. Radiological view: patient right on image left (axial, coronal),
    anterior on image left (sagittal), superior up."""
    s = load_case(case)
    if zyx is None:
        zyx = max(s.nodules, key=lambda n: n.diameter_mm).zyx
    z, y, x = (int(round(v)) for v in zyx)
    vol = np.asarray(s.vol)
    liver = np.asarray(s.lung).astype(bool)
    big = np.load(f"{CACHE}/{case}.big.npy").astype(bool)
    lab = _labels_lps(case)
    tum = (lab & TUMOUR) > 0
    sz, sy, sx = s.spacing

    def edge(m):
        return m & ~ndi.binary_erosion(m)

    def view(take, rows_mm, cols_mm, flip_rows, centre, title, letters):
        im = take(vol).astype(np.float32)
        rgb = np.stack([np.clip((im - (50 - 175)) / 350 * 255, 0, 255).astype(np.uint8)] * 3, -1)
        layers = [(edge(take(liver)), (0, 255, 0))]
        if vessels:
            layers += [(edge(take((lab & PORTAL) > 0)), (60, 120, 255)), (edge(take((lab & HEPATIC) > 0)), (0, 230, 230))]
        t, b = take(tum), take(big)
        layers += [(edge(t) & b, (255, 0, 0)), (edge(t) & ~b, (255, 160, 0))]
        for m, col in layers:
            rgb[m] = col
        r0, c0 = centre
        if flip_rows:
            rgb, r0 = rgb[::-1], rgb.shape[0] - 1 - r0
        hr, hc = half_mm / rows_mm, half_mm / cols_mm
        a0, a1 = int(max(0, r0 - hr)), int(min(rgb.shape[0], r0 + hr + 1))
        b0, b1 = int(max(0, c0 - hc)), int(min(rgb.shape[1], c0 + hc + 1))
        crop = rgb[a0:a1, b0:b1]
        sc = tile / (2 * half_mm)
        out = Image.new("RGB", (tile, tile), (0, 0, 0))
        im2 = Image.fromarray(crop).resize((max(1, int(round(crop.shape[1] * cols_mm * sc))),
                                            max(1, int(round(crop.shape[0] * rows_mm * sc)))), Image.NEAREST)
        out.paste(im2, (int(round((half_mm - (c0 - b0) * cols_mm) * sc)), int(round((half_mm - (r0 - a0) * rows_mm) * sc))))
        d = ImageDraw.Draw(out)
        m = tile // 2
        d.line([(m - 18, m), (m - 7, m)], fill=(255, 0, 255), width=2)
        d.line([(m + 7, m), (m + 18, m)], fill=(255, 0, 255), width=2)
        d.rectangle([0, 0, 7 * len(title) + 6, 13], fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 0))
        top, bottom, left, right = letters
        d.text((m - 3, 16), top, fill=(255, 255, 0))
        d.text((m - 3, tile - 13), bottom, fill=(255, 255, 0))
        d.text((3, m - 6), left, fill=(255, 255, 0))
        d.text((tile - 10, m - 6), right, fill=(255, 255, 0))
        return out

    panels = [view(lambda a: a[z], sy, sx, False, (y, x), f"axial z={z}/{vol.shape[0]}", ("A", "P", "R", "L")),
              view(lambda a: a[:, y, :], sz, sx, True, (z, x), f"coronal y={y}", ("S", "I", "R", "L")),
              view(lambda a: a[:, :, x], sz, sy, True, (z, y), f"sagittal x={x}", ("S", "I", "A", "P"))]
    sheet = Image.new("RGB", (3 * tile + 8, tile + 16), (30, 30, 30))
    for i, p in enumerate(panels):
        sheet.paste(p, (i * (tile + 4), 16))
    d = ImageDraw.Draw(sheet)
    rec = meta_record(case)["selected"]
    dmax = max(n.diameter_mm for n in s.nodules)
    nbig = sum(n.diameter_mm >= MIN_DIAM for n in s.nodules)
    d.text((4, 2), f"{case}  spacing {sz:.2f}/{sy:.2f}/{sx:.2f} mm  {rec['phase']}  {len(s.nodules)} tumour components "
                   f"({nbig} >= 10 mm, largest {dmax:.0f} mm)  parenchyma SD {rec['hu']['parenchyma']['sd']} HU",
           fill=(255, 255, 255))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
    return path


if __name__ == "__main__":
    # python -m perr.liver_crlm download | convert [case ...] (liverext venv) ; cache | list | qc [case ...] (any venv)
    import sys
    from multiprocessing import Pool

    cmd, args = sys.argv[1:2], sys.argv[2:]
    nproc = int(os.environ.get("NPROC", "12"))
    if cmd == ["download"]:
        download()
    elif cmd == ["convert"]:
        with Pool(nproc) as p:
            for r in p.imap_unordered(convert, args or [pid_to_case(x) for x in pids()]):
                sel = r.get("selected") or {}
                print(r["case"], sel.get("dz"), sel.get("exact"), sel.get("phase"), sel.get("fit_vessel"), flush=True)
    elif cmd == ["cache"]:
        with Pool(nproc) as p:
            for s in p.imap_unordered(load_case, args or cases()):
                print(s.case_id, len(s.nodules), flush=True)
    elif cmd == ["list"]:
        for c in cases():
            why = eligibility(c)
            if why:
                print(c, "excluded:", "; ".join(why))
        print(len(usable_cases(write=True)), "cases ->", LIST)
    elif cmd == ["qc"]:
        for c in args:
            print(qc_png(c, f"{QC}/{c}.png"))

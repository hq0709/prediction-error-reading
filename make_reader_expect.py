"""Reader strips that also show the world model's expectation: under each candidate's strip (what is there: axial with
the measured outline, coronal, sagittal, 40 mm each), a second row of the same three views rendered from what the
anatomy world model predicted at every voxel from its surroundings (expected = observed - residual, tag <tag>R on the
1 mm grid; same window, field of view, orientation and nearest-neighbour enlargement as the harness). A structure the
model could predict (a vessel, an airway wall) is in both rows; a finding is only in the top one. The rows of an
existing reader data folder are copied with the new image and question.

    python make_reader_expect.py data/reader_ctxfree nllfree data/reader_ctxfree_exp
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

from perr.context_wm import ORGANS
from perr.render import window

QUESTION_EXPECT = ("One candidate from a chest CT. Top row, what is there: axial (with measured outline), coronal and "
                   "sagittal views, 40 mm each; magenta ticks mark the exact address. Bottom row: the same three views as "
                   "the anatomy world model expected them from the surrounding lung (what it can predict, such as a "
                   "vessel, appears in both rows; what it cannot predict appears only in the top row). Is the structure "
                   "at the marked address a lung nodule at least 6 mm across (a compact blob, round in all three planes), "
                   "rather than a vessel, airway, chest wall or a smaller nodule? Answer with one word: yes or no.")
TILE, FOV, GAP = 200, 40, 4
ISO = ORGANS["lung"]["iso"]
LO, HI = ORGANS["lung"]["window"]


def tile(img, label):
    """A FOV x FOV mm plane (1 mm pixels, HU) as a TILE x TILE lung-window tile with the harness's label and ticks."""
    k = np.floor((np.arange(TILE) + 0.5) / TILE * img.shape[0]).astype(int)
    im = Image.fromarray(window(img[k][:, k], "lung")).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(label) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), label, fill=(255, 255, 0))
    m = TILE // 2
    d.line([(m - 14, m), (m - 6, m)], fill=(255, 0, 255))
    d.line([(m + 6, m), (m + 14, m)], fill=(255, 0, 255))
    return im


def planes(E, c):
    """Axial, coronal and sagittal FOV-mm planes of E (1 mm grid) through c; rows run superior -> inferior off axial."""
    h = FOV // 2
    z, y, x = (int(round(v)) for v in c)
    pad = np.pad(E, h + 1, mode="constant", constant_values=-1024)
    z, y, x = z + h + 1, y + h + 1, x + h + 1
    ax = pad[z, y - h:y + h, x - h:x + h]
    co = pad[z - h + 1:z + h + 1, y, x - h:x + h][::-1]
    sa = pad[z - h + 1:z + h + 1, y - h:y + h, x][::-1]
    return ax, co, sa


def expected_row(E, c):
    row = Image.new("RGB", (TILE * 3 + 2 * GAP, TILE), (40, 40, 40))
    for i, (p, name) in enumerate(zip(planes(E, c), ("axial", "coronal", "sagittal"))):
        row.paste(tile(p, f"expected {name}"), (i * (TILE + GAP), 0))
    return row


def work(job):
    uid, rows, tag, out = job
    meta = np.load(f"{ISO}/{uid}.meta.npy", allow_pickle=True).item()
    X = np.load(f"{ISO}/{uid}.vol.npy").astype(np.float32)
    R = np.load(f"{ISO}/{uid}.{tag}R.npy").astype(np.float32)
    Xn = np.clip((X - LO) / (HI - LO), 0, 1)
    E = np.where(R != 0, (Xn - R) * (HI - LO) + LO, X)          # outside the domain nothing was predicted: as observed
    res = []
    for r in rows:
        c = np.asarray(r["zyx"], float) * np.asarray(meta["spacing"], float) - np.asarray(meta["lo_mm"], float)
        top = Image.open(r["image"]).convert("RGB")
        sheet = Image.new("RGB", (top.width, top.height + GAP + TILE), (40, 40, 40))
        sheet.paste(top, (0, 0))
        sheet.paste(expected_row(E, c), (0, top.height + GAP))
        path = os.path.abspath(f"{out}/img/{os.path.basename(r['image'])}")
        sheet.save(path)
        res.append({**r, "image": path, "question": QUESTION_EXPECT})
    return res


def main():
    src, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(f"{out}/img", exist_ok=True)
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".jsonl")):
        rows = [json.loads(l) for l in open(f"{src}/{fn}")]
        by = defaultdict(list)
        for r in rows:
            by[r["uid"]].append(r)
        with Pool(24) as p:
            new = [r for rs in p.map(work, [(u, v, tag, out) for u, v in by.items()], chunksize=1) for r in rs]
        with open(f"{out}/{fn}", "w") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        print(fn, len(new), "rows", flush=True)


if __name__ == "__main__":
    main()

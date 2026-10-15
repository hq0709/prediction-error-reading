"""Follow-up reading of brain MRI (LUMIERE): two time points, four sequences, one harness.

The evidence archive and its rules are those of harness.Session. What changes is the study: a view now
names a sequence and a time point, and the same address can be re-rendered at the other time point or in
another sequence. The world model is the disease-course model "no change since the prior": its expected
view is the prior at the same address, and its error is current minus prior (sequences are z-scored per
time point inside the brain, so they can be subtracted).
"""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .harness import ToolError, ViewRecord, MAX_VIEWS_PER_CALL
from .render import Geom, plane_image, plane_shape, plane_spacing, png, sample, square_box, to_voxel, from_voxel, montage_tile
from .lumiere import SEQS

DISPLAY = (-2.5, 4.5)        # z-score range shown as black..white
CHANGE_SCALE = 3.0           # z-score difference shown at full intensity
MONTAGE_N = 9
MONTAGE_GRID = 3


def zwin(a: np.ndarray) -> np.ndarray:
    lo, hi = DISPLAY
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def change_overlay(gray: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Increase in hot (red-yellow), decrease in cool (blue), over the grayscale current image."""
    g = np.repeat(gray[..., None], 3, -1).astype(np.float32)
    up = np.clip(diff / CHANGE_SCALE, 0, 1)
    dn = np.clip(-diff / CHANGE_SCALE, 0, 1)
    hotc = np.stack([np.ones_like(up), np.clip(up * 2 - 0.5, 0, 1), np.zeros_like(up)], -1) * 255
    cool = np.stack([np.zeros_like(dn), np.clip(dn * 1.5 - 0.5, 0, 1), np.ones_like(dn)], -1) * 255
    a_up = np.clip((up - 0.33) * 1.6, 0, 0.85)[..., None]
    a_dn = np.clip((dn - 0.33) * 1.6, 0, 0.7)[..., None]
    out = g * (1 - a_up) + hotc * a_up
    out = out * (1 - a_dn) + cool * a_dn
    return out.astype(np.uint8)


def core_mask(fu, mm: float = 3.0) -> np.ndarray:
    """Brain mask eroded by `mm`, where a subtraction of the two time points is meaningful."""
    return ndi.binary_erosion(np.asarray(fu.brain) > 0, iterations=int(round(mm)))


class FollowSession:
    """cond: oneshot | harness | harness_wm (the world model adds expected views, change maps and peaks)."""

    def __init__(self, fu, cond: str, coord_mode: str = "norm1000", size: int = 512):
        if cond not in {"oneshot", "harness", "harness_wm"}:
            raise ValueError(cond)
        self.s, self.cond, self.mode, self.size = fu, cond, coord_mode, size
        self.use_wm = cond == "harness_wm"
        self.views: dict[str, ViewRecord] = {}
        self.peaks: list[tuple] = []
        self.final = None
        self.core = core_mask(fu)

    def diff(self, seq, plane, index):
        """current - prior on a plane, zero outside the eroded brain."""
        d = plane_image(self.vol("current", seq), plane, index) - plane_image(self.vol("prior", seq), plane, index)
        return d * plane_image(self.core.astype(np.float32), plane, index)

    # coordinates (same conventions as the CT harness)
    def to_disp(self, pt):
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            raise ToolError("A point must be [x, y] with two numbers.")
        x, y = float(pt[0]), float(pt[1])
        return (x / 1000 * self.size, y / 1000 * self.size) if self.mode == "norm1000" else (x, y)

    def from_disp(self, u, v):
        if self.mode == "norm1000":
            return [int(round(u / self.size * 1000)), int(round(v / self.size * 1000))]
        return [int(round(u)), int(round(v))]

    def coord_note(self):
        if self.mode == "norm1000":
            return "x and y from 0 to 1000 across each image (0,0 = top-left, 1000,1000 = bottom-right)"
        return f"pixel coordinates of each image (0,0 = top-left, {self.size},{self.size} = bottom-right)"

    def locate(self, obj):
        if isinstance(obj, dict) and "bbox_2d" in obj:
            b = obj["bbox_2d"]
            if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(t, (int, float)) for t in b)):
                raise ToolError("bbox_2d must be [x1, y1, x2, y2].")
            (u0, v0), (u1, v1) = self.to_disp(b[:2]), self.to_disp(b[2:])
            u0, u1 = sorted((u0, u1))
            v0, v1 = sorted((v0, v1))
            return (u0 + u1) / 2, (v0 + v1) / 2, (u0, v0, u1, v1)
        if isinstance(obj, dict) and "point" in obj:
            u, v = self.to_disp(obj["point"])
            return u, v, None
        raise ToolError('Give the location as "bbox_2d": [x1, y1, x2, y2].')

    # archive
    def add(self, kind, shows, geom, data, source=None):
        rec = ViewRecord(f"V{len(self.views) + 1}", kind, shows, geom, data, source)
        self.views[rec.vid] = rec
        return rec

    def get(self, vid):
        if not isinstance(vid, str) or vid not in self.views:
            raise ToolError(f"Unknown view id {vid!r}. Existing ids: {', '.join(self.views)}.")
        return self.views[vid]

    def label(self, rec):
        g = rec.geom
        _, dc, _ = plane_spacing(g.plane, self.s.spacing)
        fov = (g.box[2] - g.box[0]) * dc
        depth = plane_shape(g.plane, self.s.shape)[2]
        parts = [f'id="{rec.vid}"', f'kind="{rec.kind}"', f'shows="{rec.shows}"',
                 f'time="{g.extra.get("time")}"', f'sequence="{g.extra.get("seq")}"', f'plane="{g.plane}"']
        if g.extra.get("montage"):
            tile = g.size // g.extra["grid"]
            parts += [f'slices="{", ".join(str(k) for k in g.extra["slices"])} (tiles left-to-right, top-to-bottom)"',
                      f'tile_mm_per_px="{fov / tile:.2f}"', f'size="{g.size}x{g.size}"']
        else:
            parts += [f'slice="{g.index} (of 0-{depth - 1})"', f'fov_mm="{fov:.0f}"', f'mm_per_px="{fov / g.size:.2f}"',
                      f'size="{g.size}x{g.size}"']
            if rec.source:
                parts.append(f'of="{rec.source}"')
        return "<view " + " ".join(parts) + ">"

    # rendering
    def vol(self, time, seq):
        return self.s.series[f"{time}:{seq}"]

    def full_box(self, plane):
        R, C, _ = plane_shape(plane, self.s.shape)
        return square_box((0, 0, C, R), plane, self.s.spacing)

    def fov_box(self, plane, c, r, fov_mm):
        dr, dc, _ = plane_spacing(plane, self.s.spacing)
        h = float(fov_mm) / 2
        return (c - h / dc, r - h / dr, c + h / dc, r + h / dr)

    def render(self, g: Geom) -> bytes:
        img = plane_image(self.vol(g.extra["time"], g.extra["seq"]), g.plane, g.index)
        return png(zwin(sample(img, g.box, g.size, fill=-5)))

    def montage_slices(self):
        zs = np.nonzero(np.asarray(self.s.brain).any(axis=(1, 2)))[0]
        lo, hi = int(zs.min()), int(zs.max())
        return [int(round(v)) for v in np.linspace(lo + 0.12 * (hi - lo), hi - 0.12 * (hi - lo), MONTAGE_N)]

    def render_montage(self, g: Geom, change=False) -> bytes:
        grid, tile = g.extra["grid"], g.size // g.extra["grid"]
        canvas = np.zeros((g.size, g.size, 3), np.uint8)
        cur = self.vol("current", g.extra["seq"])
        pri = self.vol("prior", g.extra["seq"])
        for t, k in enumerate(g.extra["slices"]):
            i, j = divmod(t, grid)
            a = sample(plane_image(self.vol(g.extra["time"], g.extra["seq"]), "axial", k), g.box, tile, fill=-5)
            if change:
                d = sample(self.diff(g.extra["seq"], "axial", k), g.box, tile, fill=0)
                rgb = change_overlay(zwin(a), d)
            else:
                rgb = np.repeat(zwin(a)[..., None], 3, -1)
            canvas[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = rgb
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        for t, k in enumerate(g.extra["slices"]):
            i, j = divmod(t, grid)
            d.rectangle([j * tile, i * tile, j * tile + 52, i * tile + 15], fill=(0, 0, 0))
            d.text((j * tile + 3, i * tile + 2), f"z={k}", fill=(255, 255, 0))
        return png(np.asarray(im))

    def overview(self):
        sl = self.montage_slices()
        recs = []
        for time in ("current", "prior"):
            for seq in ("CT1", "FLAIR"):
                g = Geom("axial", None, 0, "mri", self.full_box("axial"), self.size,
                         extra={"montage": True, "grid": MONTAGE_GRID, "slices": sl, "time": time, "seq": seq})
                recs.append(self.add("observed", f"MRI {seq}, 3x3 montage of axial slices", g, self.render_montage(g)))
        if self.use_wm:
            for seq in ("CT1", "FLAIR"):
                g = Geom("axial", None, 0, "mri", self.full_box("axial"), self.size,
                         extra={"montage": True, "grid": MONTAGE_GRID, "slices": sl, "time": "current", "seq": seq})
                recs.append(self.add("predicted", f"change since the prior on {seq} (no-change world model error: "
                                     "red-yellow = brighter than the prior, blue = darker)", g,
                                     self.render_montage(g, change=True)))
            self.peaks = self.change_peaks()
        return recs

    def change_peaks(self, k=6, nms_mm=15.0):
        """Largest increases since the prior (CT1 and FLAIR), smoothed over 3 mm, inside the brain."""
        out = []
        for seq in ("CT1", "FLAIR"):
            d = ndi.gaussian_filter((self.vol("current", seq) - self.vol("prior", seq)) * self.core, 1.5) * self.core
            W = d.copy()
            r = int(nms_mm)
            for _ in range(k):
                i = int(np.argmax(W))
                z, y, x = np.unravel_index(i, W.shape)
                v = float(W[z, y, x])
                if v <= 0.5:
                    break
                out.append((seq, float(z), float(y), float(x), v))
                W[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
        out.sort(key=lambda t: -t[4])
        return out[:k]

    def peak_text(self):
        if not self.peaks:
            return ""
        lines = ["Largest increases since the prior (no-change world model errors, candidate addresses, not "
                 'findings); open one with {"plane": "axial", "at": {"peak": n}, "fov_mm": 80}:']
        for n, (seq, z, y, x, v) in enumerate(self.peaks, 1):
            side = "right" if x < self.s.shape[2] / 2 else "left"
            lines.append(f"  peak {n}: {seq} brighter by {v:.1f} SD, axial slice {int(round(z))}, {side} hemisphere")
        return "\n".join(lines)

    # tools
    def _point_voxel(self, at):
        if not isinstance(at, dict):
            raise ToolError('"at" must be {"view": id, "bbox_2d": [...]}' + (' or {"peak": n}' if self.use_wm else "") + ".")
        if "peak" in at:
            if not self.use_wm or not self.peaks:
                raise ToolError("There are no world-model peaks in this setting.")
            n = int(at["peak"])
            if not 1 <= n <= len(self.peaks):
                raise ToolError(f"peak must be 1-{len(self.peaks)}.")
            _, z, y, x, _ = self.peaks[n - 1]
            return (z, y, x)
        rec = self.get(at.get("view"))
        u, v, _ = self.locate(at)
        return to_voxel(rec.geom, u, v, self.s.shape)

    def geom_from_item(self, it) -> Geom:
        if not isinstance(it, dict):
            raise ToolError("Each view item must be an object.")
        time = it.get("time")
        seq = it.get("sequence", it.get("series"))
        for name, val, allowed in (("time", time, ("current", "prior")), ("sequence", seq, SEQS)):
            if val is not None and val not in allowed:
                raise ToolError(f"{name} must be one of {', '.join(allowed)}.")
        Z = self.s.shape[0]
        if "from" in it:
            base = self.get(it["from"]).geom
            ex = dict(base.extra)
            if "bbox_2d" in it and "box" not in it:
                it = {**it, "box": it["bbox_2d"]}
            if ex.get("montage"):
                if "box" not in it:
                    raise ToolError('A montage holds several slices: give a bbox_2d inside one tile, or open a slice '
                                    'with {"plane": "axial", "index": k}.')
                b = it["box"]
                (u0, v0), (u1, v1) = self.to_disp(b[:2]), self.to_disp(b[2:])
                k0, a0, c0_ = montage_tile(base, min(u0, u1), min(v0, v1))
                k1, a1, c1_ = montage_tile(base, max(u0, u1) - 1e-6, max(v0, v1) - 1e-6)
                if k0 != k1:
                    raise ToolError("The box spans several montage tiles; keep it inside one tile.")
                base = Geom("axial", k0, 0, base.window, base.box, base.size, extra={"time": ex["time"], "seq": ex["seq"]})
                it = {**it, "box": [*self.from_disp(a0, c0_), *self.from_disp(a1, c1_)]}
                ex = dict(base.extra)
            plane, index = base.plane, base.index
            depth = plane_shape(plane, self.s.shape)[2]
            if "step" in it:
                index = int(np.clip(index + int(it["step"]), 0, depth - 1))
            box = base.box
            if "box" in it:
                b = it["box"]
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    raise ToolError("bbox_2d must be [x1, y1, x2, y2].")
                (u0, v0), (u1, v1) = self.to_disp(b[:2]), self.to_disp(b[2:])
                u0, u1 = sorted((u0, u1))
                v0, v1 = sorted((v0, v1))
                c0, r0, c1, r1 = base.box
                f = lambda u, lo, hi: lo + u / base.size * (hi - lo)
                box = square_box((f(u0, c0, c1), f(v0, r0, r1), f(u1, c0, c1), f(v1, r0, r1)), plane, self.s.spacing)
            ex = {"time": time or ex.get("time", "current"), "seq": seq or ex.get("seq", "CT1")}
            return Geom(plane, index, 0, "mri", box, self.size, extra=ex)
        plane = it.get("plane")
        if plane not in ("axial", "coronal", "sagittal"):
            raise ToolError('Give "from" (an existing view), or "plane" with "at" or "index".')
        depth = plane_shape(plane, self.s.shape)[2]
        R, C, _ = plane_shape(plane, self.s.shape)
        c, r = C / 2, R / 2
        if "at" in it:
            z, y, x = self._point_voxel(it["at"])
            normal = {"axial": z, "coronal": y, "sagittal": x}[plane]
            index = int(np.clip(round(normal), 0, depth - 1))
            c, r = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[plane]
        elif "index" in it:
            index = int(np.clip(int(it["index"]), 0, depth - 1))
        else:
            raise ToolError('A "plane" item needs "at" or "index".')
        box = self.fov_box(plane, c, r, it["fov_mm"]) if it.get("fov_mm") else self.full_box(plane)
        return Geom(plane, index, 0, "mri", box, self.size, extra={"time": time or "current", "seq": seq or "CT1"})

    def tool_view(self, args):
        items = args.get("views")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_VIEWS_PER_CALL:
            raise ToolError(f'"views" must be a list of 1-{MAX_VIEWS_PER_CALL} items.')
        geoms = [self.geom_from_item(it) for it in items]
        recs = [self.add("observed", f"MRI {g.extra['seq']}", g, self.render(g)) for g in geoms]
        return f"Rendered {', '.join(r.vid for r in recs)} from the raw studies.", recs

    def tool_predict(self, args):
        if not self.use_wm:
            raise ToolError("There is no world model in this setting.")
        vids = args.get("views")
        if not isinstance(vids, list) or not 1 <= len(vids) <= 2:
            raise ToolError('"views" must list 1-2 view ids.')
        recs, notes = [], []
        for vid in vids:
            src = self.get(vid)
            g = src.geom
            if src.kind != "observed" or g.extra.get("montage") or g.extra.get("time") != "current":
                raise ToolError(f"{vid} must be an observed single slice of the current study.")
            seq = g.extra["seq"]
            exp = sample(plane_image(self.vol("prior", seq), g.plane, g.index), g.box, g.size, fill=-5)
            cur = sample(plane_image(self.vol("current", seq), g.plane, g.index), g.box, g.size, fill=-5)
            gp = Geom(g.plane, g.index, 0, "mri", g.box, g.size, extra={"time": "prior", "seq": seq})
            recs.append(self.add("predicted", "expected if nothing changed since the prior (the prior at this address)",
                                 gp, png(zwin(exp)), vid))
            dd = sample(self.diff(seq, g.plane, g.index), g.box, g.size, fill=0)
            recs.append(self.add("predicted", "change since the prior over the current image (red-yellow brighter, "
                                 "blue darker)", g, png(change_overlay(zwin(cur), dd)), vid))
            notes.append(f"{vid}: largest increase {float((cur - exp).max()):.1f} SD, largest decrease {float((exp - cur).max()):.1f} SD")
        return "World-model views: " + ", ".join(r.vid for r in recs) + ". " + "; ".join(notes) + ".", recs

    def _single(self, rec):
        if not rec.evidence:
            raise ToolError(f"{rec.vid} is a predicted view; measure works on observed views only.")
        if rec.geom.extra.get("montage"):
            raise ToolError(f"{rec.vid} is a montage; open the slice first.")
        return rec

    def tool_measure(self, args):
        """Bidimensional (RANO/Macdonald) measurement; with "both_times": true the same box is measured on the
        current and the prior study and the change of the product is reported."""
        if args.get("both_times"):
            rec = self._single(self.get(args.get("view")))
            g = rec.geom
            other = "prior" if g.extra.get("time") == "current" else "current"
            g2 = Geom(g.plane, g.index, 0, "mri", g.box, g.size, extra={**g.extra, "time": other})
            rec2 = self.add("observed", f"MRI {g.extra['seq']}", g2, self.render(g2), rec.vid)
            a = {k: v for k, v in args.items() if k != "both_times"}
            r1, i1 = self._measure_one(rec, a)
            r2, i2 = self._measure_one(rec2, {**a, "view": rec2.vid})
            m1, m2 = json.loads(r1), json.loads(r2)
            cur, pri = (m1, m2) if g.extra.get("time") == "current" else (m2, m1)
            out = {"current": cur, "prior": pri}
            if cur.get("product_mm2") and pri.get("product_mm2"):
                out["product_change_percent"] = round(100 * (cur["product_mm2"] - pri["product_mm2"]) / pri["product_mm2"], 1)
            return json.dumps(out), [rec2] + i1 + i2
        return self._measure_one(self._single(self.get(args.get("view"))), args)

    def _measure_one(self, rec, args):
        """Threshold at half the box maximum (z-score) and take the component under the box centre."""
        g = rec.geom
        if g.plane != "axial":
            raise ToolError("measure needs an axial slice.")
        u, v, box = self.locate(args)
        if box is None:
            raise ToolError("Give a tight bbox_2d around the lesion.")
        img = plane_image(self.vol(g.extra["time"], g.extra["seq"]), "axial", g.index)
        z0, y0, x0 = to_voxel(g, box[0], box[1], self.s.shape)
        _, y1, x1 = to_voxel(g, box[2], box[3], self.s.shape)
        ya, yb = int(max(0, np.floor(min(y0, y1)))), int(min(img.shape[0], np.ceil(max(y0, y1)) + 1))
        xa, xb = int(max(0, np.floor(min(x0, x1)))), int(min(img.shape[1], np.ceil(max(x0, x1)) + 1))
        sub = img[ya:yb, xa:xb]
        if sub.size < 4:
            raise ToolError("The box is too small or outside the image.")
        thr = float(sub.max()) / 2
        m = sub > thr
        lab, n = ndi.label(m)
        cy, cx = int(round((y0 + y1) / 2)) - ya, int(round((x0 + x1) / 2)) - xa
        cy, cx = int(np.clip(cy, 0, sub.shape[0] - 1)), int(np.clip(cx, 0, sub.shape[1] - 1))
        lid = lab[cy, cx] or (np.bincount(lab.ravel())[1:].argmax() + 1 if n else 0)
        if not lid:
            return json.dumps({"view": rec.vid, "result": "nothing brighter than half the box maximum"}), []
        comp = lab == lid
        yy, xx = np.nonzero(comp)
        P = np.stack([xx, yy], 1).astype(float)
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)) if len(P) <= 600 else None
        if D is None:
            P = P[np.random.default_rng(0).choice(len(P), 600, replace=False)]
            D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        i, j = np.unravel_index(int(D.argmax()), D.shape)
        L = float(D[i, j])
        ax = (P[j] - P[i]) / max(L, 1e-6)
        proj = (P - P[i]) @ np.array([-ax[1], ax[0]])
        W = float(proj.max() - proj.min())
        out = {"view": rec.vid, "long_axis_mm": round(L, 1), "perpendicular_mm": round(W, 1),
               "product_mm2": round(L * W, 1), "area_mm2": float(comp.sum())}
        im = Image.open(io.BytesIO(rec.png)).convert("RGB")
        d = ImageDraw.Draw(im)
        edge = comp & ~ndi.binary_erosion(comp)
        for a, b in zip(*np.nonzero(edge)):
            pu, pv = from_voxel(g, (g.index, ya + a, xa + b), self.s.shape)
            d.rectangle([pu - 1, pv - 1, pu + 1, pv + 1], outline=(0, 255, 0))
        ann = self.add("observed", "MRI with the measured outline drawn (green)", g, png(np.asarray(im)), rec.vid)
        out["drawn_on"] = ann.vid
        return json.dumps(out), [ann]

    def tool_report(self, args):
        prog = args.get("progression")
        if not isinstance(prog, bool):
            raise ToolError('"progression" must be true or false.')
        zyx = None
        if prog:
            rec = self.get(args.get("view"))
            if not rec.evidence or rec.geom.extra.get("time") != "current":
                raise ToolError("Locate the progression on an observed view of the current study.")
            u, v, _ = self.locate(args)
            zyx = to_voxel(rec.geom, u, v, self.s.shape)
            if None in zyx:
                raise ToolError("That location is not fixed in 3-D.")
        self.final = {"progression": prog, "zyx": zyx, "rationale": str(args.get("rationale", ""))[:500]}
        return "Report recorded.", []

    def located(self, rec, box, target: str):
        """Archive the focused reader's box as an annotated observed view."""
        if box is None:
            return json.dumps({"view": rec.vid, "target": target, "bbox_2d": None,
                               "result": "the focused reader did not find it on this view"}), []
        (u0, v0), (u1, v1) = self.to_disp(box[:2]), self.to_disp(box[2:])
        im = Image.open(io.BytesIO(rec.png)).convert("RGB")
        ImageDraw.Draw(im).rectangle([min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)], outline=(0, 255, 0), width=2)
        ann = self.add("observed", f"CT/MRI with the located box drawn (green): {target[:60]}", rec.geom, png(np.asarray(im)), rec.vid)
        return json.dumps({"view": rec.vid, "target": target, "bbox_2d": [round(float(t)) for t in box],
                           "drawn_on": ann.vid}), [ann]

    def call(self, name, args):
        fn = {"view": self.tool_view, "predict": self.tool_predict, "measure": self.tool_measure,
              "report": self.tool_report}.get(name)
        if fn is None or (name == "predict" and not self.use_wm):
            raise ToolError(f"Unknown tool {name!r}.")
        if self.cond == "oneshot" and name != "report":
            raise ToolError("Only report is available in this setting.")
        return fn(args if isinstance(args, dict) else {})


def score_followup(fu, final) -> dict:
    truth = fu.rating == "PD"
    out = {"answered": final is not None, "truth_pd": truth, "pred_pd": None, "correct": False, "evidence_ok": None}
    if final is None:
        return out
    out["pred_pd"] = final["progression"]
    out["correct"] = final["progression"] == truth
    if truth and final["progression"] and final["zyx"] is not None:
        z, y, x = (int(round(c)) for c in final["zyx"])
        tumour = ndi.binary_dilation(fu.seg_current > 0, iterations=3)
        Z, Y, X = tumour.shape
        out["evidence_ok"] = bool(0 <= z < Z and 0 <= y < Y and 0 <= x < X and tumour[z, y, x])
    return out

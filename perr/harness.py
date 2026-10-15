"""The reading harness: an evidence archive plus the tools a model uses to read one study.

Adapted from VISTA's session/inspection contracts (see third_party/NOTICE.md):
- every image the model sees is archived under an id with its full geometry, and is labelled with that
  address before it is shown;
- views are re-rendered from the raw volume, never re-encoded from an earlier PNG;
- evidence has a kind: `observed` views can support a finding, `predicted` views (world-model output)
  cannot. A report that cites a predicted view is refused.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .render import (WINDOWS, Geom, from_voxel, hot, montage_tile, overlay, plane_image, plane_shape, plane_spacing,
                     png, sample, square_box, to_voxel, window)
from .worldmodels import ShapePrior, lung_domain

MAX_VIEWS_PER_CALL = 4
MONTAGE_SLICES = 16
MONTAGE_GRID = 2


@dataclass
class ViewRecord:
    vid: str
    kind: str          # observed | predicted
    shows: str
    geom: Geom
    png: bytes
    source: str | None = None

    @property
    def evidence(self) -> bool:
        return self.kind == "observed"


class ToolError(ValueError):
    pass


class Session:
    def __init__(self, study, cond: str, coord_mode: str = "norm1000", size: int = 512, wm=None):
        if cond not in {"oneshot", "harness", "harness_wm"}:
            raise ValueError(cond)
        self.s = study
        self.cond = cond
        self.mode = coord_mode
        self.size = size
        self.wm = wm or ShapePrior()
        self.use_wm = cond.endswith("_wm")
        self.views: dict[str, ViewRecord] = {}
        self.domain = lung_domain(study)
        self.peaks: list[tuple] = []
        self.final: dict | None = None

    # ---------- coordinates ----------
    def to_disp(self, pt) -> tuple[float, float]:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            raise ToolError("A point must be [x, y] with two numbers.")
        x, y = float(pt[0]), float(pt[1])
        if self.mode == "norm1000":
            return x / 1000 * self.size, y / 1000 * self.size
        return x, y

    def locate(self, obj) -> tuple[float, float, tuple | None]:
        """Read a location given as {"bbox_2d": [x1, y1, x2, y2]} (preferred) or {"point": [x, y]}."""
        if isinstance(obj, dict) and isinstance(obj.get("target"), dict) and "bbox_2d" not in obj:
            obj = {**obj, **obj["target"]}
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

    def from_disp(self, u: float, v: float) -> list[int]:
        if self.mode == "norm1000":
            return [int(round(u / self.size * 1000)), int(round(v / self.size * 1000))]
        return [int(round(u)), int(round(v))]

    def coord_note(self) -> str:
        if self.mode == "norm1000":
            return "x and y from 0 to 1000 across each image (0,0 = top-left, 1000,1000 = bottom-right)"
        return f"pixel coordinates of each image (0,0 = top-left, {self.size},{self.size} = bottom-right)"

    # ---------- archive ----------
    def add(self, kind, shows, geom, data, source=None) -> ViewRecord:
        rec = ViewRecord(f"V{len(self.views) + 1}", kind, shows, geom, data, source)
        self.views[rec.vid] = rec
        return rec

    def get(self, vid) -> ViewRecord:
        if not isinstance(vid, str) or vid not in self.views:
            raise ToolError(f"Unknown view id {vid!r}. Existing ids: {', '.join(self.views)}.")
        return self.views[vid]

    def label(self, rec: ViewRecord) -> str:
        g = rec.geom
        dr, dc, dn = plane_spacing(g.plane, self.s.spacing)
        c0, r0, c1, r1 = g.box
        fov = (c1 - c0) * dc
        depth = plane_shape(g.plane, self.s.shape)[2]
        parts = [f'id="{rec.vid}"', f'kind="{rec.kind}"', f'shows="{rec.shows}"', f'plane="{g.plane}"']
        if g.extra.get("montage"):
            tiles = g.size // g.extra["grid"]
            parts += [f'slices="{", ".join(str(k) for k in g.extra["slices"])} (tiles left-to-right, top-to-bottom)"',
                      f'window="{g.window}"', f'tile_mm_per_px="{fov / tiles:.2f}"', f'size="{g.size}x{g.size}"']
            return "<view " + " ".join(parts) + ">"
        parts.append(f'slice="{g.index} (of 0-{depth - 1})"')
        if g.slab:
            parts.append(f'slab_mm="{(2 * g.slab + 1) * dn:.1f}"')
        if rec.source:
            parts.append(f'of="{rec.source}"')
        parts += [f'window="{g.window}"', f'fov_mm="{fov:.0f}"', f'mm_per_px="{fov / g.size:.2f}"',
                  f'size="{g.size}x{g.size}"']
        return "<view " + " ".join(parts) + ">"

    # ---------- rendering ----------
    def full_box(self, plane):
        R, C, _ = plane_shape(plane, self.s.shape)
        return square_box((0, 0, C, R), plane, self.s.spacing)

    def fov_box(self, plane, c, r, fov_mm):
        dr, dc, _ = plane_spacing(plane, self.s.spacing)
        h = float(fov_mm) / 2
        return (c - h / dc, r - h / dr, c + h / dc, r + h / dr)

    def render_observed(self, g: Geom) -> bytes:
        img = plane_image(self.s.vol, g.plane, g.index, g.slab)
        return png(window(sample(img, g.box, g.size), g.window))

    def montage_slices(self) -> list[int]:
        zs = np.nonzero(np.asarray(self.s.lung).any(axis=(1, 2)))[0]
        lo, hi = int(zs.min()), int(zs.max())
        pad = 0.04 * (hi - lo)
        return [int(round(v)) for v in np.linspace(lo + pad, hi - pad, MONTAGE_SLICES)]

    def render_montage(self, g: Geom, errors=None) -> bytes:
        grid, tile = g.extra["grid"], g.size // g.extra["grid"]
        canvas = np.zeros((g.size, g.size, 3), np.uint8)
        for t, k in enumerate(g.extra["slices"]):
            i, j = divmod(t, grid)
            gray = window(sample(plane_image(self.s.vol, "axial", k), g.box, tile), g.window)
            if errors is None:
                rgb = np.repeat(gray[..., None], 3, -1)
            else:
                rgb = overlay(gray, sample(errors[t], g.box, tile, fill=0) / self.wm.scale(self.s))
            canvas[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = rgb
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        for t, k in enumerate(g.extra["slices"]):
            i, j = divmod(t, grid)
            d.rectangle([j * tile, i * tile, j * tile + 52, i * tile + 15], fill=(0, 0, 0))
            d.text((j * tile + 3, i * tile + 2), f"z={k}", fill=(255, 255, 0))
        for q in range(1, grid):
            d.line([(q * tile, 0), (q * tile, g.size)], fill=(90, 90, 90))
            d.line([(0, q * tile), (g.size, q * tile)], fill=(90, 90, 90))
        return png(np.asarray(im))

    # ---------- initial observation ----------
    def overview(self) -> list[ViewRecord]:
        slices = self.montage_slices()
        per = MONTAGE_GRID * MONTAGE_GRID
        pages = [slices[p:p + per] for p in range(0, len(slices), per)]
        recs = []
        for sl in pages:
            g = Geom("axial", None, 0, "lung", self.full_box("axial"), self.size,
                     extra={"montage": True, "grid": MONTAGE_GRID, "slices": sl})
            recs.append(self.add("observed", "CT axial slices, 2x2 montage", g, self.render_montage(g)))
        ys = np.nonzero(np.asarray(self.s.lung).any(axis=(0, 2)))[0]
        g = Geom("coronal", int((ys.min() + ys.max()) / 2), 0, "lung", self.full_box("coronal"), self.size)
        recs.append(self.add("observed", "CT coronal slice through the middle of the lungs", g, self.render_observed(g)))
        if self.use_wm:
            errs = self.wm.slab_errors(self.s, slices)
            for p, sl in enumerate(pages):
                g = Geom("axial", None, 0, "lung", self.full_box("axial"), self.size,
                         extra={"montage": True, "grid": MONTAGE_GRID, "slices": sl})
                recs.append(self.add("predicted", "shape world-model error over the same montage (largest error in the "
                                     "slab each tile stands for; hot = compact structure that is neither tube nor sheet)",
                                     g, self.render_montage(g, errs[p * per:(p + 1) * per])))
            self.peaks = self.wm.peaks(self.s)
        return recs

    def peak_text(self, k: int = 5) -> str:
        if not self.peaks:
            return ""
        top = self.peaks[0][3]
        lines = ["Largest shape world-model errors (candidate addresses to check, not findings); open one with "
                 '{"plane": "axial", "at": {"peak": n}, "fov_mm": 60}:']
        for n, (z, y, x, val) in enumerate(self.peaks[:k], 1):
            side = "right" if x < self.s.shape[2] / 2 else "left"
            lines.append(f"  peak {n}: error {val / top:.2f} of the largest, axial slice {int(round(z))}, {side} side")
        return "\n".join(lines)

    # ---------- tools ----------
    def _window_and_slab(self, it, plane, default_slab=0, default_window="lung"):
        win = it.get("window", default_window)
        if win not in WINDOWS:
            raise ToolError(f"Unknown window {win!r}; use one of {', '.join(WINDOWS)}.")
        if "slab_mm" in it:
            dn = plane_spacing(plane, self.s.spacing)[2]
            slab = max(0, int(round((float(it["slab_mm"]) / dn - 1) / 2)))
        else:
            slab = default_slab
        return win, slab

    def _point_voxel(self, at) -> tuple:
        if not isinstance(at, dict):
            raise ToolError('"at" must be {"view": id, "point": [x, y]}' + (' or {"peak": n}' if self.use_wm else "") + ".")
        if "peak" in at:
            if not self.use_wm or not self.peaks:
                raise ToolError("There are no world-model peaks in this setting.")
            n = int(at["peak"])
            if not 1 <= n <= len(self.peaks):
                raise ToolError(f"peak must be 1-{len(self.peaks)}.")
            z, y, x, _ = self.peaks[n - 1]
            return (z, y, x)
        rec = self.get(at.get("view"))
        u, v, _ = self.locate(at)
        return to_voxel(rec.geom, u, v, self.s.shape)

    def _tile_geom(self, g: Geom, u: float, v: float):
        """A montage point -> (full-size geometry of that tile's slice, u, v in it)."""
        k, uu, vv = montage_tile(g, u, v)
        return Geom("axial", k, 0, g.window, g.box, g.size), uu, vv

    def geom_from_item(self, it) -> Geom:
        if not isinstance(it, dict):
            raise ToolError("Each view item must be an object.")
        Z = self.s.shape[0]
        if "from" in it:
            base = self.get(it["from"]).geom
            if "bbox_2d" in it and "box" not in it:
                it = {**it, "box": it["bbox_2d"]}
            if base.extra.get("montage"):
                if "box" not in it:
                    raise ToolError("A montage holds several slices: give a box inside one tile, or open a slice with "
                                    '{"plane": "axial", "index": k}.')
                b = it["box"]
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    raise ToolError("box must be [x0, y0, x1, y1].")
                (u0, v0), (u1, v1) = self.to_disp(b[:2]), self.to_disp(b[2:])
                g0, a0, c0_ = self._tile_geom(base, min(u0, u1), min(v0, v1))
                g1, a1, c1_ = self._tile_geom(base, max(u0, u1) - 1e-6, max(v0, v1) - 1e-6)
                if g0.index != g1.index:
                    raise ToolError("The box spans several montage tiles; keep it inside one tile.")
                base = g0
                it = {**it, "box": [*self.from_disp(a0, c0_), *self.from_disp(a1, c1_)]}
            plane, index = base.plane, base.index
            depth = plane_shape(plane, self.s.shape)[2]
            if "step" in it:
                index = int(np.clip(index + int(it["step"]), 0, depth - 1))
            box = base.box
            if "box" in it:
                b = it["box"]
                if not (isinstance(b, (list, tuple)) and len(b) == 4):
                    raise ToolError("box must be [x0, y0, x1, y1].")
                u0, v0 = self.to_disp(b[:2])
                u1, v1 = self.to_disp(b[2:])
                u0, u1 = sorted((u0, u1))
                v0, v1 = sorted((v0, v1))
                c0, r0, c1, r1 = base.box
                f = lambda u, lo, hi: lo + u / base.size * (hi - lo)
                box = square_box((f(u0, c0, c1), f(v0, r0, r1), f(u1, c0, c1), f(v1, r0, r1)), plane, self.s.spacing)
            win, slab = self._window_and_slab(it, plane, base.slab, base.window)
            return Geom(plane, index, slab, win, box, self.size)
        plane = it.get("plane")
        if plane not in ("axial", "coronal", "sagittal"):
            raise ToolError('Give "from" (an existing view), or "plane" (axial, coronal or sagittal) with "at" or "index".')
        depth = plane_shape(plane, self.s.shape)[2]
        R, C, _ = plane_shape(plane, self.s.shape)
        c, r = C / 2, R / 2
        if "at" in it:
            z, y, x = self._point_voxel(it["at"])
            normal = {"axial": z, "coronal": y, "sagittal": x}[plane]
            index = int(np.clip(round(normal), 0, depth - 1))
            cc, rr = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[plane]
            c, r = cc, rr
        elif "index" in it:
            index = int(np.clip(int(it["index"]), 0, depth - 1))
        else:
            raise ToolError('A "plane" item needs "at" or "index".')
        box = self.fov_box(plane, c, r, it["fov_mm"]) if it.get("fov_mm") else self.full_box(plane)
        win, slab = self._window_and_slab(it, plane)
        return Geom(plane, index, slab, win, box, self.size)

    def tool_view(self, args):
        items = args.get("views")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_VIEWS_PER_CALL:
            raise ToolError(f'"views" must be a list of 1-{MAX_VIEWS_PER_CALL} items.')
        geoms = [self.geom_from_item(it) for it in items]
        recs = [self.add("observed", "CT", g, self.render_observed(g)) for g in geoms]
        return f"Rendered {', '.join(r.vid for r in recs)} from the raw study.", recs

    def tool_predict(self, args):
        if not self.use_wm:
            raise ToolError("There is no world model in this setting.")
        vids = args.get("views")
        if not isinstance(vids, list) or not 1 <= len(vids) <= 2:
            raise ToolError('"views" must list 1-2 view ids.')
        recs, notes = [], []
        scale = self.wm.scale(self.s)
        for vid in vids:
            src = self.get(vid)
            g = src.geom
            if src.kind != "observed" or g.extra.get("montage") or g.slab:
                raise ToolError(f"{vid} must be an observed single-slice view (not a montage, slab or prediction).")
            exp = sample(self.wm.expected(self.s, g.plane, g.index), g.box, g.size)
            obs = sample(plane_image(self.s.vol, g.plane, g.index), g.box, g.size)
            err = sample(self.wm.error_plane(self.s, g.plane, g.index), g.box, g.size, fill=0)
            recs.append(self.add("predicted", "expected by the shape world model (structures it cannot explain removed)",
                                 g, png(window(exp, g.window)), vid))
            recs.append(self.add("predicted", "shape world-model error over the CT (hot = neither tube nor sheet)", g,
                                 png(overlay(window(obs, g.window), err / scale)), vid))
            v, u = np.unravel_index(int(np.argmax(err)), err.shape)
            notes.append(f"{vid}: largest error {err.max() / scale:.2f} of the study's largest, at {self.from_disp(u + 0.5, v + 0.5)}")
        return "World-model views: " + ", ".join(r.vid for r in recs) + ". " + "; ".join(notes) + ".", recs

    def _single(self, rec: ViewRecord) -> ViewRecord:
        if not rec.evidence:
            raise ToolError(f"{rec.vid} is a predicted view; read and measure work on observed views only.")
        if rec.geom.extra.get("montage"):
            raise ToolError(f"{rec.vid} is a montage; open the slice first.")
        if rec.geom.extra.get("sheet"):
            raise ToolError(f"{rec.vid} is a contact sheet; use the per-peak views listed with it.")
        return rec

    def tool_read(self, args):
        rec = self._single(self.get(args.get("view")))
        g = rec.geom
        img = plane_image(self.s.vol, g.plane, g.index, g.slab)
        b = args.get("box")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            raise ToolError("box must be [x0, y0, x1, y1].")
        (u0, v0), (u1, v1) = self.to_disp(b[:2]), self.to_disp(b[2:])
        c0, r0, c1, r1 = g.box
        cs = sorted((c0 + u0 / g.size * (c1 - c0), c0 + u1 / g.size * (c1 - c0)))
        rs = sorted((r0 + v0 / g.size * (r1 - r0), r0 + v1 / g.size * (r1 - r0)))
        R, C = img.shape
        ca, cb = int(max(0, np.floor(cs[0]))), int(min(C, np.ceil(cs[1])))
        ra, rb = int(max(0, np.floor(rs[0]))), int(min(R, np.ceil(rs[1])))
        if cb <= ca or rb <= ra:
            raise ToolError("The box is outside the image.")
        vals = img[ra:rb, ca:cb].astype(np.float32).ravel()
        dr, dc, _ = plane_spacing(g.plane, self.s.spacing)
        stats = {"view": rec.vid, "voxels": int(vals.size), "area_mm2": round(vals.size * dr * dc, 1),
                 "mean_HU": round(float(vals.mean()), 1), "sd_HU": round(float(vals.std()), 1),
                 "min_HU": int(vals.min()), "p10_HU": int(np.percentile(vals, 10)), "median_HU": int(np.median(vals)),
                 "p90_HU": int(np.percentile(vals, 90)), "max_HU": int(vals.max())}
        if g.slab:
            stats["note"] = "values come from a slab (maximum along the normal), not one slice"
        return json.dumps(stats), []

    def _annotated(self, rec: ViewRecord, draw_fn, shows: str) -> ViewRecord:
        im = Image.open(io.BytesIO(rec.png)).convert("RGB")
        draw_fn(ImageDraw.Draw(im))
        return self.add("observed", shows, rec.geom, png(np.asarray(im)), rec.vid)

    def tool_measure(self, args):
        """Two modes: a caliper between two points, or a seed point that a mechanical region grow
        (threshold at half the local maximum, 3-D connected component) turns into an outline."""
        if "peaks" in args:
            return self._measure_peaks(args["peaks"])
        if "peak" in args:
            if not self.use_wm or not self.peaks:
                raise ToolError("There are no world-model peaks in this setting.")
            n = int(args["peak"])
            if not 1 <= n <= len(self.peaks):
                raise ToolError(f"peak must be 1-{len(self.peaks)}.")
            z, y, x, _ = self.peaks[n - 1]
            g = Geom("axial", int(round(z)), 0, "lung", self.fov_box("axial", x, y, 60), self.size)
            rec = self.add("observed", "CT", g, self.render_observed(g))
            res, recs = self._measure_seed(rec, {"point": self.from_disp(self.size / 2, self.size / 2)})
            return res, [rec] + recs
        rec = self._single(self.get(args.get("view")))
        g = rec.geom
        c0, r0, c1, r1 = g.box
        dr, dc, dn = plane_spacing(g.plane, self.s.spacing)
        if "bbox_2d" in args or "point" in args:
            return self._measure_seed(rec, args)
        pts, disp = [], []
        for key in ("p1", "p2"):
            u, v = self.to_disp(args.get(key))
            disp.append((u, v))
            pts.append((c0 + u / g.size * (c1 - c0), r0 + v / g.size * (r1 - r0)))
        d = float(np.hypot((pts[0][0] - pts[1][0]) * dc, (pts[0][1] - pts[1][1]) * dr))

        def draw(dd):
            dd.line([disp[0], disp[1]], fill=(0, 255, 0), width=2)
            for (u, v) in disp:
                dd.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(0, 255, 0), width=2)
        ann = self._annotated(rec, draw, f"CT with the caliper drawn ({d:.1f} mm)")
        return json.dumps({"view": rec.vid, "distance_mm": round(d, 1), "drawn_on": ann.vid}), [ann]

    def _peak_view(self, n: int) -> ViewRecord:
        z, y, x, _ = self.peaks[n - 1]
        g = Geom("axial", int(round(z)), 0, "lung", self.fov_box("axial", x, y, 60), self.size)
        return self.add("observed", f"CT centred on world-model peak {n}", g, self.render_observed(g))

    def _plane_tile(self, plane: str, zyx, fov: float, tile: int, label: str) -> Image.Image:
        z, y, x = zyx
        Z = self.s.shape[0]
        idx = {"axial": z, "coronal": y, "sagittal": x}[plane]
        c, r = {"axial": (x, y), "coronal": (x, Z - 1 - z), "sagittal": (y, Z - 1 - z)}[plane]
        g = Geom(plane, int(round(idx)), 0, "lung", self.fov_box(plane, c, r, fov), tile)
        img = Image.open(io.BytesIO(self.render_observed(g))).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8 * len(label) + 6, 14], fill=(0, 0, 0))
        d.text((3, 1), label, fill=(255, 255, 0))
        m = tile // 2
        d.line([(m - 14, m), (m - 6, m)], fill=(255, 0, 255))
        d.line([(m + 6, m), (m + 14, m)], fill=(255, 0, 255))
        return img

    def _measure_peaks(self, peaks):
        """Measure several world-model candidates in one call. Each candidate gets a row of three 40 mm views
        through its exact address (axial with the measured outline, coronal, sagittal): a nodule is round in all
        three planes, a vessel is a tube in at least one."""
        if not self.use_wm or not self.peaks:
            raise ToolError("There are no world-model peaks in this setting.")
        if not isinstance(peaks, list) or not 1 <= len(peaks) <= 6:
            raise ToolError('"peaks" must list 1-6 peak numbers.')
        rows, strips, recs = [], [], []
        centre = self.from_disp(self.size / 2, self.size / 2)
        tile = 200
        for n in peaks:
            n = int(n)
            if not 1 <= n <= len(self.peaks):
                raise ToolError(f"peak must be 1-{len(self.peaks)}.")
            rec = self._peak_view(n)
            res, ann = self._measure_seed(rec, {"point": centre})
            row = {"peak": n, "view": rec.vid, "slice": rec.geom.index,
                   **{k: v for k, v in json.loads(res).items() if k not in ("view", "drawn_on")}}
            rows.append(row)
            recs.append(rec)
            zyx = tuple(self.peaks[n - 1][:3])
            ax = Image.open(io.BytesIO((ann[0] if ann else rec).png)).convert("RGB")
            ax = ax.crop((ax.width // 6, ax.height // 6, ax.width * 5 // 6, ax.height * 5 // 6)).resize((tile, tile))
            d = ImageDraw.Draw(ax)
            lab = f"peak {n} axial"
            d.rectangle([0, 0, 8 * len(lab) + 6, 14], fill=(0, 0, 0))
            d.text((3, 1), lab, fill=(255, 255, 0))
            strip = Image.new("RGB", (tile * 3 + 8, tile), (40, 40, 40))
            strip.paste(ax, (0, 0))
            strip.paste(self._plane_tile("coronal", zyx, 40, tile, f"peak {n} coronal"), (tile + 4, 0))
            strip.paste(self._plane_tile("sagittal", zyx, 40, tile, f"peak {n} sagittal"), (2 * tile + 8, 0))
            strips.append(strip)
        sheet = Image.new("RGB", (tile * 3 + 8, (tile + 4) * len(strips)), (40, 40, 40))
        for i, st in enumerate(strips):
            sheet.paste(st, (0, i * (tile + 4)))
        g0 = recs[0].geom
        pic = self.add("observed", "candidates, one row per peak: axial with the measured outline (green), coronal, "
                       "sagittal; 40 mm each, magenta ticks mark the exact address", g0, png(np.asarray(sheet)), None)
        pic.geom = Geom(g0.plane, None, 0, g0.window, g0.box, g0.size, extra={"sheet": True})
        return json.dumps({"candidates": rows, "picture": pic.vid}), [pic]

    def _measure_seed(self, rec: ViewRecord, args):
        g = rec.geom
        if g.plane != "axial" or g.slab:
            raise ToolError("A seed measurement needs an observed single axial slice.")
        u, v, box = self.locate(args)
        z, y, x = to_voxel(g, u, v, self.s.shape)
        vol = self.s.vol
        sz, sy, sx = self.s.spacing
        zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
        hz, hy, hx = (max(1, int(np.ceil(20.0 / sp))) for sp in (sz, sy, sx))
        Z, Y, X = vol.shape
        za, zb, ya, yb, xa, xb = max(0, zi - hz), min(Z, zi + hz + 1), max(0, yi - hy), min(Y, yi + hy + 1), max(0, xi - hx), min(X, xi + hx + 1)
        cube = np.asarray(vol[za:zb, ya:yb, xa:xb]).astype(np.float32)
        lz, ly, lx = zi - za, yi - ya, xi - xa
        ry, rx = max(1, int(round(2.0 / sy))), max(1, int(round(2.0 / sx)))
        local = cube[lz, max(0, ly - ry):ly + ry + 1, max(0, lx - rx):lx + rx + 1]
        peak = float(local.max())
        if peak < -750:
            return json.dumps({"view": rec.vid, "result": "the seed is in aerated lung: no structure denser than "
                               "-750 HU within 2 mm"}), []
        thr = (peak + (-850.0)) / 2
        from scipy import ndimage as ndi
        dom = np.asarray(self.domain[za:zb, ya:yb, xa:xb]) > 0
        base = (cube > thr) & dom
        comp, touches, used_r = None, True, None
        for r_mm in (0.0, 1.5, 2.5, 3.5):
            m = base
            if r_mm > 0:
                rz, ry2, rx2 = (max(0, int(round(r_mm / sp))) for sp in (sz, sy, sx))
                zz, yy2, xx2 = np.ogrid[-rz:rz + 1, -ry2:ry2 + 1, -rx2:rx2 + 1]
                ball = ((zz * sz) ** 2 + (yy2 * sy) ** 2 + (xx2 * sx) ** 2) <= r_mm ** 2
                m = ndi.binary_opening(base, structure=ball)
            lab, _ = ndi.label(m)
            sub = lab[lz, max(0, ly - ry):ly + ry + 1, max(0, lx - rx):lx + rx + 1]
            ids = sub[sub > 0]
            if ids.size == 0:
                break
            comp = lab == np.bincount(ids).argmax()
            used_r = r_mm
            touches = comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any() or comp[:, :, 0].any() or comp[:, :, -1].any()
            if not touches:
                break
        if comp is None:
            return json.dumps({"view": rec.vid, "result": "no connected structure at the seed"}), []
        if box is not None:
            (bu0, bv0, bu1, bv1) = box
            cu, cv, hu_, hv_ = (bu0 + bu1) / 2, (bv0 + bv1) / 2, 0.75 * (bu1 - bu0) + 2, 0.75 * (bv1 - bv0) + 2
            pz0, py0, px0 = to_voxel(g, cu - hu_, cv - hv_, self.s.shape)
            pz1, py1, px1 = to_voxel(g, cu + hu_, cv + hv_, self.s.shape)
            ylo, yhi = sorted((py0 - ya, py1 - ya))
            xlo, xhi = sorted((px0 - xa, px1 - xa))
            keep = np.zeros_like(comp)
            keep[:, max(0, int(np.floor(ylo))):int(np.ceil(yhi)) + 1, max(0, int(np.floor(xlo))):int(np.ceil(xhi)) + 1] = True
            hz_mm = 0.75 * max((bu1 - bu0), (bv1 - bv0)) / g.size * (g.box[2] - g.box[0]) * self.s.spacing[2] + 2
            nz = max(1, int(np.ceil(hz_mm / sz)))
            keep[:max(0, lz - nz)] = False
            keep[lz + nz + 1:] = False
            comp = comp & keep
            if not comp.any():
                return json.dumps({"view": rec.vid, "result": "no structure inside the box at the seed"}), []
        vol_mm3 = float(comp.sum() * sz * sy * sx)
        sl = comp[lz]
        out = {"view": rec.vid, "threshold_HU": round(thr), "vessel_cut_mm": used_r, "volume_mm3": round(vol_mm3, 1),
               "equivalent_diameter_mm": round((6 * vol_mm3 / np.pi) ** (1 / 3), 1),
               "mean_HU": round(float(cube[comp].mean()), 1)}
        if touches:
            out["warning"] = ("the region reached the 40 mm search box: it continues beyond it, as a vessel, the "
                              "chest wall or a larger mass would")
        yy, xx = np.nonzero(sl)
        if yy.size:
            P = np.stack([xx * sx, yy * sy], 1)
            if len(P) > 400:
                P = P[np.random.default_rng(0).choice(len(P), 400, replace=False)]
            D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
            i, j = np.unravel_index(int(D.argmax()), D.shape)
            out["long_axis_mm_on_slice"] = round(float(D[i, j]), 1)
            ax = (P[j] - P[i]) / max(D[i, j], 1e-6)
            proj = (P - P[i]) @ np.array([-ax[1], ax[0]])
            out["short_axis_mm_on_slice"] = round(float(proj.max() - proj.min()), 1)
        edge = sl & ~ndi.binary_erosion(sl)
        ey, ex = np.nonzero(edge)
        disp = [from_voxel(g, (zi, ya + a, xa + b), self.s.shape) for a, b in zip(ey, ex)]

        def draw(dd):
            for (pu, pv) in disp:
                dd.point((pu, pv), fill=(0, 255, 0))
                dd.rectangle([pu - 1, pv - 1, pu + 1, pv + 1], outline=(0, 255, 0))
            dd.ellipse([u - 3, v - 3, u + 3, v + 3], outline=(255, 0, 255), width=2)
        ann = self._annotated(rec, draw, "CT with the measured outline drawn (green) and the seed (magenta)")
        out["drawn_on"] = ann.vid
        return json.dumps(out), [ann]

    def resolve_points(self, points):
        if not isinstance(points, list) or not points:
            raise ToolError('"points" must be a non-empty list of {"view": id, "point": [x, y]}.')
        coords, used = [], []
        for p in points:
            if not isinstance(p, dict):
                raise ToolError('Each point must be {"view": id, "point": [x, y]}.')
            rec = self.get(p.get("view"))
            if not rec.evidence:
                raise ToolError(f"{rec.vid} is a predicted view; a finding must be located on an observed view.")
            if rec.geom.extra.get("sheet"):
                raise ToolError(f"{rec.vid} is a contact sheet; report on a per-peak view or with {{\"peak\": n}}.")
            u, v, _ = self.locate(p)
            coords.append(to_voxel(rec.geom, u, v, self.s.shape))
            used.append(rec.vid)
        out = []
        for axis in range(3):
            vals = [c[axis] for c in coords if c[axis] is not None]
            out.append(float(np.mean(vals)) if vals else None)
        return (tuple(out) if None not in out else None), used

    def tool_report(self, args):
        present = args.get("present")
        if not isinstance(present, bool):
            raise ToolError('"present" must be true or false.')
        zyx, used = (None, [])
        if present and "peak" in args:
            if not self.use_wm or not self.peaks:
                raise ToolError("There are no world-model peaks in this setting.")
            n = int(args["peak"])
            if not 1 <= n <= len(self.peaks):
                raise ToolError(f"peak must be 1-{len(self.peaks)}.")
            rec = self._peak_view(n)
            zyx, used = tuple(self.peaks[n - 1][:3]), [rec.vid]
            present = True
        elif present:
            pts = args.get("points")
            if pts is None and "view" in args:
                pts = [{k: args[k] for k in ("view", "bbox_2d", "point", "target") if k in args}]
            zyx, used = self.resolve_points(pts)
            if zyx is None:
                raise ToolError("These points do not fix a 3-D location.")
        diam = args.get("diameter_mm")
        self.final = {"present": present, "zyx": zyx, "views": used,
                      "diameter_mm": float(diam) if isinstance(diam, (int, float)) else None,
                      "side": args.get("side")}
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

    def call(self, name: str, args: dict):
        fn = {"view": self.tool_view, "predict": self.tool_predict, "read": self.tool_read,
              "measure": self.tool_measure, "report": self.tool_report}.get(name)
        if fn is None or (name == "predict" and not self.use_wm):
            raise ToolError(f"Unknown tool {name!r}.")
        if self.cond == "oneshot" and name != "report":
            raise ToolError("Only report is available in this setting.")
        return fn(args if isinstance(args, dict) else {})

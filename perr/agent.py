"""Model loop for open models served by vLLM (OpenAI-compatible API).

One tool call per reply, written as <tool_call>{"name": ..., "arguments": ...}</tool_call>. Tool results
come back as a user turn: the result text, then each new image preceded by its address label. Old images
leave the context once more than `max_ctx_images` are present; they stay in the archive and can be
re-rendered by id (VISTA's lossless memory: the context holds reasoning, the archive holds pixels).
"""

from __future__ import annotations

import base64
import json
import re
import time

import numpy as np

from .harness import Session, ToolError

CONVENTIONS = (
    "Images use radiological convention: the patient's right is on the image left. Axial views are seen from "
    "the feet with anterior at the top; coronal and sagittal views have superior at the top. Axial slice "
    "indices run from inferior (0) to superior. Locations are boxes, {{\"bbox_2d\": [x1, y1, x2, y2]}}, in {coords}."
)

CALL_FORMAT = (
    "Call exactly one tool per reply, after brief reasoning, in this format:\n<tool_call>\n"
    '{"name": "<tool>", "arguments": {...}}\n</tool_call>'
)

ONESHOT_REPORT_SPEC = (
    '- report: your final answer; it ends the task. Arguments: {"present": true or false, "view": "V2", '
    '"bbox_2d": [x1, y1, x2, y2], "diameter_mm": <number>, "side": "right" or "left"}. The box is in that view\'s '
    "coordinates and encloses the nodule; on a montage it must lie inside one tile."
)

REPORT_SPEC = (
    '- report: your final answer; it ends the task. Arguments: {"present": true or false, "view": "V7", '
    '"target": "the nodule you measured" (or "bbox_2d": [x1, y1, x2, y2]), "diameter_mm": <number>, "side": '
    '"right" or "left"}. The view must be an observed single-slice view (or a montage tile) that shows the nodule.'
)

VIEW_SPEC = (
    '- view: render up to 4 new views from the raw study. Arguments: {"question": "<what this look should '
    'settle>", "expect": "<what you expect to see>", "views": [items]}. Each item is one of\n'
    '    {"from": "V3", "bbox_2d": [x1, y1, x2, y2]}  zoom into a box of an existing view (same plane and slice; '
    "on a montage the box must lie inside one tile and opens that tile's slice);\n"
    '    {"from": "V5", "step": -2}  the same region two slices further along the view\'s normal (scrolling);\n'
    '    {"plane": "axial", "at": {"view": "V2", "bbox_2d": [x1, y1, x2, y2]}, "fov_mm": 60}  the slice through '
    "the centre of a box seen in another view, centred on it (omit fov_mm for the whole slice);\n"
    '    {"plane": "axial", "index": 120}  a whole slice by index.\n'
    '  Optional in any item: "window": "lung" | "mediastinum" | "bone" | "soft" (default lung), "slab_mm": '
    "thickness of a maximum-intensity slab (default: one slice)."
)

PREDICT_SPEC = (
    "- predict: ask the shape world model what an observed single slice should show. The model expects lung "
    "anatomy to be made of tubes (vessels, airways) and sheets (fissures, pleura). It returns the expected view, "
    "with structures it cannot explain removed, and its prediction error over the CT (hot = a compact structure "
    "that is neither tube nor sheet). Arguments: {\"question\": \"...\", \"views\": [\"V5\"]} (1-2 ids). A hot spot is "
    "a candidate to check, not a finding: vessels seen end-on or branching can also look compact on one slice.\n"
    '  In view items, "at" also accepts {"peak": n}: the n-th largest world-model error listed in the first message. '
    'measure also accepts {"peak": n}, or {"peaks": [1, 2, ...]} for several candidates at once: it outlines the '
    "structure at each exact address and returns sizes plus one picture of all outlines. report accepts "
    '{"present": true, "peak": n, "diameter_mm": ..., "side": ...}.'
)

LOCATE_SPEC = (
    "- Naming instead of pointing: wherever a tool takes a location (a zoom item with \"from\", an \"at\", measure, "
    'read, report) you may give "target": "<what you mean, e.g. the round nodule near the centre>" instead of '
    "bbox_2d. A focused reader then finds that target on the named view and the tool uses its box. Prefer this "
    'to writing coordinates. locate, {"view": "V10", "target": "..."}, shows you the box it would use.'
)

READ_MEASURE_SPEC = (
    '- read: CT numbers (HU) inside a box of an observed view. Arguments: {"view": "V5", "bbox_2d": [x1, y1, x2, y2]}.\n'
    '- measure: on an observed single axial slice, {"view": "V5", "target": "the round nodule near the centre"} '
    "(or a tight bbox_2d) around one structure: the viewer outlines it by thresholding and returns its long and "
    "short axis on that slice, its volume and equivalent diameter, plus the view with the outline drawn. A warning means the "
    "outline ran beyond the search box (a vessel, the chest wall or a large mass). A caliper, {\"view\": \"V5\", "
    "\"p1\": [x, y], \"p2\": [x, y]}, returns a distance. Look at the returned drawing to check what you measured."
)

READING_PROTOCOL = (
    "How radiologists search a chest CT for nodules: read the lungs in 10 mm maximum-intensity slabs from apex to "
    'base ({{"plane": "axial", "index": k, "slab_mm": 10}}, whole slice, four slabs per call); in a slab a nodule '
    "is a round dot and a vessel a branching line. Zoom into each dot, confirm it on single slices, then size it "
    "with measure. The lungs span axial slices {lo}-{hi}."
)

PROTOCOL = (
    "Before each view, state what it should settle and what you expect to see. Afterward, state what it "
    "showed, expected or not.{wm} Report only when the evidence supports one answer. You have at most "
    "{max_calls} tool calls."
)

QUESTION = (
    "Clinical question: Does this chest CT show a lung nodule at least 6 mm in diameter? If it does, locate "
    "the largest such nodule, measure its longest diameter in millimetres, and give its side."
)


def system_prompt(s: Session, max_calls: int) -> str:
    conv = CONVENTIONS.format(coords=s.coord_note())
    if s.cond == "oneshot":
        return ("You are reading a chest CT study to answer a clinical question from the views given below. "
                + conv + " You must answer now with a single report call.\n\n" + CALL_FORMAT + "\n\nTool:\n"
                + ONESHOT_REPORT_SPEC + " A box on a montage tile fixes the location: the tile's slice and the position in it.")
    tools = [VIEW_SPEC] + ([PREDICT_SPEC] if s.use_wm else []) + [LOCATE_SPEC, READ_MEASURE_SPEC, REPORT_SPEC]
    wm = (' Start with measure {"peaks": [1, 2, 3, 4, 5]}: you get every world-model candidate\'s size and a row '
          "of three views through it (axial with the measured outline, coronal, sagittal). A nodule is a compact "
          "blob that is round in all three planes; a vessel is a tube in at least one plane (or its outline gives a "
          "warning); chest wall sits on the edge of the lung. Report the largest candidate that is a nodule with "
          '{"present": true, "peak": n, "diameter_mm": <its equivalent diameter>, "side": ...}; always report a '
          "candidate by its peak number. Answer present: false only if every candidate is clearly a vessel or wall "
          "and the slabs show no nodule either.") if s.use_wm else ""
    zs = np.nonzero(np.asarray(s.s.lung).any(axis=(1, 2)))[0]
    protocol = READING_PROTOCOL.format(lo=int(zs.min()), hi=int(zs.max()))
    return ("You are reading a chest CT study to answer a clinical question. A viewer holds the full study "
            "losslessly; you see only the views you ask for, each rendered from the raw data and labelled with "
            "its id before it is shown. " + conv + "\n\n" + CALL_FORMAT + "\n\nTools:\n" + "\n".join(tools)
            + "\n\n" + protocol + "\n\n" + PROTOCOL.format(wm=wm, max_calls=max_calls))


def image_part(rec) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(rec.png).decode()}}


def visual_parts(s, recs) -> list[dict]:
    parts = []
    for r in recs:
        parts.append({"type": "text", "text": s.label(r)})
        parts.append(image_part(r))
    return parts


TOOL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def parse_call(text: str):
    cands = [m.group(1) for m in TOOL_RE.finditer(text or "")]
    if not cands:
        cands = [text or ""]
    dec = json.JSONDecoder()
    for chunk in reversed(cands):
        for i, ch in enumerate(chunk):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(chunk[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return obj["name"], args if isinstance(args, dict) else {}
    return None


LOCATE_Q = ("Locate {target} in this image. Output its bounding box as JSON: {{\"bbox_2d\": [x1, y1, x2, y2]}}. "
            "If it is not visible, output {{\"bbox_2d\": null}}.")


def parse_box(text: str):
    m = re.search(r"bbox_2d\"?\s*:\s*(null|\[[^\]]*\])", text or "")
    if not m or m.group(1) == "null":
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
    return [float(t) for t in nums[:4]] if len(nums) >= 4 else None


async def focused_box(client, model, s, rec, target: str, trace):
    msg = [{"role": "user", "content": [image_part(rec), {"type": "text", "text": LOCATE_Q.format(target=target)}]}]
    r = await client.chat.completions.create(model=model, messages=msg, temperature=0, max_tokens=120)
    if r.usage:
        trace["usage"]["prompt"] += r.usage.prompt_tokens
        trace["usage"]["completion"] += r.usage.completion_tokens
    trace["locate_calls"] = trace.get("locate_calls", 0) + 1
    return parse_box(r.choices[0].message.content)


def _grow(box, f, lim):
    cx, cy, hw, hh = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (box[2] - box[0]) * f / 2, (box[3] - box[1]) * f / 2
    h = max(hw, hh, lim * 0.06)
    return [max(0, cx - h), max(0, cy - h), min(lim, cx + h), min(lim, cy + h)]


async def ground_args(client, model, s, name, args, trace):
    """Replace every {"target": "..."} with the bbox_2d a focused reader finds on the named view."""
    lim = 1000 if s.mode == "norm1000" else s.size

    async def box_for(vid, target, grow=1.0):
        rec = s.get(vid)
        if isinstance(target, dict) and isinstance(target.get("bbox_2d"), list):
            return target["bbox_2d"]
        b = await focused_box(client, model, s, rec, str(target)[:200], trace)
        if b is None:
            raise ToolError(f"The focused reader did not find {str(target)[:80]!r} on {vid}.")
        return _grow(b, grow, lim) if grow != 1.0 else b

    if name in ("measure", "report", "read") and "target" in args and "bbox_2d" not in args and "view" in args:
        args = {**args, "bbox_2d": await box_for(args["view"], args["target"])}
    if name == "view" and isinstance(args.get("views"), list):
        items = []
        for it in args["views"]:
            if isinstance(it, dict) and "from" in it and "target" in it and "bbox_2d" not in it and "box" not in it:
                it = {**it, "bbox_2d": await box_for(it["from"], it["target"], grow=2.5)}
            at = it.get("at") if isinstance(it, dict) else None
            if isinstance(at, dict) and "target" in at and "view" in at and "bbox_2d" not in at:
                it = {**it, "at": {**at, "bbox_2d": await box_for(at["view"], at["target"])}}
            items.append(it)
        args = {**args, "views": items}
    return args


async def locate(client, model, s, args, trace):
    """Focused perception: the same model boxes the target on one image, in a clean context."""
    if s.cond == "oneshot":
        raise ToolError("Only report is available in this setting.")
    rec = s.get(args.get("view"))
    if not rec.evidence:
        raise ToolError("locate works on observed views.")
    target = str(args.get("target") or "the most suspicious lesion")[:200]
    msg = [{"role": "user", "content": [image_part(rec), {"type": "text", "text": LOCATE_Q.format(target=target)}]}]
    r = await client.chat.completions.create(model=model, messages=msg, temperature=0, max_tokens=120)
    if r.usage:
        trace["usage"]["prompt"] += r.usage.prompt_tokens
        trace["usage"]["completion"] += r.usage.completion_tokens
    trace.setdefault("locate_calls", 0)
    trace["locate_calls"] += 1
    return s.located(rec, parse_box(r.choices[0].message.content), target)


def evict_images(messages: list, keep: int) -> int:
    """Replace the oldest images beyond `keep` with a note; returns how many were evicted."""
    locs = [(i, j) for i, m in enumerate(messages) if isinstance(m["content"], list)
            for j, p in enumerate(m["content"]) if p.get("type") == "image_url"]
    drop = locs[:max(0, len(locs) - keep)]
    for i, j in drop:
        messages[i]["content"][j] = {"type": "text", "text": "[image left the context; it stays in the archive: "
                                                               "re-render it with view {\"from\": <its id>}]"}
    return len(drop)


async def run_episode(client, model: str, s, max_calls: int = 20, max_ctx_images: int = 16,
                      max_tokens: int = 1536, temperature: float = 0.0, system: str | None = None,
                      task: str | None = None) -> dict:
    if task is None:
        sp = s.s.spacing
        task = (QUESTION + f"\n\nStudy: {s.s.shape[0]} axial slices, {sp[0]:.2f} mm apart; in-plane pixel "
                f"{sp[2]:.2f} mm.")
    overview = s.overview()
    if s.use_wm:
        task += "\n\n" + s.peak_text()
    messages = [{"role": "system", "content": system if system is not None else system_prompt(s, max_calls)},
                {"role": "user", "content": [{"type": "text", "text": task + "\n\nInitial views:"}]
                 + visual_parts(s, overview)}]
    trace = {"steps": [], "usage": {"prompt": 0, "completion": 0}, "calls": 0, "invalid": 0, "forced": False,
             "evicted": 0, "views_rendered": 0}
    # one-shot settings may only call report; retries are allowed only to fix a malformed report
    budget = 3 if s.cond == "oneshot" else max_calls
    t0 = time.time()
    finished = False
    while not finished:
        if trace["calls"] >= budget:
            if trace["forced"] or s.cond == "oneshot":
                break
            trace["forced"] = True
            messages.append({"role": "user", "content": [{"type": "text", "text":
                             "You have used all tool calls. Call report now with your best answer."}]})
        trace["evicted"] += evict_images(messages, max_ctx_images)
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens)
        except Exception as e:  # context overflow or server error: end the episode, keep the trace
            trace["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            break
        text = resp.choices[0].message.content or ""
        if resp.usage:
            trace["usage"]["prompt"] += resp.usage.prompt_tokens
            trace["usage"]["completion"] += resp.usage.completion_tokens
        messages.append({"role": "assistant", "content": text})
        step = {"assistant": text}
        parsed = parse_call(text)
        if parsed is None:
            # some models think in one turn and act in the next: tolerate up to 2 turns without a call in a row
            trace["invalid"] += 1
            trace["streak"] = trace.get("streak", 0) + 1
            step["error"] = "no tool call"
            trace["steps"].append(step)
            if trace["streak"] >= 3 or trace["invalid"] >= 12:
                break
            messages.append({"role": "user", "content": [{"type": "text", "text":
                             "Now call one tool: reply with exactly one <tool_call>{...}</tool_call>."}]})
            continue
        trace["streak"] = 0
        name, args = parsed
        step["call"] = {"name": name, "arguments": args}
        trace["calls"] += 1
        if trace["forced"] and name != "report":
            step["error"] = "forced turn without report"
            trace["steps"].append(step)
            break
        try:
            if name == "locate":
                result, recs = await locate(client, model, s, args, trace)
            else:
                if s.cond != "oneshot":
                    args = await ground_args(client, model, s, name, args, trace)
                    step["call"]["grounded_arguments"] = args
                result, recs = s.call(name, args)
        except ToolError as e:
            result, recs = f"Error: {e}", []
        except Exception as e:  # malformed arguments the validators did not anticipate
            result, recs = f"Error: {type(e).__name__}: {e}", []
        step["result"] = result
        step["views"] = [s.label(r) for r in recs]
        trace["views_rendered"] += len(recs)
        trace["steps"].append(step)
        if name == "report" and s.final is not None:
            finished = True
            break
        messages.append({"role": "user", "content": [{"type": "text", "text": f"Result of {name}: {result}"}]
                         + visual_parts(s, recs)})
    trace["seconds"] = round(time.time() - t0, 1)
    trace["final"] = s.final
    trace["overview"] = [s.label(r) for r in overview]
    return trace

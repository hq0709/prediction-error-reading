"""LaTeX draft of the CheXlocalize unseen-finding table (results/open_set_chexloc.json, open_set_chexloc.py), in the
layout of the paper's tables/supp_open_set.tex: hit inside the segmentation at K = 10; the detector column is the
compared method with the highest mean over types in that panel. Writes results/open_set_chexloc.tex.
"""
import json

CITE = {"Faster R-CNN": ("Faster R-CNN", "ren2015faster"), "RetinaNet": ("RetinaNet", "lin2017focal"),
        "FCOS": ("FCOS", "tian2019fcos"), "YOLO11m": ("YOLO11m", "jocher2024yolo11"),
        "RT-DETRv2-R50": ("RT-DETRv2", "lv2024rtdetrv2"), "Deformable DETR": ("Deform. DETR", "zhu2021deformable"),
        "one-shot Qwen3-VL-8B": (r"\makecell[l]{Qwen3-VL-8B\\(LoRA, answer + point)}", "qwen3vl2025")}
TWO, ONE, RND = "world model, two-sided", "world model, one-sided (focal setting)", "random thorax points"


def cells(vals):
    order = sorted({round(v, 2) for v in vals}, reverse=True)
    out = []
    for v in vals:
        s = f"{v:.2f}"
        if round(v, 2) == order[0]:
            s = rf"\textbf{{{s}}}"
        elif len(order) > 1 and round(v, 2) == order[1]:
            s = rf"\second{{{s}}}"
        out.append(s)
    return out


SHORT = {"Enlarged Cardiomediastinum": r"\makecell[l]{Enlarged\\cardiomediastinum}"}


def minipage(title, tab):
    return (r"\begin{minipage}[t]{0.495\textwidth}\centering" + "\n" + rf"\textit{{{title}}}\par\smallskip" + "\n"
            + tab + "\n" + r"\end{minipage}")


def panel(rows, size):
    # Support Devices is a CheXlocalize class but not a finding, so it is left out.
    r = [x for x in rows if x["boxes"] == size and x["region"] == "mask" and x["K"] == 10
         and x["finding"] != "Support Devices"]
    fs = list(dict.fromkeys(x["finding"] for x in r))
    L = {(x["method"], x["finding"]): x for x in r}
    dets = [m for m in dict.fromkeys(x["method"] for x in r) if m in CITE]
    mean = lambda m: sum(L[(m, f)]["hit"] for f in fs) / len(fs)  # noqa: E731
    best = max(dets, key=mean)
    lab, cite = CITE[best]
    lines = [r"\toprule", r" & & \multicolumn{2}{c}{World model} & & \\", r"\cmidrule(lr){3-4}",
             rf"Finding & $n$ & two-sided & one-sided & {lab}\tabcite{{{cite}}} & \makecell{{Random\\points}} \\", r"\midrule"]
    for f in fs:
        lines.append(f"{SHORT.get(f, f)} & {L[(TWO, f)]['images']} & " + " & ".join(cells([L[(m, f)]["hit"] for m in (TWO, ONE, best, RND)])) + r" \\")
    lines += [r"\midrule", "Mean over types & -- & " + " & ".join(cells([mean(m) for m in (TWO, ONE, best, RND)])) + r" \\", r"\bottomrule"]
    return "\\begin{tabular}{@{}lccccc@{}}\n" + "\n".join(lines) + "\n\\end{tabular}"


rows = json.load(open("results/open_set_chexloc.json"))
tex = (r"\begin{table*}[!htbp]" + "\n" + r"\centering\tablestyle" + "\n" + r"\setlength{\tabcolsep}{2.5pt}" + "\n"
       + minipage(r"Regions $<2\%$ of image", panel(rows, "boxes < 2% of the image")) + r"\hfill" + "\n"
       + minipage("All regions", panel(rows, "all boxes")) + "\n"
       + r"\caption{\floattitle{Unseen finding types on CheXlocalize at $K=10$}Hit inside the radiologist segmentation; $n$: images; all models unchanged from ChestX-Det; means weight types equally.}" + "\n"
       + r"\label{tab:supp-open-set-chexloc}" + "\n" + r"\end{table*}" + "\n")
open("results/open_set_chexloc.tex", "w").write(tex)
print(tex)

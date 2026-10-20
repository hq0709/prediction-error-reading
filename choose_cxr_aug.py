"""Augmentation on or off for every trained radiograph method and label fraction, chosen by the mean image-level AUROC
over the four focal findings on the same held-out tenth of the ChestX-Det training images:

  detectors (train_cxr_detector.py, train_cxr_detr.py; cxr_aug.py)   best checkpoint's score ("best validation" in the log)
  one-step Qwen3-VL-8B (train_cxr_oneshot.py --aug)                  presence AUROC of the best epoch (log)
  PER verifier (train_cxr_focal_reader.py --aug)                     max-p over the image's candidates, from the scored
                                                                     held-out tenth (wmu_cxdet_val.jsonl); the run's own
                                                                     epoch is chosen by candidate-level AUROC (best.json)

YOLO11m keeps its library recipe. A missing augmented run leaves the plain run in place (reported).

    python choose_cxr_aug.py
Writes results/cxr_aug_choice.json: {method: {fraction: {"plain": score, "aug": score, "run": chosen run}}}; the
collectors (collect_cxr_focal.py, collect_cxr_curve.py, collect_cxr_nih.py, open_set_cxr.py, collect_supp.py) read the
chosen run through chosen().
"""
import json
import os
import re

BUDGETS = (1, 5, 10, 25, 50, 100)
CHOICE = "results/cxr_aug_choice.json"


def from_log(pattern):
    def read(path):
        if not os.path.exists(path):
            return None
        v = [float(x) for x in re.findall(pattern, open(path).read())]
        return max(v) if v else None
    return read


det = from_log(r"best validation ([0-9.]+)")
oneshot = from_log(r"validation mean presence AUROC ([0-9.]+)")


def verifier(run):
    """Image-level mean AUROC of a verifier run on the held-out tenth: max-p over the up to 60 union candidates of each
    image (the test readout), from its scored results/cxr_focal/<run>/wmu_cxdet_val.jsonl; the same image-level
    criterion as the detectors and the one-step model. (The run's own epoch was chosen by candidate-level AUROC.)"""
    path = f"results/cxr_focal/{run}/wmu_cxdet_val.jsonl"
    if not os.path.exists(path):
        return None
    from cxr_recall import boxes
    from score_cxr_focal import TYPES, evaluate
    b = boxes("cxdet")
    rows = [json.loads(line) for line in open(path)]
    return float(evaluate(rows, b[b.split == "train"], TYPES, ks=(60,), rules=("max-p",)).AUROC.mean())


def run_name(base, bud):
    return base if bud == 100 else f"{base}_b{bud}bal"


# label: (plain run, augmented run, score of a run at a fraction)
METHODS = {
    "Faster R-CNN": ("cxr_detector", "cxr_detectoraug",
                     lambda run, b, aug: det(f"logs/aug_frcnn_b{b}.log" if aug else
                                             ("logs/tr_cxr_detector.log" if b == 100 else f"logs/bal_frcnn_b{b}.log"))),
    "RetinaNet": ("cxr_retinanet", "cxr_retinanetaug",
                  lambda run, b, aug: det(f"logs/aug_retinanet_b{b}.log" if aug else
                                          ("logs/tr_cxr_retinanet.log" if b == 100 else f"logs/bal_retinanet_b{b}.log"))),
    "FCOS": ("cxr_fcos", "cxr_fcosaug",
             lambda run, b, aug: det(f"logs/aug_fcos_b{b}.log" if aug else
                                     ("logs/tr_cxr_fcos.log" if b == 100 else f"logs/bal_fcos_b{b}.log"))),
    "RT-DETRv2-R50": ("cxr_rtdetrv2", "cxr_rtdetrv2aug",
                      lambda run, b, aug: det(f"logs/aug_rtdetrv2aug_b{b}.log" if aug else
                                              ("logs/tr_cxr_rtdetrv2.log" if b == 100 else f"logs/bal_rtdetrv2_b{b}.log"))),
    "Deformable DETR": ("cxr_deformdetr", "cxr_deformdetraug",
                        lambda run, b, aug: det(f"logs/aug_deformdetraug_b{b}.log" if aug else
                                                ("logs/tr_cxr_deformdetr.log" if b == 100 else f"logs/bal_deformdetr_b{b}.log"))),
    "one-shot Qwen3-VL-8B": ("cxr_oneshot", "cxr_oneshotaug",
                             lambda run, b, aug: oneshot(f"logs/os_aug_b{b}.log" if aug else
                                                         ("logs/tr_cxr_oneshot.log" if b == 100 else f"logs/bal_oneshot_b{b}.log"))),
    "PER (ours)": ("cxr_focal_wmu", "cxr_focal_wmuaug", lambda run, b, aug: verifier(run)),
}


# further augmented runs compared with the augmented run, {(method, fraction): (run, ...)}
EXTRA = {}


def choice():
    out = {}
    for label, (plain, aug, score) in METHODS.items():
        out[label] = {}
        for bud in BUDGETS:
            rp = run_name(plain, bud)
            vp = score(rp, bud, False)
            tried = {}
            for ra in (run_name(aug, bud),) + EXTRA.get((label, bud), ()):
                va = score(ra, bud, True)
                if va is not None and os.path.isdir(f"results/cxr_focal/{ra}"):
                    tried[ra] = va
            ra = max(tried, key=tried.get) if tried else None
            va = tried.get(ra)
            out[label][str(bud)] = {"plain": vp, "aug": va, "run": ra if ra and (vp is None or va > vp) else rp}
            if len(tried) > 1:
                out[label][str(bud)]["aug_runs"] = tried
    return out


def chosen(label, bud=100, path=CHOICE):
    """The chosen run of a radiograph method at a label fraction (percent of the focal-boxed training images)."""
    return json.load(open(path))[label][str(bud)]["run"]


def main():
    out = choice()
    json.dump(out, open(CHOICE, "w"), indent=1)
    f = lambda v: "  -  " if v is None else f"{v:.3f}"  # noqa: E731
    for label, d in out.items():
        print(f"{label:21s} " + "  ".join(f"{b}%: {f(v['plain'])}/{f(v['aug'])}{'*' if v['run'].startswith(METHODS[label][1]) else ' '}"
                                          for b, v in d.items()))
    print("(plain/augmented validation score; * = augmented run chosen; - = not run or not finished)")


if __name__ == "__main__":
    main()

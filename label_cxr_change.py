"""Interval-change labels for MIMIC-CXR pairs (results/cxr_pairs.csv: each frontal study with the patient's previous
frontal study) from the current study's report (R2Gen annotation of MIMIC-CXR reports), read by an open LLM
(Qwen3-8B, no thinking): for each finding, whether the report says it got worse (new, increased, larger, worsened),
stayed the same (unchanged, stable, similar), got better (decreased, improved, smaller, resolved), or does not compare
it (not mentioned, or mentioned without a comparison). The MS-CXR-T findings (consolidation, oedema, effusion,
pneumonia, pneumothorax) plus atelectasis, cardiomegaly and support devices.

Writes results/cxr_change_labels.csv (key, prior_key, split, one column per finding with worse / same / better / none).

    CUDA_VISIBLE_DEVICES=0 python label_cxr_change.py
"""
import json
import os
import re

import pandas as pd

from perr.cxr import mimic_frontal

ANN = "/home/local/PARTNERS/hj880/dMLLM_in_Medicine/data/mimic/mimic_cxr/annotation.json"
MODEL = "/raid/home/CAMCA/hj880/hf_home/hub/models--Qwen--Qwen3-8B"
FINDINGS = ("consolidation", "pulmonary edema", "pleural effusion", "pneumonia", "pneumothorax", "atelectasis",
            "cardiomegaly", "support devices")
PROMPT = """Below is the report of a chest radiograph that may be compared with the patient's previous radiograph.
For each finding, say how the report describes its change since the previous study:
- "worse": new, increased, larger, worsened, progressed, developing
- "same": unchanged, stable, similar, persistent, again seen, no change
- "better": decreased, improved, smaller, resolving, resolved, removed (for devices)
- "none": not mentioned, mentioned without any comparison, or absent in both studies
Answer with one JSON object with exactly these keys: {keys}.

Report:
{report}"""


KEYWORDS = {"consolidation": r"consolidat", "pulmonary edema": r"edema|congestion", "pleural effusion": r"effusion",
            "pneumonia": r"pneumonia", "pneumothorax": r"pneumothora", "atelectasis": r"atelecta",
            "cardiomegaly": r"cardiomegaly|heart size|cardiac silhouette|enlarged heart|cardiac enlargement",
            "support devices": r"tube|line|catheter|pacemaker|pacer|wire|device|port|picc|drain|stent|valve|lead"}
CUES = {"worse": r"increas|worsen|new|larger|more|progress|develop|enlarg|greater|extend|accumulat",
        "same": r"unchanged|stable|similar|persist|again|no change|not changed|no significant change|redemonstrat|remain",
        "better": r"decreas|improv|resolv|smaller|less|remov|clear|withdraw|interval removal|no longer"}


def keep_compared(lab, reports):
    """The LLM's answer survives only when a sentence of the report names the finding together with a word of that
    kind of change (e.g. "No pneumothorax." is not a comparison and becomes "none")."""
    out = lab.copy()
    for i, rep in enumerate(reports):
        sents = [x.lower() for x in re.split(r"(?<=[.;])\s+", rep.replace("\n", " "))]
        for f in FINDINGS:
            v = out.at[i, f]
            if v == "none":
                continue
            if not any(re.search(KEYWORDS[f], x) and re.search(CUES[v], x) for x in sents):
                out.at[i, f] = "none"
    return out


def snapshot(path):
    snaps = os.path.join(path, "snapshots")
    return os.path.join(snaps, sorted(os.listdir(snaps))[0])


def main():
    from vllm import LLM, SamplingParams
    pairs = pd.read_csv("results/cxr_pairs.csv")[["key", "prior_key", "split"]]
    m = mimic_frontal()
    m["key"] = "mimic/" + m.dicom_id
    pairs = pairs.merge(m[["key", "study_id"]], on="key", how="left")
    ann = json.load(open(ANN))
    report = {}
    for split in ann.values():
        for r in split:
            report.setdefault(int(r["study_id"]), r["report"])
    pairs["report"] = pairs.study_id.map(lambda s: report.get(int(s)))
    pairs = pairs[pairs.report.notna()].reset_index(drop=True)
    print(f"{len(pairs)} pairs with a report ({pairs.split.value_counts().to_dict()})", flush=True)
    llm = LLM(model=snapshot(MODEL), max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.35")),
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    keys = ", ".join(f'"{f}"' for f in FINDINGS)
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(keys=keys, report=r.strip())}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for r in pairs.report]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=160))
    rows, bad = [], 0
    for o in outs:
        txt = o.outputs[0].text
        mm = re.search(r"\{.*\}", txt, re.S)
        try:
            d = json.loads(mm.group(0)) if mm else {}
        except json.JSONDecodeError:
            d = {}
        bad += not d
        rows.append({f: (str(d.get(f, "none")).lower() if str(d.get(f, "none")).lower() in ("worse", "same", "better") else "none")
                     for f in FINDINGS})
    out = pd.concat([pairs[["key", "prior_key", "split"]], pd.DataFrame(rows)], axis=1)
    out = keep_compared(out, pairs.report.tolist())
    out.to_csv("results/cxr_change_labels.csv", index=False)
    print(f"unparsed answers: {bad}", flush=True)
    for split in ("train", "validate", "test"):
        s = out[out.split == split]
        print(split, len(s), {f: s[f].value_counts().to_dict() for f in FINDINGS}, flush=True)


if __name__ == "__main__":
    main()

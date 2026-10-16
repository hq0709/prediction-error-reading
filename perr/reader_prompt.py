"""The reader's question about one candidate strip, shared by training, probing and scoring (no heavy imports)."""

import json

QUESTION = ("One candidate from a chest CT: axial (with measured outline), coronal and sagittal views, 40 mm each; "
            "magenta ticks mark the exact address. Measurement: {m}. Is the structure at the marked address a lung "
            "nodule at least 6 mm across (a compact blob, round in all three planes), rather than a vessel, airway, "
            "chest wall or a smaller nodule? Answer with one word: yes or no.")


def question(measurement: dict) -> str:
    return QUESTION.format(m=json.dumps(measurement))


# Ablation variants of what the reader is shown (the same transform is applied in training and in scoring).
VARIANTS = ("full", "axial", "nomeas")
QUESTION_AXIAL = ("One candidate from a chest CT: an axial view (with measured outline), 40 mm; the magenta dot marks "
                  "the exact address. Measurement: {m}. Is the structure at the marked address a lung nodule at least "
                  "6 mm across (a compact round blob), rather than a vessel, airway, chest wall or a smaller nodule? "
                  "Answer with one word: yes or no.")
QUESTION_NOMEAS = QUESTION.replace(" Measurement: {m}.", "")


def prepare(row: dict, variant: str = "full"):
    """The (image, question) pair shown to the reader for one candidate row."""
    from PIL import Image
    im = Image.open(row["image"]).convert("RGB")
    if row.get("question"):
        return im, row["question"]
    if variant == "axial":
        return im.crop((0, 0, 200, 200)), QUESTION_AXIAL.format(m=json.dumps(row["measurement"]))
    if variant == "nomeas":
        return im, QUESTION_NOMEAS
    return im, question(row["measurement"])

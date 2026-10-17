"""Latent chest-radiograph world model: predict the representation of hidden image regions from the visible ones, in
the patch-feature space of a frozen vision encoder, with the model's own uncertainty (cf. JEPA-style world models).

Pixel prediction errors on radiographs are dominated by exposure, positioning and texture; a prediction error in a
learned feature space is closer to a semantic surprise. The encoder (DINOv2, or the radiograph-specific RAD-DINO) maps
a 518 x 518 radiograph to a 37 x 37 grid of 768-d patch features. A small transformer sees the grid with square blocks
replaced by a mask token and predicts the hidden features and a per-patch log-variance (beta-NLL). It is trained on
radiographs labelled "No Finding". Dense reading: blocks of 6 x 6 patches on a stride-5 lattice are hidden in turn,
and each patch's error is averaged over the variants that hid it.

No torch import at module level.
"""

from __future__ import annotations

import numpy as np

GRID, DIM, BLOCK, STRIDE = 37, 768, 6, 5
ENCODERS = {"dinov2": "facebook/dinov2-base", "raddino": "microsoft/rad-dino"}


def block_masks():
    """Dense-reading masks: (n, GRID, GRID) booleans, one hidden 6 x 6 block each, covering the grid."""
    starts = list(range(0, GRID - BLOCK + 1, STRIDE))
    if starts[-1] != GRID - BLOCK:
        starts.append(GRID - BLOCK)
    out = []
    for r in starts:
        for c in starts:
            m = np.zeros((GRID, GRID), bool)
            m[r:r + BLOCK, c:c + BLOCK] = True
            out.append(m)
    return np.stack(out)


def make_encoder(name):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    path = ENCODERS[name]
    proc = AutoImageProcessor.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).eval()
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    return model, mean, std


def make_predictor(dim=512, depth=6, heads=8):
    import torch
    from torch import nn

    class Predictor(nn.Module):
        """Patch features (B, N, DIM) and a hidden mask (B, N) -> predicted features and log-variance at every patch."""

        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(DIM, dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, GRID * GRID, dim) * 0.02)
            layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(dim)
            self.mean = nn.Linear(dim, DIM)
            self.logvar = nn.Linear(dim, 1)

        def forward(self, feats, hidden):
            x = self.inp(feats)
            x = torch.where(hidden[..., None], self.mask_token.expand_as(x), x) + self.pos
            x = self.norm(self.blocks(x))
            return self.mean(x), self.logvar(x).squeeze(-1)

    return Predictor()

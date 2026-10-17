"""Learned chest-radiograph world model: predict a hidden patch of a frontal radiograph from its surroundings, with the
model's own uncertainty, and read the precision-weighted prediction error.

The 2-D counterpart of the anatomy world model for CT (context_wm.py): a U-Net sees a 128 x 128 window of the
canonical 512 x 512 radiograph (about 9 cm) with its central 48 x 48 hidden and predicts the hidden part and its
log-variance (beta-NLL). It is trained on radiographs whose report labels say "No Finding", so it learns what normal
anatomy, projection and positioning look like. Dense prediction: every 16 x 16 block is predicted from a window
centred on it, so every pixel is predicted from image content at least 16 px away. Findings can be brighter
(opacity, effusion, nodule) or darker (pneumothorax) than expected, so the error is read on both signs.

No torch import at module level.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CTX, HOLE, BLOCK = 128, 48, 16


def norm(img):
    return np.asarray(img, np.float32) / 255.0


def downscale(img, s):
    """Canonical 512 x 512 radiograph -> (512/s)^2 by block means (float32 in [0, 255]); s = 1 returns it as is. A world
    model trained on the downscaled canvas has the same window and hidden part in its own pixels, so it hides an s
    times larger region (s = 4: a 192 px = ~16 cm square of the 512 canvas, predicted from the whole radiograph)."""
    img = np.asarray(img, np.float32)
    if s == 1:
        return img
    n = img.shape[0] // s
    return img.reshape(n, s, n, s).mean((1, 3))


def hole_mask():
    m = np.zeros((CTX, CTX), np.float32)
    a, b = (CTX - HOLE) // 2, (CTX + HOLE) // 2
    m[a:b, a:b] = 1
    return m


def make_unet2d(ch=(48, 96, 192, 384, 512), in_ch=2, out_ch=2):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                                     nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

        def forward(self, x):
            return self.net(x)

    class UNet2D(nn.Module):
        """Input (hidden-zeroed image, hidden mask[, prior image]); output (predicted image, log-variance)."""

        def __init__(self):
            super().__init__()
            self.enc, cin = nn.ModuleList(), in_ch
            for c in ch:
                self.enc.append(Block(cin, c))
                cin = c
            self.up, self.dec = nn.ModuleList(), nn.ModuleList()
            for c in reversed(ch[:-1]):
                self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
                self.dec.append(Block(2 * c, c))
                cin = c
            self.out = nn.Conv2d(cin, out_ch, 1)

        def forward(self, x):
            skips = []
            for i, b in enumerate(self.enc):
                x = b(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for up, dec in zip(self.up, self.dec):
                x = dec(torch.cat([up(x), skips.pop()], 1))
            y = self.out(x)
            return torch.cat([torch.sigmoid(y[:, :1]), y[:, 1:]], 1)

    return UNet2D()


def error_map(R, S, sigma=3.0, sign=0):
    """Smoothed signed error divided by the smoothed predicted s.d.; sign +1 brighter than expected, -1 darker, 0 both
    (absolute value)."""
    W = ndi.gaussian_filter(np.asarray(R, np.float32), sigma) / np.sqrt(ndi.gaussian_filter(np.exp(np.asarray(S, np.float32)), sigma))
    return np.abs(W) if sign == 0 else sign * W


def peaks(W, valid, k=20, nms=24):
    """Top-k peaks (row, col, value) of a map inside the valid (non-padding) region, with square suppression."""
    W = np.where(valid, W, -np.inf).astype(np.float32)
    out = []
    for _ in range(k):
        i = int(np.argmax(W))
        r, c = np.unravel_index(i, W.shape)
        v = float(W[r, c])
        if not np.isfinite(v):
            break
        out.append((int(r), int(c), v))
        W[max(0, r - nms):r + nms + 1, max(0, c - nms):c + nms + 1] = -np.inf
    return out


def valid_region(img, margin=8):
    """Pixels of the radiograph itself (not the letterbox padding), shrunk by a margin."""
    m = np.asarray(img) > 0
    rows, cols = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    v = np.zeros(m.shape, bool)
    if len(rows) and len(cols):
        v[rows.min() + margin:rows.max() - margin + 1, cols.min() + margin:cols.max() - margin + 1] = True
    return v

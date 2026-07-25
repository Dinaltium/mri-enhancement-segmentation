"""
model_inspect.py   --   white-box view of the networks

Opens the model up completely: every layer, the tensor shape flowing through
it, its parameter count, and — from a real forward pass on a real slice — the
actual feature maps each stage produces. Nothing is illustrative; every number
and every image comes from executing the model.

Used by the demo web app's "Inside the model" page.
"""

import numpy as np
import torch
import torch.nn as nn

from nifti_utils import IMG_SIZE
from models import EnhancementUNet, SegmentationUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} k"
    return str(n)


def trace_model(model, x):
    """Run one forward pass, recording every leaf module's output shape and
    parameter count in execution order."""
    rows, handles, seen = [], [], set()

    def hook(name, mod):
        def fn(_m, inp, out):
            if not torch.is_tensor(out):
                return
            p = sum(q.numel() for q in _m.parameters(recurse=False))
            rows.append({
                "name": name,
                "type": type(_m).__name__,
                "in_shape": tuple(inp[0].shape[1:]) if torch.is_tensor(inp[0]) else None,
                "out_shape": tuple(out.shape[1:]),
                "params": p,
            })
        return fn

    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and name not in seen:
            seen.add(name)
            handles.append(mod.register_forward_hook(hook(name, mod)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return rows


def capture_stages(model, x, layer_names):
    """Return {label: activation tensor} for the named modules."""
    caught, handles = {}, []

    def mk(label):
        def fn(_m, _i, o):
            if torch.is_tensor(o):
                caught[label] = o.detach()
        return fn

    lut = dict(model.named_modules())
    for label, mod_name in layer_names.items():
        if mod_name in lut:
            handles.append(lut[mod_name].register_forward_hook(mk(label)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return caught


def featuremap_grid(act, n=6):
    """Turn a 1xCxHxW activation into a single wide grayscale strip of the n
    highest-energy channels, normalised per channel for visibility."""
    a = act[0].cpu().numpy()
    if a.ndim != 3:
        return None
    energy = a.reshape(a.shape[0], -1).std(axis=1)
    idx = np.argsort(energy)[::-1][:n]
    tiles = []
    for i in idx:
        t = a[i]
        lo, hi = float(t.min()), float(t.max())
        t = (t - lo) / (hi - lo + 1e-8)
        t = np.pad(t, 2, constant_values=1.0)
        tiles.append(t)
    if not tiles:
        return None
    return np.concatenate(tiles, axis=1)


def enhancement_report(sample_slice=None):
    """Full white-box report for the enhancement U-Net."""
    model = EnhancementUNet(base_filters=32).to(DEVICE).eval()
    x = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if sample_slice is not None:
        x = torch.from_numpy(sample_slice).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    rows = trace_model(model, x)
    total = sum(p.numel() for p in model.parameters())
    stages = capture_stages(model, x, {
        "Encoder 1 — edges & texture": "backbone.enc1",
        "Encoder 2 — local shapes": "backbone.enc2",
        "Encoder 3 — regions": "backbone.enc3",
        "Bottleneck — compressed understanding": "backbone.bottleneck",
        "Decoder 2 — rebuilding detail": "backbone.dec2",
        "Decoder 1 — final detail": "backbone.dec1",
    })
    return {"rows": rows, "total_params": total, "stages": stages, "model": model}


def segmentation_report(sample_stack=None):
    model = SegmentationUNet(num_classes=4, in_channels=4, base_filters=32).to(DEVICE).eval()
    x = torch.zeros(1, 4, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if sample_stack is not None:
        x = torch.from_numpy(sample_stack).float().unsqueeze(0).to(DEVICE)
    rows = trace_model(model, x)
    total = sum(p.numel() for p in model.parameters())
    return {"rows": rows, "total_params": total, "model": model}


def summarise(rows):
    """Collapse the per-leaf trace into readable blocks for display."""
    out = []
    for r in rows:
        out.append({
            "layer": r["name"] or "(input)",
            "op": r["type"],
            "shape": "x".join(str(v) for v in r["out_shape"]),
            "params": _fmt(r["params"]),
            "raw_params": r["params"],
        })
    return out


ARCH_NOTES = {
    "Conv2d": "learns filters — each one responds to a specific visual pattern",
    "BatchNorm2d": "re-centres the numbers so training stays stable",
    "ReLU": "keeps positive responses, discards negative ones (adds non-linearity)",
    "MaxPool2d": "halves the resolution, keeping the strongest response in each 2x2 block",
    "ConvTranspose2d": "doubles the resolution back up (learned upsampling)",
    "Sigmoid": "squashes the output into the 0–1 intensity range of an image",
}

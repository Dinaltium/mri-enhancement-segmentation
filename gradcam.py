"""
gradcam.py   --   Grad-CAM attention maps for the segmentation model

The problem statement lists Grad-CAM / attention maps as a Stage-4 evaluation
method. Grad-CAM highlights WHERE the network looks when it decides "tumour",
giving an explainability heatmap - important for clinical trust ("show me why
the AI flagged this region").

We hook the U-Net bottleneck (the deepest, most semantic features), backprop
the tumour-class score, and weight the feature maps by their gradients.

Reusable: grad_cam(model, x) -> 224x224 heatmap in [0,1]; cam_overlay(...).
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nifti_utils import IMG_SIZE, load_volume, normalize_volume
from models import SegmentationUNet
from brain_dataset import MODALITY_ORDER


def grad_cam(model, x: torch.Tensor, target_classes=(1, 2, 3)) -> np.ndarray:
    """Seg-Grad-CAM: x is 1x4xHxW. Returns a HxW heatmap in [0,1] showing where
    the model looks to decide 'tumour'. The class score is restricted to the
    predicted-tumour pixels (not the whole image), which localises the map on
    the lesion instead of smearing it to the edges."""
    model.eval()
    store = {}

    def fwd_hook(_m, _i, o):
        store["act"] = o.detach()

    def bwd_hook(_m, _gi, go):
        store["grad"] = go[0].detach()

    layer = model.backbone.dec2                             # finer than bottleneck
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)                                   # 1,C,H,W
        tumour = logits[:, list(target_classes), :, :].sum(dim=1)  # 1,H,W
        pred = logits.argmax(dim=1)                         # 1,H,W
        region = torch.zeros_like(pred, dtype=torch.bool)
        for c in target_classes:
            region |= (pred == c)
        score = tumour[region].sum() if region.any() else tumour.sum()
        model.zero_grad()
        score.backward()
        A = store["act"]                                    # 1,K,h,w
        G = store["grad"]                                   # 1,K,h,w
        w = G.mean(dim=(2, 3), keepdim=True)                # 1,K,1,1
        cam = F.relu((w * A).sum(dim=1))[0]                 # h,w
    finally:
        h1.remove(); h2.remove()
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
    # restrict to the brain (foreground) so edge/background upsampling
    # artefacts (the red border) don't show, and normalise within the brain
    fg = (x[0, -1].detach().cpu().numpy() > 0.05)   # FLAIR channel
    cam = cam * fg
    cam = np.clip(cam - cam.min(), 0, None)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cam * fg
    return cam


def cam_overlay(gray_slice: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Jet heatmap over the grayscale slice — only on the brain, so the black
    background stays clean (no coloured border artefact)."""
    mask = gray_slice > 0.05
    base = cv2.cvtColor((np.clip(gray_slice, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base, 0.55, heat, 0.55, 0)
    out = base.copy()
    out[mask] = blended[mask]
    return out


def load_seg(ckpt="segmentation_model.pt", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(ckpt, map_location=device)
    m = SegmentationUNet(num_classes=c.get("num_classes", 4),
                         in_channels=c.get("in_channels", 4),
                         base_filters=c.get("base_filters", 32)).to(device)
    m.load_state_dict(c["model_state_dict"])
    return m, device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case_dir", required=True,
                    help="a folder with <name>_{flair,t1,t1ce,t2}.nii")
    ap.add_argument("--out_dir", default="outputs/gradcam")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, device = load_seg()
    name = os.path.basename(args.case_dir.rstrip("/\\"))
    mods = {}
    for m in MODALITY_ORDER:
        p = os.path.join(args.case_dir, f"{name}_{m}.nii")
        if os.path.exists(p):
            mods[m] = normalize_volume(load_volume(p))
    if len(mods) < 4:
        print("need all 4 modalities"); return
    # prefer the slice with the most tumour (from ground truth if available),
    # else the slice with the most brain signal
    segp = os.path.join(args.case_dir, f"{name}_seg.nii")
    if os.path.exists(segp):
        seg = load_volume(segp)
        z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    else:
        z = int(np.argmax([np.count_nonzero(mods["flair"][:, :, k])
                           for k in range(mods["flair"].shape[2])]))
    stack = [np.clip(cv2.resize(mods[m][:, :, z], (IMG_SIZE, IMG_SIZE)), 0, 1) for m in MODALITY_ORDER]
    x = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(device)
    cam = grad_cam(model, x)
    cv2.imwrite(os.path.join(args.out_dir, f"{name}_gradcam.png"),
                cam_overlay(stack[3], cam))
    print(f"[gradcam] wrote {args.out_dir}/{name}_gradcam.png")


if __name__ == "__main__":
    main()

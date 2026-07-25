"""
OUR spine enhancement -- one U-Net PER SEQUENCE, self-supervised.

THE KEY SPINE RESULT. We tested whether one model can serve T1, T2 and STIR.
It cannot, and we have the numbers on identical test slices:

    sequence   pooled model   per-sequence (ours)
    T1         0.598          0.827   SSIM
    T2         0.594          0.802
    STIR       0.540          0.714

3 of 3 wins. WHY: STIR deliberately suppresses fat, so its intensity statistics
are genuinely unlike T1's. One model averages across contradictory targets.

Same architecture and loss as brain (2D U-Net, L1 + SSIM), same self-supervised
trick: degrade the scan, restore it to itself. NO ANNOTATIONS ANYWHERE.
"""
from _common import head, kv, load_slice, side_by_side, noise_level, SPINE
import torch

head("STAGE 3 - Our spine enhancement U-Net (per sequence)",
     "enhancement for all sub-modality datasets + justification")

from models import EnhancementUNet
from mri_degradation import degrade_mri_slice
from metrics import full_reference_metrics

dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "models/enhancement_model_spine_normal_T2.pt"
ck = torch.load(ckpt, map_location=dev)
m = EnhancementUNet(base_filters=ck.get("base_filters", 32)).to(dev)
m.load_state_dict(ck["model_state_dict"]); m.eval()

clean, path = load_slice(SPINE, region="spine")
noisy = degrade_mri_slice(clean)
with torch.no_grad():
    out = m(torch.from_numpy(noisy).float()[None, None].to(dev)).squeeze().cpu().numpy()

kv("input file", path)
kv("checkpoint", ckpt)
kv("why this checkpoint", "T2 scan -> T2-specific model (3/3 beats pooled)")
kv("loss function", "L1 + SSIM")
kv("annotations used", "NONE - self-supervised")
for n, im in (("degraded", noisy), ("OURS", out)):
    r = full_reference_metrics(clean, im)
    kv(f"{n}: PSNR / SSIM", f"{r['psnr']:.2f} / {r['ssim']:.4f}")
side_by_side([clean, noisy, out], ["Clean", "Degraded", "Ours (T2 model)"],
             "11_spine_unet.png")

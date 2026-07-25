"""
compare_modality_models.py

Fair head-to-head: does a MODALITY-SPECIFIC spine enhancement model beat the
POOLED one (trained on all spine modalities together)?

The trap this avoids: each training run reports metrics on its own test set,
so "pooled 0.82 vs T1-specific 0.49" is meaningless — different test images.
Here BOTH models are evaluated on the SAME per-modality held-out test slices,
which is the only fair comparison.

Output: modality_comparison.json + a console table.
"""

import json
import os

import numpy as np
import torch

from nifti_utils import IMG_SIZE, load_volume
from enhancement_dataset import extract_training_slices, split_offline_cases
from offline_dataset import OFFLINE_ROOTS, classify_case_files
from mri_degradation import degrade_mri_slice
from models import EnhancementUNet
from spine_pipeline import clahe_enhance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUP = "spine_normal"
MODALITIES = ["T1", "T2", "STIR"]
MAX_SLICES = 60


def load_enh(path):
    if not os.path.exists(path):
        return None
    c = torch.load(path, map_location=DEVICE)
    m = EnhancementUNet(base_filters=c.get("base_filters", 32)).to(DEVICE).eval()
    m.load_state_dict(c["model_state_dict"])
    return m


def run(model, x):
    with torch.no_grad():
        t = torch.from_numpy(x).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
        return np.clip(model(t)[0, 0].cpu().numpy(), 0, 1)


def iqa(clean, test):
    from metrics import _get_iqa_metric, _to_tensor
    ct, tt = _to_tensor(clean), _to_tensor(test)
    return (float(_get_iqa_metric("psnr")(tt, ct).item()),
            float(_get_iqa_metric("ssim")(tt, ct).item()))


def test_slices(modality):
    """Held-out (test-half) slices of one modality."""
    _, test_cases = split_offline_cases(OFFLINE_ROOTS[GROUP])
    out = []
    for cd in test_cases:
        info = classify_case_files(cd)
        for p in info["buckets"].get(modality, []):
            try:
                out.extend(extract_training_slices(load_volume(p)))
            except Exception:
                continue
    return out[:MAX_SLICES]


def main():
    pooled = load_enh(f"models/enhancement_model_{GROUP}.pt")
    if pooled is None:
        print("pooled model missing"); return
    results = {}
    rng_seed = 11

    print(f"\n{'modality':8} {'n':>4} | {'input':>14} | {'CLAHE':>14} | "
          f"{'pooled':>14} | {'modality-specific':>18}")
    print("-" * 86)

    for mod in MODALITIES:
        spec = load_enh(f"models/enhancement_model_{GROUP}_{mod}.pt")
        sls = test_slices(mod)
        if not sls:
            continue
        acc = {k: [] for k in ["input", "clahe", "pooled", "specific"]}
        for i, clean in enumerate(sls):
            rng = np.random.default_rng(rng_seed + i)
            deg = degrade_mri_slice(clean, rng)
            acc["input"].append(iqa(clean, deg))
            acc["clahe"].append(iqa(clean, clahe_enhance(deg)))
            acc["pooled"].append(iqa(clean, run(pooled, deg)))
            if spec is not None:
                acc["specific"].append(iqa(clean, run(spec, deg)))

        def mean(k):
            if not acc[k]:
                return None
            a = np.array(acc[k])
            return {"psnr": round(float(a[:, 0].mean()), 2), "ssim": round(float(a[:, 1].mean()), 3)}

        r = {k: mean(k) for k in acc}
        r["n_slices"] = len(sls)
        results[mod] = r

        def fmt(k):
            v = r[k]
            return f"{v['psnr']:.1f}/{v['ssim']:.3f}" if v else "—"
        print(f"{mod:8} {len(sls):>4} | {fmt('input'):>14} | {fmt('clahe'):>14} | "
              f"{fmt('pooled'):>14} | {fmt('specific'):>18}")

    # verdict
    wins = 0
    for mod, r in results.items():
        if r.get("specific") and r.get("pooled"):
            if r["specific"]["ssim"] > r["pooled"]["ssim"]:
                wins += 1
    verdict = (f"modality-specific wins on {wins}/{len(results)} modalities "
               f"(SSIM, same test slices)")
    print(f"\nVERDICT: {verdict}")
    results["_verdict"] = verdict
    results["_note"] = ("Both models evaluated on the SAME held-out per-modality test slices "
                        "with identical synthetic degradation, so the comparison is fair.")
    with open("results/modality_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[compare] wrote modality_comparison.json")


if __name__ == "__main__":
    main()

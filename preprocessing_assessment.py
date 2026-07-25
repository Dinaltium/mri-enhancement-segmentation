"""
preprocessing_assessment.py

Stage-2 gap-fill. The problem statement says explicitly:

  "Once again, the image property assessment of the preprocessed MRI Dataset
   must be done, with Contrast, complexity, Sharpness, Edge strength, Noise
   level, Mean, Deviation parameters."

Stage 1 measured those 7 properties on the RAW volumes. This measures the SAME
7 properties again after each processing step, so the effect of preprocessing
is quantified rather than asserted:

    RAW  ->  PREPROCESSED (normalise + resize + slice)
         ->  HE           (classical)
         ->  CLAHE        (classical)
         ->  AI U-Net     (ours)

Output: stats/preprocessing_assessment.json + .csv + console table.
"""

import csv
import json
import os

import cv2
import numpy as np
import torch

from dataset_stats import estimate_noise_immerkaer
from nifti_utils import IMG_SIZE, load_volume, normalize_volume
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files
from models import EnhancementUNet
from spine_pipeline import clahe_enhance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SLICES_PER_GROUP = 40


def slice_properties(img: np.ndarray) -> dict:
    """The 7 mandated properties, measured on a single 2D slice in [0,1]."""
    fg = img[img > 0.02]
    if fg.size < 10:
        fg = img.flatten()
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    contrast = float((p99 - p1) / (p99 + p1 + 1e-8))
    hist, _ = np.histogram((img * 255).astype(np.uint8), bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    complexity = float(-(p * np.log2(p)).sum())
    sharpness = float(cv2.Laplacian(img, cv2.CV_32F).var())
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return {
        "mean": float(fg.mean()), "deviation": float(fg.std()),
        "contrast": contrast, "complexity": complexity,
        "sharpness": sharpness, "edge_strength": edge,
        "noise_level": float(estimate_noise_immerkaer(img)),
    }


def he_enhance(img):
    return cv2.equalizeHist((np.clip(img, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255


def load_enh(path):
    if not os.path.exists(path):
        return None
    c = torch.load(path, map_location=DEVICE)
    m = EnhancementUNet(base_filters=c.get("base_filters", 32)).to(DEVICE).eval()
    m.load_state_dict(c["model_state_dict"])
    return m


def run_ai(model, img):
    if model is None:
        return None
    with torch.no_grad():
        x = torch.from_numpy(img).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
        return np.clip(model(x)[0, 0].cpu().numpy(), 0, 1)


def collect(group, limit=MAX_SLICES_PER_GROUP):
    """(raw_volume_slice, preprocessed_slice) pairs. 'raw' = native resolution
    slice with only [0,1] scaling; 'preprocessed' = our Stage-2 pipeline."""
    out = []
    for cd in find_offline_cases(OFFLINE_ROOTS[group]):
        info = classify_case_files(cd)
        for mod, paths in info["buckets"].items():
            if mod == "unclassified":
                continue
            for p in paths:
                try:
                    vol = load_volume(p)
                except Exception:
                    continue
                normed = normalize_volume(vol)
                pre = extract_training_slices(normed)      # Stage-2 output
                if not pre:
                    continue
                z = normed.shape[2] // 2
                raw = normed[:, :, z]                       # native size, no resize
                out.append((raw, pre[len(pre) // 2]))
                if len(out) >= limit:
                    return out
    return out


def main():
    os.makedirs("stats", exist_ok=True)
    groups = ["brain_normal", "brain_pathological", "spine_normal", "spine_pathological"]
    enh_models = {
        "brain_normal": load_enh("enhancement_model_brain.pt"),
        "brain_pathological": load_enh("enhancement_model_brain.pt"),
        "spine_normal": load_enh("enhancement_model_spine_normal.pt"),
        "spine_pathological": load_enh("enhancement_model_spine_pathological.pt"),
    }
    report, csv_rows = {}, []
    props7 = ["mean", "deviation", "contrast", "complexity", "sharpness",
              "edge_strength", "noise_level"]

    for g in groups:
        pairs = collect(g)
        if not pairs:
            continue
        model = enh_models.get(g)
        stages = {k: [] for k in ["raw", "preprocessed", "HE", "CLAHE", "AI_UNet"]}
        for raw, pre in pairs:
            stages["raw"].append(slice_properties(raw))
            stages["preprocessed"].append(slice_properties(pre))
            stages["HE"].append(slice_properties(he_enhance(pre)))
            stages["CLAHE"].append(slice_properties(clahe_enhance(pre)))
            ai = run_ai(model, pre)
            if ai is not None:
                stages["AI_UNet"].append(slice_properties(ai))

        report[g] = {}
        for stage, lst in stages.items():
            if not lst:
                continue
            avg = {p: round(float(np.mean([d[p] for d in lst])), 4) for p in props7}
            report[g][stage] = avg
            csv_rows.append({"dataset": g, "stage": stage, "n_slices": len(lst), **avg})

        print(f"\n=== {g}  (n={len(pairs)} slices) ===")
        print(f"{'stage':14}" + "".join(f"{p[:9]:>11}" for p in props7))
        print("-" * (14 + 11 * len(props7)))
        for stage in ["raw", "preprocessed", "HE", "CLAHE", "AI_UNet"]:
            if stage in report[g]:
                a = report[g][stage]
                print(f"{stage:14}" + "".join(f"{a[p]:>11.4f}" for p in props7))

    report["_note"] = ("Same 7 properties as Stage 1, re-measured after each processing step. "
                       "'raw' = native-resolution slice, intensity-scaled only. "
                       "'preprocessed' = Stage-2 pipeline (robust normalise, empty-slice removal, "
                       "224x224 resample with clipping). HE/CLAHE/AI_UNet are applied to the "
                       "preprocessed slice.")
    with open("stats/preprocessing_assessment.json", "w") as f:
        json.dump(report, f, indent=2)
    if csv_rows:
        with open("stats/preprocessing_assessment.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
    print("\n[preproc-assess] wrote stats/preprocessing_assessment.json + .csv")


if __name__ == "__main__":
    main()

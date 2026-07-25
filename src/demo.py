"""
demo.py   --   Judge-facing visual demonstration (Stage 1 + 2 + Stage 3)

Produces the "dirty MRI in -> clean MRI out" visuals and the dataset-analysis
charts used to explain the project to judges (technical + non-technical).

Outputs (outputs/demo/):
    brain_enhancement_demo.png   clean | degraded | CLAHE | AI-enhanced  (+PSNR/SSIM)
    spine_enhancement_demo.png   same, for spine
    real_scan_enhancement.png    a real offline scan, raw vs AI-enhanced
    dataset_properties.png       grouped bar chart of the 7 image properties
    pipeline_diagram.png         the end-to-end flow, drawn simply

Run LIVE in front of judges:
    python demo.py                       # regenerates every panel
    python demo.py --brats_root data/brats_subset
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nifti_utils import IMG_SIZE, load_volume, normalize_volume, find_brats_cases, load_brats_case
from enhancement_dataset import extract_training_slices, split_offline_cases
from mri_degradation import degrade_mri_slice
from models import EnhancementUNet
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files
from spine_pipeline import clahe_enhance

OUT = "outputs/demo"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_enh(ckpt):
    c = torch.load(ckpt, map_location=DEVICE)
    m = EnhancementUNet(base_filters=c.get("base_filters", 32)).to(DEVICE).eval()
    m.load_state_dict(c["model_state_dict"])
    return m


def _metrics(clean, test):
    """PSNR + SSIM via pyiqa (cached)."""
    try:
        from metrics import _get_iqa_metric, _to_tensor
        ct, tt = _to_tensor(clean), _to_tensor(test)
        psnr = float(_get_iqa_metric("psnr")(tt, ct).item())
        ssim = float(_get_iqa_metric("ssim")(tt, ct).item())
        return psnr, ssim
    except Exception:
        return None, None


def enhance(model, img):
    with torch.no_grad():
        x = torch.from_numpy(img).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
        return np.clip(model(x)[0, 0].cpu().numpy(), 0, 1)


def four_panel(clean, model, title, path, seed=7):
    """clean | degraded | CLAHE | AI-enhanced, annotated with PSNR/SSIM."""
    rng = np.random.default_rng(seed)
    degraded = degrade_mri_slice(clean, rng)
    clahe = clahe_enhance(degraded)
    ai = enhance(model, degraded)

    panels = [("Original (clean)", clean, None),
              ("Degraded input\n(noise + artifact)", degraded, _metrics(clean, degraded)),
              ("Classical CLAHE\n(baseline)", clahe, _metrics(clean, clahe)),
              ("AI-Enhanced (ours)", ai, _metrics(clean, ai))]

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    for a, (label, img, m) in zip(ax, panels):
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        sub = label
        if m and m[0] is not None:
            sub += f"\nPSNR {m[0]:.1f} dB  SSIM {m[1]:.2f}"
        a.set_title(sub, fontsize=11)
        a.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[demo] wrote {path}")


def real_scan_panel(model, img, title, path):
    ai = enhance(model, img)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.8))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[0].set_title("Real scan (raw)"); ax[0].axis("off")
    ax[1].imshow(ai, cmap="gray", vmin=0, vmax=1); ax[1].set_title("After AI enhancement"); ax[1].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[demo] wrote {path}")


def pick_brain_flair(brats_root):
    cases = find_brats_cases(brats_root)
    for cd in cases:
        c = load_brats_case(cd)
        if "flair" in c["modalities"]:
            sls = extract_training_slices(c["modalities"]["flair"])
            if len(sls) > 20:
                return sls[len(sls) // 2]
    return None


def pick_spine_t2(group="spine_normal"):
    _, test = split_offline_cases(OFFLINE_ROOTS[group])
    for cd in test:
        info = classify_case_files(cd)
        for path in info["buckets"].get("T2", []):
            sls = extract_training_slices(load_volume(path))
            if len(sls) > 5:
                return sls[len(sls) // 2]
    return None


def dataset_properties_chart():
    """Grouped bar chart of the 7 properties across the datasets, from the
    stats CSV if present."""
    csv = "stats/dataset_stats.csv"
    if not os.path.exists(csv):
        print("[demo] stats CSV missing, skip properties chart"); return
    import csv as csvmod
    rows = list(csvmod.DictReader(open(csv)))
    # average per dataset across modalities for a clean summary
    props = ["contrast", "complexity", "sharpness", "edge_strength", "noise_level"]
    datasets = ["brats2020", "brain_pathological", "spine_normal", "spine_pathological"]
    labels = {"brats2020": "Brain (BraTS)", "brain_pathological": "Brain offline",
              "spine_normal": "Spine normal", "spine_pathological": "Spine path."}
    data = {d: {p: [] for p in props} for d in datasets}
    for r in rows:
        if r["dataset"] in data:
            for p in props:
                try:
                    data[r["dataset"]][p].append(float(r[f"{p}_mean"]))
                except (ValueError, KeyError):
                    pass
    means = {d: [np.mean(data[d][p]) if data[d][p] else 0 for p in props] for d in datasets}

    x = np.arange(len(props)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, d in enumerate(datasets):
        ax.bar(x + (i - 1.5) * w, means[d], w, label=labels[d])
    ax.set_xticks(x); ax.set_xticklabels([p.replace("_", " ").title() for p in props])
    ax.set_ylabel("mean value"); ax.set_title("Dataset Image-Property Analysis (Stage 1)",
                                               fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "dataset_properties.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[demo] wrote {p}")


def pipeline_diagram():
    """Simple left-to-right flow diagram of the framework."""
    fig, ax = plt.subplots(figsize=(13, 3.2))
    steps = ["Raw MRI\n(.nii)", "Stage 1\nAnalyze\nproperties",
             "Stage 2\nPreprocess\n+ CLAHE", "Stage 3\nAI Enhance\n(U-Net)",
             "Stage 4\nROI Segment", "Clean scan\n+ ROI mask\n(COCO)"]
    colors = ["#cfd8dc", "#b3e5fc", "#c8e6c9", "#fff9c4", "#ffccbc", "#d1c4e9"]
    n = len(steps)
    for i, (s, c) in enumerate(zip(steps, colors)):
        ax.add_patch(plt.Rectangle((i * 2.1, 0), 1.8, 2, facecolor=c, edgecolor="black"))
        ax.text(i * 2.1 + 0.9, 1, s, ha="center", va="center", fontsize=10, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(i * 2.1 + 2.1, 1), xytext=(i * 2.1 + 1.8, 1),
                        arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(-0.2, n * 2.1); ax.set_ylim(-0.3, 2.3); ax.axis("off")
    ax.set_title("End-to-End MRI Enhancement + Segmentation Framework",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    p = os.path.join(OUT, "pipeline_diagram.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[demo] wrote {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brats_root", default="data/brats_subset")
    args = ap.parse_args()

    print(f"[demo] device={DEVICE}")
    pipeline_diagram()
    dataset_properties_chart()

    # brain enhancement demo (model trained on BraTS FLAIR)
    if os.path.exists("models/enhancement_model_brain.pt"):
        clean = pick_brain_flair(args.brats_root)
        if clean is not None:
            four_panel(clean, load_enh("models/enhancement_model_brain.pt"),
                       "Brain MRI (FLAIR) — AI Enhancement Demonstration",
                       os.path.join(OUT, "brain_enhancement_demo.png"))

    # spine enhancement demo
    if os.path.exists("models/enhancement_model_spine_normal.pt"):
        clean = pick_spine_t2("spine_normal")
        if clean is not None:
            four_panel(clean, load_enh("models/enhancement_model_spine_normal.pt"),
                       "Spine MRI (T2) — AI Enhancement Demonstration",
                       os.path.join(OUT, "spine_enhancement_demo.png"))
        # real raw scan, raw vs enhanced (no synthetic degradation)
        if clean is not None:
            real_scan_panel(load_enh("models/enhancement_model_spine_normal.pt"), clean,
                            "Real Spine Scan — Raw vs AI-Enhanced",
                            os.path.join(OUT, "real_scan_enhancement.png"))

    print("[demo] done -> outputs/demo/")


if __name__ == "__main__":
    main()

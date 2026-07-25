"""
generate_demo_assets.py

Generates the full set of figures for the comprehensive demo page:
  - enhancement_compare_brain/spine.png : Original|Degraded|HE|CLAHE|AI (+PSNR/SSIM)
  - seg_confusion_matrix.png            : segmentation confusion matrix (val)
  - seg_perclass_metrics.png            : per-class Dice/Jaccard bar chart
  - tumor_vs_gt.png                     : AI prediction vs doctor ground truth
  - tissue_csf_gm_wm.png                : CSF/GM/WM healthy tissue segmentation
  - gradcam_attention.png               : Grad-CAM attention on the tumour
(plus it reuses dataset_properties / pipeline_diagram / *_curves already made)
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nifti_utils import (IMG_SIZE, load_volume, normalize_volume, remap_brats_labels,
                         find_brats_cases, load_brats_case)
from brain_dataset import split_cases, MODALITY_ORDER, BrainSegmentationDataset
from enhancement_dataset import extract_training_slices, split_offline_cases
from mri_degradation import degrade_mri_slice
from models import EnhancementUNet, SegmentationUNet
from spine_pipeline import clahe_enhance, colorize_labels
from offline_dataset import OFFLINE_ROOTS, classify_case_files
from tissue_segmentation import segment_tissues, tissue_overlay, pick_t1_slice, tissue_fractions
from gradcam import grad_cam, cam_overlay

OUT = "outputs/demo"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["background", "necrotic", "edema", "enhancing"]


def he_enhance(img01):
    return cv2.equalizeHist((np.clip(img01, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255


def _iqa(clean, test):
    from metrics import _get_iqa_metric, _to_tensor
    ct, tt = _to_tensor(clean), _to_tensor(test)
    return (round(float(_get_iqa_metric("psnr")(tt, ct).item()), 1),
            round(float(_get_iqa_metric("ssim")(tt, ct).item()), 2))


def load_enh(ck):
    c = torch.load(ck, map_location=DEV)
    m = EnhancementUNet(base_filters=c.get("base_filters", 32)).to(DEV).eval()
    m.load_state_dict(c["model_state_dict"]); return m


def load_seg():
    c = torch.load("segmentation_model.pt", map_location=DEV)
    m = SegmentationUNet(num_classes=4, in_channels=4, base_filters=c.get("base_filters", 32)).to(DEV).eval()
    m.load_state_dict(c["model_state_dict"]); return m


def enhance(model, s):
    with torch.no_grad():
        x = torch.from_numpy(s).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(DEV)
        return np.clip(model(x)[0, 0].cpu().numpy(), 0, 1)


def five_panel(clean, model, title, path, seed=7):
    rng = np.random.default_rng(seed)
    deg = degrade_mri_slice(clean, rng)
    panels = [("Original\n(clean)", clean, None),
              ("Degraded\ninput", deg, _iqa(clean, deg)),
              ("HE\n(classical)", he_enhance(deg), _iqa(clean, he_enhance(deg))),
              ("CLAHE\n(classical)", clahe_enhance(deg), _iqa(clean, clahe_enhance(deg))),
              ("AI U-Net\n(ours)", enhance(model, deg), _iqa(clean, enhance(model, deg)))]
    fig, ax = plt.subplots(1, 5, figsize=(17, 4.2))
    for a, (lab, im, m) in zip(ax, panels):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        t = lab + (f"\nPSNR {m[0]}  SSIM {m[1]}" if m else "")
        a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=125, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def confusion_and_perclass():
    """Confusion matrix + per-class Dice/Jaccard on a small val set."""
    tr, va = split_cases("data/brats_subset")
    ds = BrainSegmentationDataset(va[:4], augment=False)
    model = load_seg()
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8)
    cm = np.zeros((4, 4), dtype=np.int64)
    inter = np.zeros(4); psum = np.zeros(4); gsum = np.zeros(4); union = np.zeros(4)
    with torch.no_grad():
        for xb, yb in dl:
            pr = torch.argmax(model(xb.to(DEV)), 1).cpu().numpy()
            gt = yb.numpy()
            for c_t in range(4):
                for c_p in range(4):
                    cm[c_t, c_p] += int(((gt == c_t) & (pr == c_p)).sum())
            for c in range(4):
                p = pr == c; g = gt == c
                inter[c] += (p & g).sum(); psum[c] += p.sum(); gsum[c] += g.sum(); union[c] += (p | g).sum()
    # row-normalized confusion matrix (recall per class)
    cmn = cm / (cm.sum(1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (ground truth)")
    ax.set_title("Segmentation Confusion Matrix\n(row-normalised = recall)", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seg_confusion_matrix.png"), dpi=125); plt.close(fig)
    print("wrote seg_confusion_matrix.png")

    dice = [2 * inter[c] / (psum[c] + gsum[c] + 1e-9) for c in range(1, 4)]
    jac = [inter[c] / (union[c] + 1e-9) for c in range(1, 4)]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, dice, w, label="Dice", color="#4fc3f7")
    ax.bar(x + w/2, jac, w, label="Jaccard (IoU)", color="#66bb6a")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES[1:])
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.legend()
    ax.set_title("Per-class Segmentation Accuracy (validation)", fontweight="bold")
    for i, (d, j) in enumerate(zip(dice, jac)):
        ax.text(i - w/2, d + .02, f"{d:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, j + .02, f"{j:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seg_perclass_metrics.png"), dpi=125); plt.close(fig)
    print("wrote seg_perclass_metrics.png")


def _first_case():
    root = "showcase/for_tumor_detection"
    cs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    return os.path.join(root, cs[0]) if cs else None


def tumor_vs_gt(case=None):
    case = case or _first_case()
    name = os.path.basename(case)
    mods = {m: normalize_volume(load_volume(os.path.join(case, f"{name}_{m}.nii"))) for m in MODALITY_ORDER}
    seg = load_volume(os.path.join(case, f"{name}_seg.nii"))
    z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    model = load_seg()
    stack = [np.clip(cv2.resize(mods[m][:, :, z], (IMG_SIZE, IMG_SIZE)), 0, 1) for m in MODALITY_ORDER]
    x = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(DEV)
    with torch.no_grad():
        pred = torch.argmax(model(x), 1)[0].cpu().numpy().astype(np.uint8)
    gt = remap_brats_labels(cv2.resize(seg[:, :, z], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.uint8))
    flair = stack[3]
    d = 2*np.logical_and(pred > 0, gt > 0).sum()/((pred > 0).sum()+(gt > 0).sum()+1e-8)
    fb = cv2.cvtColor((flair*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(flair, cmap="gray"); ax[0].set_title("Brain scan (FLAIR)"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(cv2.addWeighted(fb, .7, colorize_labels(pred, 3), .5, 0), cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"AI detection — {d*100:.0f}% overlap"); ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(cv2.addWeighted(fb, .7, colorize_labels(gt, 3), .5, 0), cv2.COLOR_BGR2RGB))
    ax[2].set_title("Doctor's label (ground truth)"); ax[2].axis("off")
    fig.suptitle("Brain Tumour Detection vs Expert Label", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(os.path.join(OUT, "tumor_vs_gt.png"), dpi=125); plt.close(fig)
    print("wrote tumor_vs_gt.png")


def tissue_figure(case=None):
    case = case or _first_case()
    name = os.path.basename(case)
    t1 = pick_t1_slice(load_volume(os.path.join(case, f"{name}_t1.nii")))
    labels = segment_tissues(t1); fr = tissue_fractions(labels)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ax[0].imshow(t1, cmap="gray"); ax[0].set_title("Healthy brain (T1)"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(tissue_overlay(t1, labels), cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"CSF/GM/WM segmentation\nCSF {fr.get('CSF',0):.0%}  GM {fr.get('grey_matter',0):.0%}  WM {fr.get('white_matter',0):.0%}")
    ax[1].axis("off")
    fig.suptitle("Healthy-Brain Tissue Segmentation (unsupervised)  ·  blue=CSF green=grey red=white",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(OUT, "tissue_csf_gm_wm.png"), dpi=125); plt.close(fig)
    print("wrote tissue_csf_gm_wm.png")


def gradcam_figure(case=None):
    case = case or _first_case()
    name = os.path.basename(case)
    mods = {m: normalize_volume(load_volume(os.path.join(case, f"{name}_{m}.nii"))) for m in MODALITY_ORDER}
    seg = load_volume(os.path.join(case, f"{name}_seg.nii"))
    z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
    stack = [np.clip(cv2.resize(mods[m][:, :, z], (IMG_SIZE, IMG_SIZE)), 0, 1) for m in MODALITY_ORDER]
    x = torch.from_numpy(np.stack(stack)).float().unsqueeze(0).to(DEV)
    cam = grad_cam(load_seg(), x)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ax[0].imshow(stack[3], cmap="gray"); ax[0].set_title("Brain scan (FLAIR)"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(cam_overlay(stack[3], cam), cv2.COLOR_BGR2RGB))
    ax[1].set_title("Grad-CAM: where the AI looks\n(red = focus)"); ax[1].axis("off")
    fig.suptitle("Explainability — Grad-CAM Attention", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(OUT, "gradcam_attention.png"), dpi=125); plt.close(fig)
    print("wrote gradcam_attention.png")


def spine_anomaly_figure():
    """Self-supervised lesion localisation on a pathological spine."""
    from spine_autoencoder import load_model, anomaly_map, overlay_anomaly
    from offline_dataset import find_offline_cases
    ae, mod = load_model()
    best = None
    for cdir in find_offline_cases(OFFLINE_ROOTS["spine_pathological"]):
        info = classify_case_files(cdir)
        for p in info["buckets"].get(mod, [])[:1]:
            sls = extract_training_slices(load_volume(p))
            if not sls:
                continue
            img = sls[len(sls) // 2]
            recon, heat = anomaly_map(ae, img)
            score = float(heat.mean())
            if best is None or score > best[0]:
                best = (score, img, recon, heat, os.path.basename(cdir))
            break
    if best is None:
        print("no spine anomaly case"); return
    score, img, recon, heat, name = best
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("1. Pathological spine (input)"); ax[0].axis("off")
    ax[1].imshow(recon, cmap="gray")
    ax[1].set_title("2. AI's 'healthy' reconstruction"); ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(overlay_anomaly(img, heat), cv2.COLOR_BGR2RGB))
    ax[2].set_title(f"3. Anomaly map — suspected region\n(score {score:.3f})"); ax[2].axis("off")
    fig.suptitle("Spine — Self-Supervised Lesion Localisation (no labels used)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "spine_anomaly.png"), dpi=125); plt.close(fig)
    print("wrote spine_anomaly.png")


def spine_slic_figure():
    """k-means (old) vs SLIC superpixels (new) on the same spine slice."""
    from spine_pipeline import kmeans_roi, slic_roi
    from enhancement_dataset import split_offline_cases as _split
    _, test = _split(OFFLINE_ROOTS["spine_pathological"])
    img = None
    for cdir in test:
        info = classify_case_files(cdir)
        for p in info["buckets"].get("T2", [])[:1]:
            sls = extract_training_slices(load_volume(p))
            if sls:
                img = sls[len(sls) // 2]
            break
        if img is not None:
            break
    if img is None:
        print("no spine slice"); return
    enh = clahe_enhance(img)
    km, sl = kmeans_roi(enh, k=4), slic_roi(enh, n_segments=250, k=4)
    base = cv2.cvtColor((enh * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Spine scan"); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(cv2.addWeighted(base, .65, colorize_labels(km, 3), .6, 0), cv2.COLOR_BGR2RGB))
    ax[1].set_title("Pixel k-means (baseline)\nspeckly"); ax[1].axis("off")
    ax[2].imshow(cv2.cvtColor(cv2.addWeighted(base, .65, colorize_labels(sl, 3), .6, 0), cv2.COLOR_BGR2RGB))
    ax[2].set_title("SLIC superpixels (ours)\ncoherent regions"); ax[2].axis("off")
    fig.suptitle("Spine ROI Segmentation — method comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "spine_slic_compare.png"), dpi=125); plt.close(fig)
    print("wrote spine_slic_compare.png")


def main():
    # enhancement comparisons (with HE + CLAHE)
    tr, va = split_cases("data/brats_subset")
    for cd in find_brats_cases("data/brats_subset"):
        c = load_brats_case(cd)
        if "flair" in c["modalities"]:
            sls = extract_training_slices(c["modalities"]["flair"])
            if len(sls) > 20:
                five_panel(sls[len(sls)//2], load_enh("enhancement_model_brain.pt"),
                           "Brain MRI (FLAIR) — Enhancement: classical methods vs our AI",
                           os.path.join(OUT, "enhancement_compare_brain.png")); break
    _, spine_test = split_offline_cases(OFFLINE_ROOTS["spine_normal"])
    for cd in spine_test:
        info = classify_case_files(cd)
        done = False
        for p in info["buckets"].get("T2", []):
            sls = extract_training_slices(load_volume(p))
            if len(sls) > 5:
                five_panel(sls[len(sls)//2], load_enh("enhancement_model_spine_normal.pt"),
                           "Spine MRI (T2) — Enhancement: classical methods vs our AI",
                           os.path.join(OUT, "enhancement_compare_spine.png")); done = True; break
        if done:
            break
    confusion_and_perclass()
    tumor_vs_gt()
    tissue_figure()
    gradcam_figure()
    print("\nall demo assets generated -> outputs/demo/")


if __name__ == "__main__":
    main()

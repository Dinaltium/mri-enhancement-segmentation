"""
spine_pipeline.py   --   Spine track (classical / unsupervised)

Per the strategic decision in CLAUDE.md: the offline Spine dataset has NO
ground-truth ROI annotations and no external data is permitted, so a
"trained supervised segmentation model" would be dishonest. The problem
statement itself suggests self-supervised / unsupervised methods for exactly
this situation. We therefore use:

    Enhancement    : CLAHE (Contrast Limited Adaptive Histogram Equalization)
                     - classical, fast, and named directly in the problem
                     statement's Stage 2 suggestions.
    Segmentation   : UNSUPERVISED intensity clustering (k-means) with an
                     Otsu multi-threshold alternative - framed honestly as
                     "exploratory ROI segmentation", separating candidate
                     disc / vertebra / CSF-cord / soft-tissue regions.

(The trained-U-Net enhancement path for Spine lives in
train_enhancement_offline.py; this file is the classical baseline + the
segmentation the Spine track relies on.)

Reusable functions (clahe_enhance, kmeans_roi, otsu_roi) are imported by
coco_export.py and inference_report.py. The CLI runs the full pipeline over
a spine dataset group and writes enhanced images, ROI masks, and overlay
panels for the slides.

Usage:
    python spine_pipeline.py --group spine_pathological --out_dir outputs/spine_path
    python spine_pipeline.py --group spine_normal --k 4
"""

import argparse
import json
import os

import cv2
import numpy as np

from nifti_utils import IMG_SIZE, load_volume, normalize_volume, save_slice_png
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files


def clahe_enhance(img01: np.ndarray, clip_limit: float = 2.0,
                  tile: int = 8) -> np.ndarray:
    """CLAHE on a [0,1] grayscale image -> [0,1]."""
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(u8).astype(np.float32) / 255.0


def kmeans_roi(img01: np.ndarray, k: int = 4, seed: int = 42,
               smooth: bool = True) -> np.ndarray:
    """Unsupervised intensity k-means on foreground pixels. Returns an int
    label map 0..k-1 where clusters are ordered by ascending mean intensity
    and label 0 is forced to background (near-zero pixels). A light median
    filter (smooth=True) removes salt-and-pepper speckle so ROI regions are
    spatially coherent - pure per-pixel intensity clustering has no spatial
    term, so this cheap regularisation noticeably improves the masks."""
    h, w = img01.shape
    flat = img01.reshape(-1, 1).astype(np.float32)
    fg_mask = (flat[:, 0] > 0.02)
    labels_full = np.zeros(h * w, dtype=np.int32)

    if fg_mask.sum() < k:
        return labels_full.reshape(h, w)

    fg = flat[fg_mask]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # cluster foreground into k-1 tissue classes (label 0 reserved for background)
    kk = max(2, k - 1)
    _compact, lbl, centers = cv2.kmeans(fg, kk, None, criteria, 3,
                                        cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers.flatten())          # dark -> bright
    remap = {old: new + 1 for new, old in enumerate(order)}  # 1..kk
    fg_labels = np.array([remap[int(c)] for c in lbl.flatten()], dtype=np.int32)
    labels_full[fg_mask] = fg_labels
    labels_map = labels_full.reshape(h, w)
    if smooth:
        labels_map = cv2.medianBlur(labels_map.astype(np.uint8), 3).astype(np.int32)
    return labels_map


def slic_roi(img01: np.ndarray, n_segments: int = 250, k: int = 4,
             compactness: float = 0.08) -> np.ndarray:
    """SLIC-superpixel ROI segmentation — an upgrade over per-pixel k-means.
    SLIC groups nearby similar pixels into ~n_segments spatially-coherent
    superpixels; we then cluster those superpixels by mean intensity into k
    tissue classes (ordered dark->bright). Result: clean, connected regions
    (disc/vertebra/cord/soft-tissue) instead of speckle."""
    from skimage.segmentation import slic
    from skimage.measure import regionprops

    fg = img01 > 0.02
    segments = slic(img01, n_segments=n_segments, compactness=compactness,
                    channel_axis=None, start_label=1, mask=fg)
    props = regionprops(segments, intensity_image=img01)
    if len(props) < k:
        return np.zeros_like(img01, dtype=np.int32)

    means = np.array([[p.mean_intensity] for p in props], dtype=np.float32)
    ids = [p.label for p in props]
    kk = max(2, k - 1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _c, lbl, cen = cv2.kmeans(means, kk, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    order = {int(old): new + 1 for new, old in enumerate(np.argsort(cen.flatten()))}
    out = np.zeros_like(segments, dtype=np.int32)
    for sid, cl in zip(ids, lbl.flatten()):
        out[segments == sid] = order[int(cl)]
    return out


def otsu_roi(img01: np.ndarray, classes: int = 3) -> np.ndarray:
    """Multi-level Otsu thresholding alternative. Returns int label map."""
    from skimage.filters import threshold_multiotsu
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    fg = u8[u8 > 5]
    if fg.size < classes or np.unique(fg).size < classes:
        return np.zeros_like(u8, dtype=np.int32)
    try:
        thresholds = threshold_multiotsu(u8[u8 > 5], classes=classes)
    except Exception:
        return np.zeros_like(u8, dtype=np.int32)
    labels = np.digitize(u8, bins=thresholds).astype(np.int32)
    labels[u8 <= 5] = 0
    return labels


def colorize_labels(labels: np.ndarray, k: int) -> np.ndarray:
    """Map an int label map to a BGR color image for overlays."""
    palette = np.array([
        [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0],
        [0, 255, 255], [255, 0, 255], [255, 255, 0],
    ], dtype=np.uint8)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for c in range(1, k + 1):
        out[labels == c] = palette[c % len(palette)]
    return out


def overlay_panel(orig01: np.ndarray, enhanced01: np.ndarray,
                  labels: np.ndarray, k: int) -> np.ndarray:
    """3-up panel: original | CLAHE-enhanced | ROI overlay (BGR uint8)."""
    o = cv2.cvtColor((orig01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    e = cv2.cvtColor((enhanced01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = colorize_labels(labels, k)
    ov = cv2.addWeighted(e, 0.7, color, 0.3, 0)
    return np.concatenate([o, e, ov], axis=1)


def pick_segmentation_slice(case_dir: str, prefer=("T2", "STIR", "T1")) -> np.ndarray | None:
    """Choose a representative sagittal slice for ROI work - prefer T2/STIR
    (best disc/CSF contrast). Returns a normalized [0,1] slice at IMG_SIZE."""
    info = classify_case_files(case_dir)
    for mod in prefer:
        for path in info["buckets"].get(mod, []):
            try:
                sls = extract_training_slices(load_volume(path))
            except Exception:
                continue
            if sls:
                return sls[len(sls) // 2]  # middle slice
    return None


def run_case(case_dir: str, out_dir: str, k: int) -> dict:
    """Full classical pipeline on one case's representative slices, writing
    enhanced PNGs, mask arrays, and overlay panels. Returns a manifest entry."""
    case_id = os.path.basename(case_dir)
    info = classify_case_files(case_dir)
    entries = []
    for mod in ["T2", "STIR", "T1", "T1c"]:
        paths = info["buckets"].get(mod, [])
        if not paths:
            continue
        # use the middle slice of the first volume of this modality
        try:
            sls = extract_training_slices(load_volume(paths[0]))
        except Exception:
            continue
        if not sls:
            continue
        orig = sls[len(sls) // 2]
        enhanced = clahe_enhance(orig)
        labels = kmeans_roi(enhanced, k=k)

        base = f"{case_id}_{mod}"
        save_slice_png(orig, os.path.join(out_dir, f"{base}_orig.png"))
        save_slice_png(enhanced, os.path.join(out_dir, f"{base}_clahe.png"))
        np.save(os.path.join(out_dir, f"{base}_roi.npy"), labels)
        cv2.imwrite(os.path.join(out_dir, f"{base}_panel.png"),
                    overlay_panel(orig, enhanced, labels, k))
        entries.append({"modality": mod, "base": base,
                        "roi_classes_present": [int(c) for c in np.unique(labels) if c > 0]})
    return {"case": case_id, "outputs": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[g for g in OFFLINE_ROOTS if g.startswith("spine")],
                        default="spine_pathological")
    parser.add_argument("--root", default=None)
    parser.add_argument("--k", type=int, default=4, help="number of intensity clusters (incl. bg)")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()

    root = args.root or OFFLINE_ROOTS[args.group]
    out_dir = args.out_dir or os.path.join("outputs", args.group)
    os.makedirs(out_dir, exist_ok=True)

    cases = find_offline_cases(root)
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"[spine_pipeline] {args.group}: {len(cases)} cases -> {out_dir}")

    manifest = []
    for case_dir in cases:
        entry = run_case(case_dir, out_dir, args.k)
        manifest.append(entry)
        print(f"   {entry['case']}: {[e['modality'] for e in entry['outputs']]}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"group": args.group, "k": args.k, "cases": manifest}, f, indent=2)
    print(f"[spine_pipeline] wrote overlays + masks + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()

"""
tissue_segmentation.py   --   Stage 4 for HEALTHY brains (CSF / GM / WM)

The problem statement asks, for *healthy* subjects, to segment the region of
interest into CSF (cerebrospinal fluid), grey matter, and white matter. There
are no CSF/GM/WM labels in the data, so we use the standard classical,
UNSUPERVISED approach: intensity-based tissue clustering (Gaussian Mixture,
falling back to k-means) on the brain region of a T1 scan. This is exactly how
classical tools (e.g. FSL FAST) partition brain tissue when no labels exist.

Tissue intensity ordering on T1: CSF darkest < grey matter < white matter.

Reusable functions (segment_tissues, tissue_overlay) are imported by the demo
webapp. CLI runs over BraTS T1 (skull-stripped -> cleanest) or any T1 volume.

Usage:
    python tissue_segmentation.py --brats_root data/brats_subset --n 4
"""

import argparse
import os

import cv2
import numpy as np

from nifti_utils import IMG_SIZE, load_volume, normalize_volume

TISSUE_NAMES = {1: "CSF", 2: "grey_matter", 3: "white_matter"}
# BGR colours: CSF blue, GM green, WM red-ish
TISSUE_COLORS = {1: (255, 90, 0), 2: (0, 210, 0), 3: (40, 40, 220)}


def _brain_mask(sl: np.ndarray) -> np.ndarray:
    """Foreground brain mask: threshold + keep largest component + fill holes.
    (For BraTS this is already the brain; for skull-on scans it trims a lot of
    the background but is only approximate.)"""
    m = (sl > 0.06).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == largest).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def segment_tissues(t1_slice: np.ndarray) -> np.ndarray:
    """Return an int label map: 0=background, 1=CSF, 2=GM, 3=WM.
    Unsupervised 3-class intensity clustering within the brain mask."""
    sl = np.clip(t1_slice, 0, 1).astype(np.float32)
    mask = _brain_mask(sl)
    labels = np.zeros_like(sl, dtype=np.int32)
    vals = sl[mask > 0].reshape(-1, 1)
    if vals.shape[0] < 30:
        return labels

    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=3, covariance_type="full",
                             random_state=42, max_iter=100)
        pred = gm.fit_predict(vals)
        centers = gm.means_.flatten()
    except Exception:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _c, lbl, cen = cv2.kmeans(vals.astype(np.float32), 3, None, crit, 5,
                                  cv2.KMEANS_PP_CENTERS)
        pred = lbl.flatten()
        centers = cen.flatten()

    order = np.argsort(centers)                # dark -> bright  => CSF,GM,WM
    remap = {int(old): new + 1 for new, old in enumerate(order)}
    flat = np.array([remap[int(p)] for p in pred], dtype=np.int32)
    labels[mask > 0] = flat
    labels = cv2.medianBlur(labels.astype(np.uint8), 3).astype(np.int32)
    return labels


def tissue_overlay(t1_slice: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """BGR overlay of the tissue map on the T1 slice."""
    base = cv2.cvtColor((np.clip(t1_slice, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base)
    for c, col in TISSUE_COLORS.items():
        color[labels == c] = col
    return cv2.addWeighted(base, 0.65, color, 0.55, 0)


def tissue_fractions(labels: np.ndarray) -> dict:
    """Volume fraction of each tissue within the brain (a real clinical-style
    readout: e.g. grey/white ratio)."""
    total = int((labels > 0).sum())
    if total == 0:
        return {}
    return {TISSUE_NAMES[c]: round(float((labels == c).sum()) / total, 3)
            for c in (1, 2, 3)}


def pick_t1_slice(vol: np.ndarray) -> np.ndarray:
    v = normalize_volume(vol)
    # a central axial slice with lots of brain
    best, best_frac = None, 0
    for z in range(v.shape[2] // 3, 2 * v.shape[2] // 3):
        sl = v[:, :, z]
        frac = np.count_nonzero(sl) / sl.size
        if frac > best_frac:
            best_frac, best = frac, sl
    if best is None:
        best = v[:, :, v.shape[2] // 2]
    return np.clip(cv2.resize(best, (IMG_SIZE, IMG_SIZE)), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brats_root", default="data/brats_subset")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out_dir", default="outputs/tissue")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from nifti_utils import find_brats_cases, load_brats_case
    cases = find_brats_cases(args.brats_root)[:args.n]
    for cd in cases:
        c = load_brats_case(cd)
        if "t1" not in c["modalities"]:
            continue
        t1 = pick_t1_slice(c["modalities"]["t1"])
        labels = segment_tissues(t1)
        fr = tissue_fractions(labels)
        name = c["case_name"]
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_t1.png"),
                    (t1 * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(args.out_dir, f"{name}_tissue.png"),
                    tissue_overlay(t1, labels))
        np.save(os.path.join(args.out_dir, f"{name}_tissue.npy"), labels)
        print(f"[tissue] {name}: fractions {fr} -> {args.out_dir}")
    print(f"[tissue] CSF=blue, grey matter=green, white matter=red. Done.")


if __name__ == "__main__":
    main()

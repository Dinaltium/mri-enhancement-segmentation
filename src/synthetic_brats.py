"""
synthetic_brats.py

Generates fake volumes that match BraTS2020's exact folder/file naming
convention and label values (0/1/2/4), so nifti_utils.py, dataset_stats.py,
and train_brain.py can all be built and tested tonight before the real
BraTS2020 Kaggle dataset is downloaded onto a laptop with real internet.

This is TEST-ONLY. Once you download the real Kaggle dataset, point every
script at that folder instead and skip this file entirely.

Usage:
    python synthetic_brats.py --out data/synthetic_brats --n_cases 6
"""

import argparse
import os

import nibabel as nib
import numpy as np

VOLUME_SHAPE = (240, 240, 155)  # matches real BraTS volume dimensions


def make_synthetic_case(seed: int, shape=VOLUME_SHAPE) -> dict:
    rng = np.random.default_rng(seed)
    h, w, d = shape

    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    zz, yy, xx = zz.transpose(1, 2, 0), yy.transpose(1, 2, 0), xx.transpose(1, 2, 0)

    # rough "brain-shaped" blob so most slices aren't empty
    cx, cy, cz = w / 2, h / 2, d / 2
    brain_mask = (((xx - cx) / (w * 0.4)) ** 2 + ((yy - cy) / (h * 0.4)) ** 2 +
                  ((zz - cz) / (d * 0.45)) ** 2) < 1.0

    base = rng.uniform(0.3, 0.6) * brain_mask.astype(np.float32)
    base += rng.normal(0, 0.03, shape).astype(np.float32) * brain_mask

    modalities = {}
    for mod, scale in [("flair", 1.1), ("t1", 0.9), ("t1ce", 1.0), ("t2", 1.05)]:
        vol = base * scale + rng.normal(0, 0.02, shape).astype(np.float32) * brain_mask
        modalities[mod] = np.clip(vol, 0, None).astype(np.float32)

    # fake tumor: a small sphere with the 3 BraTS label classes as concentric shells
    seg = np.zeros(shape, dtype=np.uint8)
    tx, ty, tz = (rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.65) * h,
                  rng.uniform(0.35, 0.65) * d)
    dist = np.sqrt((xx - tx) ** 2 + (yy - ty) ** 2 + (zz - tz) ** 2)
    r_core, r_necrotic, r_edema = 6, 10, 18
    seg[dist < r_edema] = 2       # edema (outer shell)
    seg[dist < r_necrotic] = 1    # necrotic/non-enhancing core
    seg[dist < r_core] = 4        # enhancing tumor (innermost) - real BraTS skips label 3

    # bump up modality intensities inside the fake tumor, so it's visually/statistically
    # distinct (otherwise the segmentation task is trivially impossible even for testing)
    for mod in modalities:
        modalities[mod][dist < r_edema] *= 1.6

    return {"modalities": modalities, "seg": seg}


def save_case(case: dict, case_dir: str, case_name: str) -> None:
    os.makedirs(case_dir, exist_ok=True)
    affine = np.eye(4)
    for mod, vol in case["modalities"].items():
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(case_dir, f"{case_name}_{mod}.nii.gz"))
    nib.save(nib.Nifti1Image(case["seg"], affine), os.path.join(case_dir, f"{case_name}_seg.nii.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic_brats")
    parser.add_argument("--n_cases", type=int, default=6)
    args = parser.parse_args()

    root = os.path.join(args.out, "MICCAI_BraTS2020_TrainingData")
    for i in range(1, args.n_cases + 1):
        case_name = f"BraTS20_Training_{i:03d}"
        case = make_synthetic_case(seed=i)
        save_case(case, os.path.join(root, case_name), case_name)
        print(f"[synthetic_brats] wrote {case_name}")

    print(f"[synthetic_brats] done -> {root}")


if __name__ == "__main__":
    main()

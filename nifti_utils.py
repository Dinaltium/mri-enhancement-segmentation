"""
nifti_utils.py

Shared utilities for loading NIfTI (.nii/.nii.gz) volumes and extracting
2D axial slices - used by every other script in this project.

Why 2D slices instead of full 3D: the recommended pretrained 3D BraTS
models need 16GB+ GPU memory (MONAI's brats_mri_segmentation bundle
docs state this explicitly). A laptop RTX 4050 (6GB) can't run that.
Slice-based 2D processing is a standard, well-documented simplification
for exactly this constraint - it's a legitimate design choice to justify
in your report, not corner-cutting.
"""

import os

import cv2
import nibabel as nib
import numpy as np

IMG_SIZE = 224  # standard-ish size, divisible by 16 for U-Net pooling


def load_volume(path: str) -> np.ndarray:
    """Loads a .nii/.nii.gz file, returns HxWxD float32 array."""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize a whole volume to [0, 1] (per-volume, not per-slice,
    so relative intensity relationships across slices are preserved)."""
    v = volume - np.percentile(volume, 0.5)
    p99 = np.percentile(volume, 99.5)
    if p99 > 0:
        v = v / p99
    return np.clip(v, 0, 1).astype(np.float32)


def extract_axial_slices(volume: np.ndarray, min_nonzero_frac: float = 0.02) -> list[np.ndarray]:
    """
    Extracts axial (third-axis) slices from an HxWxD volume, skipping
    near-empty slices (mostly background - common at the top/bottom of
    a brain/spine scan, not useful for training).

    Returns a list of 2D float32 slices resized to IMG_SIZE x IMG_SIZE.
    """
    slices = []
    depth = volume.shape[2]
    for z in range(depth):
        sl = volume[:, :, z]
        nonzero_frac = np.count_nonzero(sl) / sl.size
        if nonzero_frac < min_nonzero_frac:
            continue
        sl_resized = cv2.resize(sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        sl_resized = np.clip(sl_resized, 0.0, 1.0)  # cubic interpolation can overshoot near edges
        slices.append(sl_resized.astype(np.float32))
    return slices


def extract_axial_slices_with_mask(volume: np.ndarray, seg: np.ndarray,
                                    min_nonzero_frac: float = 0.02) -> tuple[list, list]:
    """Same as extract_axial_slices, but also extracts the matching label slice
    (nearest-neighbor resize, to avoid inventing fractional label values)."""
    img_slices, mask_slices = [], []
    depth = volume.shape[2]
    for z in range(depth):
        sl = volume[:, :, z]
        nonzero_frac = np.count_nonzero(sl) / sl.size
        if nonzero_frac < min_nonzero_frac:
            continue
        sl_resized = cv2.resize(sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        sl_resized = np.clip(sl_resized, 0.0, 1.0)  # cubic interpolation can overshoot near edges
        mask_sl = seg[:, :, z]
        mask_resized = cv2.resize(mask_sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        img_slices.append(sl_resized.astype(np.float32))
        mask_slices.append(mask_resized.astype(np.uint8))
    return img_slices, mask_slices


def remap_brats_labels(mask: np.ndarray) -> np.ndarray:
    """
    BraTS label convention: 0=background, 1=necrotic/non-enhancing core,
    2=edema, 4=enhancing tumor (label 3 is intentionally unused in BraTS).
    Remap 4 -> 3 so classes are contiguous (0,1,2,3) for CrossEntropyLoss.
    """
    remapped = mask.copy()
    remapped[remapped == 4] = 3
    return remapped


def save_slice_png(slice_2d: np.ndarray, path: str) -> None:
    """Save a float32 [0,1] slice as an 8-bit PNG."""
    img_uint8 = (np.clip(slice_2d, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


def find_brats_cases(brats_root: str) -> list[str]:
    """
    Finds patient case folders in a BraTS-structured root, e.g.:
        <root>/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/...
    Returns a list of case folder paths, each expected to contain
    <case>_flair.nii.gz, _t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, _seg.nii.gz
    """
    cases = []
    for dirpath, dirnames, filenames in os.walk(brats_root):
        if any(f.endswith("_seg.nii.gz") or f.endswith("_seg.nii") for f in filenames):
            cases.append(dirpath)
    return sorted(cases)


def load_brats_case(case_dir: str) -> dict:
    """Loads all 4 modalities + segmentation mask for one BraTS case."""
    case_name = os.path.basename(case_dir)
    modalities = {}
    for mod in ["flair", "t1", "t1ce", "t2"]:
        for ext in ["nii.gz", "nii"]:
            path = os.path.join(case_dir, f"{case_name}_{mod}.{ext}")
            if os.path.exists(path):
                modalities[mod] = load_volume(path)
                break
    seg = None
    for ext in ["nii.gz", "nii"]:
        path = os.path.join(case_dir, f"{case_name}_seg.{ext}")
        if os.path.exists(path):
            seg = load_volume(path)
            break
    return {"modalities": modalities, "seg": seg, "case_name": case_name}

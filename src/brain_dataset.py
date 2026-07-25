"""
brain_dataset.py

Builds two PyTorch datasets from a BraTS-structured root folder:
    - BrainEnhancementDataset: FLAIR slices, clean vs synthetically-degraded
    - BrainSegmentationDataset: all 4 modalities stacked as input channels,
      paired with the (remapped) ground-truth segmentation mask

IMPORTANT: splits are done at the CASE level (whole patients), not the
slice level - slices from the same patient are highly correlated, so a
slice-level split would leak information between train/val and inflate
your reported metrics. Always split by case.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from mri_degradation import degrade_mri_slice
from nifti_utils import (
    extract_axial_slices, extract_axial_slices_with_mask, find_brats_cases,
    load_brats_case, normalize_volume, remap_brats_labels,
)

MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]
NUM_SEG_CLASSES = 4  # background, necrotic/non-enhancing, edema, enhancing (remapped 4->3)


def split_cases(brats_root: str, val_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    cases = find_brats_cases(brats_root)
    if len(cases) < 2:
        raise ValueError(
            f"only found {len(cases)} case(s) under {brats_root} - need at least 2 "
            f"(one for train, one for val). Check the folder path."
        )
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


class BrainEnhancementDataset(Dataset):
    """Yields (degraded_flair, clean_flair) pairs, both 1xHxW in [0,1]."""

    def __init__(self, case_dirs: list[str], augment: bool = True, seed: int = 0):
        self.slices = []
        for case_dir in case_dirs:
            case = load_brats_case(case_dir)
            if "flair" not in case["modalities"]:
                continue
            vol_norm = normalize_volume(case["modalities"]["flair"])
            self.slices.extend(extract_axial_slices(vol_norm))
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        clean = self.slices[idx]
        if self.augment and random.random() < 0.5:
            clean = np.fliplr(clean).copy()

        degraded = degrade_mri_slice(clean, self._rng)

        clean_t = torch.from_numpy(clean).float().unsqueeze(0)
        degraded_t = torch.from_numpy(degraded).float().unsqueeze(0)
        return degraded_t, clean_t


class BrainSegmentationDataset(Dataset):
    """Yields (stacked_modalities, remapped_mask). stacked_modalities is
    len(MODALITY_ORDER)xHxW in [0,1]; mask is HxW int64 with class indices
    0..NUM_SEG_CLASSES-1."""

    def __init__(self, case_dirs: list[str], augment: bool = True):
        self.samples = []  # list of (case_dir, slice_index_within_case)
        self._case_cache = {}
        self.augment = augment

        for case_dir in case_dirs:
            case = load_brats_case(case_dir)
            if case["seg"] is None or len(case["modalities"]) < len(MODALITY_ORDER):
                continue

            normed = {mod: normalize_volume(vol) for mod, vol in case["modalities"].items()}
            # use one modality's slice extraction to decide which slice indices are valid,
            # then reuse those same indices for every modality + the mask
            ref_mod = MODALITY_ORDER[0]
            ref_slices, mask_slices = extract_axial_slices_with_mask(normed[ref_mod], case["seg"])

            per_modality_slices = {}
            for mod in MODALITY_ORDER:
                mod_slices, _ = extract_axial_slices_with_mask(normed[mod], case["seg"])
                per_modality_slices[mod] = mod_slices

            n_slices = len(ref_slices)
            self._case_cache[case_dir] = {
                "modalities": per_modality_slices,
                "masks": [remap_brats_labels(m) for m in mask_slices],
            }
            for i in range(n_slices):
                self.samples.append((case_dir, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        case_dir, slice_idx = self.samples[idx]
        cached = self._case_cache[case_dir]

        stacked = np.stack(
            [cached["modalities"][mod][slice_idx] for mod in MODALITY_ORDER], axis=0
        )  # CxHxW
        mask = cached["masks"][slice_idx]  # HxW

        if self.augment and random.random() < 0.5:
            stacked = np.ascontiguousarray(stacked[:, :, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        stacked_t = torch.from_numpy(stacked).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return stacked_t, mask_t

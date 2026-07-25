"""
enhancement_dataset.py

A modality-agnostic enhancement dataset for the OFFLINE hackathon data
(Spine + offline Brain). Unlike BrainEnhancementDataset (BraTS FLAIR only),
this builds (degraded, clean) training pairs from ANY in-scope sub-modality
volume, discovered and classified by offline_dataset.classify_modality.

Same synthetic-degradation trick as the BraTS enhancement pipeline: the
offline scans have no "noisy version", so we synthesize MRI-realistic
degradation (Rician noise + bias field + mild blur) and learn to invert it.
This needs no ground-truth annotations - only the clean scans themselves -
which is exactly why a *trained* enhancement model is legitimate here even
though segmentation must stay unsupervised.

Handles both shapes present in the offline data:
    - full 3D volumes            -> many 2D slices along the through-plane axis
    - single-slice 2D niftis     -> one slice each (the ..._i0000N.nii.gz files)
"""

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from nifti_utils import IMG_SIZE, load_volume, normalize_volume
from mri_degradation import degrade_mri_slice
from offline_dataset import classify_case_files


def extract_training_slices(volume: np.ndarray, size: int = IMG_SIZE,
                            min_nonzero: float = 0.05) -> list[np.ndarray]:
    """Normalize to [0,1] and return every sufficiently-populated 2D slice,
    resized to size x size. Slices along the smallest axis (through-plane)
    for 3D; handles 2D and HxWx1 too."""
    v = np.squeeze(normalize_volume(volume))
    if v.ndim == 2:
        candidates = [v]
    elif v.ndim == 3:
        ax = int(np.argmin(v.shape))
        candidates = [np.take(v, z, axis=ax) for z in range(v.shape[ax])]
    else:
        return []

    out = []
    for sl in candidates:
        if sl.size == 0 or np.count_nonzero(sl) / sl.size < min_nonzero:
            continue
        r = cv2.resize(sl.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        r = np.clip(r, 0.0, 1.0)  # cubic can overshoot near sharp edges
        out.append(r.astype(np.float32))
    return out


class OfflineEnhancementDataset(Dataset):
    """Yields (degraded, clean) 1xHxW pairs in [0,1] from a list of offline
    case dirs, pooling every in-scope sub-modality volume."""

    def __init__(self, case_dirs: list[str], modalities: set[str] | None = None,
                 augment: bool = True, seed: int = 0, max_slices_per_volume: int | None = None):
        self.slices: list[np.ndarray] = []
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        for case_dir in case_dirs:
            info = classify_case_files(case_dir)
            for mod, paths in info["buckets"].items():
                if mod == "unclassified":
                    continue
                if modalities is not None and mod not in modalities:
                    continue
                for path in paths:
                    try:
                        vol = load_volume(path)
                    except Exception:
                        continue
                    sls = extract_training_slices(vol)
                    if max_slices_per_volume and len(sls) > max_slices_per_volume:
                        # evenly subsample to cap huge multi-slice sequences
                        idx = np.linspace(0, len(sls) - 1, max_slices_per_volume).astype(int)
                        sls = [sls[i] for i in idx]
                    self.slices.extend(sls)

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


def split_offline_cases(root: str, train_frac: float = 0.5) -> tuple[list[str], list[str]]:
    """Deterministic sorted split of a dataset root's case folders.
    Default 50/50 == the coordinator's '5 train / 5 test per group' rule.
    Returns (train_case_dirs, test_case_dirs) as absolute paths."""
    from offline_dataset import find_offline_cases
    cases = find_offline_cases(root)
    n_train = round(len(cases) * train_frac)
    return cases[:n_train], cases[n_train:]

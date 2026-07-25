"""
metrics.py

Every metric the problem statement asks for, in one place:

Enhancement/IQA metrics (Stage 2 & 3):
    PSNR, SSIM, MSE, RMSE, UQI, FSIM, GMSD, VIF, BRISQUE, NIQE, PIQE,
    Entropy, LPIPS

Segmentation metrics (Stage 4):
    Dice, Jaccard, accuracy, sensitivity, specificity, precision, F1,
    Hausdorff Distance, Average Surface Distance

Full-reference metrics (PSNR/SSIM/MSE/RMSE/UQI/FSIM/GMSD/VIF/LPIPS) need
a clean reference image - only available where you have real paired data
(synthetic degradation pairs, or BraTS's own clean scans). No-reference
metrics (BRISQUE/NIQE/PIQE/Entropy) work on any single image - these are
what you report on the hackathon's own raw offline dataset, since there's
no "clean version" of those real scans to compare against.
"""

import numpy as np
import torch
from medpy.metric.binary import (
    asd, dc, hd, jc, precision as medpy_precision, recall as medpy_recall,
    sensitivity as medpy_sensitivity, specificity as medpy_specificity,
)
from scipy.stats import entropy as scipy_entropy
from skimage.measure import shannon_entropy

import pyiqa

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lazily created on first use, since loading BRISQUE/NIQE/LPIPS etc. downloads
# small pretrained weight files the first time
_iqa_metrics_cache: dict = {}


def _get_iqa_metric(name: str):
    if name not in _iqa_metrics_cache:
        _iqa_metrics_cache[name] = pyiqa.create_metric(name, device=_device)
    return _iqa_metrics_cache[name]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """img: HxW float32 in [0,1] -> 1x3xHxW tensor (pyiqa expects 3-channel)."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)  # fake RGB from grayscale
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img.transpose(2, 0, 1), 3, axis=0)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    return tensor.to(_device)


# ---------------------------------------------------------------------------
# Full-reference enhancement metrics (need a clean reference image)
# ---------------------------------------------------------------------------

def mse(clean: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((clean.astype(np.float64) - test.astype(np.float64)) ** 2))


def rmse(clean: np.ndarray, test: np.ndarray) -> float:
    return float(np.sqrt(mse(clean, test)))


def uqi(clean: np.ndarray, test: np.ndarray) -> float:
    """Universal Quality Index (Wang & Bovik 2002) - simple global version."""
    c, t = clean.astype(np.float64).flatten(), test.astype(np.float64).flatten()
    mean_c, mean_t = c.mean(), t.mean()
    var_c, var_t = c.var(), t.var()
    cov = np.mean((c - mean_c) * (t - mean_t))
    numerator = 4 * cov * mean_c * mean_t
    denominator = (var_c + var_t) * (mean_c ** 2 + mean_t ** 2)
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)


def _safe_iqa(name: str, *tensors) -> float | None:
    """Run a pyiqa metric, returning None if its pretrained weights can't be
    fetched (e.g. no internet on first use) instead of crashing the whole
    suite. Hand-rolled metrics (mse/rmse/uqi) never hit this path."""
    try:
        return float(_get_iqa_metric(name)(*tensors).item())
    except Exception as e:
        print(f"[metrics] '{name}' unavailable ({type(e).__name__}); skipped. "
              f"(pyiqa downloads weights on first use - needs internet once)")
        return None


def full_reference_metrics(clean: np.ndarray, test: np.ndarray) -> dict:
    """clean, test: HxW float32 in [0,1]. Returns every full-reference metric.
    pyiqa-backed metrics that can't load weights are omitted (not fatal)."""
    clean_t, test_t = _to_tensor(clean), _to_tensor(test)

    results = {
        "mse": mse(clean, test),
        "rmse": rmse(clean, test),
        "uqi": uqi(clean, test),
    }
    for name in ["psnr", "ssim", "fsim", "gmsd", "vif", "lpips"]:
        v = _safe_iqa(name, test_t, clean_t)
        if v is not None:
            results[name] = v
    return results


# ---------------------------------------------------------------------------
# No-reference metrics (work on a single image, no clean version needed -
# use these on the hackathon's own raw dataset)
# ---------------------------------------------------------------------------

def no_reference_metrics(img: np.ndarray) -> dict:
    """img: HxW float32 in [0,1]. Entropy is always available; the pyiqa
    no-reference metrics are omitted if their weights can't be fetched."""
    img_t = _to_tensor(img)
    # entropy on the 8-bit-quantized image (0-8 bits), so it is comparable
    # across quantized inputs and continuous-valued model outputs - raw
    # shannon_entropy on a float image inflates with the number of unique
    # float values, which is a metric artifact, not real information gain.
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    results = {"entropy": float(shannon_entropy(img_u8))}
    for name in ["brisque", "niqe", "piqe"]:
        v = _safe_iqa(name, img_t)
        if v is not None:
            results[name] = v
    return results


# ---------------------------------------------------------------------------
# Segmentation metrics (need predicted mask + ground-truth mask, both
# binary/boolean arrays for a single class - loop over classes for multi-class)
# ---------------------------------------------------------------------------

def segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    pred_mask, gt_mask: boolean or 0/1 arrays, same shape, for ONE class.
    Returns dice, jaccard, accuracy, sensitivity, specificity, precision, f1,
    hausdorff distance, average surface distance.

    Note: hd/asd require both masks to have at least one foreground voxel -
    returns None for those two if either mask is empty (common for slices
    with no lesion present), rather than crashing.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = medpy_precision(pred, gt) if pred.any() else 0.0
    rec = medpy_recall(pred, gt) if gt.any() else 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0

    results = {
        "dice": dc(pred, gt) if (pred.any() or gt.any()) else 1.0,
        "jaccard": jc(pred, gt) if (pred.any() or gt.any()) else 1.0,
        "accuracy": float(accuracy),
        "sensitivity": medpy_sensitivity(pred, gt) if gt.any() else None,
        "specificity": medpy_specificity(pred, gt) if (~gt).any() else None,
        "precision": prec,
        "f1": f1,
    }

    if pred.any() and gt.any():
        results["hausdorff_distance"] = hd(pred, gt)
        results["average_surface_distance"] = asd(pred, gt)
    else:
        results["hausdorff_distance"] = None
        results["average_surface_distance"] = None

    return results


def relative_volume_error(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Relative Volume Error = |vol(pred) - vol(gt)| / vol(gt), a metric the
    problem statement lists for segmentation. 0 = identical volume."""
    gt_vol = float(gt_mask.astype(bool).sum())
    pred_vol = float(pred_mask.astype(bool).sum())
    if gt_vol == 0:
        return 0.0 if pred_vol == 0 else 1.0
    return abs(pred_vol - gt_vol) / gt_vol


def multiclass_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                     class_names: list[str]) -> dict:
    """pred_mask, gt_mask: HxW int arrays with class indices 0..N-1.
    Returns per-class metrics dict, skipping background (class 0) by convention."""
    per_class = {}
    for idx, name in enumerate(class_names):
        if idx == 0:
            continue  # skip background
        per_class[name] = segmentation_metrics(pred_mask == idx, gt_mask == idx)
    return per_class

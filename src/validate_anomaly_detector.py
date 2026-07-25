"""
validate_anomaly_detector.py

Honest validation of the spine anomaly autoencoder.

The method is attractive in principle: train an autoencoder on healthy spines,
and whatever it cannot reconstruct should be the pathology. This script tests
that claim instead of assuming it — by scoring held-out NORMAL and PATHOLOGICAL
spines and asking whether the two distributions actually separate.

Result (see results/anomaly_validation.json): they do not. The score is driven
by image texture and complexity rather than by disease, so it must NOT be used
as a detector. The reconstruction-difference map is retained only as a
visualisation of where a scan departs from the learned healthy appearance.

Reporting this negative result is deliberate — an unvalidated detector that
fires on healthy patients is worse than no detector at all.
"""

import json
import os
from itertools import product

import numpy as np

from spine_autoencoder import load_model, anomaly_map
from nifti_utils import load_volume
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files

SLICES_PER_CASE = 3


def group_scores(ae, group, modality="T2"):
    out = []
    for cd in find_offline_cases(OFFLINE_ROOTS[group]):
        info = classify_case_files(cd)
        for p in info["buckets"].get(modality, [])[:1]:
            try:
                sls = extract_training_slices(load_volume(p))
            except Exception:
                continue
            if not sls:
                continue
            mid = len(sls) // 2
            for s in sls[mid:mid + SLICES_PER_CASE]:
                _recon, heat = anomaly_map(ae, s)
                out.append(float(heat.mean()))
            break
    return np.array(out)


def auc(neg, pos):
    """Probability a random pathological scan scores above a random normal one.
    0.5 = no discrimination at all."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    wins = [1.0 if b > a else 0.5 if b == a else 0.0 for a, b in product(neg, pos)]
    return float(np.mean(wins))


def main():
    os.makedirs("results", exist_ok=True)
    ae, modality = load_model()
    normal = group_scores(ae, "spine_normal", modality)
    patho = group_scores(ae, "spine_pathological", modality)
    a = auc(normal, patho)
    thr = float(np.percentile(normal, 95)) if len(normal) else float("nan")
    caught = float((patho > thr).mean()) if len(patho) else float("nan")

    verdict = (
        "NOT A VALID DETECTOR — the anomaly score does not separate pathological "
        "from normal spines (AUC below 0.5 means it is worse than chance, and the "
        "distributions overlap completely). The reconstruction error is dominated "
        "by image texture and anatomical complexity, not by disease. The map is "
        "therefore presented only as a reconstruction-difference visualisation and "
        "no diagnostic claim is made from it."
    ) if a < 0.65 else (
        "Usable as a weak screening signal; report the operating threshold with it."
    )

    res = {
        "modality": modality,
        "normal": {"n": len(normal), "mean": round(float(normal.mean()), 5),
                   "min": round(float(normal.min()), 5), "max": round(float(normal.max()), 5)},
        "pathological": {"n": len(patho), "mean": round(float(patho.mean()), 5),
                         "min": round(float(patho.min()), 5), "max": round(float(patho.max()), 5)},
        "auc": round(a, 3),
        "threshold_95th_pct_of_normal": round(thr, 5),
        "pathological_detected_at_that_threshold": round(caught, 3),
        "verdict": verdict,
    }
    with open("results/anomaly_validation.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"normal        n={len(normal):3d}  mean={normal.mean():.4f}  "
          f"range {normal.min():.4f}-{normal.max():.4f}")
    print(f"pathological  n={len(patho):3d}  mean={patho.mean():.4f}  "
          f"range {patho.min():.4f}-{patho.max():.4f}")
    print(f"AUC = {a:.3f}   (0.5 = no better than guessing)")
    print(f"\nVERDICT: {verdict}")
    print("\n[validate] wrote results/anomaly_validation.json")


if __name__ == "__main__":
    main()

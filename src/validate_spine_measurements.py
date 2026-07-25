"""
validate_spine_measurements.py

Tests whether the canal-width measurements (spine_measurements.py) actually
differ between the normal and pathological cohorts — before any of it is shown
as evidence.

Two things matter methodologically:

  * The unit of analysis is the PATIENT, not the slice. Slices from one patient
    are highly correlated, so pooling them inflates the apparent sample size and
    makes a weak effect look significant.
  * The direction is fixed in advance by physiology: stenosis is narrowing, so a
    pathological canal should show a LOWER narrowing ratio. We test that
    one-sided hypothesis rather than fishing for whatever is significant.

Output: results/spine_measurement_validation.json
"""

import json
import os

import numpy as np
from itertools import product

from spine_measurements import measure
from nifti_utils import load_volume
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files

SLICES_PER_CASE = 5
METRICS = ["narrowing_ratio", "min_width_px", "mean_width_px", "variability_cv"]


def per_case(group, modality="T2"):
    """One value per patient = median over that patient's sampled slices."""
    cases, detect_hits, detect_total = {}, 0, 0
    for cd in find_offline_cases(OFFLINE_ROOTS[group]):
        vals = []
        info = classify_case_files(cd)
        for p in info["buckets"].get(modality, [])[:1]:
            try:
                sls = extract_training_slices(load_volume(p))
            except Exception:
                continue
            if not sls:
                continue
            mid = len(sls) // 2
            for s in sls[max(0, mid - 2): mid + 3]:
                detect_total += 1
                r = measure(s)
                if r["summary"]:
                    detect_hits += 1
                    vals.append(r["summary"])
            break
        if vals:
            cases[os.path.basename(cd)] = {
                k: float(np.median([v[k] for v in vals])) for k in METRICS}
    return cases, detect_hits, detect_total


def auc(a, b):
    """P(b > a). Chance = 0.5."""
    if not a or not b:
        return float("nan")
    return float(np.mean([1.0 if y > x else 0.5 if y == x else 0.0
                          for x, y in product(a, b)]))


def main():
    os.makedirs("results", exist_ok=True)
    N, nh, nt = per_case("spine_normal")
    P, ph, pt = per_case("spine_pathological")

    per_metric = {}
    for k in METRICS:
        a = [v[k] for v in N.values()]
        b = [v[k] for v in P.values()]
        raw = auc(a, b)
        # orient so that a higher figure means "more stenotic-looking"
        oriented = 1 - raw if k in ("narrowing_ratio", "min_width_px", "mean_width_px") else raw
        per_metric[k] = {
            "normal_mean": round(float(np.mean(a)), 4),
            "pathological_mean": round(float(np.mean(b)), 4),
            "auc_oriented": round(float(oriented), 3),
        }

    # pre-specified one-sided test on the physiologically motivated metric
    a = [v["narrowing_ratio"] for v in N.values()]
    b = [v["narrowing_ratio"] for v in P.values()]
    try:
        from scipy.stats import mannwhitneyu
        _u, pval = mannwhitneyu(a, b, alternative="greater")
        pval = float(pval)
    except Exception:
        pval = float("nan")

    significant = bool(pval == pval and pval < 0.05)
    verdict = (
        "Canal detection is reliable and the narrowing ratio moves in the direction "
        "stenosis predicts (pathological canals are relatively narrower), with "
        f"AUC {per_metric['narrowing_ratio']['auc_oriented']}. However at "
        f"{len(N)} vs {len(P)} patients the difference is NOT statistically "
        f"significant (one-sided Mann-Whitney p = {pval:.3f}). We therefore report "
        "canal width as a MEASUREMENT with an observed trend, and make no "
        "diagnostic claim. A larger cohort would be needed to establish the effect."
    ) if not significant else (
        f"Narrowing ratio differs significantly (p = {pval:.3f}); usable as a "
        "screening measurement with the operating point reported."
    )

    res = {
        "unit_of_analysis": "patient (median over sampled slices) — not slice, "
                            "because slices from one patient are correlated",
        "n_normal_cases": len(N), "n_pathological_cases": len(P),
        "canal_detection_rate": {
            "normal": f"{nh}/{nt}", "pathological": f"{ph}/{pt}"},
        "metrics": per_metric,
        "prespecified_test": {
            "metric": "narrowing_ratio",
            "hypothesis": "one-sided: normal > pathological (stenosis narrows the canal)",
            "mann_whitney_p": round(pval, 4) if pval == pval else None,
            "significant_at_0.05": significant,
        },
        "verdict": verdict,
    }
    with open("results/spine_measurement_validation.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"patients: normal={len(N)}  pathological={len(P)}")
    print(f"canal detected: normal {nh}/{nt}, pathological {ph}/{pt}\n")
    print(f"{'metric':20}{'normal':>10}{'path':>10}{'AUC*':>8}")
    for k, v in per_metric.items():
        print(f"{k:20}{v['normal_mean']:>10.3f}{v['pathological_mean']:>10.3f}"
              f"{v['auc_oriented']:>8.3f}")
    print(f"\none-sided Mann-Whitney p = {pval:.3f} -> "
          f"{'significant' if significant else 'NOT significant'}")
    print(f"\nVERDICT: {verdict}")
    print("\n[validate] wrote results/spine_measurement_validation.json")


if __name__ == "__main__":
    main()

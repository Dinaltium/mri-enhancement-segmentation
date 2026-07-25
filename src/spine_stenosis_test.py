"""
spine_stenosis_test.py -- can we actually separate a pathological spine from a
normal one?

THE QUESTION: our own canal detector gives a narrowing ratio that trends the
right way (normal 0.557 vs pathological 0.485) but is not statistically
significant (AUC 0.69, p = 0.089, 10 vs 9 patients). Is that the biology being
weak, or our CSF-column heuristic being noisy?

THE TEST: measure the same quantity on SPINEPS's canal segmentation instead.
SPINEPS reports Dice 0.958 on the spinal canal, far better than an intensity
heuristic, so if the signal exists this should find it. If the separation is
still weak with a near-perfect canal mask, the honest conclusion is that canal
width alone does not identify these particular patients -- which is a real
finding, not a failure.

METHOD, per case:
  * take SPINEPS label 61 (spinal canal) + 60 (cord) as the canal region
  * for each sagittal slice, measure the canal's width row by row
  * narrowing ratio = 10th percentile width / median width, within that same
    canal, so patient size and resolution cancel out
  * aggregate per patient by median across slices (a single slice is far too
    noisy -- one normal scan ranges 0.29-0.66 across its own slices)

Then compare the normal group against the pathological group with a
Mann-Whitney U test and an ROC AUC. Nothing here is fitted, so there is no
train/test leak to worry about.

Output: results/spine_stenosis_test.json
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CANAL_LABELS = [60, 61]      # spinal cord + spinal canal
OUT = "results/spine_stenosis_test.json"


def canal_ratio(mask3d: np.ndarray) -> float | None:
    """Median-over-slices narrowing ratio from a labelled canal mask."""
    canal = np.isin(mask3d, CANAL_LABELS)
    if canal.sum() < 200:
        return None
    ax = int(np.argmin(canal.shape))            # through-plane axis
    ratios = []
    for k in range(canal.shape[ax]):
        sl = np.take(canal, k, axis=ax)
        widths = sl.sum(axis=1)                 # canal width per row
        widths = widths[widths > 0]
        if widths.size < 12:                    # too little canal on this slice
            continue
        med = float(np.median(widths))
        if med <= 0:
            continue
        # 10th percentile rather than the raw minimum: the single narrowest row
        # is usually a segmentation edge effect, not anatomy
        narrow = float(np.percentile(widths, 10))
        ratios.append(narrow / med)
    if len(ratios) < 3:
        return None
    return float(np.median(ratios))


def main():
    from spineps_runner import mask_in_scan_space

    rows = []
    for scan in sorted(glob.glob("showcase/for_spineps/*.nii.gz")):
        name = os.path.basename(scan)
        key = "".join(c for c in name.replace(".nii.gz", "") if c.isalnum())[:32]
        hits = glob.glob(os.path.join("outputs/spineps/live", key, "**",
                                      "*seg-spine*.nii.gz"), recursive=True)
        if not hits:
            continue
        try:
            m = mask_in_scan_space(hits[0], scan)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        r = canal_ratio(m)
        if r is None:
            print(f"  skip {name}: canal too small to measure")
            continue
        grp = "pathological" if "PATH" in name else "normal"
        rows.append({"case": name, "group": grp, "narrowing_ratio": round(r, 4)})
        print(f"  {grp:<14} {name:<32} ratio {r:.3f}")

    norm = [r["narrowing_ratio"] for r in rows if r["group"] == "normal"]
    path = [r["narrowing_ratio"] for r in rows if r["group"] == "pathological"]

    result = {"n_normal": len(norm), "n_pathological": len(path), "cases": rows}
    print(f"\n  normal n={len(norm)}   pathological n={len(path)}")

    if len(norm) >= 3 and len(path) >= 3:
        from scipy.stats import mannwhitneyu
        result["mean_normal"] = round(float(np.mean(norm)), 4)
        result["mean_pathological"] = round(float(np.mean(path)), 4)
        # one-sided: stenosis predicts pathological canals are NARROWER
        u, p = mannwhitneyu(path, norm, alternative="less")
        # AUC for "lower ratio => pathological"
        auc = sum(1.0 if a < b else 0.5 if a == b else 0.0
                  for a in path for b in norm) / (len(path) * len(norm))
        result.update(p_value=round(float(p), 4), auc=round(float(auc), 4))
        print(f"  mean normal        {result['mean_normal']}")
        print(f"  mean pathological  {result['mean_pathological']}")
        print(f"  AUC                {result['auc']}")
        print(f"  p (one-sided)      {result['p_value']}")
        sig = p < 0.05
        result["significant"] = bool(sig)
        print("\n  VERDICT: " + (
            "separates the groups (p < 0.05). Report it, with the n."
            if sig else
            "does NOT reach significance. Report the trend and say so — canal "
            "width alone does not identify these patients."))
    else:
        print("  not enough cases measured yet — let the precompute finish")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()

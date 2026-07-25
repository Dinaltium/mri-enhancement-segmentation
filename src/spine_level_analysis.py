"""
spine_level_analysis.py -- WHERE along the spine is the canal narrowest?

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
Our earlier work collapsed each patient to a single narrowing ratio, and that
number does not separate normal from pathological (AUC 0.583, p = 0.331, on
SPINEPS canal masks). So we do not claim to classify a spine as diseased.

But a whole-spine score was always the wrong output anyway. A radiologist does
not say "this spine is stenotic" -- they say "canal narrowing at L4-L5". The
clinically useful quantity is WHERE, not WHETHER, and where is a measurement we
can make honestly: walk down the canal, measure its width at every level, and
report the narrowest one.

This is localisation by MEASUREMENT, not by prediction. It flags the tightest
point of the canal in this scan. It does not diagnose stenosis, and a healthy
spine also has a narrowest point -- what matters is how narrow it is relative to
that same canal's own typical width, which is what the ratio reports.

METHOD
------
1. Canal = SPINEPS labels 60 (cord) + 61 (canal), mapped back into the scan's
   own voxel grid through the image affine.
2. Walk along the spine's long axis (found from the canal's own extent, so no
   assumption about scan orientation).
3. At each step measure canal width, then smooth along the axis -- a single row
   can be a segmentation edge artefact rather than anatomy.
4. Report the narrowest level, its ratio against that canal's median width, and
   draw it on the slice.

Optionally anchors the level to the nearest intervertebral disc (label 100), so
the output reads as "narrowest beside disc 3 from the top" instead of a bare
pixel row.

Output: results/spine_level_analysis.json
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CANAL = [60, 61]        # spinal cord + spinal canal
DISC = 100              # intervertebral disc
OUT = "results/spine_level_analysis.json"


def _smooth(v, k=5):
    if len(v) < k:
        return v
    kern = np.ones(k) / k
    return np.convolve(v, kern, mode="same")


def analyse(mask3d: np.ndarray, min_run: int = 12) -> dict | None:
    """Width profile along the canal, and the narrowest level."""
    canal = np.isin(mask3d, CANAL)
    if canal.sum() < 200:
        return None

    ax = int(np.argmin(canal.shape))                    # through-plane axis
    # the slice carrying most canal is the one to measure and draw on
    z = int(np.argmax([np.take(canal, k, axis=ax).sum() for k in range(canal.shape[ax])]))
    sl = np.take(canal, z, axis=ax)

    # Long axis of the spine = whichever image axis the canal spans further.
    # Derived from the data, so a rotated or differently-stored scan still works.
    ys, xs = np.nonzero(sl)
    vertical = (ys.max() - ys.min()) >= (xs.max() - xs.min())
    widths = sl.sum(axis=1 if vertical else 0).astype(float)

    idx = np.nonzero(widths > 0)[0]
    if idx.size < min_run:
        return None
    lo, hi = int(idx.min()), int(idx.max())
    prof = _smooth(widths[lo:hi + 1], k=5)

    # Ignore the top and bottom 15%: the canal tapers where it leaves the field
    # of view, and that taper is not a finding.
    m = max(1, int(0.15 * len(prof)))
    core = prof[m:len(prof) - m]
    if core.size < 5:
        return None
    med = float(np.median(core))
    if med <= 0:
        return None
    i_local = int(np.argmin(core))
    narrowest = float(core[i_local])
    pos = lo + m + i_local                              # row/col in the slice

    disc_level = None
    discs = (np.take(mask3d, z, axis=ax) == DISC)
    if discs.sum() > 20:
        lbl_n, lbl = cv2.connectedComponents(discs.astype(np.uint8))
        cents = []
        for c in range(1, lbl_n):
            yy, xx = np.nonzero(lbl == c)
            if yy.size > 20:
                cents.append(float(yy.mean() if vertical else xx.mean()))
        if cents:
            cents.sort()
            disc_level = int(np.argmin([abs(c - pos) for c in cents])) + 1

    return {
        "slice_index": z,
        "axis": "vertical" if vertical else "horizontal",
        "position_px": int(pos),
        "narrowest_width_px": round(narrowest, 2),
        "median_width_px": round(med, 2),
        "narrowing_ratio": round(narrowest / med, 4),
        "nearest_disc_from_top": disc_level,
        "profile": [round(float(v), 2) for v in prof],
        "profile_start_px": lo,
        "vertical": bool(vertical),
    }


def overlay(img01: np.ndarray, mask2d: np.ndarray, res: dict) -> np.ndarray:
    """Draw the canal, and mark the narrowest level with a line and a label."""
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canal = np.isin(mask2d, CANAL)
    col = np.zeros_like(base)
    col[canal] = (0, 200, 255)                          # amber canal
    out = cv2.addWeighted(base, 1.0, col, 0.45, 0)

    p = res["position_px"]
    h, w = img01.shape
    if res["vertical"]:
        cv2.line(out, (0, p), (w, p), (60, 60, 255), 1, cv2.LINE_AA)
        org = (6, max(14, p - 6))
    else:
        cv2.line(out, (p, 0), (p, h), (60, 60, 255), 1, cv2.LINE_AA)
        org = (min(w - 120, p + 6), 16)
    cv2.putText(out, f"narrowest {res['narrowing_ratio']:.2f}", org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 255), 1, cv2.LINE_AA)
    return out


def main():
    from spineps_runner import mask_in_scan_space
    import nibabel as nib

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
            r = analyse(m)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        if not r:
            continue
        r["case"] = name
        r["group"] = "pathological" if "PATH" in name else "normal"
        r.pop("profile", None)
        rows.append(r)
        disc = r["nearest_disc_from_top"]
        where = f"disc {disc} from top" if disc else f"row {r['position_px']}"
        print(f"  {r['group']:<14} {name:<32} narrowest {r['narrowing_ratio']:.2f} at {where}")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"n": len(rows), "cases": rows}, f, indent=2)
    print(f"\n  wrote {OUT}  ({len(rows)} cases)")


if __name__ == "__main__":
    main()

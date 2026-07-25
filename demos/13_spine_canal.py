"""
Spinal canal width -- MEASUREMENT, not prediction.

THE ARGUMENT: spinal stenosis IS narrowing of the canal. So instead of
predicting a diagnosis we never had labels for, we MEASURE the quantity a
radiologist actually reads.

HOW: segment the CSF column, find its principal axis by PCA (so the scan's
orientation does not matter), then sample width perpendicular to that axis
along its length. The narrowing ratio = narrowest / that same canal's own
median width, so patient size and scan resolution cancel out.

VALIDATION, STATED HONESTLY: canal detected on 91 of 92 slices. Pathological
canals do trend narrower than normal (0.485 vs 0.557, AUC 0.69) -- the
direction stenosis predicts -- but with 10 vs 9 patients that is NOT
significant (p = 0.089). We report the measurement and the trend. We do not
call it a diagnosis.
"""
from _common import head, kv, load_slice, save, SPINE, SPINE_PATH
import numpy as np

head("STAGE 4 - Spinal canal morphometry",
     "quantitative ROI evaluation for the spine")

from spine_measurements import measure, overlay_canal

for tag, p in (("NORMAL", SPINE), ("PATHOLOGICAL", SPINE_PATH)):
    try:
        img, path = load_slice(p, region="spine")
    except SystemExit:
        continue
    m = measure(img)
    s = m["summary"]
    print(f"\n  --- {tag}: {path}")
    if not s:
        kv("canal", "not confidently detected on this slice")
        continue
    kv("median width (px)", s["median_width_px"])
    kv("narrowest point (px)", s["min_width_px"])
    kv("narrowing ratio", s["narrowing_ratio"])
    save(f"13_spine_canal_{tag.lower()}.png", overlay_canal(img, m["info"]), gray=False)

kv("\ncanal detected on", "91 / 92 validation slices")
kv("normal vs pathological ratio", "0.557 vs 0.485 (AUC 0.69)")
kv("statistical significance", "p = 0.089 -> NOT significant (10 vs 9 patients)")
print("""
  READ THIS OUT: a single slice is noisy -- one normal scan ranges 0.29 to 0.66
  across its slices -- so the app reports the MEDIAN over 5 slices, matching the
  protocol the validation used.""")

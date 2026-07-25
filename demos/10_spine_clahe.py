"""
CLAHE on a SPINE MRI -- classical baseline for the spine track.

WHY SPINE IS DIFFERENT: spine scans measure about 2x the complexity of brain
scans (more distinct structures per slice), and slice thickness runs to 13 mm.
The same classical method therefore behaves differently here than on brain.
"""
from _common import head, kv, load_slice, side_by_side, noise_level, SPINE

head("STAGE 2 - CLAHE (classical) on spine",
     "preprocessing justified per sub-modality (T1/T2/STIR)")

from spine_pipeline import clahe_enhance

img, path = load_slice(SPINE, region="spine")
cl = clahe_enhance(img)

kv("input file", path)
kv("method", "CLAHE, clipLimit 2.0")
kv("noise before -> after", f"{noise_level(img):.4f} -> {noise_level(cl):.4f}")
kv("contrast before -> after", f"{img.std():.4f} -> {cl.std():.4f}")
side_by_side([img, cl], ["Spine original", "CLAHE"], "10_spine_clahe.png")

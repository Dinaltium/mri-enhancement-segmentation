"""
Healthy-tissue segmentation -- CSF / grey matter / white matter.

WHY IT EXISTS: the brief asks for the region of interest, and for a HEALTHY
brain the ROI is not a tumour -- it is the tissue compartments. This covers the
normal-brain half of the offline dataset, which has no tumour to find.

METHOD: Gaussian Mixture Model over intensity, 3 components, on brain-masked
pixels. Unsupervised -- no labels needed, which matters because the offline
brain data has none. Components are ordered by mean intensity, which is what
maps them to CSF (darkest on T1), grey, then white matter.
"""
from _common import head, kv, load_slice, save
import numpy as np

head("STAGE 4 - Healthy tissue ROI: CSF / grey / white (brain)",
     "ROI segmentation for non-pathological scans")

from tissue_segmentation import segment_tissues, tissue_overlay, tissue_fractions

sl, path = load_slice(region="brain")
labels = segment_tissues(sl)
fr = tissue_fractions(labels)

kv("input file", path)
kv("method", "Gaussian Mixture Model, 3 components, unsupervised")
kv("annotations used", "none")
for k, v in fr.items():
    kv(f"fraction {k}", f"{v:.1%}")
print("""
  READ THIS OUT: no labels were used. The three compartments separate by
  intensity statistics alone, which is why this works on the offline brain data
  that has no ground truth at all.""")
save("06_brain_tissue.png", tissue_overlay(sl, labels), gray=False)

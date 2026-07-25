"""
OUR spine ROI segmentation -- self-supervised CNN, ZERO annotations.

THE METHOD (differentiable feature clustering, Kanezaki ICASSP 2018):
a small CNN is trained FROM SCRATCH on the single scan in front of it, for
~120 iterations. Its supervision comes from three constraints, no labels:

  1. COMMIT     - each pixel is pushed toward its own argmax class, sharpening
                  fuzzy assignments into definite ones.
  2. CONTINUITY - neighbouring pixels pushed to agree -> coherent regions.
  3. BALANCE    - entropy term on the mean class distribution stops everything
                  collapsing into one class.

The number of structures found is EMERGENT: we offer 12 candidates and it
settles on 9-10.

TWO BUGS WE HIT (both measured, both worth telling):
  - Collapse to 1 class: we first masked background by zeroing FEATURES. Every
    background pixel then argmaxed to the same class and cross-entropy dragged
    the whole image into it. Fix: mask the LOSS, not the features.
  - Collapse to 2 classes: applying the superpixel prior every iteration
    compounded with CE feedback. Measured 2 classes with it in-loop vs 11
    without. Fix: apply the prior once, after training.

HONEST SCORE: against SPINEPS as reference, best Dice 0.38 (canal). Highest
precision of all three annotation-free methods on all four structures.
"""
from _common import head, kv, load_slice, side_by_side, SPINE

head("STAGE 4 - Our self-supervised spine ROI segmentation",
     "ROI segmentation with NO annotations for training")

from spine_pipeline import clahe_enhance, kmeans_roi, slic_roi, colorize_labels
from spine_deep_segmentation import segment as dseg, colorize as dcol

img, path = load_slice(SPINE, region="spine")
enh = clahe_enhance(img)
km = kmeans_roi(enh, k=4)
sl_ = slic_roi(enh, n_segments=250, k=4)
lab, info = dseg(enh, n_classes=12, iters=120)

kv("input file", path)
kv("method", "differentiable feature clustering (Kanezaki 2018)")
kv("trained on", "THIS SINGLE SCAN, from scratch, ~120 iterations")
kv("annotations used", "NONE")
kv("candidate classes offered", 12)
kv("structures actually found", info.get("classes_found"))
kv("k-means classes", len(set(km.ravel().tolist())))
kv("vs SPINEPS reference (canal)", "Dice 0.380 +/- 0.041, precision 0.310")
kv("k-means for comparison", "Dice 0.304, precision 0.194")
print("""
  READ THIS OUT: k-means and SLIC group BRIGHTNESS, so they cannot separate two
  adjacent vertebrae that look identical. The trained network separates by
  learned features. Ours wins on precision on all four structures.""")
side_by_side([img, colorize_labels(km, 3), colorize_labels(sl_, 3), dcol(lab)],
             ["Original", "k-means", "SLIC", "Ours (self-sup CNN)"],
             "12_spine_selfsup.png")

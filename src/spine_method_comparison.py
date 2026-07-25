"""
spine_method_comparison.py

Side-by-side of every spine segmentation approach we implemented, on the same
slice, so the report can show what each one buys:

    original | CLAHE | k-means | SLIC | self-supervised CNN | SPINEPS

The first five need no annotations and no external data — they are our own
work under the competition constraint. SPINEPS is a published pretrained model
(external training data) included as a REFERENCE UPPER BOUND, clearly labelled
as such. Presented this way the comparison is honest under either reading of
the rules, and it quantifies how much of the achievable result our
annotation-free pipeline recovers.

Output: outputs/demo/spine_method_comparison.png
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nifti_utils import load_volume
from enhancement_dataset import extract_training_slices
from spine_pipeline import clahe_enhance, kmeans_roi, slic_roi, colorize_labels

OUT = "outputs/demo/spine_method_comparison.png"


def _ov(img01, labels, colour_fn=None):
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = colour_fn(labels) if colour_fn else colorize_labels(labels, 3)
    return cv2.cvtColor(cv2.addWeighted(base, 0.7, col, 0.6, 0), cv2.COLOR_BGR2RGB)


def build(nifti_path: str, spineps_instance: str = None):
    sls = extract_training_slices(load_volume(nifti_path))
    if not sls:
        print("no usable slices"); return
    img = sls[len(sls) // 2]
    enh = clahe_enhance(img)

    panels = [("Original", cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB), None)]
    panels.append(("CLAHE\n(classical)",
                   cv2.cvtColor((enh * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB), None))
    panels.append(("k-means\n(intensity only)", _ov(enh, kmeans_roi(enh, k=4)), "annotation-free"))
    panels.append(("SLIC superpixels\n(intensity + edges)",
                   _ov(enh, slic_roi(enh, n_segments=250, k=4)), "annotation-free"))

    try:
        from spine_deep_segmentation import segment as dseg, colorize as dcol
        lab, info = dseg(enh, n_classes=8, iters=80)
        panels.append((f"Self-supervised CNN\n({info['classes_found']} structures, ours)",
                       _ov(enh, lab, dcol), "annotation-free"))
    except Exception as e:
        print("deep segmentation unavailable:", e)

    if spineps_instance and os.path.exists(spineps_instance):
        try:
            from spineps_runner import instance_slice, overlay as sp_overlay
            inst = instance_slice(spineps_instance)
            panels.append(("SPINEPS\n(pretrained reference)",
                           cv2.cvtColor(sp_overlay(enh, inst), cv2.COLOR_BGR2RGB),
                           "external weights"))
        except Exception as e:
            print("SPINEPS overlay unavailable:", e)

    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(2.9 * n, 3.9))
    for a, (title, im, tag) in zip(np.atleast_1d(ax), panels):
        a.imshow(im)
        a.set_title(title, fontsize=9.5)
        a.axis("off")
        if tag:
            a.text(0.5, -0.06, tag, transform=a.transAxes, ha="center",
                   fontsize=7.5, color="#1f6f4a" if tag == "annotation-free" else "#b4432c")
    fig.suptitle("Spine ROI segmentation — every method on the same slice",
                 fontsize=13, fontweight="700", x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=135); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        import glob
        c = glob.glob("showcase/for_enhancement/SPINE_*T2*")
        path = c[0] if c else None
    else:
        path = sys.argv[1]
    inst = sys.argv[2] if len(sys.argv) > 2 else None
    if path:
        build(path, inst)

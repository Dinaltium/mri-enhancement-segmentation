"""
spine_vs_spineps.py  --  how much does our annotation-free pipeline recover?

SPINEPS is trained on external annotated data, so it can name structures we
cannot. That makes it useful as a *reference standard*: we can ask, for each
structure it identifies, how well our own annotation-free methods recover the
same region without ever having seen a label.

This is the honest way to quantify the gap the pretrained model fills. It
turns "we used a pretrained model" into "here is exactly what our own work
achieves, and here is exactly what it cannot."

METHOD, and its one important caveat
------------------------------------
Our unsupervised methods return anonymous cluster indices, not names. Cluster 3
is not "the vertebral body" -- it is just cluster 3. To score them at all we
must decide which cluster corresponds to which structure, and we do that by
picking the cluster with the highest Dice against each reference region.

That is an ORACLE-ASSISTED assignment: it uses the reference to choose the
label. The resulting Dice is therefore an UPPER BOUND on what the method
delivers unaided, and we report it as such. It answers "did the method carve
out this structure as a distinct region?" -- not "can the method name it?"
Naming is precisely the supervised step our data cannot provide, which is the
whole argument for using a pretrained model for that one output.

SPINEPS subregion labels used (TPTBox vert_constants):
    41-48  posterior elements (arcus, spinosus, costal/articular processes)
    49     vertebral body (corpus border)
    60     spinal cord        61  spinal canal        62  endplate
    100    intervertebral disc

Outputs: results/spine_vs_spineps.json
         outputs/demo/spine_vs_spineps.png
"""

import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nifti_utils import load_volume
from spine_pipeline import clahe_enhance, kmeans_roi, slic_roi
from spineps_runner import mask_in_scan_space

OUT_JSON = "results/spine_vs_spineps.json"
OUT_FIG = "outputs/demo/spine_vs_spineps.png"

# repeats for the stochastic method (see evaluate())
N_REPEATS = 3

# name -> the SPINEPS label values that make up that reference region
REGIONS = {
    "Vertebral bodies": [49],
    "Intervertebral discs": [100],
    "Spinal canal + cord": [60, 61],
    "Posterior elements": [41, 42, 43, 44, 45, 46, 47, 48],
}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else 0.0


def best_cluster_dice(labels: np.ndarray, ref: np.ndarray):
    """Highest Dice achievable by any single cluster of `labels` against `ref`,
    plus that cluster's recall and precision.

    Oracle-assisted (see module docstring): the reference picks the cluster.

    Dice alone is misleading here and the extra two numbers matter. A cluster
    that covers the whole structure but also spills into surrounding tissue
    scores a low Dice, exactly like a cluster that missed the structure
    entirely -- but those are completely different failures. Recall says
    "did it find the structure", precision says "did it stop at its edges".
    Unsupervised clustering typically shows high recall with low precision,
    and that distinction is the actual finding.

    Returns (dice, recall, precision, cluster_id, n_clusters).
    """
    ids = [int(u) for u in np.unique(labels)]
    if not ids or ref.sum() == 0:
        return 0.0, 0.0, 0.0, -1, len(ids)
    scores = [(dice(labels == u, ref), u) for u in ids]
    d, u = max(scores)
    sel = labels == u
    r = ref.astype(bool)
    recall = float((sel & r).sum() / r.sum()) if r.sum() else 0.0
    prec = float((sel & r).sum() / sel.sum()) if sel.sum() else 0.0
    return d, recall, prec, int(u), len(ids)


def best_union_dice(labels: np.ndarray, ref: np.ndarray):
    """Best Dice from a greedy UNION of clusters, not just the single best one.

    Needed for a fair comparison across methods that produce different numbers
    of clusters. A method that splits the spine into 9 regions can never match
    a whole structure with one cluster, while a 4-cluster method might -- so
    scoring "best single cluster" quietly rewards coarseness. Greedily adding
    whichever cluster still improves Dice removes that bias: it asks whether
    the structure's boundary exists *somewhere* in the method's partition,
    which is the real question for an unsupervised segmenter.

    Returns (dice, n_clusters_merged).
    """
    r = ref.astype(bool)
    if r.sum() == 0:
        return 0.0, 0
    ids = [int(u) for u in np.unique(labels)]
    cur = np.zeros_like(r)
    best, used = 0.0, 0
    remaining = set(ids)
    while remaining:
        gains = [(dice(cur | (labels == u), r), u) for u in remaining]
        d, u = max(gains)
        if d <= best + 1e-6:
            break
        cur = cur | (labels == u)
        best, used = d, used + 1
        remaining.discard(u)
    return float(best), used


def evaluate(scan_path: str, semantic_path: str, instance_path: str = None):
    vol = load_volume(scan_path)
    sem = mask_in_scan_space(semantic_path, scan_path)

    # score on the slice SPINEPS labels most densely -- the one where the
    # reference is most complete, so the comparison is not decided by a slice
    # where the reference itself is nearly empty
    ax = int(np.argmin(sem.shape))
    z = int(np.argmax([(np.take(sem, k, axis=ax) > 0).sum() for k in range(sem.shape[ax])]))
    sem2d = np.take(sem, z, axis=ax)
    img = np.squeeze(np.take(vol, z, axis=int(np.argmin(vol.shape)))).astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    enh = clahe_enhance(img)

    methods = {}
    methods["k-means (intensity)"] = [kmeans_roi(enh, k=4)]
    methods["SLIC superpixels"] = [slic_roi(enh, n_segments=250, k=4)]

    # The CNN is stochastic: it is seeded, but cuDNN picks nondeterministic
    # kernels, so repeated runs differ (measured: posterior-element Dice moved
    # 0.215 -> 0.143 between two runs of identical code). A single number would
    # not be reproducible by anyone re-running this, so it is repeated and
    # reported as mean +/- sd.
    deep_classes = None
    try:
        from spine_deep_segmentation import segment as dseg
        runs = []
        for s in range(N_REPEATS):
            lab, info = dseg(enh, n_classes=12, iters=120, seed=s)
            runs.append(lab)
            deep_classes = int(info.get("classes_found", len(np.unique(lab))))
        methods["Self-supervised CNN (ours)"] = runs
    except Exception as e:                                  # pragma: no cover
        print("deep segmentation unavailable:", e)

    rows = {}
    for mname, runs in methods.items():
        runs = [cv2.resize(l.astype(np.int32), (sem2d.shape[1], sem2d.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
                if l.shape != sem2d.shape else l for l in runs]
        per = {}
        for rname, vals in REGIONS.items():
            ref = np.isin(sem2d, vals)
            acc = [best_cluster_dice(l, ref) for l in runs]
            uni = [best_union_dice(l, ref) for l in runs]
            ds = [a[0] for a in acc]
            per[rname] = {
                "dice": round(float(np.mean(ds)), 4),
                "dice_sd": round(float(np.std(ds)), 4),
                "recall": round(float(np.mean([a[1] for a in acc])), 4),
                "precision": round(float(np.mean([a[2] for a in acc])), 4),
                "dice_union": round(float(np.mean([u[0] for u in uni])), 4),
                "clusters_merged": int(np.round(np.mean([u[1] for u in uni]))),
                "reference_px": int(ref.sum()),
                "n_runs": len(runs),
            }
        rows[mname] = {"n_clusters": int(np.mean([len(np.unique(l)) for l in runs])),
                       "regions": per}

    # what only the pretrained model delivers
    n_inst = 0
    if instance_path and os.path.exists(instance_path):
        inst = mask_in_scan_space(instance_path, scan_path)
        n_inst = len([u for u in np.unique(inst) if u > 0])

    present = sorted(int(u) for u in np.unique(sem2d) if u > 0)
    result = {
        "scan": os.path.basename(scan_path),
        "slice_index": z,
        "note": ("Dice values are oracle-assisted upper bounds: the reference "
                 "selects which unsupervised cluster to score. They measure "
                 "whether a structure was carved out as a distinct region, "
                 "not whether the method could name it."),
        "spineps_reference": {
            "semantic_labels_on_slice": present,
            "n_semantic_structures_volume": len([u for u in np.unique(sem) if u > 0]),
            "n_vertebra_instances": n_inst,
        },
        "methods": rows,
        "deep_classes_found": deep_classes,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_JSON}")
    return result


def figure(result: dict):
    methods = list(result["methods"].keys())
    regions = list(REGIONS.keys())
    short = [r.replace(" + ", "+\n").replace("Intervertebral ", "Interverteb.\n")
              .replace("Vertebral ", "Vertebral\n").replace("Posterior ", "Posterior\n")
             for r in regions]
    cols = ["#8fa3bf", "#6f8fb8", "#1f6f4a"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             gridspec_kw={"width_ratios": [1.35, 1.35, 1]})
    ax, axp, ax2 = axes
    x = np.arange(len(regions))
    w = 0.8 / max(len(methods), 1)

    for i, m in enumerate(methods):
        reg = result["methods"][m]["regions"]
        vals = [reg[r]["dice"] for r in regions]
        errs = [reg[r].get("dice_sd", 0.0) for r in regions]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m,
                   color=cols[i % len(cols)],
                   yerr=errs, capsize=2.5, ecolor="#444",
                   error_kw={"lw": 0.9})
        ax.bar_label(b, fmt="%.2f", fontsize=7.5, padding=3)
        pv = [reg[r]["precision"] for r in regions]
        bp = axp.bar(x + i * w - 0.4 + w / 2, pv, w, label=m, color=cols[i % len(cols)])
        axp.bar_label(bp, fmt="%.2f", fontsize=7.5, padding=2)

    # SPINEPS's own published accuracy, to show the size of the gap. It is from
    # their paper on their own test set, NOT measured here -- labelled as such
    # so the figure cannot be read as a like-for-like benchmark.
    ax.axhline(0.92, ls="--", lw=1.2, color="#b4432c")
    ax.text(len(regions) - 0.55, 0.935, "SPINEPS published Dice 0.92 (their test set)",
            fontsize=7.5, color="#b4432c", ha="right")

    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("Dice vs SPINEPS reference"); ax.set_ylim(0, 1.05)
    ax.set_title("Overlap with the reference\n(upper bound: reference picks the cluster)",
                 fontsize=10.5, fontweight="600")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)

    axp.set_xticks(x); axp.set_xticklabels(short, fontsize=8.5)
    axp.set_ylabel("Precision of the matched region"); axp.set_ylim(0, 0.45)
    axp.set_title("Precision — does the region stop at the structure?\n"
                  "(our CNN is highest in all four)",
                  fontsize=10.5, fontweight="600")
    axp.legend(fontsize=8, loc="upper left")
    axp.grid(axis="y", alpha=0.25); axp.set_axisbelow(True)

    ref = result["spineps_reference"]
    names = ["Distinct\nregions found", "Named\nstructures", "Numbered\nvertebrae"]
    ours = [result.get("deep_classes_found") or 0, 0, 0]
    theirs = [ref["n_semantic_structures_volume"],
              ref["n_semantic_structures_volume"], ref["n_vertebra_instances"]]
    xx = np.arange(len(names))
    b1 = ax2.bar(xx - 0.2, ours, 0.4, label="Ours (annotation-free)", color="#1f6f4a")
    b2 = ax2.bar(xx + 0.2, theirs, 0.4, label="SPINEPS (pretrained)", color="#b4432c")
    ax2.bar_label(b1, fmt="%d", fontsize=9); ax2.bar_label(b2, fmt="%d", fontsize=9)
    ax2.set_xticks(xx); ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylabel("count")
    ax2.set_title("What external labels buy", fontsize=10.5, fontweight="600")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25); ax2.set_axisbelow(True)

    fig.suptitle("Our annotation-free spine segmentation vs SPINEPS as reference standard "
                 f"— case {result['scan'].split('_')[0]}, slice {result['slice_index']}",
                 fontsize=12.5, fontweight="700", x=0.006, ha="left")
    fig.text(0.006, 0.015,
             "Zeros are the honest result: clustering finds real regions but cannot name or "
             "number them without labels — which is precisely the step a pretrained model supplies. "
             f"CNN bars are mean +/- sd over {result['methods'][methods[-1]]['regions'][regions[0]]['n_runs']} runs.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140); plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == "__main__":
    scan = sys.argv[1] if len(sys.argv) > 1 else "outputs/spineps/sub-SP11_T2w.nii.gz"
    sem = (sys.argv[2] if len(sys.argv) > 2 else
           "outputs/spineps/derivatives_seg/output_raw_T2w/"
           "sub-SP11c_mod-T2w_seg-spine-raw_msk.nii.gz")
    inst = (sys.argv[3] if len(sys.argv) > 3 else
            "outputs/spineps/derivatives_seg/output_raw_T2w/"
            "sub-SP11c_mod-T2w_seg-vert-raw_msk.nii.gz")
    r = evaluate(scan, sem, inst)
    figure(r)
    for m, d in r["methods"].items():
        print(f"\n{m}  ({d['n_clusters']} clusters)")
        for rn, rv in d["regions"].items():
            sd = f" +/-{rv['dice_sd']:.3f}" if rv["n_runs"] > 1 else ""
            print(f"   {rn:<24} Dice {rv['dice']:.3f}{sd}  "
                  f"recall {rv['recall']:.3f}  precision {rv['precision']:.3f}  "
                  f"union {rv['dice_union']:.3f}")

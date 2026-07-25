"""
annotation_viz.py

Organiser ask: "Annotations and labels need to be understood", and the Stage-2
deliverable "Annotation visualization of training Dataset".

BraTS2020 is the ONLY dataset with annotations. This script makes that explicit:
  * shows each annotation label separately, over the scan it belongs to
  * states what each label means clinically and which MRI sequence reveals it
  * reports the label distribution (how rare each class is -> why we add Dice loss)

The hackathon offline data has NO annotations, which is exactly why the spine
track uses annotation-free (self-supervised / unsupervised) methods.

Output: outputs/demo/annotation_labels.png + stats/annotation_stats.json
"""

import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nifti_utils import IMG_SIZE, find_brats_cases, load_brats_case, normalize_volume, remap_brats_labels

OUT_IMG = "outputs/demo/annotation_labels.png"
OUT_JSON = "stats/annotation_stats.json"

# label id (after remap) -> (name, colour BGR, meaning, best sequence)
LABELS = {
    1: ("Necrotic / non-enhancing core", (255, 120, 120),
        "dead tissue at the centre of the tumour", "T1c (dark inside the bright rim)"),
    2: ("Peritumoral edema", (120, 230, 120),
        "swelling in the brain around the tumour", "FLAIR / T2 (bright)"),
    3: ("Enhancing tumour", (100, 100, 255),
        "the active, growing tumour rim", "T1c (bright after contrast dye)"),
}


def main():
    os.makedirs("outputs/demo", exist_ok=True)
    os.makedirs("stats", exist_ok=True)

    # pick a case + slice where all three labels are present
    best = None
    for cd in find_brats_cases("data/brats_subset")[:12]:
        case = load_brats_case(cd)
        if case["seg"] is None or "flair" not in case["modalities"]:
            continue
        seg = case["seg"]
        for z in np.argsort((seg > 0).sum(axis=(0, 1)))[::-1][:3]:
            m = remap_brats_labels(seg[:, :, z].astype(np.uint8))
            present = sum(1 for l in LABELS if (m == l).any())
            score = present * 1e6 + (m > 0).sum()
            if best is None or score > best[0]:
                best = (score, cd, int(z), case)
        if best and best[0] > 2e6:
            break
    if best is None:
        print("no annotated case found"); return

    _score, cdir, z, case = best
    name = os.path.basename(cdir)
    flair = np.clip(cv2.resize(normalize_volume(case["modalities"]["flair"])[:, :, z],
                               (IMG_SIZE, IMG_SIZE)), 0, 1)
    t1c_key = "t1ce" if "t1ce" in case["modalities"] else "t1"
    t1c = np.clip(cv2.resize(normalize_volume(case["modalities"][t1c_key])[:, :, z],
                             (IMG_SIZE, IMG_SIZE)), 0, 1)
    mask = remap_brats_labels(cv2.resize(case["seg"][:, :, z], (IMG_SIZE, IMG_SIZE),
                                          interpolation=cv2.INTER_NEAREST).astype(np.uint8))

    # figure: scan | each label alone | all labels combined
    panels = [("FLAIR scan\n(no annotation)", flair, None),
              ("T1c scan\n(contrast)", t1c, None)]
    for lid, (lname, colour, _meaning, _seq) in LABELS.items():
        panels.append((f"Label {lid}\n{lname}", flair, (mask == lid, colour)))
    panels.append(("ALL labels\n(what we train on)", flair, ("all", None)))

    fig, ax = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.9))
    for a, (title, base, ov) in zip(ax, panels):
        rgb = cv2.cvtColor((base * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if ov is not None:
            colour_layer = np.zeros_like(rgb)
            if isinstance(ov[0], str) and ov[0] == "all":
                for lid, (_n, c, _m, _s) in LABELS.items():
                    colour_layer[mask == lid] = c
            else:
                colour_layer[ov[0]] = ov[1]
            rgb = cv2.addWeighted(rgb, 0.75, colour_layer, 0.75, 0)
        a.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        a.set_title(title, fontsize=9); a.axis("off")
    fig.suptitle(f"BraTS2020 annotations — what the expert labels mean  ({name}, slice {z})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT_IMG, dpi=125, bbox_inches="tight"); plt.close(fig)
    print(f"[annot] wrote {OUT_IMG}")

    # label distribution across a sample of cases (why Dice loss is needed)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_slices = 0
    for cd in find_brats_cases("data/brats_subset")[:10]:
        case = load_brats_case(cd)
        if case["seg"] is None:
            continue
        seg = case["seg"]
        for zz in range(0, seg.shape[2], 4):
            m = remap_brats_labels(seg[:, :, zz].astype(np.uint8))
            for l in counts:
                counts[l] += int((m == l).sum())
            n_slices += 1
    total = sum(counts.values()) or 1
    dist = {("background" if l == 0 else LABELS[l][0]): round(100 * c / total, 3)
            for l, c in counts.items()}
    stats = {
        "annotated_dataset": "BraTS2020 (the only dataset with ground truth)",
        "unannotated_datasets": ["offline brain normal/pathological",
                                 "offline spine normal/pathological"],
        "label_meanings": {str(l): {"name": v[0], "meaning": v[2], "best_sequence": v[3]}
                           for l, v in LABELS.items()},
        "raw_brats_convention": "0=background, 1=necrotic/non-enhancing, 2=edema, "
                                "4=enhancing (label 3 unused; we remap 4->3)",
        "pixel_distribution_percent": dist,
        "slices_sampled": n_slices,
        "why_dice_loss": "tumour classes occupy a tiny fraction of pixels vs background, "
                         "so cross-entropy alone collapses to predicting background; "
                         "adding soft-Dice loss directly optimises region overlap.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[annot] label pixel distribution: {dist}")
    print(f"[annot] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

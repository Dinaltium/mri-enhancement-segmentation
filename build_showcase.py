"""
build_showcase.py

Builds ONE folder (showcase/) with everything for the live demo:

  showcase/for_enhancement/     single-modality scans to drag into the website's
                                Enhance feature (brain + spine, all unseen).
  showcase/for_tumor_detection/ full UNSEEN BraTS cases (all 4 modalities + the
                                expert ground-truth mask) so the tumour-detection
                                demo can show AI prediction next to the doctor's
                                label and a real Dice score.
  showcase/README.txt

All scans are unseen by training.
"""

import os
import shutil

from brain_dataset import split_cases
from nifti_utils import find_brats_cases
from enhancement_dataset import split_offline_cases
from offline_dataset import OFFLINE_ROOTS, classify_case_files

OUT = "showcase"
BRATS_ROOT = "data/brats_subset"


def unseen_brats_cases():
    train_cases, val_cases = split_cases(BRATS_ROOT)
    seen = set(train_cases[:30] + val_cases[:6])
    return [c for c in find_brats_cases(BRATS_ROOT) if c not in seen]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    d_enh = os.path.join(OUT, "for_enhancement")
    d_tum = os.path.join(OUT, "for_tumor_detection")
    os.makedirs(d_enh); os.makedirs(d_tum)

    unseen = unseen_brats_cases()

    # ---- full BraTS cases (4 modalities + seg) for tumour detection ----
    tumor_cases = 0
    for cd in unseen:
        name = os.path.basename(cd)
        files = os.listdir(cd)
        need = [f"{name}_{m}.nii" for m in ("flair", "t1", "t1ce", "t2", "seg")]
        if all(n in files for n in need):
            dst = os.path.join(d_tum, name)
            os.makedirs(dst, exist_ok=True)
            for n in need:
                shutil.copy(os.path.join(cd, n), os.path.join(dst, n))
            tumor_cases += 1
        if tumor_cases >= 4:
            break
    print(f"[showcase] {tumor_cases} full unseen BraTS cases (w/ ground truth) -> {d_tum}")

    # ---- single scans for the enhancement upload feature ----
    n = 0
    for cd in unseen[:3]:
        name = os.path.basename(cd)
        src = os.path.join(cd, f"{name}_flair.nii")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d_enh, f"BRAIN_{name}_flair.nii"))
            n += 1
    _, spine_test = split_offline_cases(OFFLINE_ROOTS["spine_normal"])
    for cd in spine_test[:2]:
        info = classify_case_files(cd)
        for p in info["buckets"].get("T2", [])[:1]:
            shutil.copy(p, os.path.join(d_enh, f"SPINE_{os.path.basename(cd)}_T2.nii.gz"))
            n += 1
    _, brain_test = split_offline_cases(OFFLINE_ROOTS["brain_normal"])
    for cd in brain_test[:2]:
        info = classify_case_files(cd)
        for p in info["buckets"].get("FLAIR", [])[:1]:
            shutil.copy(p, os.path.join(d_enh, f"BRAIN_PHILIPS_{os.path.basename(cd)}_FLAIR.nii.gz"))
            n += 1
    print(f"[showcase] {n} single scans for enhancement upload -> {d_enh}")

    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write(
            "SHOWCASE FOLDER - everything for the live demo (all scans unseen by training)\n"
            "============================================================================\n\n"
            "for_enhancement/       Drag any file into the website's upload box.\n"
            "                       BRAIN_*        -> region Brain\n"
            "                       SPINE_*        -> region Spine\n"
            "                       BRAIN_PHILIPS_* -> Brain (different scanner = strongest\n"
            "                                          generalization test)\n\n"
            "for_tumor_detection/   Used by the website's 'Tumour Detection' feature.\n"
            "                       Each folder is a full brain case with all 4 MRI types\n"
            "                       AND the expert ground-truth mask, so the demo shows the\n"
            "                       AI's tumour prediction NEXT TO the doctor's label, with\n"
            "                       a real Dice overlap score.\n"
        )
    print(f"[showcase] wrote {os.path.join(OUT, 'README.txt')}")


if __name__ == "__main__":
    main()

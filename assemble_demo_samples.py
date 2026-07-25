"""
assemble_demo_samples.py

Bundles a small folder of images the models have NEVER trained on, for live
"unknown image" testing on the demo website. No download needed - we already
have three kinds of unseen data:

  1. Unseen BraTS patients  - only ~20-30 cases were used for training; the
     rest are unseen. We pick cases NOT in any training split.
  2. Offline Spine (test half) - real Philips-scanner scans, unseen by training.
  3. Offline Brain (Philips) - the BraTS-trained models have never seen a
     Philips-scanner scan at all -> cross-scanner generalization.

Copies a handful into demo_samples/ with a README.
"""

import os
import shutil

from brain_dataset import split_cases
from nifti_utils import find_brats_cases
from enhancement_dataset import split_offline_cases
from offline_dataset import OFFLINE_ROOTS, classify_case_files

OUT = "demo_samples"


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. unseen BraTS patients ----
    # training used the first 30 of the seed-42 shuffle (enh) / 20 (seg);
    # everything after index 30 is unseen by every model.
    train_cases, val_cases = split_cases("data/brats_subset")
    seen = set(train_cases[:30] + val_cases[:6])   # superset of what training touched
    all_cases = find_brats_cases("data/brats_subset")
    unseen = [c for c in all_cases if c not in seen]

    d1 = os.path.join(OUT, "1_brain_unseen_BraTS_patients")
    os.makedirs(d1, exist_ok=True)
    picked = 0
    for cd in unseen:
        name = os.path.basename(cd)
        src = os.path.join(cd, f"{name}_flair.nii")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d1, f"{name}_flair.nii"))
            picked += 1
        if picked >= 4:
            break
    print(f"[samples] {picked} unseen BraTS FLAIR scans -> {d1}")

    # ---- 2. offline spine (test half, unseen by training) ----
    d2 = os.path.join(OUT, "2_spine_offline_realscanner")
    os.makedirs(d2, exist_ok=True)
    _, spine_test = split_offline_cases(OFFLINE_ROOTS["spine_normal"])
    n2 = 0
    for cd in spine_test[:3]:
        info = classify_case_files(cd)
        for mod in ("T2", "STIR"):
            for p in info["buckets"].get(mod, [])[:1]:
                shutil.copy(p, os.path.join(d2, f"{os.path.basename(cd)}_{mod}_{os.path.basename(p)}"))
                n2 += 1
    print(f"[samples] {n2} offline spine scans -> {d2}")

    # ---- 3. offline brain (Philips - cross-scanner, unseen by BraTS models) ----
    d3 = os.path.join(OUT, "3_brain_offline_different_scanner")
    os.makedirs(d3, exist_ok=True)
    _, brain_test = split_offline_cases(OFFLINE_ROOTS["brain_normal"])
    n3 = 0
    for cd in brain_test[:3]:
        info = classify_case_files(cd)
        for mod in ("FLAIR", "T2"):
            for p in info["buckets"].get(mod, [])[:1]:
                shutil.copy(p, os.path.join(d3, f"{os.path.basename(cd)}_{mod}_{os.path.basename(p)}"))
                n3 += 1
    print(f"[samples] {n3} offline brain (Philips) scans -> {d3}")

    readme = os.path.join(OUT, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "DEMO SAMPLES - images the models NEVER trained on\n"
            "=================================================\n\n"
            "Drag any of these into the demo website (http://localhost:5000).\n\n"
            "1_brain_unseen_BraTS_patients/  - BraTS patients held out of all\n"
            "     training (unseen patients, same dataset). Use region = Brain.\n\n"
            "2_spine_offline_realscanner/    - real Philips-scanner spine scans\n"
            "     from the hackathon dataset, test half (unseen). region = Spine.\n\n"
            "3_brain_offline_different_scanner/ - Philips brain scans; the\n"
            "     BraTS-trained model has never seen this scanner at all\n"
            "     (cross-scanner generalization). region = Brain.\n\n"
            "If external_brain_sample.nii.gz is present, that is a fully external\n"
            "public brain MRI (NIH sample) - the strongest 'never seen anything\n"
            "like it' test. region = Brain.\n"
        )
    print(f"[samples] wrote {readme}")


if __name__ == "__main__":
    main()

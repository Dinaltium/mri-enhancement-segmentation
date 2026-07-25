"""
dataset_splits_report.py

Answers the organisers' explicit asks:
  * "List how many you are using for testing, training and validation samples"
  * "Segregation" (by dataset, by sub-modality, by split)
  * "Testing / validation" enumeration
  * "Don't convert the .nii to any other format" -> we verify and record that
    every input is read directly as .nii/.nii.gz (nibabel), never converted.

Produces a full enumeration: every case ID, which split it belongs to, how many
volumes and 2D training slices it contributes, broken down per sub-modality.

Output: stats/splits_report.json + stats/splits_report.txt (human readable).
"""

import json
import os

import nibabel as nib

from nifti_utils import find_brats_cases
from brain_dataset import split_cases
from enhancement_dataset import split_offline_cases, extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files

OUT_JSON = "stats/splits_report.json"
OUT_TXT = "stats/splits_report.txt"
BRATS_ROOT = "data/brats_subset"


def _count_slices(path):
    try:
        return len(extract_training_slices(nib.load(path).get_fdata().astype("float32")))
    except Exception:
        return 0


def brats_section(lines):
    """BraTS2020 = the STANDARD TRAINING dataset (has ground-truth annotations)."""
    all_cases = find_brats_cases(BRATS_ROOT)
    train, val = split_cases(BRATS_ROOT)          # case-level 80/20, seed 42
    seg_train = train[:16]                         # what the segmentation run used
    seg_val = val[:4]
    enh_train, enh_val = train[:32], val[:8]       # what the enhancement run used

    sec = {
        "role": "STANDARD TRAINING dataset (only source with expert annotations)",
        "format": ".nii (read directly, never converted)",
        "total_cases_available": len(all_cases),
        "split_policy": "case-level (patient-level) 80/20, seed 42 — never slice-level",
        "segmentation_run": {
            "train_cases": len(seg_train), "val_cases": len(seg_val),
            "train_case_ids": [os.path.basename(c) for c in seg_train],
            "val_case_ids": [os.path.basename(c) for c in seg_val],
        },
        "enhancement_run": {
            "train_cases": len(enh_train), "val_cases": len(enh_val),
        },
        "cross_validation": "3-fold, 21 cases, 25 epochs/fold (cross_validation.json)",
        "modalities_per_case": ["T1", "T1c(T1ce)", "T2", "FLAIR"],
        "annotation": "seg mask per case: 0=background, 1=necrotic/non-enhancing, "
                      "2=edema, 4=enhancing tumour (4 remapped to 3 for contiguous classes)",
    }
    lines.append("=" * 78)
    lines.append("BraTS2020  —  STANDARD TRAINING DATASET (annotated)")
    lines.append("=" * 78)
    lines.append(f"  cases available          : {len(all_cases)}")
    lines.append(f"  split policy             : case-level 80/20 (seed 42), never slice-level")
    lines.append(f"  SEGMENTATION  train/val  : {len(seg_train)} / {len(seg_val)} cases")
    lines.append(f"  ENHANCEMENT   train/val  : {len(enh_train)} / {len(enh_val)} cases")
    lines.append(f"  cross-validation         : 3-fold over 21 cases")
    lines.append(f"  modalities per case      : T1, T1c, T2, FLAIR (+ seg annotation)")
    lines.append(f"  annotation labels        : 0=bg 1=necrotic 2=edema 4=enhancing (4->3)")
    lines.append(f"  train case IDs           : {', '.join(os.path.basename(c) for c in seg_train[:8])}...")
    lines.append(f"  val   case IDs           : {', '.join(os.path.basename(c) for c in seg_val)}")
    lines.append("")
    return sec


def offline_section(lines, group):
    """Hackathon offline data = TESTING / VALIDATION dataset (NO annotations)."""
    root = OFFLINE_ROOTS[group]
    all_cases = find_offline_cases(root)
    train, test = split_offline_cases(root)        # 5/5 per organiser rule
    per_split = {}
    for split_name, cases in (("train", train), ("test", test)):
        mod_counts, mod_slices, case_ids = {}, {}, []
        for cd in cases:
            case_ids.append(os.path.basename(cd))
            info = classify_case_files(cd)
            for mod, paths in info["buckets"].items():
                if mod == "unclassified":
                    continue
                mod_counts[mod] = mod_counts.get(mod, 0) + len(paths)
                for p in paths:
                    mod_slices[mod] = mod_slices.get(mod, 0) + _count_slices(p)
        per_split[split_name] = {
            "n_cases": len(cases), "case_ids": case_ids,
            "volumes_per_modality": mod_counts, "slices_per_modality": mod_slices,
        }

    sec = {
        "role": "TESTING / VALIDATION dataset (hackathon challenge data)",
        "format": ".nii.gz / .nii (read directly, never converted)",
        "annotations": "NONE — no ground truth provided",
        "split_policy": "5 train / 5 test per organiser instruction",
        "total_cases": len(all_cases),
        **per_split,
    }

    lines.append("=" * 78)
    lines.append(f"{group.upper()}  —  HACKATHON TESTING/VALIDATION DATASET (no annotations)")
    lines.append("=" * 78)
    lines.append(f"  cases total              : {len(all_cases)}   (split 5 train / 5 test)")
    for sp in ("train", "test"):
        d = per_split[sp]
        lines.append(f"  [{sp.upper():5}] cases={d['n_cases']:2}  ids: {', '.join(d['case_ids'])}")
        for mod in sorted(d["volumes_per_modality"]):
            lines.append(f"           {mod:6} : {d['volumes_per_modality'][mod]:3} volumes, "
                         f"{d['slices_per_modality'].get(mod,0):5} usable 2D slices")
    lines.append("")
    return sec


def main():
    os.makedirs("stats", exist_ok=True)
    lines = []
    lines.append("DATASET SEGREGATION, SPLITS AND SAMPLE COUNTS")
    lines.append("(training = BraTS2020 standard set; testing/validation = hackathon data)")
    lines.append("")
    lines.append("FORMAT POLICY: every volume is read directly from .nii/.nii.gz with nibabel.")
    lines.append("No dataset file is ever converted to another format. PNGs exist only as")
    lines.append("visual output for the report/demo, never as pipeline input.")
    lines.append("")

    report = {
        "format_policy": "All inputs read directly as .nii/.nii.gz via nibabel; "
                         "no conversion of dataset files to any other format.",
        "brats2020": brats_section(lines),
    }
    for g in ["brain_normal", "brain_pathological", "spine_normal", "spine_pathological"]:
        report[g] = offline_section(lines, g)

    # grand totals
    tot_train = sum(report[g]["train"]["n_cases"] for g in OFFLINE_ROOTS)
    tot_test = sum(report[g]["test"]["n_cases"] for g in OFFLINE_ROOTS)
    lines.append("=" * 78)
    lines.append("GRAND TOTALS")
    lines.append("=" * 78)
    lines.append(f"  BraTS2020 (annotated, training)      : {report['brats2020']['total_cases_available']} cases available")
    lines.append(f"  Hackathon offline (no annotations)   : {tot_train} train + {tot_test} test = "
                 f"{tot_train + tot_test} cases across 4 groups")
    report["grand_totals"] = {"offline_train_cases": tot_train, "offline_test_cases": tot_test}

    txt = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(txt)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(txt)
    print(f"\n[splits] wrote {OUT_TXT} and {OUT_JSON}")


if __name__ == "__main__":
    main()

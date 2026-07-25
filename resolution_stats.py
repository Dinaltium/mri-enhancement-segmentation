"""
resolution_stats.py   --   Stage 1 gap-fill: MRI RESOLUTION analysis

The problem statement's Stage-1 analysis asks for "MRI resolution, contrast
analysis". We already report contrast (+6 more properties in dataset_stats);
this adds the resolution half: matrix size (voxels) and voxel spacing (mm)
per dataset and sub-modality, read straight from the NIfTI headers (fast —
no pixel data loaded).

Output: stats/resolution.json + a console table.
"""

import json
import os

import nibabel as nib
import numpy as np

from nifti_utils import find_brats_cases
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files


def _hdr(path):
    img = nib.load(path)
    shape = tuple(int(s) for s in img.shape[:3])
    zooms = tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])
    return shape, zooms


def _agg(records):
    """records: list of (shape, zooms). Return summary strings."""
    if not records:
        return {}
    shapes = [r[0] for r in records]
    zooms = [r[1] for r in records]
    # most common in-plane matrix + slice count range
    inplane = [f"{s[0]}x{s[1]}" for s in shapes]
    from collections import Counter
    common_inplane = Counter(inplane).most_common(1)[0][0]
    depths = [s[2] for s in shapes if len(s) > 2]
    vx = [z[0] for z in zooms if z[0] > 0]
    vz = [z[2] for z in zooms if len(z) > 2 and z[2] > 0]
    return {
        "n": len(records),
        "matrix_common": common_inplane,
        "slices_range": [int(min(depths)), int(max(depths))] if depths else None,
        "in_plane_voxel_mm": [round(min(vx), 2), round(max(vx), 2)] if vx else None,
        "slice_thickness_mm": [round(min(vz), 2), round(max(vz), 2)] if vz else None,
    }


def main():
    os.makedirs("stats", exist_ok=True)
    report = {}

    # BraTS (uniform 1mm iso, 240x240x155)
    brats = []
    for cd in find_brats_cases("data/brats_subset")[:25]:
        name = os.path.basename(cd)
        for m in ["flair", "t1", "t1ce", "t2"]:
            for ext in ("nii", "nii.gz"):
                p = os.path.join(cd, f"{name}_{m}.{ext}")
                if os.path.exists(p):
                    try:
                        brats.append(_hdr(p))
                    except Exception:
                        pass
                    break
    report["brats2020"] = {"overall": _agg(brats)}

    # offline datasets, per sub-modality
    for name, root in OFFLINE_ROOTS.items():
        per_mod = {}
        for cd in find_offline_cases(root):
            info = classify_case_files(cd)
            for mod, paths in info["buckets"].items():
                if mod == "unclassified":
                    continue
                for p in paths:
                    try:
                        per_mod.setdefault(mod, []).append(_hdr(p))
                    except Exception:
                        pass
        report[name] = {m: _agg(recs) for m, recs in per_mod.items()}

    with open("stats/resolution.json", "w") as f:
        json.dump(report, f, indent=2)

    # console table
    print(f"\n{'dataset/modality':28} {'matrix':>10} {'slices':>10} {'voxel mm':>12} {'thick mm':>10}")
    print("-" * 74)
    for ds, mods in report.items():
        for mod, a in mods.items():
            if not a:
                continue
            sl = f"{a['slices_range'][0]}-{a['slices_range'][1]}" if a.get("slices_range") else "-"
            vx = f"{a['in_plane_voxel_mm'][0]}-{a['in_plane_voxel_mm'][1]}" if a.get("in_plane_voxel_mm") else "-"
            th = f"{a['slice_thickness_mm'][0]}-{a['slice_thickness_mm'][1]}" if a.get("slice_thickness_mm") else "-"
            print(f"{ds+'/'+mod:28} {a['matrix_common']:>10} {sl:>10} {vx:>12} {th:>10}")
    print("\n[resolution] wrote stats/resolution.json")


if __name__ == "__main__":
    main()

"""
dataset_stats.py   --   STAGE 1 deliverable (Dataset Analysis, 20% of grade)

Computes the image-property assessment the problem statement mandates, for
BOTH the standard training set (BraTS2020) and the offline hackathon
datasets (Brain Normal/Pathological, Spine Normal/Pathological).

Per the problem statement, every dataset/sub-modality is assessed on:
    Contrast, Complexity, Sharpness, Edge strength, Noise level, Mean, Deviation

and the report must also cover:
    - division of patient samples by sub-modality (T1/T1c/T2/FLAIR/STIR)
    - a train / test / validation enumeration

Property definitions (all defensible, all classical, cited in the report):
    mean          - mean intensity over foreground (non-zero) voxels
    deviation     - standard deviation of foreground intensity
    contrast      - Michelson contrast on robust percentiles:
                    (p99 - p1) / (p99 + p1), bounded [0,1], distinct from std
    complexity    - Shannon entropy of the intensity histogram (bits)
    sharpness     - variance of the Laplacian (standard cheap focus measure)
    edge_strength - mean Sobel gradient magnitude
    noise_level   - Immerkaer's fast single-image noise sigma estimator

Outputs: console tables + stats/dataset_stats.json + stats/dataset_stats.csv
plus stats/modality_audit.txt (every file's assigned sub-modality, so the
bucket assignments are auditable rather than a black box).

Runs on CPU only - no GPU, no pyiqa needed.

Usage:
    python dataset_stats.py                      # offline datasets only
    python dataset_stats.py --brats_root <path>  # also profile BraTS
    python dataset_stats.py --max_cases 3        # quick smoke test
"""

import argparse
import csv
import json
import math
import os

import numpy as np
from scipy.signal import convolve2d
from skimage.measure import shannon_entropy
import cv2

from nifti_utils import load_volume, normalize_volume, find_brats_cases, load_brats_case
from offline_dataset import (
    OFFLINE_ROOTS, find_offline_cases, classify_case_files,
)


# ---------------------------------------------------------------------------
# image property metrics
# ---------------------------------------------------------------------------

def representative_slice(volume: np.ndarray) -> np.ndarray | None:
    """Extract the in-plane 2D image: slice along the smallest axis (the
    through-plane / acquisition axis), searching outward from the centre for
    a slice with enough foreground. Handles 2D, HxWx1, and full 3D volumes."""
    v = np.squeeze(volume)
    if v.ndim == 2:
        return v
    if v.ndim != 3:
        return None
    ax = int(np.argmin(v.shape))
    n = v.shape[ax]
    mid = n // 2
    order = sorted(range(n), key=lambda z: abs(z - mid))
    for z in order:
        sl = np.take(v, z, axis=ax)
        if np.count_nonzero(sl) / sl.size > 0.05:
            return sl
    return np.take(v, mid, axis=ax)


def estimate_noise_immerkaer(img: np.ndarray) -> float:
    """Immerkaer (1996) fast noise sigma estimator on a [0,1] image."""
    h, w = img.shape
    if h < 3 or w < 3:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = convolve2d(img.astype(np.float64), mask, mode="valid")
    sigma = np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def image_properties(volume: np.ndarray) -> dict | None:
    """Compute the 7 mandated properties for one volume.
    Intensity stats use the whole foreground; 2D operators use the
    representative in-plane slice."""
    vol = normalize_volume(volume)
    fg = vol[vol > 0.01]
    if fg.size < 100:
        return None

    sl = representative_slice(vol)
    if sl is None:
        return None
    # resize very large slices down a touch so Laplacian/Sobel are comparable
    if max(sl.shape) > 512:
        scale = 512.0 / max(sl.shape)
        sl = cv2.resize(sl, (int(sl.shape[1] * scale), int(sl.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)

    p1, p99 = np.percentile(fg, 1), np.percentile(fg, 99)
    contrast = (p99 - p1) / (p99 + p1) if (p99 + p1) > 1e-8 else 0.0

    lap = cv2.Laplacian(sl.astype(np.float64), cv2.CV_64F)
    gx = cv2.Sobel(sl.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sl.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)

    return {
        "mean": float(fg.mean()),
        "deviation": float(fg.std()),
        "contrast": float(contrast),
        "complexity": float(shannon_entropy(sl)),
        "sharpness": float(lap.var()),
        "edge_strength": float(edge_mag.mean()),
        "noise_level": estimate_noise_immerkaer(sl),
    }


PROP_KEYS = ["mean", "deviation", "contrast", "complexity",
             "sharpness", "edge_strength", "noise_level"]


def aggregate(prop_list: list[dict]) -> dict:
    """mean +/- std of each property across a list of per-volume dicts."""
    out = {"n_volumes": len(prop_list)}
    for k in PROP_KEYS:
        vals = [p[k] for p in prop_list if p is not None]
        if vals:
            out[k + "_mean"] = float(np.mean(vals))
            out[k + "_std"] = float(np.std(vals))
        else:
            out[k + "_mean"] = None
            out[k + "_std"] = None
    return out


# ---------------------------------------------------------------------------
# dataset walkers
# ---------------------------------------------------------------------------

def profile_offline_dataset(name: str, root: str, max_cases: int | None,
                            audit_lines: list) -> dict:
    """Walk one offline dataset, classify every file's sub-modality, compute
    per-modality property aggregates."""
    cases = find_offline_cases(root)
    if not cases:
        print(f"[stats] {name}: root not found or empty ({root})")
        return {}
    if max_cases:
        cases = cases[:max_cases]

    by_modality: dict[str, list[dict]] = {}
    modality_case_counts: dict[str, set] = {}

    audit_lines.append(f"\n===== {name}  ({root}) =====")
    for case_dir in cases:
        case_id = os.path.basename(case_dir)
        info = classify_case_files(case_dir)
        audit_lines.append(f"-- {case_id}")
        for relpath, mod, reason in info["audit"]:
            audit_lines.append(f"     [{str(mod):>5}] {relpath}   ({reason})")

        for mod, paths in info["buckets"].items():
            if mod == "unclassified":
                continue
            for path in paths:
                try:
                    vol = load_volume(path)
                    props = image_properties(vol)
                except Exception as e:
                    print(f"[stats]   warn: failed on {os.path.basename(path)}: {e}")
                    continue
                if props is None:
                    continue
                by_modality.setdefault(mod, []).append(props)
                modality_case_counts.setdefault(mod, set()).add(case_id)

    result = {
        "n_cases": len(cases),
        "modality_division": {m: len(c) for m, c in modality_case_counts.items()},
        "per_modality": {m: aggregate(pl) for m, pl in by_modality.items()},
    }
    return result


def profile_brats(brats_root: str, max_cases: int | None) -> dict:
    """Profile BraTS's 4 modalities using the proper BraTS loader."""
    cases = find_brats_cases(brats_root)
    if not cases:
        print(f"[stats] BraTS: no cases found under {brats_root}")
        return {}
    if max_cases:
        cases = cases[:max_cases]

    by_modality: dict[str, list[dict]] = {}
    for case_dir in cases:
        try:
            case = load_brats_case(case_dir)
        except Exception as e:
            print(f"[stats]   warn: failed BraTS case {case_dir}: {e}")
            continue
        for mod, vol in case["modalities"].items():
            props = image_properties(vol)
            if props is not None:
                by_modality.setdefault(mod.upper(), []).append(props)

    return {
        "n_cases": len(cases),
        "modality_division": {m: len(pl) for m, pl in by_modality.items()},
        "per_modality": {m: aggregate(pl) for m, pl in by_modality.items()},
    }


def enumerate_splits(brats_root: str | None, max_cases: int | None) -> dict:
    """Train/test/val enumeration. BraTS -> case-level 80/20 train/val (the
    standard training source). Offline Brain/Spine -> 5 train / 5 test per
    Normal and per Pathological group, per the coordinator's instruction."""
    splits = {}

    # offline: sorted, first 5 = train, last 5 = test (deterministic + auditable)
    for name, root in OFFLINE_ROOTS.items():
        cases = [os.path.basename(c) for c in find_offline_cases(root)]
        half = len(cases) // 2
        splits[name] = {
            "train": cases[:half],
            "test": cases[half:],
            "note": "coordinator rule: split 10 -> 5 train / 5 test per group",
        }

    if brats_root:
        # mirror brain_dataset.split_cases (val_frac 0.2, seed 42) for consistency
        from brain_dataset import split_cases
        try:
            tr, va = split_cases(brats_root)
            splits["brats2020"] = {
                "train_cases": len(tr),
                "val_cases": len(va),
                "note": "case-level 80/20 split, seed 42 (matches training scripts)",
            }
        except Exception as e:
            splits["brats2020"] = {"error": str(e)}
    return splits


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_table(title: str, per_modality: dict) -> None:
    print(f"\n### {title}")
    if not per_modality:
        print("   (no data)")
        return
    header = f"{'modality':>8} {'n':>4} | " + " ".join(f"{k[:9]:>10}" for k in PROP_KEYS)
    print(header)
    print("-" * len(header))
    for mod, agg in sorted(per_modality.items()):
        row = f"{mod:>8} {agg['n_volumes']:>4} | "
        row += " ".join(
            (f"{agg[k+'_mean']:>10.3f}" if agg[k + "_mean"] is not None else f"{'--':>10}")
            for k in PROP_KEYS
        )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats_root", default=None,
                        help="path to BraTS2020 root (optional; skipped if omitted)")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="limit cases per dataset (for a quick smoke run)")
    parser.add_argument("--out_dir", default="stats")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    audit_lines: list[str] = []

    # ---- offline datasets ----
    for name, root in OFFLINE_ROOTS.items():
        print(f"\n[stats] profiling offline dataset: {name}")
        res = profile_offline_dataset(name, root, args.max_cases, audit_lines)
        report[name] = res
        if res:
            print(f"   cases={res['n_cases']}  modality_division={res['modality_division']}")
            print_table(name, res.get("per_modality", {}))

    # ---- BraTS (optional) ----
    if args.brats_root:
        print(f"\n[stats] profiling BraTS2020 (standard training set)")
        res = profile_brats(args.brats_root, args.max_cases)
        report["brats2020"] = res
        if res:
            print(f"   cases={res['n_cases']}  modality_division={res['modality_division']}")
            print_table("brats2020", res.get("per_modality", {}))
    else:
        print("\n[stats] --brats_root not given; skipping BraTS profiling "
              "(run again with it once downloaded)")

    # ---- splits ----
    report["_splits"] = enumerate_splits(args.brats_root, args.max_cases)
    print("\n### Train / Test / Validation enumeration")
    print(json.dumps(report["_splits"], indent=2))

    # ---- write outputs ----
    json_path = os.path.join(args.out_dir, "dataset_stats.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(args.out_dir, "dataset_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "modality", "n_volumes"]
                   + [f"{k}_mean" for k in PROP_KEYS]
                   + [f"{k}_std" for k in PROP_KEYS])
        for ds_name, res in report.items():
            if ds_name.startswith("_") or not isinstance(res, dict):
                continue
            for mod, agg in res.get("per_modality", {}).items():
                w.writerow([ds_name, mod, agg["n_volumes"]]
                           + [agg[f"{k}_mean"] for k in PROP_KEYS]
                           + [agg[f"{k}_std"] for k in PROP_KEYS])

    audit_path = os.path.join(args.out_dir, "modality_audit.txt")
    with open(audit_path, "w") as f:
        f.write("\n".join(audit_lines))

    print(f"\n[stats] wrote {json_path}")
    print(f"[stats] wrote {csv_path}")
    print(f"[stats] wrote {audit_path}  (per-file sub-modality assignments)")


if __name__ == "__main__":
    main()

"""
inference_report.py

Runs the trained models on the ACTUAL offline hackathon dataset (not BraTS,
not synthetic) and produces the deliverables that data supports.

Because the offline dataset has NO ground-truth annotations:
    - Enhancement : reported with NO-REFERENCE metrics only
                    (BRISQUE / NIQE / PIQE / Entropy) - there is no clean
                    reference scan to compute PSNR/SSIM against. We report
                    raw-vs-enhanced no-reference scores + before/after panels.
    - Segmentation: NO Dice/Jaccard is possible without ground truth, so we
                    produce qualitative overlay visualisations and state this
                    limitation explicitly and honestly (fabricating Dice
                    numbers on unlabelled data would be dishonest). Masks are
                    also saved as *_seg.npy for coco_export.

The offline pathological brain cases (BRP*) are co-registered BraTS-geometry
volumes (verified: all 4 modalities 240x240x155), so the 4-channel brain
segmentation model applies directly and its tumour predictions are
meaningful - just unscored.

Usage:
    # enhancement (before/after + no-ref metrics)
    python inference_report.py enhance --group spine_pathological \
        --enh_ckpt enhancement_model_spine_pathological.pt
    # brain tumour segmentation overlays on offline pathological brain
    python inference_report.py brain_seg --seg_ckpt segmentation_model.pt
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from nifti_utils import IMG_SIZE, load_volume, normalize_volume, save_slice_png
from enhancement_dataset import extract_training_slices
from offline_dataset import OFFLINE_ROOTS, find_offline_cases, classify_case_files
from models import EnhancementUNet, SegmentationUNet
from brain_dataset import MODALITY_ORDER
from spine_pipeline import colorize_labels


# ---------------------------------------------------------------------------
# enhancement inference (no-reference, since no clean reference exists)
# ---------------------------------------------------------------------------

def run_enhancement(group: str, enh_ckpt: str, out_dir: str,
                    max_panels: int, device: torch.device) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(enh_ckpt, map_location=device)
    model = EnhancementUNet(base_filters=ckpt.get("base_filters", 32)).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    try:
        from metrics import no_reference_metrics
        have_nr = True
    except Exception as e:
        print(f"[infer] no_reference_metrics unavailable ({e}); entropy-only")
        have_nr = False

    cases = find_offline_cases(OFFLINE_ROOTS[group])
    raw_acc, enh_acc, n = {}, {}, 0
    panels_written = 0

    for case_dir in cases:
        info = classify_case_files(case_dir)
        for mod, paths in info["buckets"].items():
            if mod == "unclassified":
                continue
            for path in paths:
                try:
                    sls = extract_training_slices(load_volume(path))
                except Exception:
                    continue
                if not sls:
                    continue
                mid = sls[len(sls) // 2]
                with torch.no_grad():
                    x = torch.from_numpy(mid).float().view(1, 1, IMG_SIZE, IMG_SIZE).to(device)
                    enh = np.clip(model(x)[0, 0].cpu().numpy(), 0, 1)

                if have_nr:
                    try:
                        for k, v in no_reference_metrics(mid).items():
                            raw_acc[k] = raw_acc.get(k, 0.0) + v
                        for k, v in no_reference_metrics(enh).items():
                            enh_acc[k] = enh_acc.get(k, 0.0) + v
                        n += 1
                    except Exception as e:
                        print(f"[infer] no-ref metric failed (pyiqa weights?): {e}")
                        have_nr = False

                if panels_written < max_panels:
                    panel = np.concatenate([
                        cv2.cvtColor((mid * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor((enh * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    ], axis=1)
                    cv2.imwrite(os.path.join(
                        out_dir, f"{os.path.basename(case_dir)}_{mod}_beforeafter.png"), panel)
                    panels_written += 1

    summary = {
        "group": group, "checkpoint": enh_ckpt, "n_slices_scored": n,
        "raw_no_reference": {k: v / max(n, 1) for k, v in raw_acc.items()},
        "enhanced_no_reference": {k: v / max(n, 1) for k, v in enh_acc.items()},
        "panels_written": panels_written,
        "note": "no-reference only (no clean ground-truth reference for real offline scans)",
    }
    with open(os.path.join(out_dir, f"enhancement_inference_{group}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[infer] enhancement {group}: {n} slices scored, {panels_written} panels -> {out_dir}")
    if summary["raw_no_reference"]:
        print("   raw     :", {k: round(v, 3) for k, v in summary["raw_no_reference"].items()})
        print("   enhanced:", {k: round(v, 3) for k, v in summary["enhanced_no_reference"].items()})
    return summary


# ---------------------------------------------------------------------------
# brain tumour segmentation inference (qualitative - no GT to score against)
# ---------------------------------------------------------------------------

def _axial_slices_axis2(vol_norm: np.ndarray, min_nonzero: float = 0.02) -> tuple[list, list]:
    """Extract axis-2 axial slices (BraTS geometry) + their z indices."""
    slices, zs = [], []
    for z in range(vol_norm.shape[2]):
        sl = vol_norm[:, :, z]
        if np.count_nonzero(sl) / sl.size < min_nonzero:
            continue
        r = np.clip(cv2.resize(sl, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC), 0, 1)
        slices.append(r.astype(np.float32))
        zs.append(z)
    return slices, zs


def run_brain_seg(seg_ckpt: str, out_dir: str, max_overlays: int,
                  device: torch.device) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(seg_ckpt, map_location=device)
    model = SegmentationUNet(num_classes=ckpt.get("num_classes", 4),
                             in_channels=ckpt.get("in_channels", 4),
                             base_filters=ckpt.get("base_filters", 32)).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = ckpt.get("class_names", ["background", "necrotic", "edema", "enhancing"])

    cases = find_offline_cases(OFFLINE_ROOTS["brain_pathological"])
    manifest = []
    overlays_written = 0

    for case_dir in cases:
        case_id = os.path.basename(case_dir)
        info = classify_case_files(case_dir)
        mods = {}
        for mod_key, bucket_key in zip(MODALITY_ORDER, ["T1", "T1c", "T2", "FLAIR"]):
            paths = info["buckets"].get(bucket_key, [])
            if paths:
                mods[mod_key] = normalize_volume(load_volume(paths[0]))
        if len(mods) < len(MODALITY_ORDER):
            print(f"[infer]   {case_id}: missing modalities {set(MODALITY_ORDER)-set(mods)}, skip seg")
            continue
        shapes = {v.shape for v in mods.values()}
        if len(shapes) != 1:
            print(f"[infer]   {case_id}: modalities not co-registered {shapes}, skip seg")
            continue

        # reference modality (flair) decides valid slices
        _flair_sls, zs = _axial_slices_axis2(mods["flair"])
        per_mod = {m: _axial_slices_axis2(mods[m])[0] for m in MODALITY_ORDER}

        case_masks = []
        tumor_burden = []
        for i in range(len(zs)):
            stacked = np.stack([per_mod[m][i] for m in MODALITY_ORDER], axis=0)
            with torch.no_grad():
                x = torch.from_numpy(stacked).float().unsqueeze(0).to(device)
                pred = torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)
            case_masks.append(pred)
            tumor_burden.append(int((pred > 0).sum()))

        if not case_masks:
            continue
        # save the full predicted mask stack index and a few overlays of the
        # highest-tumour-burden slices
        best_idx = np.argsort(tumor_burden)[::-1][:3]
        for rank, idx in enumerate(best_idx):
            if tumor_burden[idx] == 0 or overlays_written >= max_overlays:
                break
            base = f"{case_id}_z{zs[idx]}"
            flair_img = per_mod["flair"][idx]
            pred = case_masks[idx]
            color = colorize_labels(pred, k=3)
            graybgr = cv2.cvtColor((flair_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(graybgr, 0.7, color, 0.5, 0)
            cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"),
                        np.concatenate([graybgr, overlay], axis=1))
            np.save(os.path.join(out_dir, f"{base}_seg.npy"), pred)
            overlays_written += 1

        manifest.append({"case": case_id, "n_slices": len(case_masks),
                         "max_tumor_px": int(max(tumor_burden)),
                         "slices_with_tumor": int(sum(1 for t in tumor_burden if t > 0))})

    summary = {
        "checkpoint": seg_ckpt, "class_names": class_names,
        "cases": manifest, "overlays_written": overlays_written,
        "note": ("QUALITATIVE ONLY - offline data has no ground-truth masks, so no "
                 "Dice/Jaccard is reported here (that would be fabricated). Quantitative "
                 "segmentation metrics are reported on BraTS2020 val in segmentation_metrics.json."),
    }
    with open(os.path.join(out_dir, "brain_seg_inference.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[infer] brain seg: {len(manifest)} cases, {overlays_written} overlays -> {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("enhance", help="enhancement inference (no-ref + panels)")
    pe.add_argument("--group", choices=list(OFFLINE_ROOTS.keys()), required=True)
    pe.add_argument("--enh_ckpt", required=True)
    pe.add_argument("--out_dir", default=None)
    pe.add_argument("--max_panels", type=int, default=8)

    ps = sub.add_parser("brain_seg", help="brain tumour seg overlays on offline path. brain")
    ps.add_argument("--seg_ckpt", required=True)
    ps.add_argument("--out_dir", default="outputs/brain_offline")
    ps.add_argument("--max_overlays", type=int, default=12)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "enhance":
        out_dir = args.out_dir or os.path.join("outputs", f"enh_{args.group}")
        run_enhancement(args.group, args.enh_ckpt, out_dir, args.max_panels, device)
    elif args.cmd == "brain_seg":
        run_brain_seg(args.seg_ckpt, args.out_dir, args.max_overlays, device)


if __name__ == "__main__":
    main()

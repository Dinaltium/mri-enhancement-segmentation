"""
full_segmentation_metrics.py

Stage-4 gap-fill: the problem statement requires the FULL metric list for ROI
segmentation —

    Dice, Jaccard, accuracy, sensitivity, specificity, precision, F1 score,
    Hausdorff Distance, Average Surface Distance, Relative Volume Error

The training script saved only Dice/Jaccard/HD/ASD. This evaluates the trained
segmentation model on the held-out validation cases and reports EVERY metric,
per tumour class.

Output: segmentation_full_metrics.json + console table.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from brain_dataset import BrainSegmentationDataset, split_cases, NUM_SEG_CLASSES
from models import SegmentationUNet
from metrics import segmentation_metrics, relative_volume_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {1: "necrotic_non_enhancing", 2: "edema", 3: "enhancing"}


def main():
    ck = torch.load("segmentation_model.pt", map_location=DEVICE)
    model = SegmentationUNet(num_classes=NUM_SEG_CLASSES, in_channels=4,
                             base_filters=ck.get("base_filters", 32)).to(DEVICE).eval()
    model.load_state_dict(ck["model_state_dict"])

    _train, val = split_cases("data/brats_subset")
    ds = BrainSegmentationDataset(val[:4], augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    print(f"[full-metrics] evaluating on {len(ds)} validation slices "
          f"({len(val[:4])} held-out patients)")

    # accumulate dataset-level counts (not per-slice averages, which inflate)
    acc = {c: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "pred_vol": 0, "gt_vol": 0}
           for c in CLASSES}
    per_slice = {c: {"hd": [], "asd": []} for c in CLASSES}

    with torch.no_grad():
        for xb, yb in dl:
            pred = torch.argmax(model(xb.to(DEVICE)), 1).cpu().numpy()
            gt = yb.numpy()
            for b in range(pred.shape[0]):
                for c in CLASSES:
                    p, g = pred[b] == c, gt[b] == c
                    a = acc[c]
                    a["tp"] += int(np.logical_and(p, g).sum())
                    a["tn"] += int(np.logical_and(~p, ~g).sum())
                    a["fp"] += int(np.logical_and(p, ~g).sum())
                    a["fn"] += int(np.logical_and(~p, g).sum())
                    a["pred_vol"] += int(p.sum()); a["gt_vol"] += int(g.sum())
                    if p.any() and g.any():
                        m = segmentation_metrics(p, g)
                        if m["hausdorff_distance"] is not None:
                            per_slice[c]["hd"].append(m["hausdorff_distance"])
                            per_slice[c]["asd"].append(m["average_surface_distance"])

    results, rows = {}, []
    for c, name in CLASSES.items():
        a = acc[c]
        tp, tn, fp, fn = a["tp"], a["tn"], a["fp"], a["fn"]
        eps = 1e-9
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        jacc = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        sens = tp / (tp + fn + eps)                      # recall / sensitivity
        spec = tn / (tn + fp + eps)
        prec = tp / (tp + fp + eps)
        f1 = 2 * prec * sens / (prec + sens + eps)
        rve = abs(a["pred_vol"] - a["gt_vol"]) / (a["gt_vol"] + eps)
        hd = float(np.mean(per_slice[c]["hd"])) if per_slice[c]["hd"] else None
        asd = float(np.mean(per_slice[c]["asd"])) if per_slice[c]["asd"] else None
        results[name] = {
            "dice": round(dice, 4), "jaccard": round(jacc, 4),
            "accuracy": round(accuracy, 4), "sensitivity_recall": round(sens, 4),
            "specificity": round(spec, 4), "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "hausdorff_distance_mean": round(hd, 3) if hd else None,
            "average_surface_distance_mean": round(asd, 3) if asd else None,
            "relative_volume_error": round(rve, 4),
        }
        rows.append((name, results[name]))

    mean_dice = float(np.mean([r["dice"] for _n, r in rows]))
    results["_mean_tumor_dice"] = round(mean_dice, 4)
    results["_protocol"] = ("dataset-level accumulation of TP/TN/FP/FN over all held-out "
                            "validation slices (not per-slice averaging, which inflates scores "
                            "because most slices contain no tumour). HD/ASD averaged over "
                            "slices where both prediction and ground truth are non-empty.")

    hdr = (f"{'class':24} {'Dice':>6} {'Jacc':>6} {'Acc':>7} {'Sens':>6} {'Spec':>7} "
           f"{'Prec':>6} {'F1':>6} {'HD':>7} {'ASD':>6} {'RVE':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for name, r in rows:
        print(f"{name:24} {r['dice']:>6.3f} {r['jaccard']:>6.3f} {r['accuracy']:>7.4f} "
              f"{r['sensitivity_recall']:>6.3f} {r['specificity']:>7.4f} {r['precision']:>6.3f} "
              f"{r['f1_score']:>6.3f} {str(r['hausdorff_distance_mean']):>7} "
              f"{str(r['average_surface_distance_mean']):>6} {r['relative_volume_error']:>6.3f}")
    print(f"\nmean tumour Dice = {mean_dice:.4f}")

    with open("segmentation_full_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[full-metrics] wrote segmentation_full_metrics.json")


if __name__ == "__main__":
    main()

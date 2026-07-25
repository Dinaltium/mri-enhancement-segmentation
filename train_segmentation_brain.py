"""
train_segmentation_brain.py

Trains the multi-class segmentation U-Net on BraTS. Input = 4 stacked
modalities (T1/T1ce/T2/FLAIR), target = the remapped tumor mask with
classes 0..3 (background, necrotic/non-enhancing core, edema, enhancing).

Mirrors train_enhancement_brain.py's structure (same CLI conventions,
same checkpoint/JSON-dump pattern), but for the segmentation task:
    - Dataset:  BrainSegmentationDataset  (4-channel in, int mask target)
    - Model:    SegmentationUNet(num_classes=4, in_channels=4)
    - Loss:     combined CrossEntropy + soft Dice  (see note below)
    - Metrics:  per-class Dice/Jaccard + Hausdorff/ASD on the val set

Loss choice (documented design decision):
    The BraTS tumour classes occupy a tiny fraction of each slice - plain
    cross-entropy on this severe foreground/background imbalance drives the
    model toward predicting "background everywhere" and yields poor
    enhancing-tumour Dice. We therefore use CE + soft multi-class Dice by
    default (`--loss ce_dice`), the standard remedy for BraTS-style class
    imbalance. `--loss ce` reproduces the plain-CE baseline for comparison.

Dice/Jaccard reporting (documented design decision):
    We report *dataset-aggregated* Dice/Jaccard per class - intersection and
    region sizes are summed across every validation slice, then the ratio is
    taken once. Mean-per-slice Dice would be inflated: slices containing none
    of a given class score a perfect 1.0 by convention, and most axial slices
    contain no tumour, so a per-slice mean mostly measures empty slices.

Usage:
    python train_segmentation_brain.py --brats_root <root> --epochs 40
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from brain_dataset import BrainSegmentationDataset, split_cases, NUM_SEG_CLASSES
from models import SegmentationUNet
from metrics import segmentation_metrics

# class 0 is background; these are the 3 scored tumour sub-regions, in label order
CLASS_NAMES = ["background", "necrotic_non_enhancing", "edema", "enhancing"]


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss over softmax probabilities. Complements
    cross-entropy under heavy class imbalance (counts region overlap directly
    rather than per-pixel likelihood)."""

    def __init__(self, num_classes: int, smooth: float = 1.0,
                 include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                       # N,C,H,W
        target_1h = F.one_hot(target, self.num_classes)        # N,H,W,C
        target_1h = target_1h.permute(0, 3, 1, 2).float()      # N,C,H,W

        start = 0 if self.include_background else 1
        dice = 0.0
        n = 0
        for c in range(start, self.num_classes):
            p = probs[:, c]
            g = target_1h[:, c]
            inter = (p * g).sum()
            denom = p.sum() + g.sum()
            dice = dice + (2 * inter + self.smooth) / (denom + self.smooth)
            n += 1
        return 1.0 - dice / max(n, 1)


def _amp_ctx(device: torch.device, enabled: bool):
    """autocast context that works on both new (torch.amp) and CPU paths."""
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return torch.amp.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def evaluate_segmentation(model, loader, device) -> dict:
    """Dataset-aggregated per-class Dice/Jaccard, plus mean Hausdorff/ASD over
    the slices where a class is actually present. Returns a nested dict."""
    model.eval()
    n_cls = NUM_SEG_CLASSES

    # aggregated overlap accumulators, per class
    inter = np.zeros(n_cls, dtype=np.float64)   # |pred & gt|
    pred_sum = np.zeros(n_cls, dtype=np.float64)
    gt_sum = np.zeros(n_cls, dtype=np.float64)
    union = np.zeros(n_cls, dtype=np.float64)

    hd_vals = {c: [] for c in range(1, n_cls)}
    asd_vals = {c: [] for c in range(1, n_cls)}
    present_slices = {c: 0 for c in range(1, n_cls)}

    for stacked, mask in loader:
        stacked = stacked.to(device)
        logits = model(stacked)
        preds = torch.argmax(logits, dim=1).cpu().numpy()   # N,H,W
        gts = mask.numpy()                                  # N,H,W

        for b in range(preds.shape[0]):
            pred, gt = preds[b], gts[b]
            for c in range(1, n_cls):
                p = (pred == c)
                g = (gt == c)
                i = np.logical_and(p, g).sum()
                inter[c] += i
                pred_sum[c] += p.sum()
                gt_sum[c] += g.sum()
                union[c] += np.logical_or(p, g).sum()
                # HD/ASD only defined when both masks non-empty for this slice
                if p.any() and g.any():
                    m = segmentation_metrics(p, g)
                    if m["hausdorff_distance"] is not None:
                        hd_vals[c].append(m["hausdorff_distance"])
                        asd_vals[c].append(m["average_surface_distance"])
                if g.any():
                    present_slices[c] += 1

    results = {}
    for c in range(1, n_cls):
        name = CLASS_NAMES[c]
        agg_dice = (2 * inter[c]) / (pred_sum[c] + gt_sum[c]) if (pred_sum[c] + gt_sum[c]) > 0 else 1.0
        agg_jac = inter[c] / union[c] if union[c] > 0 else 1.0
        results[name] = {
            "dice": float(agg_dice),
            "jaccard": float(agg_jac),
            "hausdorff_distance_mean": float(np.mean(hd_vals[c])) if hd_vals[c] else None,
            "average_surface_distance_mean": float(np.mean(asd_vals[c])) if asd_vals[c] else None,
            "val_slices_with_class": int(present_slices[c]),
        }
    # a single mean-over-tumour-classes Dice, handy for the headline number
    results["mean_tumor_dice"] = float(
        np.mean([results[CLASS_NAMES[c]]["dice"] for c in range(1, n_cls)])
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats_root", default="data/synthetic_brats")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default="ce_dice",
                        help="ce_dice (default) counters BraTS class imbalance; "
                             "ce reproduces the plain cross-entropy baseline")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="cap total cases (RAM safety - the dataset caches all slices)")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="run the (heavier) per-class Dice/HD eval every N epochs")
    parser.add_argument("--no_amp", action="store_true",
                        help="disable mixed precision (AMP is on by default on CUDA)")
    parser.add_argument("--out", default="segmentation_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"[train_segmentation] device: {device}  amp: {use_amp}  loss: {args.loss}")

    train_cases, val_cases = split_cases(args.brats_root)
    if args.max_cases:
        n_val = max(1, int(args.max_cases * 0.2))
        val_cases = val_cases[:n_val]
        train_cases = train_cases[:args.max_cases - n_val]
    print(f"[train_segmentation] train cases={len(train_cases)} val cases={len(val_cases)}")

    train_ds = BrainSegmentationDataset(train_cases, augment=True)
    val_ds = BrainSegmentationDataset(val_cases, augment=False)
    print(f"[train_segmentation] train slices={len(train_ds)} val slices={len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "no usable segmentation slices - every case needs all 4 modalities "
            "AND a _seg mask. Check the dataset (offline hackathon Brain data has "
            "no masks; this trainer is for BraTS2020 only)."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SegmentationUNet(
        num_classes=NUM_SEG_CLASSES, in_channels=4, base_filters=args.base_filters
    ).to(device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = SoftDiceLoss(num_classes=NUM_SEG_CLASSES)

    def compute_loss(logits, target):
        if args.loss == "ce":
            return criterion_ce(logits, target)
        if args.loss == "dice":
            return criterion_dice(logits, target)
        return criterion_ce(logits, target) + criterion_dice(logits, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "val_metrics": {}}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for stacked, mask in train_loader:
            stacked, mask = stacked.to(device), mask.to(device)
            optimizer.zero_grad()
            with _amp_ctx(device, use_amp):
                logits = model(stacked)
                loss = compute_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for stacked, mask in val_loader:
                stacked, mask = stacked.to(device), mask.to(device)
                with _amp_ctx(device, use_amp):
                    logits = model(stacked)
                    loss = compute_loss(logits, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        msg = (f"[train_segmentation] epoch {epoch:02d}/{args.epochs} "
               f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            vm = evaluate_segmentation(model, val_loader, device)
            history["val_metrics"][epoch] = vm
            msg += f" | mean_tumor_dice={vm['mean_tumor_dice']:.4f}"

        print(msg)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_SEG_CLASSES,
                "in_channels": 4,
                "base_filters": args.base_filters,
                "class_names": CLASS_NAMES,
            }, args.out)

    # convergence epoch = best val-loss epoch; overfitting gap = val-train at that epoch
    overfit_gap = history["val_loss"][best_epoch - 1] - history["train_loss"][best_epoch - 1]
    print(f"[train_segmentation] best val_loss={best_val_loss:.4f} at epoch {best_epoch} "
          f"(convergence epoch), overfitting_gap={overfit_gap:.4f}, saved -> {args.out}")

    print("[train_segmentation] final per-class validation metrics:")
    final_metrics = evaluate_segmentation(model, val_loader, device)
    for c in range(1, NUM_SEG_CLASSES):
        name = CLASS_NAMES[c]
        m = final_metrics[name]
        print(f"    {name:24s} dice={m['dice']:.4f} jaccard={m['jaccard']:.4f} "
              f"HD={m['hausdorff_distance_mean']} ASD={m['average_surface_distance_mean']} "
              f"(val slices w/ class={m['val_slices_with_class']})")
    print(f"    mean_tumor_dice = {final_metrics['mean_tumor_dice']:.4f}")

    summary = {
        "best_epoch_convergence": best_epoch,
        "best_val_loss": best_val_loss,
        "overfitting_gap": overfit_gap,
        "loss_type": args.loss,
        "final_val_metrics": final_metrics,
        "history": history,
    }
    with open("segmentation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[train_segmentation] saved -> segmentation_metrics.json "
          "(loss curves + per-class Dice/Jaccard/HD/ASD for the report)")


if __name__ == "__main__":
    main()

"""
train_enhancement_brain.py

Trains the enhancement U-Net on BraTS FLAIR slices (degraded -> clean),
reporting the full IQA metric suite the problem statement asks for.

Usage:
    python train_enhancement_brain.py --brats_root data/synthetic_brats --epochs 15
"""

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_dataset import BrainEnhancementDataset, split_cases
from models import EnhancementUNet
from ssim import SSIMLoss
from metrics import full_reference_metrics


def evaluate_full_metrics(model, loader, device, max_batches: int = 8) -> dict:
    """Runs pyiqa's full metric suite on a handful of validation batches
    (pyiqa metrics are heavier than a simple loss, so we sample rather than
    running all of them on the entire val set every epoch)."""
    model.eval()
    accum = {}
    n = 0
    with torch.no_grad():
        for i, (degraded, clean) in enumerate(loader):
            if i >= max_batches:
                break
            degraded, clean = degraded.to(device), clean.to(device)
            output = model(degraded)
            for b in range(output.shape[0]):
                out_np = output[b, 0].cpu().numpy()
                clean_np = clean[b, 0].cpu().numpy()
                try:
                    m = full_reference_metrics(clean_np, out_np)
                except Exception as e:
                    print(f"[warn] metric computation failed (likely no internet for "
                          f"LPIPS/BRISQUE pretrained weights on first run): {e}")
                    continue
                for k, v in m.items():
                    accum[k] = accum.get(k, 0.0) + v
                n += 1
    return {k: v / max(n, 1) for k, v in accum.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brats_root", default="data/synthetic_brats")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--max_cases", type=int, default=None,
                        help="cap total cases (RAM safety - dataset caches all slices)")
    parser.add_argument("--out", default="enhancement_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_enhancement] device: {device}")

    train_cases, val_cases = split_cases(args.brats_root)
    if args.max_cases:
        n_val = max(1, int(args.max_cases * 0.2))
        val_cases = val_cases[:n_val]
        train_cases = train_cases[:args.max_cases - n_val]
    print(f"[train_enhancement] train cases={len(train_cases)} val cases={len(val_cases)}")

    train_ds = BrainEnhancementDataset(train_cases, augment=True)
    val_ds = BrainEnhancementDataset(val_cases, augment=False)
    print(f"[train_enhancement] train slices={len(train_ds)} val slices={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = EnhancementUNet(base_filters=args.base_filters).to(device)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for degraded, clean in train_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(degraded)
            loss = criterion_l1(output, clean) + criterion_ssim(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded, clean = degraded.to(device), clean.to(device)
                output = model(degraded)
                loss = criterion_l1(output, clean) + criterion_ssim(output, clean)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"[train_enhancement] epoch {epoch:02d}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "base_filters": args.base_filters,
            }, args.out)

    print(f"[train_enhancement] best val_loss={best_val_loss:.4f}, saved -> {args.out}")

    print("[train_enhancement] computing full IQA metric suite on validation set...")
    full_metrics = evaluate_full_metrics(model, val_loader, device)
    print("[train_enhancement] validation metrics (full-reference, vs clean BraTS scans):")
    for k, v in full_metrics.items():
        print(f"    {k}: {v:.4f}")

    with open("enhancement_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    print("[train_enhancement] saved -> enhancement_metrics.json (put these numbers in your report)")


if __name__ == "__main__":
    main()

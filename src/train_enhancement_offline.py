"""
train_enhancement_offline.py

Trains the enhancement U-Net on an OFFLINE hackathon dataset (Spine or
offline Brain), following the coordinator's rule: split the 10 samples per
group into 5 training / 5 testing. Training pairs are built by synthetic
MRI-realistic degradation (Rician + bias field + blur) of the clean scans -
no ground-truth annotations required, which is why a trained enhancement
model is legitimate here (unlike segmentation, which stays unsupervised).

The evaluation runs a THREE-WAY comparison on the held-out test cases, all
against the clean reference, giving the "systematic comparison study" the
problem statement asks for, for free:
    1. degraded input      (no enhancement  - lower bound)
    2. CLAHE               (classical baseline named in the problem statement)
    3. our trained U-Net   (the deep model)

Usage:
    python train_enhancement_offline.py --group spine_normal --epochs 30
    python train_enhancement_offline.py --group brain_pathological --epochs 30
    python train_enhancement_offline.py --root "Spine DATASETS/..." --epochs 30
"""

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from enhancement_dataset import OfflineEnhancementDataset, split_offline_cases
from models import EnhancementUNet
from ssim import SSIMLoss
from offline_dataset import OFFLINE_ROOTS


def clahe_enhance(img01: np.ndarray) -> np.ndarray:
    """Classical CLAHE on a [0,1] image (the problem statement's own Stage 2
    suggestion). Returns [0,1]."""
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return (clahe.apply(u8).astype(np.float32)) / 255.0


def _accumulate(accum: dict, m: dict) -> None:
    for k, v in m.items():
        accum[k] = accum.get(k, 0.0) + v


def evaluate_three_way(model, loader, device, max_batches: int = 12) -> dict:
    """Full-reference IQA for degraded-input / CLAHE / model-output, each vs
    the clean reference. Sampled over a handful of batches (pyiqa is heavy)."""
    from metrics import full_reference_metrics
    model.eval()
    acc = {"input": {}, "clahe": {}, "model": {}}
    n = 0
    with torch.no_grad():
        for i, (degraded, clean) in enumerate(loader):
            if i >= max_batches:
                break
            degraded_d = degraded.to(device)
            output = model(degraded_d).cpu()
            for b in range(output.shape[0]):
                clean_np = clean[b, 0].numpy()
                deg_np = degraded[b, 0].numpy()
                out_np = np.clip(output[b, 0].numpy(), 0, 1)
                clahe_np = clahe_enhance(deg_np)
                try:
                    _accumulate(acc["input"], full_reference_metrics(clean_np, deg_np))
                    _accumulate(acc["clahe"], full_reference_metrics(clean_np, clahe_np))
                    _accumulate(acc["model"], full_reference_metrics(clean_np, out_np))
                except Exception as e:
                    print(f"[warn] IQA failed (pyiqa weights need internet on first run): {e}")
                    continue
                n += 1
    return {method: {k: v / max(n, 1) for k, v in d.items()} for method, d in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=list(OFFLINE_ROOTS.keys()), default=None,
                        help="named offline dataset group")
    parser.add_argument("--root", default=None, help="explicit dataset root (overrides --group)")
    parser.add_argument("--modalities", default=None,
                        help="comma list to restrict, e.g. T1,T2,STIR (default: all in-scope)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--max_slices_per_volume", type=int, default=40,
                        help="cap slices per volume so huge multi-slice sequences don't dominate")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = args.root or (OFFLINE_ROOTS[args.group] if args.group else None)
    if root is None:
        parser.error("give --group or --root")
    tag = args.group or "offline"
    mod_filter = set(args.modalities.split(",")) if args.modalities else None
    if mod_filter:
        # modality-specific run (e.g. spine_normal + T2) gets its own tag/files,
        # so per-modality models never overwrite the pooled one
        tag = f"{tag}_{'-'.join(sorted(mod_filter))}"
    out_path = args.out or f"models/enhancement_model_{tag}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"[enh_offline:{tag}] device={device} amp={use_amp} root={root}")

    train_cases, test_cases = split_offline_cases(root)
    print(f"[enh_offline:{tag}] train cases={len(train_cases)} test cases={len(test_cases)}")
    if not train_cases or not test_cases:
        raise ValueError(f"need >=1 train and >=1 test case under {root}")

    train_ds = OfflineEnhancementDataset(train_cases, modalities=mod_filter, augment=True,
                                         max_slices_per_volume=args.max_slices_per_volume)
    test_ds = OfflineEnhancementDataset(test_cases, modalities=mod_filter, augment=False,
                                        max_slices_per_volume=args.max_slices_per_volume)
    print(f"[enh_offline:{tag}] train slices={len(train_ds)} test slices={len(test_ds)}")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("no usable slices - check modality filter / data")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = EnhancementUNet(base_filters=args.base_filters).to(device)
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def amp_ctx():
        if use_amp:
            return torch.amp.autocast(device_type="cuda")
        return torch.amp.autocast(device_type="cpu", enabled=False)

    history = {"train_loss": [], "test_loss": []}
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for degraded, clean in train_loader:
            degraded, clean = degraded.to(device), clean.to(device)
            optimizer.zero_grad()
            with amp_ctx():
                out = model(degraded)
                loss = criterion_l1(out, clean) + criterion_ssim(out, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr += loss.item()
        tr /= len(train_loader)

        model.eval()
        te = 0.0
        with torch.no_grad():
            for degraded, clean in test_loader:
                degraded, clean = degraded.to(device), clean.to(device)
                with amp_ctx():
                    out = model(degraded)
                    loss = criterion_l1(out, clean) + criterion_ssim(out, clean)
                te += loss.item()
        te /= len(test_loader)

        history["train_loss"].append(tr)
        history["test_loss"].append(te)
        print(f"[enh_offline:{tag}] epoch {epoch:02d}/{args.epochs} train_loss={tr:.4f} test_loss={te:.4f}")

        if te <= best:
            best = te
            torch.save({"model_state_dict": model.state_dict(),
                        "base_filters": args.base_filters, "group": tag}, out_path)

    best_epoch = int(np.argmin(history["test_loss"])) + 1
    overfit_gap = history["test_loss"][best_epoch - 1] - history["train_loss"][best_epoch - 1]
    print(f"[enh_offline:{tag}] best test_loss={best:.4f} at epoch {best_epoch}, "
          f"overfitting_gap={overfit_gap:.4f}, saved -> {out_path}")

    print(f"[enh_offline:{tag}] three-way IQA comparison on held-out test set...")
    cmp = evaluate_three_way(model, test_loader, device)
    for method in ["input", "clahe", "model"]:
        d = cmp[method]
        if d:
            print(f"    {method:6s} PSNR={d.get('psnr',0):.2f} SSIM={d.get('ssim',0):.3f} "
                  f"LPIPS={d.get('lpips',0):.3f} MSE={d.get('mse',0):.4f}")

    summary = {
        "group": tag, "best_epoch": best_epoch, "best_test_loss": best,
        "overfitting_gap": overfit_gap, "history": history,
        "three_way_iqa": cmp,
        "note": "input=degraded (no enhancement), clahe=classical baseline, model=trained U-Net",
    }
    out_json = f"results/enhancement_metrics_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[enh_offline:{tag}] saved -> {out_json}")


if __name__ == "__main__":
    main()

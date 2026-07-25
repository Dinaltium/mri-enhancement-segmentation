"""
cross_validation.py   --   Stage 3/4 gap-fill: CROSS-VALIDATION ACCURACY

The problem statement asks for "Cross validation accuracy" of the DL model.
We run K-fold cross-validation on the brain segmentation model (accuracy =
mean tumour Dice) and report per-fold + mean +/- std across folds — a proper
cross-validated accuracy, not a single split.

Kept lightweight (few cases/epochs) so it finishes quickly and barely competes
with the live demo. Output: cross_validation.json.
"""

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_dataset import BrainSegmentationDataset, NUM_SEG_CLASSES
from nifti_utils import find_brats_cases
from models import SegmentationUNet
from train_segmentation_brain import evaluate_segmentation, SoftDiceLoss, _amp_ctx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 3
N_CASES = 21
EPOCHS = 25


def main():
    cases = find_brats_cases("data/brats_subset")[:N_CASES]
    folds = [list(f) for f in np.array_split(range(len(cases)), K)]
    use_amp = DEVICE.type == "cuda"
    fold_dice = []
    per_class = []

    for k in range(K):
        val_idx = folds[k]
        train_idx = [i for i in range(len(cases)) if i not in val_idx]
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]

        train_ds = BrainSegmentationDataset(train_cases, augment=True)
        val_ds = BrainSegmentationDataset(val_cases, augment=False)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        model = SegmentationUNet(num_classes=NUM_SEG_CLASSES, in_channels=4, base_filters=32).to(DEVICE)
        ce, dice = nn.CrossEntropyLoss(), SoftDiceLoss(NUM_SEG_CLASSES)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for ep in range(EPOCHS):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with _amp_ctx(DEVICE, use_amp):
                    logits = model(xb)
                    loss = ce(logits, yb) + dice(logits, yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        vm = evaluate_segmentation(model, val_loader, DEVICE)
        fold_dice.append(vm["mean_tumor_dice"])
        per_class.append({c: vm[c]["dice"] for c in ["necrotic_non_enhancing", "edema", "enhancing"]})
        print(f"[cv] fold {k+1}/{K}: mean_tumor_dice={vm['mean_tumor_dice']:.4f} "
              f"(train={len(train_cases)} val={len(val_cases)} cases)", flush=True)
        del train_ds, val_ds, model
        if use_amp:
            torch.cuda.empty_cache()

    result = {
        "method": f"{K}-fold cross-validation, brain segmentation ({N_CASES} cases, {EPOCHS} epochs/fold)",
        "per_fold_mean_tumor_dice": [round(d, 4) for d in fold_dice],
        "cv_accuracy_mean_dice": round(float(np.mean(fold_dice)), 4),
        "cv_accuracy_std_dice": round(float(np.std(fold_dice)), 4),
        "per_fold_per_class": per_class,
    }
    with open("results/cross_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[cv] CROSS-VALIDATION ACCURACY: {result['cv_accuracy_mean_dice']:.3f} "
          f"+/- {result['cv_accuracy_std_dice']:.3f} mean tumour Dice across {K} folds")
    print("[cv] wrote cross_validation.json")


if __name__ == "__main__":
    main()

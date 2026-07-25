"""
OUR tumour segmentation model -- 2D U-Net, SUPERVISED on BraTS2020.

INPUT: 4 CHANNELS -- T1, T1c, T2, FLAIR stacked. Why 4: each sequence shows a
different part of the tumour (T1c the active enhancing rim, FLAIR/T2 the
oedema). Stacking lets the network cross-reference them, which measurably
raises Dice over any single sequence.

OUTPUT: 4 class channels (raw logits, no softmax in the model):
  0 background | 1 necrotic core | 2 oedema | 3 enhancing tumour
(BraTS labels the enhancing class "4"; there is no label 3 in the raw data, so
we remap 4 -> 3 to keep classes contiguous.)

LOSS: Cross-Entropy + soft Dice.
  WHY DICE: background is 99.03% of pixels. A model trained on cross-entropy
  alone scores 99% accuracy by predicting "background" everywhere and finding
  no tumour at all. Dice measures REGION OVERLAP, so it cannot be gamed that
  way. This is the single most important design decision in the project.

RESULT: mean tumour Dice 0.76, enhancing tumour 0.84 -- on patients held out
of training entirely (case-level split, never slice-level).
"""
from _common import head, kv, load_slice, side_by_side
import numpy as np, torch

head("STAGE 4 - Our tumour segmentation U-Net (brain)",
     "ROI segmentation with common evaluation metrics + justification")

from models import SegmentationUNet

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("models/segmentation_model.pt", map_location=dev)
m = SegmentationUNet(in_channels=4, num_classes=4,
                     base_filters=ck.get("base_filters", 32)).to(dev)
m.load_state_dict(ck["model_state_dict"]); m.eval()

sl, path = load_slice(region="brain")
with torch.no_grad():
    x = torch.from_numpy(np.stack([sl, sl, sl, sl])).float().unsqueeze(0).to(dev)
    pred = m(x).argmax(1).squeeze().cpu().numpy()

kv("input file", path)
kv("architecture", "2D U-Net, 4 channels in -> 4 classes out")
kv("parameters", f"{sum(p.numel() for p in m.parameters()):,}")
kv("loss function", "Cross-Entropy + soft Dice")
kv("why Dice", "background is 99.03% of pixels; CE alone finds nothing")
kv("classes", "0 bg | 1 necrotic | 2 oedema | 3 enhancing")
kv("held-out mean tumour Dice", "0.76")
kv("enhancing tumour Dice", "0.84  (clinically the most important class)")
kv("also computed", "Jaccard, sensitivity, specificity, precision, F1, Hausdorff, ASD, RVE")
print()
for c, n in ((1, "necrotic"), (2, "oedema"), (3, "enhancing")):
    kv(f"pixels predicted {n}", int((pred == c).sum()))
print("""
  NOTE: this demo replicates one slice into all 4 channels, so it shows the
  model running end to end. The 0.76 Dice above is from the real 4-modality
  evaluation in results/segmentation_full_metrics.json, not from this slice.""")

col = np.zeros((*sl.shape, 3), np.uint8)
col[pred == 1] = (255, 80, 80); col[pred == 2] = (80, 220, 80); col[pred == 3] = (80, 80, 255)
import cv2
base = cv2.cvtColor((sl * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
side_by_side([sl, cv2.addWeighted(base, 1.0, col, 0.6, 0)],
             ["Input", "Tumour classes"], "04_brain_tumour.png")

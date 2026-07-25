"""
SPINEPS -- the PRETRAINED model. Per-vertebra instances.

WHAT IT IS: Moller et al., European Radiology 2025, Apache-2.0, built on
nnU-Net. Two phases:
  SEMANTIC  - labels 13-14 structure TYPES (body, disc, canal, cord, processes)
  INSTANCE  - numbers INDIVIDUAL vertebrae

TRAINED ON: the public SPIDER dataset + the German National Cohort, about
1,600+ annotated subjects. Published Dice 0.92 vertebrae / 0.967 discs /
0.958 canal. THOSE ARE THEIR NUMBERS ON THEIR TEST SET -- we claim none of them.

WHY WE USE IT: naming a herniated disc, or numbering a vertebra, is SUPERVISED
by nature -- a model can only output "L4" if it has seen examples labelled L4.
With 20 unlabelled cases and no external data, no model we train can produce
it. We proved that with four annotation-free methods first, one of which
failed outright and was withdrawn.

WHAT THE NUMBERS MEAN -- two different systems, do not confuse them:
  INSTANCE IDs (1..17) = "this is a separate bone from the one above". NOT a
                         diagnosis, NOT a severity score.
  LABEL VALUES         = structure types: 49 vertebral body, 60 cord,
                         61 canal, 100 disc, 41-48 arch and processes.

MEMORY NOTE: the instance phase forwards its whole working volume at once and
needs ~12.4 GiB. This card has 6 GB, so instance runs on CPU (401 s) while the
semantic phase runs on GPU in under a minute.
"""
from _common import head, kv, save, SPINE, SPINE_PATH
import os, numpy as np, cv2, nibabel as nib

head("STAGE 4 - SPINEPS pretrained per-vertebra segmentation",
     "per-vertebra ROI, with the pretrained model justified")

import sys
sys.path.insert(0, "src")
from spineps_runner import run_semantic_live, mask_in_scan_space, SPINEPS_LABELS_HELP

kv("model", "SPINEPS (Moller et al., European Radiology 2025)")
kv("licence", "Apache-2.0")
kv("trained on", "SPIDER + German National Cohort, ~1600+ subjects")
kv("published Dice", "0.92 vertebrae / 0.967 discs / 0.958 canal (THEIRS)")
kv("we train it?", "NO - we supply no annotations and do not train it")
kv("our own best vs it", "Dice 0.38 canal, and 0 numbered vertebrae")

path = SPINE if os.path.exists(SPINE) else SPINE_PATH
print(f"\n  running semantic phase on {path} (GPU, cached after first run)...")
r = run_semantic_live(path, key=os.path.basename(path))
if not r.get("ok"):
    kv("result", f"FAILED: {r.get('error')}")
    raise SystemExit(0)

m = mask_in_scan_space(r["semantic"], path)
labs = sorted(int(u) for u in np.unique(m) if u > 0)
kv("seconds", f"{r.get('seconds')}  (cached={r.get('cached')})")
kv("structures found", len(labs))
for u in labs:
    kv(f"  label {u}", SPINEPS_LABELS_HELP.get(u, "?"))

vol = nib.load(path).get_fdata().astype(np.float32)
k = int(np.argmax([(m[..., i] > 0).sum() for i in range(m.shape[2])]))
img = vol[..., k]; img = (img - img.min()) / (np.ptp(img) + 1e-8)
base = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
pal = np.random.RandomState(0).randint(60, 240, (150, 3)).astype(np.uint8)
col = np.zeros_like(base)
for u in labs:
    col[m[..., k] == u] = pal[u % 150]
save("14_spine_spineps.png", cv2.addWeighted(base, 1.0, col, 0.55, 0), gray=False)

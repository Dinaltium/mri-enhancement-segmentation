# MASTER KNOWLEDGE BASE — everything we built, in one place

Answer any judge question from here. Organized: Models → Training → Losses →
Metrics → Preprocessing → Algorithms → Datasets → Numbers → Decisions → Q&A.

---

## 1. THE MODELS (architecture)

### The shared backbone — 2D U-Net (`models.py`)
- Shape: **encoder → bottleneck → decoder, with skip connections** (the "U").
- **conv_block** = Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU.
- **Encoder** (downsampling, MaxPool 2×2 between): channels 32 → 64 → 128 → 256.
- **Bottleneck**: 512 channels.
- **Decoder** (upsampling via ConvTranspose2d 2×2): 256 → 128 → 64 → 32,
  each concatenated with the matching encoder features (**skip connections**).
- `base_filters = 32` (the starting channel count).

### Model A — EnhancementUNet
- **1 channel in → 1 channel out**, final Conv1×1 + **sigmoid** (output 0–1).
- Job: noisy MRI slice in → clean slice out (denoising / artifact removal).
- ~**7.77 million parameters**, ~31 MB.

### Model B — SegmentationUNet
- **4 channels in → 4 channels out** (raw logits, no softmax in the model).
- 4 in = the 4 MRI modalities stacked (T1, T1c, T2, FLAIR).
- 4 out = the 4 classes (background, necrotic core, edema, enhancing tumour).
- ~**7.77 million parameters**, ~31 MB.

---

## 2. EVERY TRAINING RUN

### Run 1 — Brain Segmentation (BraTS2020)
- Data: 20 BraTS cases → **16 train / 4 val** (case-level split, seed 42).
- ~2075 train / 504 val 2D slices.
- **30 epochs**, batch size 8, **Adam**, lr 1e-3, **AMP** (mixed precision).
- Loss: **Cross-Entropy + soft Dice**.
- Result: **mean tumour Dice 0.73** (enhancing 0.796, edema 0.708, necrotic
  0.685), Jaccard 0.66/0.55/0.52, Hausdorff 8.0/23.6/9.4, ASD 0.81/2.0/0.72.
  Convergence epoch 25, overfitting gap −0.08 (no overfitting).
- File: `train_segmentation_brain.py` → `segmentation_model.pt`.

### Run 2 — Brain Enhancement (BraTS FLAIR)
- Data: 40 BraTS cases, FLAIR only. **18 epochs**, batch 8, Adam lr 1e-3, AMP.
- Loss: **L1 + SSIM**.
- Degradation to make training pairs: **Rician noise + bias field + blur**,
  noise σ range **0.02–0.20** (wide, so it cleans genuinely noisy scans).
- Result: **PSNR 30.3 dB, SSIM 0.965**, FSIM 0.985, UQI 0.988, LPIPS 0.041.
- On heavy noise: restores SSIM 0.19→**0.89**. File: `enhancement_model_brain.pt`.

### Runs 3-6 — Offline Enhancement (Spine + Brain, 4 groups)
- Groups: spine_normal, spine_pathological, brain_normal, brain_pathological.
- Each: **5 train / 5 test** cases (the coordinators' rule), 25 epochs, L1+SSIM.
- 3-way comparison per group (degraded input vs CLAHE vs our U-Net), all vs clean:
  | Group | input SSIM | CLAHE SSIM | **our U-Net SSIM** |
  |---|---|---|---|
  | spine_normal | 0.40 | 0.28 | **0.82** |
  | spine_pathological | 0.52 | 0.39 | **0.75** |
  | brain_normal | 0.35 | 0.26 | **0.87** |
  | brain_pathological | 0.19 | 0.15 | **0.96** |
- Finding: **CLAHE lowers fidelity** (redistributes contrast, doesn't restore);
  our U-Net wins every group. Files: `enhancement_model_<group>.pt`.

### Run 7 — Cross-validation (brain segmentation)
- **3-fold cross-validation**, 21 cases, 25 epochs/fold.
- **CV accuracy = 0.59 ± 0.04 mean tumour Dice** (folds 0.62 / 0.62 / 0.54).
- Why lower than the 0.73 headline: each fold trains on only ~14 cases (vs 16
  for the main model) and averages over 3 different splits. **The key number is
  the tiny ±0.04 std — it proves the model is CONSISTENT, not lucky.** That is
  exactly what cross-validation is meant to show. File: `cross_validation.py`.

---

### Run 8 — Spine Anomaly Autoencoder (self-supervised lesion localisation)
- **Trained on NORMAL spine T2 only — 719 slices**, 80 epochs, Adam **5e-4**,
  loss **L1 + 0.5·SSIM**, no AMP (stability), final loss 0.124.
- Architecture: **bottleneck convolutional autoencoder, NO skip connections**
  (encoder 1→32→64→128→128 with 4 MaxPools → 14×14 bottleneck → mirrored
  decoder with ConvTranspose). Skips are omitted *on purpose*: they would copy
  the input (lesion included) to the output and destroy the anomaly signal.
- **How detection works:** it can only rebuild "healthy spine". On a diseased
  spine the lesion cannot be reconstructed, so |input − reconstruction| peaks
  exactly at the abnormality. Post-processing: blur → background mask →
  subtract median foreground error (keep only *excess*) → ^1.5 → normalise →
  box the largest connected component above the 90th percentile.
- Result: flags a focal "suspected" region on all 9 pathological cases
  (anomaly scores 0.013–0.022). File: `spine_autoencoder.py`.

### Run 9 — Modality-specific spine enhancement
- One enhancement U-Net per spine modality (**T1, T2, STIR**) on spine_normal,
  25 epochs each, vs the pooled all-modality model.
- Fair comparison in `compare_modality_models.py`: both evaluated on the SAME
  held-out per-modality test slices (each run's own metrics use different test
  sets and must not be compared directly).

---

## 3. EVERY LOSS FUNCTION (what + why)

- **L1 loss (Mean Absolute Error)** — average pixel-wise difference. Used for
  enhancement; keeps output numerically close to the clean scan.
- **SSIM loss** = 1 − SSIM (differentiable, `ssim.py`). Preserves *structure*
  (edges, texture), so the enhanced image looks right, not just numerically close.
- **Cross-Entropy loss** — per-pixel classification loss for segmentation.
- **Soft Dice loss** — 1 − Dice overlap. Directly maximizes region overlap.
  **Why we add it:** the tumour is tiny vs background; plain cross-entropy would
  predict "background everywhere". Dice fixes this **class imbalance**.
- Enhancement total = **L1 + SSIM**. Segmentation total = **Cross-Entropy + Dice**.

**Optimizer:** Adam, learning rate 1e-3. **AMP** (Automatic Mixed Precision) for
speed + less memory. **GradScaler** to keep fp16 training stable.

---

## 4. EVERY METRIC (definition in plain words)

### Enhancement — full-reference (need clean copy to compare)
- **PSNR** (Peak Signal-to-Noise Ratio) — cleanliness in dB, higher better.
- **SSIM** (Structural Similarity) — structural closeness 0–1, higher better.
- **MSE / RMSE** — average pixel error, lower better.
- **UQI** — universal quality index. **FSIM** — feature/edge similarity.
- **GMSD** — gradient-magnitude similarity deviation, lower better.
- **VIF** — visual information fidelity. **LPIPS** — AI "looks-the-same" score, lower better.

### Enhancement — no-reference (single image, no clean copy)
- **BRISQUE / NIQE / PIQE** — blind quality scores, lower better.
- **Entropy** — information/detail content.

### Segmentation
- **Dice (DSC)** — overlap with ground truth, 0–1 (main score).
- **Jaccard (IoU)** — stricter overlap. **Hausdorff Distance** — worst-case
  boundary error (lower better). **ASD** — average surface distance.
- **Sensitivity/Recall, Specificity, Precision, F1, Relative Volume Error.**

---

## 5. EVERY PREPROCESSING STEP (Stage 2 — classical, NO model)

1. **Load** NIfTI (.nii/.nii.gz) with nibabel.
2. **Normalize** to [0,1] using robust percentiles (0.5–99.5), per volume.
3. **Slice** 3D → 2D axial; skip near-empty slices (<2% non-zero).
4. **Resize** to 224×224 (cubic) + **clip to [0,1]** (cubic overshoots at edges).
5. Masks resized **nearest-neighbour** (cubic would invent label values).
6. **Remap BraTS labels** 4 → 3 (label 3 unused → make classes contiguous 0-3).
7. **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) + **HE**.
8. **Augmentation**: horizontal flips (train only).
9. **Synthetic degradation** (for enhancement pairs): Rician + bias field + blur.

---

## 6. EVERY ALGORITHM / TECHNIQUE

- **CLAHE** — local contrast enhancement (clipLimit 2.0, 8×8 tiles).
- **HE** — histogram equalization (global contrast).
- **Rician noise** — the correct MRI magnitude-image noise model (not Gaussian).
- **Bias field** — smooth multiplicative RF-coil intensity artifact.
- **k-means** — unsupervised intensity clustering (spine ROI, tissue).
- **GMM (Gaussian Mixture)** — softer tissue clustering (CSF/GM/WM).
- **SLIC superpixels** — groups nearby similar pixels into ~250 coherent
  superpixels; we cluster those by mean intensity → connected disc/vertebra/
  cord regions (replaced the speckly per-pixel k-means for spine ROI).
- **Autoencoder anomaly detection** — train on healthy only; reconstruction
  error localises the abnormality. Self-supervised, needs no labels.
- **Otsu multi-threshold** — alternative unsupervised split.
- **Grad-CAM (Seg-Grad-CAM)** — attention heatmap, gradient of tumour score
  w.r.t. decoder features, restricted to the predicted tumour region.
- **Immerkær estimator** — single-image noise-σ. **Michelson contrast**,
  **Laplacian variance** (sharpness), **Sobel** (edge), **Shannon entropy** (complexity).

---

## 7. THE DATASETS

- **BraTS2020** (Kaggle awsaf49): **4.47 GB**, **369 cases**; we extracted
  **126 complete cases**. Each case = T1, T1c(T1ce), T2, FLAIR + seg mask,
  240×240×155, **1 mm isotropic**. Labels 0=bg, 1=necrotic, 2=edema, 4=enhancing.
- **Offline Brain**: 10 Normal (S1–S10, Philips scanner) + 10 Pathological
  (BRP1–BRP10, BraTS-geometry). No ground truth.
- **Offline Spine**: 10 Normal (SP1–SP10) + 10 Pathological (SP11+). No GT.
- **Resolution finding**: BraTS uniform 1 mm; offline highly heterogeneous
  (voxel 0.25–1.3 mm, slice thickness 3–13 mm) → why we resample to 224×224.

---

## 8. HEADLINE NUMBERS (memorize)

- Segmentation: **mean tumour Dice 0.73**, enhancing tumour **0.80**.
- Enhancement (BraTS): **PSNR 30.3, SSIM 0.965**; heavy noise 0.19→0.89.
- Offline enhancement beats CLAHE everywhere (e.g. brain-path SSIM 0.19→0.96).
- Model: **7.77 M params, 31 MB, 4.2 ms/image, 236 images/sec**, peak 390 MB GPU.
- Utilization: Enhancement 84% GPU, Segmentation 98% GPU, CPU ~15%.

---

## 9. DESIGN DECISIONS (with justification)

- **2D slices, not 3D** — 3D BraTS models need 16 GB+ GPU; we have 6 GB. 2D is a
  standard, documented workaround.
- **Case-level split** (never slice-level) — slices from one patient are
  correlated; slice-split leaks info and inflates metrics.
- **Multi-modal (4-channel) segmentation** — each modality shows a different
  tumour part (T1c→enhancing, FLAIR/T2→edema); improves Dice.
- **CE + Dice loss** — Dice counters severe tumour/background imbalance.
- **Synthetic Rician degradation** — the *correct* MRI noise model, so the
  enhancer learns to remove real MRI noise, not generic noise.
- **Spine = unsupervised** — no labels exist, no external data allowed →
  supervised would be dishonest; self/unsupervised is what the brief encourages.
- **Detection runs on the ORIGINAL scan**, not the smoothed one — so enhancement
  can never erase a tumour (verified: Dice 0.842→0.824, negligible).

---

## 10. RAPID Q&A (one-line answers)

- **Loss function?** Enhancement: L1 + SSIM. Segmentation: Cross-Entropy + Dice.
- **Architecture?** 2D U-Net, encoder-decoder + skip connections.
- **Why 4 channels in?** Stack T1/T1c/T2/FLAIR — each shows a different tumour part.
- **Optimizer?** Adam, lr 1e-3, with AMP mixed precision.
- **Dataset + size?** BraTS2020, 4.47 GB, 369 cases; we used 126.
- **Dice score?** Mean tumour 0.73, enhancing 0.80.
- **Enhancement metrics?** PSNR 30.3, SSIM 0.965.
- **Why 2D not 3D?** 3D needs 16 GB+ GPU; 2D fits our 6 GB.
- **Preprocessing model?** None — classical CLAHE + histogram equalization.
- **Spine segmentation?** Unsupervised SLIC-superpixel clustering (no labels allowed).
- **Do your spine models need annotations?** **No — none of them.** Enhancement is
  self-supervised (degrade the scan, restore it to itself), ROI is unsupervised
  clustering, lesion localisation is an autoencoder trained on healthy scans only.
- **Did you convert the .nii files?** No. Every volume is read directly from
  .nii/.nii.gz with nibabel. PNG/JSON are outputs only, never pipeline inputs.
- **How many samples train/test/val?** BraTS 127 cases available — segmentation
  16 train / 4 val, enhancement 32 / 8, plus 3-fold CV over 21. Offline data:
  **20 train / 20 test** (5+5 per each of the 4 groups). Full per-case ID
  enumeration in `stats/splits_report.txt`.
- **What do the labels mean?** BraTS: 1 = necrotic/dead core, 2 = edema
  (swelling), 3 (raw 4) = enhancing/active tumour. Distribution: background
  **99.03 %**, edema 0.71 %, enhancing 0.17 %, necrotic 0.10 % → why Dice loss.
- **Does one model fit all sub-modalities?** No — we proved it. Modality-specific
  spine models beat the pooled model on **3/3** sequences (T1 0.598→0.827,
  T2 0.594→0.802, STIR 0.540→0.714 SSIM, same test slices).
- **How do you find the spine lesion without labels?** We train an autoencoder
  on **healthy spines only**; it can't reconstruct a lesion it never saw, so the
  reconstruction-error map localises the abnormality. Self-supervised.
- **Why no skip connections in that autoencoder?** Skips would copy the input
  (lesion included) straight to the output and erase the anomaly signal.
- **Does it diagnose the spine condition?** No — it flags a suspicious region
  for a radiologist. Naming "herniation vs stenosis" needs labels we're not
  allowed to have.
- **How avoid erasing tumour?** Detect on the original scan; verified enhancement
  barely changes Dice.
- **How prevent false positives?** Non-MRI images rejected (dark-corner check);
  minimum tumour-size guard.
- **Overfitting?** No — validation tracks training, gap ≈ 0, converged ~epoch 25.
- **Metrics for segmentation?** Dice, Jaccard, Hausdorff, ASD, sensitivity,
  specificity, precision, F1, Relative Volume Error.

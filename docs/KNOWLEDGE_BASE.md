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
- **Autoencoder anomaly detection** — train on healthy only, treat reconstruction
  error as pathology. Self-supervised. **We tested this and it failed on our data
  (AUC 0.27, worse than chance); claim withdrawn.**
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
- **Spine segmentation?** Unsupervised SLIC-superpixel clustering (no labels allowed),
  plus **spinal-canal delineation with a width profile**.
- **How do you handle stenosis without labels?** Stenosis is by definition canal
  narrowing, so we *measure* canal width instead of predicting a label: segment the
  CSF column, find its axis by PCA, sample width perpendicular to it, and report the
  narrowing ratio (narrowest ÷ typical width of that same canal, so patient size and
  resolution cancel out). Canal detected on **91/92** validation slices. Pathological
  canals trend narrower (0.485 vs 0.557, AUC 0.69) but at 10 vs 9 patients this is
  **not significant** (p = 0.089) — we report the measurement and the trend, not a
  diagnosis.
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
- **How do you find the spine lesion without labels?** We tried exactly that — an
  autoencoder trained on **healthy spines only**, treating reconstruction error as
  pathology. **We validated it and it failed: AUC 0.27**, worse than chance, healthy
  spines scoring higher than diseased ones. We withdrew the claim and report the
  negative result. Spine deliverables are restoration + unsupervised region
  segmentation.
- **Why no skip connections in that autoencoder?** Skips would copy the input
  (lesion included) straight to the output and erase the anomaly signal.
- **Does it diagnose the spine condition?** No, and it does not even flag one — the
  detector failed validation, so the map is shown purely as a visualisation.
- **How avoid erasing tumour?** Detect on the original scan; verified enhancement
  barely changes Dice.
- **How prevent false positives?** Non-MRI images rejected (dark-corner check);
  minimum tumour-size guard.
- **Overfitting?** No — validation tracks training, gap ≈ 0, converged ~epoch 25.
- **Metrics for segmentation?** Dice, Jaccard, Hausdorff, ASD, sensitivity,
  specificity, precision, F1, Relative Volume Error.

---

## 11. THE SPINE TRACK — the complete story

The brain track had ground truth (BraTS) and so it is a conventional supervised
story. **The spine track is the interesting one**, because it had no labels at
all, and most of the engineering effort went into working out what is and is not
possible in that situation. Judges will probe here — this is the full account.

### 11.1 What the spine data actually is

- **20 cases total**: 10 normal (SP1–SP10), 10 pathological (SP11+).
- Sagittal MRI, sequences **T1 / T2 / STIR**.
- **No annotations of any kind.** No masks, no boxes, no labels.
- Highly heterogeneous geometry (voxel 0.25–1.3 mm, slice thickness 3–13 mm).
- Split 5 train / 5 test within each group, per the coordinator's instruction.

### 11.2 The seven things we built for spine, in order

| # | Method | Needs labels? | Outcome |
|---|---|---|---|
| 1 | CLAHE enhancement (classical) | no | baseline, works |
| 2 | **Self-supervised U-Net enhancement (ours)** | no | **beats CLAHE on 3/3 sequences** |
| 3 | k-means / SLIC ROI clustering | no | works, but groups brightness only |
| 4 | **Self-supervised CNN segmentation (ours)** | no | **best annotation-free method, measured** |
| 5 | Autoencoder anomaly detection | no | **FAILED validation — withdrawn** |
| 6 | Canal-width morphometry (ours) | no | works, 91/92 slices |
| 7 | SPINEPS per-vertebra instances | pretrained | 17 vertebrae, 13 structures |

### 11.3 The self-supervised CNN — how something with no labels trains at all

This is the question most likely to be asked, because it sounds impossible.

It is **differentiable feature clustering** (Kanezaki, ICASSP 2018). A small CNN
is trained **on the single scan in front of it**, from scratch, for ~120
iterations. Its supervision comes from three constraints, none of which need a
human label:

1. **Commit** — each pixel's feature vector is pushed toward its own argmax
   class. The network trains on its own current best guess, which sharpens fuzzy
   assignments into definite ones.
2. **Continuity** — neighbouring pixels are pushed to agree, so regions come out
   spatially coherent instead of speckled.
3. **Balance** — an entropy term on the mean class distribution stops every
   pixel collapsing into one class. Without it the whole image becomes one
   region (we hit this).

The number of structures it finds is **emergent, not set by us**: we offer 12
candidate classes and it settled on **9–10** on the test slice.

**Two real bugs, both measured, both worth telling:**

- *Collapse to 1 class.* We first masked the background by zeroing the
  **features**. That made every background pixel argmax to the same class, and
  the cross-entropy term then dragged the entire image into it. Fix: mask the
  **loss**, not the features.
- *Collapse to 2 classes.* Applying the superpixel prior every training
  iteration compounded with the cross-entropy feedback. **Measured: 2 classes
  with the prior in-loop, 11 without.** Fix: apply the prior once, after
  training.

### 11.4 What we ditched, and why (be ready for this)

| Ditched | Why |
|---|---|
| **Supervised spine segmentation** | No labels exist and no external data was permitted. Any "trained" claim would be fabricated. |
| **Autoencoder anomaly detection** | Validated at **AUC 0.266 — worse than chance**; normal spines scored *higher* (0.0199) than pathological (0.0167). We tested five alternative scoring statistics (mean 0.304, max 0.500, p99 0.388, p95 0.312, top-1% 0.413) — all at or below chance, so it was not a scoring artefact. The verdict box was removed; the map remains only as a labelled visualisation. |
| **Vertebra instance segmentation, our own** | **Four** attempts: component uniformity/linearity scoring; canal-proximity prior; periodicity + autocorrelation; tissue-constrained band. All four locked onto soft tissue rather than bone. Not shipped. |
| **3D volumetric models** | Need 16 GB+ VRAM; we have 6 GB. 2D slice-wise is the documented workaround. |
| **Diagnosis / severity claims** | We measure canal width; we do not output "stenosis". The statistics do not support a diagnostic claim (p = 0.089). |

The pattern is deliberate: **we validated before shipping, and we withdrew two
things that failed.** That is a stronger position than claiming four working
features, because a judge can break a false claim in one question.

---

## 12. SPINEPS — the pretrained model (what, why, trained on what)

### 12.1 What it is

**SPINEPS** — Möller et al., *"SPINEPS: automatic whole spine segmentation of
T2-weighted MR images using a two-phase approach to multi-class semantic and
instance segmentation"*, **European Radiology (2025)**. Apache-2.0 licence.
Built on **nnU-Net**. Runs in its own Python 3.11 conda environment.

**Two phases:**
1. **Semantic** — labels 14 spinal structure *types* (body, disc, canal, cord,
   arch, processes). Answers "what kind of tissue is this pixel?"
2. **Instance** — converts that into *individually numbered* vertebrae.
   Answers "which bone is this?"

### 12.2 What it was trained on

- The public **SPIDER** dataset (annotated lumbar spine MRI).
- The **German National Cohort (NAKO)** — a large population imaging study.
- Roughly **1,600+ subjects**, with expert annotations.

That external annotated data is exactly what we do not have and are not
permitted to collect — which is the entire argument for using it.

### 12.3 Its published accuracy (theirs, not ours)

| Structure | Dice |
|---|---|
| Vertebrae | **0.920** |
| Intervertebral discs | **0.967** |
| Spinal canal | **0.958** |

**Say clearly: these are their numbers on their test set. We claim none of them.**

### 12.4 What it produced on OUR data (case SP11, sagittal T2w 512×512×12)

| Phase | Result |
|---|---|
| Semantic | **13 structures** — labels 41–49, 60, 61, 62, 100 |
| Instance | **17 individually numbered vertebrae** |
| Runtime | **401 s** total, instance phase on CPU (~25 s per vertebra) |

### 12.5 What the numbers on the spine actually MEAN

Two different numbering systems appear, and confusing them is an easy trap:

**A. Instance IDs (1, 2, 3 … 17)** — the numbers drawn on each coloured
vertebra. They mean *"this is a distinct bone, separate from its neighbour."*
They are an **ordering down the spine**, not a diagnosis, not a severity score,
and not an anatomical name. ID 5 is simply the fifth vertebra the model
separated.

**B. Semantic label values** — the mask's pixel values, which are structure
*types* from the TPTBox convention:

| Value | Structure |
|---|---|
| 41 | Arcus vertebrae (vertebral arch) |
| 42 | Spinous process |
| 43 / 44 | Costal process left / right |
| 45 / 46 | Superior articular process left / right |
| 47 / 48 | Inferior articular process left / right |
| **49** | **Vertebral body (corpus)** |
| **60** | **Spinal cord** |
| **61** | **Spinal canal** |
| 62 | Endplate |
| **100** | **Intervertebral disc** |

So a pixel valued 100 is disc tissue; a vertebra tagged instance 7 is the
seventh bone down. Different questions, different numbers.

### 12.6 Why using it is defensible

1. **The required output is supervised by nature.** Naming a herniated disc
   requires having seen examples labelled "herniated disc". With 20 unlabelled
   cases and no external data, *no model we train can produce it.* That is a
   property of the problem, not a failure of effort.
2. **We proved the alternatives fall short** — four annotation-free methods,
   measured, one reported as an outright failure (11.4, 13).
3. **We supply it no annotations and do not train it**, which matches the
   brief's "no annotations for model training".
4. **It is auditable** — open source, peer-reviewed, published weights and
   accuracy. Anyone can re-run it.
5. **Provenance is labelled everywhere it appears** — demo, figures, report.

### 12.7 The two engineering problems it caused (good story, tells well)

- **CUDA OOM on the instance phase.** Failed at **12.44 GiB on a 6 GiB card**.
  Our first hypothesis — that the scan's 0.44 mm in-plane resolution was to
  blame — was **wrong**: downsampling 3.6x produced a byte-identical failure.
  The log revealed why: the instance phase does not read the input scan, it
  reads the **semantic mask SPINEPS just wrote**, which is stored in SPINEPS's
  own internal 0.75 mm space (512x512x53) regardless of input. Fixed by running
  that phase on CPU.
- **Misaligned overlay.** SPINEPS **resamples and reorients** to a canonical
  axis order, so a mask cannot be matched to the scan by array index. Fixed with
  `spineps_runner.mask_in_scan_space()`, which composes the two affines and
  resamples nearest-neighbour (interpolating integer labels would invent classes
  that do not exist).

---

## 13. OURS vs SPINEPS — the measured comparison

`src/spine_vs_spineps.py` produces `results/spine_vs_spineps.json` and
`outputs/demo/spine_vs_spineps.png`.

**The idea:** SPINEPS can name structures we cannot, so use it as a *reference
standard* and measure how much our annotation-free methods recover without ever
seeing a label. This converts "we used a pretrained model" into "here is exactly
what our own work achieves, and exactly what it cannot."

**Dice (case SP11, slice 5; CNN = mean ± sd over 3 runs):**

| Structure | k-means | SLIC | **Self-sup. CNN (ours)** |
|---|---|---|---|
| Vertebral bodies | 0.263 | 0.204 | **0.257 ± 0.018** |
| Intervertebral discs | 0.047 | 0.047 | **0.090 ± 0.017** |
| Spinal canal + cord | 0.304 | 0.302 | **0.380 ± 0.041** |
| Posterior elements | 0.071 | 0.097 | **0.169 ± 0.032** |

**Precision** (does the region stop at the structure's edge?):

| Structure | k-means | SLIC | **Ours** |
|---|---|---|---|
| Vertebral bodies | 0.157 | 0.121 | **0.191** |
| Discs | 0.024 | 0.024 | **0.050** |
| Canal + cord | 0.194 | 0.190 | **0.310** |
| Posterior elements | 0.038 | 0.052 | **0.116** |

**The three claims this supports:**
1. Our CNN has the **highest precision on all four structures**, and the best
   overlap on three of four (vertebral bodies is a tie within uncertainty).
2. Classical clustering shows **high recall, very low precision** (k-means
   recall 0.81 on bodies but precision 0.16) — it *finds* structures but its
   clusters flood across the image, exactly the "groups brightness" failure.
3. Even our best is **Dice 0.38** against SPINEPS's published **0.92**, and ours
   numbers **zero** vertebrae. That gap is the justification, quantified.

### 13.1 Three methodological honesty points (judges may probe these)

- **The Dice values are oracle-assisted upper bounds.** Unsupervised methods
  return anonymous cluster indices; to score them at all, the reference must
  pick which cluster to compare. So the number answers *"was this structure
  carved out as a distinct region?"* — **not** *"can the method name it?"*
  Naming is precisely the supervised step our data cannot provide.
- **We checked whether the metric was unfair** to methods with more clusters
  (ours makes 9–10, k-means 4, so no single cluster can cover a whole
  structure). We added a greedy **best-union** metric to test exactly that. It
  moved the result by 0.002 (0.226 to 0.228) — so the concern was **measured and
  rejected**, not assumed away.
- **The CNN is stochastic.** It is seeded, but cuDNN chooses nondeterministic
  kernels, so runs differ (measured: posterior-element Dice 0.215 vs 0.143 on
  two runs of identical code). Reporting one number would not be reproducible,
  so all CNN figures are **mean ± sd over 3 runs**.

---

## 14. WHAT COULD BE IMPROVED (asked in almost every viva)

Have a real answer here — "nothing" reads as not understanding the work.

**Spine, biggest wins first:**
1. **Label a small subset and fine-tune.** Even 20–30 annotated slices would
   allow a supervised head on top of our features, turning anonymous regions
   into named structures. This is the single highest-value next step.
2. **3D instead of 2D.** We segment slice by slice; vertebrae are 3D objects and
   through-plane context would improve boundaries. Blocked by 6 GB VRAM.
3. **Distil SPINEPS output into pseudo-labels** and train a small model on them
   — gets instance numbering without shipping SPINEPS at inference time.
4. **More cases.** 20 patients is small; the canal-width trend (0.485 vs 0.557,
   AUC 0.69) points the right way but is **not significant (p = 0.089)**. Around
   40 per group would likely settle it.

**Brain:**
5. **3D or 2.5D segmentation** — adjacent-slice context typically adds several
   Dice points on BraTS.
6. **Ensembling** across seeds/folds — a reliable small gain we did not spend
   time on.
7. **Test-time augmentation** (flips) — cheap, usually worth about 1 Dice point.
8. **Boundary-aware loss** (add a surface term) — our Hausdorff is the weakest
   metric relative to Dice, which points at boundary quality.

**Engineering:**
9. **Determinism** — enable deterministic cuDNN kernels so results are exactly
   reproducible rather than mean ± sd.
10. **SPINEPS on GPU** would need more than 6 GB, or patched sliding-window
    inference in its instance phase.

---

## 15. RAPID Q&A — spine and SPINEPS

- **Did you train SPINEPS?** No. We give it no annotations and do not train it.
  We run published weights and label its provenance everywhere.
- **What was it trained on?** SPIDER + the German National Cohort, about 1,600+
  annotated subjects. That external labelled data is what we lack.
- **Isn't that against the rules?** The organisers approved a public pretrained
  model and asked for justification — that document is
  `PRETRAINED_MODEL_JUSTIFICATION.md`. Our own annotation-free pipeline is shown
  *beside* it, never replaced by it.
- **What do the numbers 1–17 on the vertebrae mean?** Instance IDs — "this is a
  separate bone from the one above." Not a diagnosis, not severity, not an
  anatomical name.
- **What do 41–49, 60, 61, 62, 100 mean?** Structure *types*: 49 vertebral body,
  60 cord, 61 canal, 100 disc, 41–48 arch and processes.
- **How good is your own spine segmentation, honestly?** Best Dice **0.38**
  (canal) against SPINEPS as reference, and the highest precision of the three
  annotation-free methods on all four structures — but zero numbered vertebrae,
  because numbering needs labels.
- **Why is your Dice so low?** Two reasons, both real: unsupervised clusters are
  anonymous and bleed past structure edges (precision 0.19–0.31), and the metric
  is an oracle-assisted upper bound on a method that was never given a target to
  fit.
- **Why did downsampling not fix the OOM?** Because the instance phase reads the
  saved semantic mask in SPINEPS's own 0.75 mm space, not our input — so the
  working volume never changed. We measured that, then moved to CPU.
- **Why not run SPINEPS live in the demo?** 401 s per scan on CPU. It is shown
  as a precomputed reference on a named case, explicitly labelled as not having
  been run on the uploaded scan.
- **Which parts are yours and which are not?** Ours: all brain enhancement and
  segmentation, all spine enhancement, the self-supervised CNN segmentation, the
  canal morphometry, and every validation. Not ours: SPINEPS instance/semantic
  masks, used for one output and labelled as pretrained everywhere.

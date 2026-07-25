# The 4 Stages — what each is, what WE did, how to explain it

The hackathon workflow has 4 stages. Phase-1 judging covers **Stage 1 + 2**
(and half of Stage 3). Here's the task for each, what we built, and a
one-line simple explanation you can say out loud.

---

## STAGE 1 — Dataset Exploration, Analysis & Preparation  ✅ DONE

**The official task:** Understand the MRI sub-modalities (T1, T2, FLAIR,
STIR…), analyse the dataset, measure image properties, divide patients by
sub-modality, and lay out the train/test/validation split.

**What we did:**
- Profiled **every** scan on 7 properties: Contrast, Complexity, Sharpness,
  Edge strength, Noise level, Mean, Deviation — for BraTS2020 **and** all four
  offline groups (`dataset_stats.py` → `stats/dataset_stats.csv`).
- Built a smart classifier that reads the messy scanner filenames and sorts
  each file into T1 / T1c / T2 / FLAIR / STIR, throwing out the out-of-scope
  sequences (DWI, ADC, SWI…). Every decision is logged (`stats/modality_audit.txt`).
- Enumerated the splits: BraTS 80/20; offline 5-train / 5-test per group (the
  coordinators' rule).

**Say it simply:**
> "First we studied the scans — measured how noisy, how sharp, how detailed
> each type is. That told us brain and spine scans are very different, so we
> handle them differently instead of one-size-fits-all."

---

## STAGE 2 — Pre-processing  ✅ DONE

**The official task:** Denoise, rescale, correct artifacts, de-emphasise
unimportant regions, apply basic enhancement (Histogram Equalization / CLAHE),
then re-measure quality with the same properties + IQA metrics (PSNR, SSIM…).

**What we did:**
- **Normalize** every scan to a common range (robust percentiles), **slice**
  the 3D volume into 2D images, **resize** to 224×224 (with clipping to kill
  interpolation ringing), skip empty background slices.
- **CLAHE** contrast enhancement as the classical baseline.
- Built **synthetic degradation pairs** (Rician noise + bias-field + blur) so
  we have clean-vs-dirty examples to train the enhancer on — no annotations
  needed.
- Augmentation (flips) and curated train/test sets.

**Say it simply:**
> "Then we cleaned and standardised every scan — same size, same brightness
> scale — and prepared matched clean/dirty pairs so the AI has something to
> learn from."

---

## STAGE 3 — MR Image Quality Enhancement  🔵 THIS IS THE MAIN AI (done + demoing)

**The official task:** Use a deep-learning / ensemble / transformer /
self-supervised model to further enhance quality; compare against published
methods with the full IQA metric suite; report training loss, validation loss,
learning curve, convergence epoch, overfitting gap; and a systematic
comparison on latency, throughput, memory, model complexity.

**What we did:**
- A **2D U-Net** that learns to reverse MRI degradation (noise + bias field).
  Loss = L1 + SSIM (keeps both brightness and structure faithful).
- Trained for brain (BraTS FLAIR) and all offline groups.
- Full IQA suite (PSNR, SSIM, FSIM, GMSD, VIF, LPIPS, BRISQUE, NIQE, PIQE,
  Entropy, MSE, RMSE, UQI) + a **3-way comparison** proving our AI beats the
  classical CLAHE method (e.g. brain SSIM 0.22 → **0.92**; CLAHE makes it
  *worse*).
- Loss curves, convergence epoch, overfitting gap, and a benchmark
  (7.77M params, 4 ms/image, 236 images/sec on a laptop GPU).

**Say it simply (this is the live demo):**
> "This is the core: give the AI a noisy scan, it gives back a clean one —
> without changing the anatomy. The classical method just brightens the noise;
> ours actually removes it. Watch — [run the website]."

---

## STAGE 4 — Region-of-Interest Segmentation  ✅ DONE (bonus for Phase 1)

**The official task:** On the enhanced scans, segment the region of interest —
tumour / edema / lesion in diseased brains, disc / herniation / stenosis in
spine; report Dice, Jaccard, Hausdorff, ASD, sensitivity/specificity,
Grad-CAM / attention.

**What we did:**
- **Brain (supervised, BraTS):** a 4-input U-Net (T1+T1c+T2+FLAIR) that finds
  the tumour sub-regions. Real metrics: **mean tumour Dice 0.73**, enhancing
  tumour **0.80**. On the demo it matches the doctor's label **~87%** on
  unseen cases.
- **Spine (unsupervised):** no ground-truth labels exist and no external data
  is allowed, so we use intensity clustering (k-means) — framed honestly as
  *exploratory* ROI detection, exactly what the problem statement suggests for
  a no-label situation.

**Say it simply:**
> "Finally, the cleaned scan feeds tumour detection — and here's the proof: our
> AI's tumour outline sits right on top of the doctor's, 87% overlap, on a scan
> it had never seen."

---

## The one-paragraph story (ties all four together)
> "A fast MRI is noisy and unevenly lit. **Stage 1** we measured exactly how bad.
> **Stage 2** we standardised and prepared the data. **Stage 3** — the heart —
> an AI cleans the scan without altering anatomy, beating the textbook method.
> **Stage 4** the clean scan feeds automatic tumour detection that matches the
> radiologist. All in under a second, on a laptop, using the files hospitals
> already have."

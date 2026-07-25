# Glossary — every term on the website & in the demo, in plain words

Keep this open on your phone during judging. Format: **TERM (full form)** —
what it means simply · *(technical note if a judge digs deeper)*.

---

## Scan types (the "sub-modalities")
- **MRI (Magnetic Resonance Imaging)** — a scan that uses a strong magnet + radio
  waves (no radiation) to image soft tissue.
- **NIfTI (.nii / .nii.gz)** — the standard file format for brain/spine MRI
  volumes. `.gz` just means zipped. *(Hospitals and research both use it.)*
- **T1 (T1-weighted)** — anatomy looks "natural"; fat is bright, fluid is dark.
  Good for structure.
- **T1c / T1ce (T1 contrast-enhanced)** — a T1 scan after injecting a contrast
  dye (gadolinium); makes active tumour light up bright.
- **T2 (T2-weighted)** — fluid is bright; good for spotting swelling/lesions.
- **FLAIR (FLuid-Attenuated Inversion Recovery)** — a T2 scan with the bright
  fluid "switched off", so lesions next to fluid stand out. Brain only.
- **STIR (Short-TI Inversion Recovery)** — fat-suppressed scan; makes swelling
  and bone-marrow problems obvious. Used for spine.

## The two "dirt" types we remove
- **Rician noise** — the specific grainy noise pattern that MRI magnitude images
  have (not ordinary "Gaussian" camera noise). We simulate the *correct* kind.
- **Bias field (RF-coil inhomogeneity)** — smooth uneven brightness across the
  image (bright centre, dark edges) from the scanner's antenna.

## Enhancement quality scores (higher = better unless noted)
- **PSNR (Peak Signal-to-Noise Ratio)** — how close to the clean image, in
  decibels (dB). Higher = cleaner. ~30 dB is very good.
- **SSIM (Structural Similarity Index)** — how similar the *structure* is,
  0 to 1. 1 = identical. **This is our headline number** (0.9+ = excellent).
- **MSE / RMSE (Mean / Root-Mean-Square Error)** — average pixel error.
  **Lower = better.**
- **UQI (Universal Quality Index)** — an older combined similarity score, 0–1.
- **FSIM (Feature Similarity)** — similarity focused on edges/features, 0–1.
- **GMSD (Gradient Magnitude Similarity Deviation)** — edge-based difference.
  **Lower = better.**
- **VIF (Visual Information Fidelity)** — how much visual information survived.
- **LPIPS (Learned Perceptual Image Patch Similarity)** — an AI-based
  "does it *look* the same to a human" score. **Lower = better** (0 = identical).
- **BRISQUE / NIQE / PIQE** — "no-reference" quality scores: they rate a single
  image's quality *without* needing a clean copy to compare to. **Lower = better.**
  *(We use these on the real hospital scans, where there's no clean original.)*
- **Entropy** — how much detail/information is in the image (in bits).

## Segmentation quality scores (Stage 4)
- **Dice (Dice Similarity Coefficient, DSC)** — how much the AI's region
  overlaps the doctor's, 0–1 (or %). **The main segmentation score.** 0.8 = great.
  *(On the demo we show it as "overlap with doctor".)*
- **Jaccard (IoU, Intersection-over-Union)** — same idea as Dice, slightly
  stricter. 0–1.
- **Hausdorff Distance (HD)** — the worst-case boundary error (how far off the
  edge is at its worst). **Lower = better.**
- **ASD (Average Surface Distance)** — average boundary error. **Lower = better.**
- **Sensitivity / Recall** — of the real tumour, how much did we catch.
- **Specificity** — of the healthy tissue, how much did we correctly leave out.
- **Precision** — of what we flagged, how much was truly tumour.

## What the tumour colours mean
- **Green = Edema** — swelling around the tumour.
- **Red = Enhancing tumour** — the active, growing part.
- **Blue = Necrotic / non-enhancing core** — the dead centre.

## Methods & model
- **CLAHE (Contrast-Limited Adaptive Histogram Equalization)** — the classic
  textbook contrast-boost method. Our baseline — we beat it.
- **Histogram Equalization (HE)** — spreads out brightness levels to boost
  contrast (CLAHE is a smarter, local version).
- **U-Net** — the AI network shape we use: an encoder that compresses the image
  and a decoder that rebuilds it, with "skip connections" that preserve fine
  detail. Standard for medical imaging.
- **Skip connections** — shortcuts that carry sharp detail from early layers to
  late layers so the output isn't blurry.
- **CNN (Convolutional Neural Network)** — the family of AI models U-Net belongs
  to; learns visual patterns from data.
- **Supervised vs Unsupervised** — supervised = trained with expert labels
  (our brain tumour model); unsupervised = no labels, finds structure itself
  (our spine ROI, because no labels exist and none are allowed).

## Data & training words
- **BraTS (Brain Tumour Segmentation challenge 2020)** — the standard,
  expert-labelled brain-tumour dataset we had to use for brain.
- **Ground truth** — the expert/doctor's correct answer we compare against.
- **Unseen / held-out** — data the model never trained on (proves it
  generalises, not memorises).
- **Epoch** — one full pass through the training data.
- **Loss** — the error the model minimises while learning; lower = learning.
- **Convergence** — the point where training stops improving.
- **Overfitting** — memorising training data instead of learning general rules;
  we check the "overfitting gap" (ours is ~0 — healthy).
- **Augmentation** — making extra training examples by flipping/altering scans.

## Output & performance words
- **ROI (Region of Interest)** — the area we care about (tumour, disc, etc.).
- **COCO JSON** — a standard file format for storing segmentation masks, so
  other software can read our results.
- **Latency** — time to process one image (ours ~4 ms).
- **Throughput** — images processed per second (ours ~236/sec).
- **GPU** — the graphics chip that runs the AI fast. Ours is a 6 GB laptop GPU.
- **AMP (Automatic Mixed Precision)** — a trick to run the AI faster and use
  less memory without losing accuracy.
- **Parameters** — the AI's internal "knobs" (ours ~7.8 million; small = light).

## Anatomy words (healthy-tissue ROI)
- **CSF (Cerebrospinal Fluid)** — the fluid around the brain/spinal cord.
- **Grey matter / White matter** — the two main brain tissue types.
- **Intervertebral disc** — the cushion between spine bones (where herniation /
  degeneration happens).
- **Spinal stenosis** — narrowing of the spinal canal.

---

### 3 lines that make you sound fluent
- "SSIM 0.9 means the cleaned scan is 90%+ structurally identical to the true one."
- "Dice is just the overlap between our outline and the doctor's — 0.8 is strong."
- "CLAHE is the textbook method; it boosts contrast but amplifies noise — we beat it."

---

## Spine + pretrained-model words (added with SPINEPS)

- **Semantic segmentation** — labels each pixel by *what kind* of thing it is
  ("this is disc tissue"). Two neighbouring vertebrae get the same label.
- **Instance segmentation** — separates *individual objects* ("this is vertebra
  5, that is vertebra 6"). Harder, and the thing that needs labels.
- **Instance ID** — the number drawn on each vertebra in our SPINEPS figure. It
  means "a separate bone from the one above", **not** a diagnosis or a severity
  score.
- **Pretrained model** — a model someone else already trained on their own
  labelled data; you run it as-is. We use one (SPINEPS) for exactly one output
  and label it everywhere.
- **SPINEPS** — published spine model (Möller et al., *European Radiology* 2025,
  Apache-2.0), trained on the SPIDER dataset + the German National Cohort,
  ~1,600+ annotated subjects. Published Dice 0.92 vertebrae / 0.967 discs /
  0.958 canal. Those are **their** numbers, not ours.
- **nnU-Net** — the self-configuring U-Net framework SPINEPS is built on; it
  picks its own preprocessing and patch sizes per dataset.
- **Reference standard** — a trusted output you measure yourself against when
  you have no ground truth. We use SPINEPS this way.
- **Self-supervised** — the model makes its own training target from the data
  (e.g. degrade a scan, learn to restore it). No human labels.
- **Differentiable feature clustering** — the annotation-free segmentation
  method we use (Kanezaki 2018): a small CNN trains on one image using
  commit + continuity + balance constraints instead of labels.
- **Precision vs recall** (for a segmented region) — *recall* = how much of the
  real structure you covered; *precision* = how much of what you marked was
  actually the structure. Clustering scores **high recall, low precision**: it
  finds the structure but spills far past its edges.
- **Oracle-assisted metric** — a score where the answer key picks which of your
  anonymous clusters to grade. It is an **upper bound**, and we say so.
- **Affine (in NIfTI)** — the matrix mapping voxel indices to real-world
  millimetres. Two images line up only through their affines, never by array
  index — which is why our first SPINEPS overlay was rotated.
- **Nearest-neighbour resampling** — resizing a label map by copying the closest
  label. Required for integer labels; smooth interpolation would invent classes
  that do not exist.
- **Vertebral body (corpus)** — the big drum-shaped front part of a vertebra
  (SPINEPS label 49).
- **Posterior elements** — the bony arch and processes behind the body (labels
  41–48): arch, spinous process, costal and articular processes.
- **Spinal canal vs spinal cord** — the canal (label 61) is the bony tunnel; the
  cord (label 60) is the nerve tissue inside it. Stenosis narrows the canal.
- **Endplate** — the cartilage surface between a vertebral body and its disc
  (label 62).

### 3 more lines that make you sound fluent
- "Semantic says *what*, instance says *which one* — instance is the part that
  needs labels, which is why we use a pretrained model for it."
- "Our clustering has high recall but low precision: it finds the structure and
  then bleeds past its edge."
- "Masks only line up through the affine — SPINEPS reorients, so index-matching
  gives you a rotated overlay."

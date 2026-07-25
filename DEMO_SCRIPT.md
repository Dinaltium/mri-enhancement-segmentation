# Demo Script & Talking Points — Phase 1 (Stage 1, 2 & 3)

How to run the demo, what to say (simple words), and how to answer whatever
the judges throw at you. Open `outputs/demo/demo_page.html` full-screen in a
browser (works offline — no wifi needed).

---

## 0. Before they arrive — the LIVE WEBSITE (your best weapon)

**Double-click `start_webapp.bat`** (in `C:\Projects\Yugma`). It starts the
server and opens `http://localhost:5000` in your browser. Keep the black
window open during the demo.

The website lets the judges **upload an MRI and watch the AI clean it live**:
- Click **"Brain sample"** or **"Spine sample"** for an instant demo (no file
  needed) — shows clean → noisy → CLAHE → AI-enhanced with PSNR/SSIM.
- Or hit **"Choose file"**, pick any `.nii`/`.nii.gz` (or even a `.png`),
  select Brain/Spine, and click **"Enhance it"** — the AI cleans *their* file
  in front of them.
- **"Brain tumour segmentation"** button = the Stage-4 bonus (auto ROI overlay).
- Works 100% offline (no wifi needed) and runs on the laptop GPU.

> Killer move: ask a judge to hand you any scan file, drop it in, and let them
> watch it get cleaned. Nothing convinces like *their* data working live.

**Backups if the laptop/projector misbehaves:**
1. `outputs/demo/demo_page.html` — full-screen static slideshow (same visuals,
   also fully offline).
2. The PNGs in `outputs/demo/` — open directly.
3. Regenerate panels from the terminal:
   `"C:/Users/RAFAN AHAMAD SHEIK/.conda/envs/tfenv/python.exe" demo.py`

---

## 1. The 30-second pitch (say this first, simple words)

> "MRI machines, to scan quickly, produce images that are **grainy** and
> **unevenly lit**. That grain can hide the exact things a doctor is looking
> for — the edge of a tumour, a pinched nerve. We built an **AI that cleans up
> the scan** — removes the grain, fixes the lighting — **without changing the
> actual anatomy**. Then it **highlights the region of interest** automatically.
> It runs in a few milliseconds on a normal laptop, on the standard hospital
> file format. Let me show you."

Then open the demo page and walk through it.

---

## 2. Walk-through — what to say at each panel

### Pipeline diagram
> "This is the whole framework, left to right: we **analyse** the scan quality,
> **clean** it, then **highlight** the region of interest. Today's demo is the
> first three stages — analysis, preprocessing, and the AI enhancement."

### The main demo (brain + spine 4-panel) — spend the most time here
Point at the four images left to right:
> "Far left is a **clean scan**. Second is that same scan after we **added
> realistic MRI noise and a lighting artifact** — this is what a fast, low-quality
> scan looks like. Third is the **classical textbook method, CLAHE** — notice it
> actually looks **grainier**, because it just brightens everything, noise
> included. Far right is **our AI** — the grain is gone and it looks like the
> original again."

Then point at the numbers:
> "These numbers are the proof. **SSIM** measures how close to the original,
> 1.0 is perfect. The noisy scan is 0.22. CLAHE makes it *worse* — 0.18. **Our
> AI brings it to 0.92.** Same story on the spine scan — a completely different
> part of the body — 0.29 up to 0.89."

Key line to land:
> "So the classical method isn't just weaker — measured against the truth it
> goes the *wrong way*. Our model actually **removes** the noise."

### Dataset analysis (bar chart)
> "Before cleaning anything, we **measured every scan** on seven quality
> metrics. This is Stage 1. It's how we proved brain and spine scans behave
> differently — spine images are about **twice as complex** — so the system
> treats them appropriately instead of one-size-fits-all."

### Results tiles
> "To summarise: image quality roughly **quadruples** on the structure score,
> it runs at **236 images a second**, and — as a bonus — the segmentation stage
> hits **0.73 Dice** on brain tumours, 0.80 on the active tumour region."

### Segmentation preview (tumour overlay)
> "And here's where it's going: the cleaned scan feeds automatic
> region-of-interest detection — green is swelling, red is active tumour, blue
> is the dead core. Fully automatic."

---

## 3. Analogies for non-technical judges (use if they look lost)

- **The noise:** "Like a **photo taken in a dark room** — grainy. Our AI is the
  'night mode' that cleans it up."
- **Why not CLAHE:** "CLAHE is like **turning up the brightness** on a grainy
  photo — you see the grain *more*, not less. Ours actually removes it."
- **Training the AI:** "We showed it **thousands of clean scans and dirtied
  copies**, so it learned to undo the dirtying — like teaching someone to
  read messy handwriting by showing them many messy-vs-neat examples."
- **SSIM = 0.9:** "0.9 out of 1.0 means the cleaned scan is **90%+ identical**
  to the true scan — we cleaned it *without inventing anything*."

---

## 4. Q&A cheat sheet (the questions they WILL ask)

**Q: Is it dynamic / real-time?**
> Yes. 4 milliseconds per image, 236 images per second on a 6 GB laptop GPU.
> A whole scan volume cleans in about a second.

**Q: Can a doctor actually use this?**
> It reads the standard hospital format (NIfTI / .nii) directly and outputs the
> cleaned scan plus the region mask in the standard COCO format. No special
> hardware, no file conversion, no cloud needed.

**Q: Does it change or invent anatomy? (the critical safety question)**
> No — and this matters. It's trained *only* to reverse noise and lighting
> artifacts. The 0.9+ SSIM against the true scan is the mathematical proof that
> the structure is preserved, not altered. We never add detail that wasn't there.

**Q: Will it work easily? Is it heavy?**
> The model is 31 MB — smaller than a phone photo album. Runs on an ordinary
> laptop GPU. The *same* lightweight network handles both brain and spine; only
> the input changes.

**Q: How is this better than the classical method?**
> Show the CLAHE column again. CLAHE amplifies noise (SSIM drops to 0.18); ours
> removes it (0.92). We keep CLAHE as a baseline precisely to prove the AI adds
> real value.

**Q: What data did you train on? Is it legitimate?**
> The mandated BraTS2020 brain dataset for the brain model; for spine we used
> only the offline hackathon data as required, with the 5-train / 5-test split
> the coordinators specified. No external spine data.

**Q: You have no ground truth for the offline scans — how do you evaluate?**
> Honestly. For enhancement we degrade a clean scan, enhance it, and compare to
> the known original (full-reference metrics). For segmentation we report exact
> Dice *only* on BraTS where expert labels exist; on the unlabelled hackathon
> scans we show the segmentation visually and say plainly there's no ground
> truth to score against. We do **not** fabricate accuracy numbers.

**Q: What's the AI model? (if a technical judge probes)**
> A 2D U-Net — encoder/decoder with skip connections. For enhancement: 1 channel
> in/out, trained on clean vs synthetically-degraded pairs with an L1 + SSIM
> loss. For segmentation: 4 modalities stacked as input, cross-entropy + Dice
> loss. 2D slices (not 3D) so it fits a 6 GB laptop GPU — a standard, documented
> choice for this hardware.

**Q: Why is the noise "Rician" — why does that matter?**
> MRI magnitude images have Rician noise, not ordinary Gaussian noise. We
> degrade with the *correct* noise model, so the AI learns to remove what real
> MRI noise actually looks like — not a generic approximation.

---

## 5. If something breaks (fallbacks)
- If the live `demo.py` errors on the projector laptop: the PNGs are already in
  `outputs/demo/` — just open those. The HTML page has them baked in too.
- If they want to see it on a *real* scan (not degraded): open
  `outputs/demo/real_scan_enhancement.png` (raw vs enhanced, no synthetic step).
- Full numbers for every group: `enhancement_metrics_*.json`,
  `segmentation_metrics.json`, `stats/dataset_stats.csv`.

---

## 6. One-sentence close
> "In short: we take a fast, noisy MRI and turn it into a clean, doctor-ready
> scan with the region of interest already highlighted — in under a second, on a
> normal laptop, using the files hospitals already have."

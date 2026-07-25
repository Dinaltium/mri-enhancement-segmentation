# Demo video script

**Target length: 5–6 minutes.** Screen recording with voice-over. OBS, the Xbox Game
Bar (`Win + G`), or any screen recorder works.

**Before you record**
1. Start the demo: double-click `run_demo.bat`, wait for the browser to open at
   `http://localhost:5000`.
2. Open a second browser tab with `outputs/demo/demo_page.html`.
3. Open a file-explorer window at `showcase/for_enhancement/`.
4. Close notifications, set the browser to full screen, zoom to ~110 %.

Timings are guides, not targets. Speak slowly — the visuals do the work.

---

## [0:00–0:30] Opening — the problem

**Show:** the demo home page.

> "MRI machines trade image quality for speed. A fast scan comes out grainy and
> unevenly lit, and that grain can hide exactly what a doctor is looking for — the
> edge of a tumour, a compressed disc.
>
> We built a pipeline that restores those scans and then finds the region of
> interest. Everything you'll see runs live, on this laptop, in real time."

---

## [0:30–1:00] The pipeline at a glance

**Show:** scroll slowly across the pipeline map at the top of the page.

> "Here's the whole system. Acquire, two classical enhancement baselines, our U-Net
> restoration model, region-of-interest segmentation, then tissue mapping and
> attention. The outlined boxes are the learned models — everything else is
> deterministic image processing we keep as a measurable baseline."

---

## [1:00–2:15] Live restoration — the core demo

**Do:** click "Choose file", navigate to `showcase/for_enhancement/`, pick a
`BRAIN_BraTS20_*` file. Leave anatomy on **Brain**. Click **Run pipeline**.

> "I'm uploading a brain scan the model has never seen. Watch every stage appear."

**When the results load, scroll through them one at a time:**

> "Stage one, the scan as uploaded.
>
> Stages two and three are the classical textbook methods — histogram equalisation
> and CLAHE. Look carefully: they're *grainier* than the input. They boost contrast
> but they amplify the noise along with it.
>
> Stage four is our model. The grain is gone and the anatomy is intact."

**Point at the numbers:**

> "SSIM measures structural similarity to the true scan — one is perfect. The
> classical methods sit around 0.15. Ours reaches 0.90."

---

## [2:15–3:15] Tumour detection against the radiologist

**Do:** scroll to the "Validate against expert annotation" panel, leave the first case
selected, click **Tumour vs expert**.

> "Now the segmentation. This is a patient held out of training entirely."

**When it loads:**

> "Left is the scan. Centre is what our AI found. Right is the radiologist's own
> annotation. Green is swelling, red is the active tumour, blue is the dead core.
>
> The two masks are nearly identical — and the overlap score is printed right there.
> Across the full held-out set we get a mean tumour Dice of 0.76, and 0.84 on the
> enhancing tumour, which is the clinically important part."

**Do:** click **Attention**.

> "And this is why we trust it. Grad-CAM shows where the network actually looked to
> make that decision. The hot region sits directly on the lesion — it isn't guessing
> from an unrelated part of the image."

---

## [3:15–4:15] Spine — working without any labels

**Do:** upload a `SPINE_*` file from `showcase/for_enhancement/`, set anatomy to
**Spine**, click **Run pipeline**.

> "Spine is a harder problem, because the rules are strict: the spine data has no
> annotations, and we're not allowed to bring in outside data. So nothing here can be
> trained the normal way."

**Scroll to step 5, then step 6:**

> "Step five segments the spine into coherent regions using superpixel clustering —
> no labels needed.
>
> Step six is a research view, and it's worth being precise about. We trained an
> autoencoder on *healthy* spines only, so it rebuilds a healthy-looking version of
> any scan, and this map shows where the input differs from that.
>
> The obvious hope is that the difference marks the disease. We tested that — and it
> failed. Across held-out cases the score for healthy spines is actually higher than
> for diseased ones; the area under the curve is 0.27, worse than guessing. It's
> tracking texture, not pathology.
>
> So we removed the detection claim. This is a visualisation, not a diagnosis. A
> detector that fires on healthy patients would be worse than having none, and we'd
> rather show you the negative result than a convincing-looking false positive."

---

## [4:15–5:00] Inside the model

**Do:** click **Inside the model** in the top bar.

> "We didn't want this to be a black box, so this page opens it up completely.
>
> Every one of the sixty-three operations in a forward pass, with tensor shapes and
> parameter counts. And these feature maps are captured from a real forward pass on a
> real scan — you can see the early layers responding to edges and the deeper ones
> responding to whole regions.
>
> Both loss functions are stated explicitly. For segmentation we use cross-entropy
> plus Dice, and there's a concrete reason: tumour classes are under one percent of
> all pixels, so a model trained on cross-entropy alone would score ninety-nine
> percent by predicting 'background' everywhere and finding nothing at all."

---

## [5:00–5:45] Results summary

**Do:** switch to the `demo_page.html` tab, scroll to the comparison charts.

> "To summarise the measurements. Restoration: ours reaches 0.90 SSIM where every
> classical method scores below the noisy input. Noise: ours is the only stage that
> reduces it. Segmentation: 0.76 mean Dice against the radiologist. And for spine,
> training one model per sequence beat a single pooled model on all three sequences.
>
> Speed: four milliseconds per image, two hundred and thirty-six images a second, on
> a laptop GPU."

---

## [5:45–6:00] Close

> "Three things we deliberately don't claim. We report accuracy numbers only where
> expert annotations exist. Our spine anomaly detector failed its own validation, so we
> withdrew it rather than ship it. And the model corrects noise — it never invents
> anatomy.
>
> Every figure comes from a script in the repository and can be re-run. Thank you."

---

## Recording checklist

- [ ] Demo server running, browser at `http://localhost:5000`
- [ ] `demo_page.html` open in a second tab
- [ ] `showcase/for_enhancement/` open in file explorer
- [ ] Notifications off, browser full screen
- [ ] Microphone tested — clear, no background noise
- [ ] Practised once end-to-end before the real take

**If the pipeline is slow to respond on camera:** keep talking through what's being
computed. The first run loads models into GPU memory; run one pipeline before you
start recording so the models are already warm.

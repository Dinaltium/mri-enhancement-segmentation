# Demo video script

**Target length: 60 seconds.** Screen recording with voice-over. OBS, the Xbox Game
Bar (`Win + G`), or any screen recorder works.

At one minute there is no room for pauses — the script below is **written to time**.
It is **155 spoken words** (counted, not estimated): 62 s at a normal 150 words/minute,
56 s if you speak briskly. Say it as written; every sentence that could be cut already
has been. If your natural pace is slow, use the cutting notes below rather than rushing.

**Before you record**
1. Start the demo: double-click `run_demo.bat`, wait for the browser at
   `http://localhost:5000`.
2. **Run both pipelines once before recording** — a brain scan and a spine scan from
   `showcase/for_enhancement/`. The results stay on screen, so you can scroll through
   finished output instead of waiting for compute on camera. This is the single most
   important preparation step at this length.
3. Close notifications, browser full screen, zoom ~110 %.

---

# THE 60-SECOND SCRIPT

### [0:00–0:10] What it is

**Show:** the demo home page.

> "MRI trades image quality for speed, and the grain hides what matters — the edge of a
> tumour. We restore the scan, then find the region of interest."

### [0:10–0:24] Restoration

**Show:** scroll the finished brain result across steps 2, 3, then 4.

> "These are the textbook methods, histogram equalisation and CLAHE — both grainier than
> the input. They amplify noise while boosting contrast.
>
> This is ours. Structural similarity to the true scan: classical about 0.15, ours 0.90."

### [0:24–0:34] Segmentation

**Show:** the tumour-vs-expert panel.

> "Centre is our AI, right is the radiologist, on a patient held out of training. Mean
> tumour Dice 0.76, and 0.84 on enhancing tumour."

### [0:34–0:54] Spine, and the honest part

**Show:** scroll the finished spine result to the reference-standard section.

> "Spine has no annotations and no external data allowed, so everything here is
> self-supervised. Ours has the highest precision of every label-free method we tested.
>
> But naming a vertebra needs labels — so for that one output we use a published
> pretrained model, with approval, and we measured our gap to it."

### [0:54–1:00] Close

> "Every number comes from a script in the repo, and we report the negative results too."

---

## Cutting notes, if you overrun

Trim in this order — each cut is designed to lose the least:

1. Drop *"and 0.84 on enhancing tumour"* (saves ~2 s).
2. Shorten the opening to *"MRI trades quality for speed, and the grain hides what
   matters. We restore the scan and find the region of interest."* (~3 s).
3. Drop *"They amplify noise while boosting contrast"* — the visual already shows it
   (~3 s).

**Do not cut** the pretrained-model sentence or the closing line. Those are the two
that answer the questions a judge is most likely to ask, and dropping them makes the
video look like it is hiding something.

## Recording checklist

- [ ] Both pipelines already run, results on screen
- [ ] Demo server running at `http://localhost:5000`
- [ ] Notifications off, browser full screen
- [ ] One practice read-through against a stopwatch

---
---

# EXTENDED CUT (~6 min 40 s) — optional

Keep this if a longer submission is allowed, or use it as the source for the live
walkthrough during judging. It covers every stage in full, with the reasoning behind
each one. **The 60-second script above is the deliverable; this is the reference.**

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

**Scroll to step 5, then steps 6 and 7:**

> "Step five is our region segmentation, and it's more than clustering. This is a small
> convolutional network optimised on this one scan, from scratch, using the image's own
> structure as its supervision — pixels commit to a class, neighbours are pushed to
> agree, and a balance term stops everything collapsing into one region. No annotations
> anywhere. The number of structures it finds is emergent: we offer it twelve candidates
> and it settles on nine or ten.
>
> Steps six and seven are the measurement. Stenosis *is* narrowing of the spinal canal,
> so instead of predicting a diagnosis we measure the thing a radiologist actually
> reads. We segment the fluid column, find its axis automatically so orientation doesn't
> matter, and sample the width along it. Canal detection succeeded on ninety-one of
> ninety-two validation slices.
>
> And we stop where the evidence stops. Pathological canals do trend narrower than
> normal ones — 0.485 against 0.557 — but with ten patients against nine that isn't
> statistically significant, p of 0.089. So we report the measurement and the trend, and
> we don't call it a diagnosis."

**Optional — only if you want the negative result on camera (it plays well):**

> "We also tried autoencoder anomaly detection: train on healthy spines only, and
> whatever the model can't reconstruct should be the disease. We validated it instead of
> assuming it, and it failed — area under the curve 0.27, worse than guessing, with
> healthy spines scoring *higher* than diseased ones. So we pulled the claim out of the
> pipeline entirely. We'd rather show a negative result than a convincing-looking false
> positive."

---

## [4:15–4:50] Per-vertebra segmentation — and being straight about the pretrained model

**Show:** stay on the same spine result and keep scrolling — past the model internals
you reach a section headed **"Reference standard — how good is ours, really?"**. It
holds the SPINEPS per-vertebra overlay, the 13 named structures, and the comparison
figures.

**Say this first, so nobody thinks it ran on the upload:** everything above ran live on
the uploaded scan; this section is a precomputed result on case SP11, because the
pretrained model takes 401 seconds per scan on CPU. The page states that too.

> "One more thing on spine, and we want to be completely transparent about it.
>
> The brief asks us to delineate degenerative disc, herniation and stenosis. Those are
> named diagnoses — naming them is a supervised problem. You need labelled examples, and
> we have twenty spine cases with no annotations and no external data permitted. So no
> model we train can produce that output. That's a property of the problem, not a lack
> of effort on our part.
>
> We proved that before reaching for outside help. Four annotation-free methods, all
> measured: clustering, our own self-supervised network, an autoencoder detector that we
> validated and that failed outright, and a periodicity-based vertebra detector that
> didn't hold up either.
>
> So for per-vertebra instances specifically — and only that — we use SPINEPS, a
> peer-reviewed published model, with the organisers' approval. Dice of 0.92 on
> vertebrae, validated on over sixteen hundred subjects. We give it no annotations, we
> don't train it, and we claim none of its accuracy as ours. It's labelled as a
> pretrained model everywhere it appears.
>
> Using a model with quantified accuracy is also the safer clinical call. An
> under-constrained model we forced to produce vertebrae would be confidently wrong,
> and confidently wrong is the worst outcome in medical imaging."

**Point at the numbered vertebrae as you say this:**

> "And to be precise about what those numbers are — they're instance IDs. They mean
> 'this is a separate bone from the one above it'. They are not a diagnosis and not a
> severity score. A separate pass labels thirteen structure types: vertebral body, disc,
> canal, cord, and the arch and processes behind."

---

## [4:50–5:25] We measured the gap instead of asserting it

**Show:** the "Measured: ours vs the reference" panel in that same section
(`spine_vs_spineps.png`).

> "We didn't want to just claim our own method is good, so we scored it. We used the
> pretrained model as a reference standard and measured how much our annotation-free
> work recovers without ever seeing a label.
>
> Two things came out of that, and we report both.
>
> First, ours is the best of the label-free methods. Our self-supervised network has
> the highest precision on all four structures — on the spinal canal it's 0.31 against
> 0.19 for k-means. The classical methods have high recall and terrible precision: they
> find the structure, then bleed across the whole image, which is exactly what you'd
> expect from something that groups by brightness.
>
> Second — and we're not going to dress this up — our best overlap is 0.38, against the
> pretrained model's published 0.92. And ours numbers zero vertebrae, because numbering
> needs labels we don't have.
>
> That measured gap is the entire justification. We're not using a pretrained model
> because it was easier. We're using it for the one output we proved we couldn't reach."

**If you have time, add:**

> "One more honesty note on that chart. Those overlap scores are upper bounds — our
> clusters are anonymous, so the reference has to pick which one to grade. And the
> network is stochastic, so every number there is an average over three runs with its
> standard deviation shown."

---

## [5:25–5:50] Inside the model

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

## [5:50–6:20] Results summary

**Do:** switch to the `demo_page.html` tab, scroll to the comparison charts.

> "To summarise the measurements. Restoration: ours reaches 0.90 SSIM where every
> classical method scores below the noisy input. Noise: ours is the only stage that
> reduces it. Segmentation: 0.76 mean Dice against the radiologist. And for spine,
> training one model per sequence beat a single pooled model on all three sequences.
>
> Speed: four milliseconds per image, two hundred and thirty-six images a second, on
> a laptop GPU."

---

## [6:20–6:40] Close

> "Four things we deliberately don't claim. We report accuracy numbers only where expert
> annotations exist. Our spine anomaly detector failed its own validation, so we withdrew
> it rather than ship it. We take no credit for the pretrained model's accuracy — it's
> labelled as pretrained throughout. And our model corrects noise; it never invents
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

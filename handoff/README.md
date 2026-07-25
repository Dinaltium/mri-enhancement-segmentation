# Handoff pack

Everything needed to produce the presentation, the report and the demo video.
**All numbers here are final and measured** — please don't change them; each one
comes from a script in `src/` and is stored as JSON in `results/`.

## What's in here

| File | What to do with it |
|---|---|
| `01_PRESENTATION_PLAN.md` | 14 slides, one block each. Paste into **Gamma**. Every slide names the exact image to upload and the line to say. Ends with a Q&A cheat sheet. |
| `02_PROJECT_REPORT.md` | The 3–4 page report. Paste into Word/Docs, keep the section order, drop the four figures where marked. |
| `03_DEMO_VIDEO_SCRIPT.md` | Minute-by-minute script for the 5–6 minute demo video, with what to click and what to say. |
| `images/` | All 18 figures the plan and report reference, correctly named. |

## Suggested split of work

- **Person A — slides:** work from `01_PRESENTATION_PLAN.md`, upload from `images/`.
- **Person B — report:** work from `02_PROJECT_REPORT.md`.
- **Person C — video:** work from `03_DEMO_VIDEO_SCRIPT.md`, needs the demo running
  (`run_demo.bat` in the project root).

## The five numbers to memorise

| Claim | Number |
|---|---|
| Tumour segmentation vs radiologist | **0.76 mean Dice** (0.84 enhancing tumour) |
| Restoration quality | **0.90 SSIM** — every classical method scores *below* the noisy input |
| Noise removed | **0.0068 → 0.0043**; HE and CLAHE *raise* it |
| Spine per-sequence models | win **3 / 3** sequences |
| Speed | **4 ms/image**, 236 images/sec, 7.77 M parameters |

## The one-sentence pitch

> "We take a fast, noisy MRI, restore it with a model that beats every classical
> method, and mark the region a doctor cares about — matching the radiologist at 0.76
> Dice, in four milliseconds, on a laptop."

## The three honesty points — say these, they are a strength

1. Accuracy numbers are reported **only where expert annotations exist** (BraTS).
   On unlabelled data we show enhancement metrics and qualitative results, and say so.
2. Our spine anomaly detector **failed its own validation** (AUC 0.27 — worse than
   chance). We report the negative result and removed the claim, rather than shipping
   a detector that fires on healthy patients.
3. The model **corrects noise, it does not invent anatomy** — SSIM above 0.9 against
   the true scan is the evidence.

## Deliverables checklist

- [x] Source code — `src/` (37 modules)
- [ ] Presentation — build from `01_PRESENTATION_PLAN.md`
- [ ] Report — build from `02_PROJECT_REPORT.md`
- [x] Trained model — `models/` (12 checkpoints)
- [x] `requirements.txt` — project root
- [x] README — project root
- [ ] Demo video — record with `03_DEMO_VIDEO_SCRIPT.md`

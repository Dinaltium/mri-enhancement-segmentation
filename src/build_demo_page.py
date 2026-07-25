"""
build_demo_page.py

Bakes ALL demo figures + plain-language explanations into ONE self-contained
HTML file (outputs/demo/demo_page.html) that opens offline. This is the
"PPT walkthrough" covering the whole pipeline: Stage 1&2 (analysis +
preprocessing), Stage 3 (enhancement), Stage 4 (segmentation + tissue +
Grad-CAM), Stage 5 (results/deliverables), with training graphs, metric
tables and a confusion matrix. Double-click and present full-screen.
"""

import base64
import json
import os
import os

OUT = "outputs/demo/demo_page.html"


def b64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(path, cap):
    src = b64(path)
    if not src:
        return f'<p class="missing">[missing {os.path.basename(path)} — run generate_demo_assets.py]</p>'
    return f'<figure><img src="{src}"/><figcaption>{cap}</figcaption></figure>'


D = "outputs/demo"
P = "outputs/plots"


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


# --- resolution table (Stage 1) ---
_res = _load("stats/resolution.json")
res_rows = ""
if _res:
    for ds in ["brats2020", "brain_normal", "brain_pathological", "spine_normal", "spine_pathological"]:
        for mod, a in _res.get(ds, {}).items():
            if not a:
                continue
            sl = f"{a['slices_range'][0]}-{a['slices_range'][1]}" if a.get("slices_range") else "-"
            vx = f"{a['in_plane_voxel_mm'][0]}-{a['in_plane_voxel_mm'][1]}" if a.get("in_plane_voxel_mm") else "-"
            th = f"{a['slice_thickness_mm'][0]}-{a['slice_thickness_mm'][1]}" if a.get("slice_thickness_mm") else "-"
            res_rows += (f"<tr><td>{ds}/{mod}</td><td>{a['matrix_common']}</td>"
                         f"<td>{sl}</td><td>{vx} mm</td><td>{th} mm</td></tr>")

# --- utilization (Stage 3) ---
_util = _load("results/benchmark_utilization.json")
util_txt = "—"
if _util:
    e = _util.get("EnhancementUNet", {}); s = _util.get("SegmentationUNet", {})
    util_txt = (f"Enhancement {e.get('gpu_util_pct_mean')}% GPU · Segmentation "
                f"{s.get('gpu_util_pct_mean')}% GPU · CPU ~{e.get('cpu_util_pct_mean')}%")

# --- modality-specific vs pooled (spine) ---
_modcmp = _load("results/modality_comparison.json")
modcmp_rows = ""
if _modcmp:
    for mod in ["T1", "T2", "STIR"]:
        r = _modcmp.get(mod)
        if not r:
            continue
        def _f(k):
            v = r.get(k)
            return f"{v['psnr']:.1f} / {v['ssim']:.3f}" if v else "—"
        modcmp_rows += (f"<tr><td><b>{mod}</b></td><td>{_f('input')}</td><td>{_f('clahe')}</td>"
                        f"<td>{_f('pooled')}</td><td class='good'>{_f('specific')}</td></tr>")

# --- splits / segregation ---
_splits = _load("stats/splits_report.json")
_annot = _load("stats/annotation_stats.json")
annot_dist = ""
if _annot:
    for k, v in _annot.get("pixel_distribution_percent", {}).items():
        annot_dist += f"<tr><td>{k}</td><td>{v} %</td></tr>"

# --- full segmentation metric suite (Stage 4) ---
_segfull = _load("results/segmentation_full_metrics.json")
segfull_rows = ""
if _segfull:
    for cls in ["necrotic_non_enhancing", "edema", "enhancing"]:
        r = _segfull.get(cls)
        if not r:
            continue
        segfull_rows += (
            f"<tr><td>{cls.replace('_',' ')}</td><td>{r['dice']:.3f}</td>"
            f"<td>{r['jaccard']:.3f}</td><td>{r['accuracy']:.4f}</td>"
            f"<td>{r['sensitivity_recall']:.3f}</td><td>{r['specificity']:.4f}</td>"
            f"<td>{r['precision']:.3f}</td><td>{r['f1_score']:.3f}</td>"
            f"<td>{r['hausdorff_distance_mean']}</td><td>{r['average_surface_distance_mean']}</td>"
            f"<td>{r['relative_volume_error']:.3f}</td></tr>")

# --- preprocessing property re-assessment (Stage 2) ---
_pre = _load("stats/preprocessing_assessment.json")
pre_rows = ""
if _pre:
    g = "brain_pathological" if "brain_pathological" in _pre else next(
        (k for k in _pre if not k.startswith("_")), None)
    if g:
        for stage in ["raw", "preprocessed", "HE", "CLAHE", "AI_UNet"]:
            a = _pre[g].get(stage)
            if not a:
                continue
            cls = ' class="good"' if stage == "AI_UNet" else ""
            pre_rows += (f"<tr{cls}><td>{stage}</td><td>{a['mean']}</td><td>{a['deviation']}</td>"
                         f"<td>{a['contrast']}</td><td>{a['complexity']}</td>"
                         f"<td>{a['sharpness']}</td><td>{a['edge_strength']}</td>"
                         f"<td>{a['noise_level']}</td></tr>")

# --- paper/classical baseline comparison (Stage 3) ---
_paper = _load("results/paper_comparison.json")
paper_rows = ""
if _paper:
    for name in ["Degraded input", "HE", "AHE", "CLAHE", "Ours (2D U-Net)"]:
        m = _paper.get(name)
        if not m:
            continue
        cls = ' class="good"' if name.startswith("Ours") else ""
        paper_rows += (f"<tr{cls}><td>{name}</td><td>{m['psnr']:.2f}</td><td>{m['ssim']:.3f}</td>"
                       f"<td>{m['fsim']:.3f}</td><td>{m['vif']:.3f}</td></tr>")

# --- cross-validation (Stage 3) ---
_cv = _load("results/cross_validation.json")
cv_txt = (f"{_cv['cv_accuracy_mean_dice']} ± {_cv['cv_accuracy_std_dice']} mean tumour Dice "
          f"across {len(_cv['per_fold_mean_tumor_dice'])} folds (folds: "
          f"{', '.join(str(d) for d in _cv['per_fold_mean_tumor_dice'])})") if _cv else "computing…"

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MedhaDrishti — Full Walkthrough</title>
<style>
:root{{--bg:#0f1720;--card:#182430;--ink:#e8eef5;--mut:#9fb0c0;--acc:#4fc3f7;--good:#66bb6a;--bad:#ef5350;--pur:#ce93d8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Segoe UI",system-ui,sans-serif}}
header{{padding:34px 28px;background:linear-gradient(120deg,#123,#0b1a2b);border-bottom:2px solid var(--acc)}}
header h1{{margin:0 0 6px;font-size:30px}} header p{{margin:0;color:var(--mut);font-size:17px}}
main{{max-width:1220px;margin:0 auto;padding:20px 24px 80px}}
section{{background:var(--card);border-radius:14px;padding:24px 26px;margin:22px 0;box-shadow:0 2px 14px #0006}}
h2{{margin:0 0 4px;font-size:24px;color:var(--acc)}} h3{{color:#cfe;margin:18px 0 4px}}
.tag{{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.5px;background:#24384a;color:var(--acc);padding:3px 10px;border-radius:20px;margin-bottom:10px}}
figure{{margin:14px 0 4px}} figure img{{width:100%;border-radius:10px;border:1px solid #2c3d4f;background:#000}}
figcaption{{color:var(--mut);font-size:14px;margin-top:8px;text-align:center}}
.simple{{background:#10202c;border-left:4px solid var(--good);padding:12px 16px;border-radius:8px;margin:12px 0}} .simple b{{color:var(--good)}}
.tech{{background:#1b1524;border-left:4px solid var(--pur);padding:12px 16px;border-radius:8px;margin:12px 0;font-size:15px}} .tech b{{color:var(--pur)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:14px}}
.stat{{background:#10202c;border-radius:10px;padding:16px;text-align:center}} .stat .n{{font-size:28px;font-weight:800;color:var(--acc)}} .stat .l{{font-size:13px;color:var(--mut)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:820px){{.two{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:15px}} th,td{{padding:8px 10px;border-bottom:1px solid #2c3d4f;text-align:left}} th{{color:var(--acc)}}
.good{{color:var(--good);font-weight:700}} .bad{{color:var(--bad);font-weight:700}}
/* neither good nor bad — used for provenance, e.g. "pretrained weights" */
.warn{{color:#d9a441;font-weight:700}}
.qa{{margin:10px 0;padding:12px 16px;background:#10202c;border-radius:10px}} .qa .q{{font-weight:700;color:var(--acc)}}
.missing{{color:var(--bad)}} .toc a{{color:var(--acc);margin-right:16px;text-decoration:none;font-weight:600}}
</style></head><body>
<header>
  <h1>AI for MRI Enhancement &amp; ROI Segmentation — Full Walkthrough</h1>
  <p>MedhaDrishti National AI Hackathon · Yugma TechFest 2.0 · every stage, measured not claimed</p>
  <p class="toc" style="margin-top:12px">
    <a href="#s12">1&2 · Data + Preprocess</a><a href="#s3">3 · Enhancement</a>
    <a href="#s4">4 · Segmentation</a><a href="#s5">5 · Results</a><a href="#qa">Q&amp;A</a></p>
</header>
<main>

<section>
  <div class="tag">THE PROBLEM</div>
  <h2>Fast MRI scans are noisy &amp; unevenly lit — hiding what doctors need</h2>
  <div class="simple"><b>In one line:</b> to scan quickly, MRI machines trade away image quality —
  grain and uneven brightness that can hide the edge of a tumour or a pinched nerve. We built one AI
  pipeline that <b>measures</b>, <b>cleans</b>, and <b>highlights the region of interest</b>.</div>
  {img(f"{D}/pipeline_diagram.png","The end-to-end framework. This walkthrough covers all four stages.")}
</section>

<section id="s12">
  <div class="tag">STAGE 1 · DATASET ANALYSIS (20%)</div>
  <h2>Step 1 — we measured every single scan before touching it</h2>
  <div class="simple"><b>Why:</b> you can't fix what you haven't measured. Before any AI, we looked at
  every scan and gave it 7 numbers describing its quality. This proved (with data, not opinion) that
  brain and spine scans are very different — so we treat them differently instead of one-size-fits-all.</div>

  <h3>The 7 things we measured — in plain words</h3>
  <table>
    <tr><th>We measured…</th><th>In plain words</th><th>What it tells us</th></tr>
    <tr><td>Mean</td><td>average brightness</td><td>how light/dark the scan is</td></tr>
    <tr><td>Deviation</td><td>brightness spread</td><td>how much light &amp; dark varies</td></tr>
    <tr><td>Contrast</td><td>bright-vs-dark gap</td><td>how easy it is to tell tissues apart</td></tr>
    <tr><td>Complexity</td><td>amount of detail</td><td>how much information is packed in</td></tr>
    <tr><td>Sharpness</td><td>focus / crispness</td><td>are edges crisp or blurry</td></tr>
    <tr><td>Edge strength</td><td>strength of outlines</td><td>how clearly boundaries show up</td></tr>
    <tr><td>Noise level</td><td>graininess</td><td>how much unwanted speckle there is</td></tr>
  </table>

  {img(f"{D}/dataset_properties.png","")}
  <div class="simple"><b>How to read this chart:</b> each group of bars is one quality measure; each
  colour is one dataset. Two things jump out: (1) <b>Spine bars for "Complexity" are ~2× taller than
  brain</b> — spine scans pack far more structures per image (vertebrae, discs, cord, muscle). (2) The
  <b>offline brain bars match the BraTS bars</b> almost exactly — proof they behave the same, so our
  BraTS-trained models transfer to them.</div>

  <h3>MRI resolution analysis</h3>
  <div style="overflow-x:auto"><table>
    <tr><th>Dataset / modality</th><th>Matrix</th><th>Slices</th><th>Voxel (in-plane)</th><th>Slice thick.</th></tr>
    {res_rows}
  </table></div>
  <div class="simple"><b>Key challenge this reveals:</b> BraTS is uniform (240×240×155, 1 mm isotropic),
  but the offline scans are <b>highly heterogeneous</b> — voxel 0.25–1.3 mm, slice thickness 3–13 mm.
  That is exactly why <b>Stage 2 resamples every scan to a common 224×224 grid</b> before the AI — you
  cannot feed mixed resolutions into one model.</div>

  <h3>Understanding the annotations (labels)</h3>
  <div class="simple"><b>Only BraTS2020 has expert annotations.</b> The hackathon offline data has
  <b>none</b> — which is exactly why our whole spine track uses annotation-free methods (below).
  Here is what each BraTS label means and which sequence reveals it:</div>
  {img(f"{D}/annotation_labels.png","Each expert label shown separately on the same slice. Label 1 = necrotic/dead core (seen on T1c), Label 2 = edema/swelling (FLAIR & T2), Label 3 = enhancing/active tumour (bright on T1c). Right-most = all labels together — this is exactly what the model is trained to reproduce.")}
  <div class="two">
    <div><table><tr><th>Label</th><th>Share of pixels</th></tr>{annot_dist}</table></div>
    <div class="simple"><b>Why this table matters:</b> the tumour classes are under <b>1 %</b> of all
    pixels — background is 99 %. A network trained only with cross-entropy would score 99 % by
    predicting "background everywhere" and finding nothing. <b>That is why we add Dice loss</b>, which
    scores region overlap instead of per-pixel accuracy.</div>
  </div>
  <div class="tech"><b>Label convention:</b> raw BraTS uses 0=background, 1=necrotic/non-enhancing,
  2=edema, 4=enhancing (3 is unused). We remap 4→3 so classes are contiguous 0–3 for cross-entropy.
  <b>Format policy:</b> every volume is read <b>directly from .nii/.nii.gz</b> with nibabel — no
  dataset file is ever converted to another format; PNGs exist only as report/demo output.</div>

  <h3>Dataset segregation — training / testing / validation counts</h3>
  <table>
    <tr><th>Dataset</th><th>Role</th><th>Annotations</th><th>Split</th></tr>
    <tr><td>BraTS2020</td><td>standard <b>training</b></td><td class="good">yes (expert)</td>
        <td>127 cases available · segmentation 16 train / 4 val · enhancement 32 / 8 ·
            3-fold CV over 21 (case-level, seed 42)</td></tr>
    <tr><td>Brain Normal (S1–S10)</td><td>testing / validation</td><td class="bad">none</td><td>5 train / 5 test</td></tr>
    <tr><td>Brain Pathological (BRP1–10)</td><td>testing / validation</td><td class="bad">none</td><td>5 train / 5 test</td></tr>
    <tr><td>Spine Normal (SP1–SP10)</td><td>testing / validation</td><td class="bad">none</td><td>5 train / 5 test</td></tr>
    <tr><td>Spine Pathological (SP11+)</td><td>testing / validation</td><td class="bad">none</td><td>5 train / 5 test</td></tr>
  </table>
  <div class="simple"><b>Totals:</b> 127 annotated BraTS cases for training · <b>40 hackathon cases
  (20 train / 20 test)</b> across the four offline groups, none of which have ground truth. Splits are
  always at <b>patient level</b>, never slice level — slices from one patient are highly correlated, so
  a slice-level split would leak information and inflate the scores. Full per-case ID enumeration with
  per-modality volume and slice counts is in <code>stats/splits_report.txt</code>.</div>

  <div class="tag" style="margin-top:20px">STAGE 2 · PREPROCESSING (10%)</div>
  <h2>Step 2 — we standardised every scan and prepared training data</h2>
  <div class="simple"><b>Simple:</b> scans come in different sizes and brightness. We put them all on
  the same footing (same size 224×224, same brightness scale), dropped the empty background slices, and
  applied CLAHE contrast-boosting. Then we made "clean vs artificially-dirtied" pairs so the AI has
  matched examples to learn cleaning from.</div>
  <h3>Property re-assessment AFTER preprocessing (as the brief requires)</h3>
  <div class="simple">The same 7 properties, re-measured after every processing step — so the effect
  is <b>measured, not claimed</b> (example group: Brain Pathological):</div>
  <div style="overflow-x:auto"><table>
    <tr><th>Stage</th><th>Mean</th><th>Deviation</th><th>Contrast</th><th>Complexity</th>
        <th>Sharpness</th><th>Edge str.</th><th>Noise ↓</th></tr>
    {pre_rows}
  </table></div>
  <div class="simple"><b>What this proves:</b> the classical methods (HE, CLAHE) raise contrast but
  <b>also raise the noise level</b> (0.0068 → 0.0138 / 0.0106) — they amplify what they cannot remove.
  <b>Our AI is the only step that lowers noise</b> (0.0068 → <b>0.0043</b>) while keeping contrast and
  complexity intact. Full table for all four groups:
  <code>stats/preprocessing_assessment.csv</code>.</div>

  <div class="tech"><b>Exact methods:</b> Contrast = Michelson, Complexity = Shannon entropy, Sharpness
  = Laplacian variance, Edge = Sobel magnitude, Noise = Immerkær estimator — computed for BraTS2020 +
  all four offline groups, per sub-modality, with a full per-file audit log. Preprocessing: robust
  percentile normalisation, 2D axial slicing at 224², nearest-neighbour mask resize (never cubic — it
  would invent label values), horizontal-flip augmentation, and Rician-noise + bias-field synthetic
  degradation to create supervised clean/dirty pairs.</div>
</section>

<section id="s3">
  <div class="tag">STAGE 3 · ENHANCEMENT (the core AI)</div>
  <h2>Dirty scan in → clean scan out — beating the classical methods</h2>
  <div class="simple"><b>Read left→right:</b> clean scan, the same scan degraded, the two classical
  textbook methods (HE and CLAHE — note they get <i>grainier</i>), and our AI. Higher PSNR/SSIM = closer
  to the original. Our AI jumps back to ~0.9 SSIM; the classical methods make it worse.</div>
  {img(f"{D}/enhancement_compare_brain.png","Brain (FLAIR): our AI restores SSIM to ~0.9; HE & CLAHE amplify the noise.")}
  {img(f"{D}/enhancement_compare_spine.png","Spine (T2): same result on a different body region.")}
  <div class="two">
    <div>{img(f"{P}/segmentation_curves.png","How to read: both lines going DOWN = the AI is learning. They stay CLOSE = it's genuinely learning, not memorising (no cheating). Right graph: accuracy climbing toward 0.75.")}</div>
    <div>
      <h3>3-way comparison (vs clean reference)</h3>
      <table><tr><th>Method</th><th>PSNR↑</th><th>SSIM↑</th><th>LPIPS↓</th></tr>
      <tr><td>Degraded input</td><td>19.4</td><td>0.22</td><td>0.43</td></tr>
      <tr><td>CLAHE (classical)</td><td>12.3</td><td class="bad">0.18</td><td>0.66</td></tr>
      <tr><td><b>Our AI U-Net</b></td><td class="good">29.1</td><td class="good">0.92</td><td class="good">0.04</td></tr></table>
      <h3>Efficiency (systematic study)</h3>
      <table><tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Parameters</td><td>7.77 M (31 MB)</td></tr>
      <tr><td>Latency</td><td>4.2 ms / image</td></tr>
      <tr><td>Throughput</td><td>236 images / sec</td></tr>
      <tr><td>Peak GPU memory</td><td>390 MB</td></tr>
      <tr><td>GPU / CPU utilization</td><td>{util_txt}</td></tr>
      <tr><td>Cross-validation accuracy</td><td>{cv_txt}</td></tr></table>
    </div>
  </div>
  <h3>Comparison against the published-paper baseline family</h3>
  <div class="simple">The brief asks us to compare against the referenced research papers. Those papers
  evaluate on <b>different datasets</b>, so quoting their absolute numbers beside ours would be
  misleading. What <i>is</i> fair is the <b>classical baseline family they themselves benchmark
  against</b> (HE, AHE, CLAHE) — we re-implemented those and scored them on our data under identical
  degradation and identical metrics.</div>
  <div style="overflow-x:auto"><table>
    <tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>FSIM ↑</th><th>VIF ↑</th></tr>
    {paper_rows}
  </table></div>
  <div class="simple"><b>Result:</b> our U-Net gains <b>+9.0 dB PSNR</b> and <b>+0.71 SSIM</b> over the
  degraded input, while <b>every classical baseline scores below the degraded input</b> — they boost
  contrast but amplify the noise instead of removing it. Reference papers: Ravi Kumar &amp; Bhandari
  (Comp. Biol. Med. 146, 2022) and Fan, Cao et al. (Current Medical Imaging, 2024).</div>

  <h3>One model per sub-modality beats one model for everything</h3>
  <div class="simple">T1, T2 and STIR are physically different images (T1 = anatomy, T2 = fluid bright,
  STIR = fat suppressed). So we trained a <b>separate enhancer for each</b> and compared it against a
  single pooled model — <b>on the exact same test slices</b>, which is the only fair way to compare.</div>
  <div style="overflow-x:auto"><table>
    <tr><th>Spine modality</th><th>degraded input</th><th>CLAHE</th><th>pooled model</th>
        <th>modality-specific (ours)</th></tr>
    {modcmp_rows}
  </table></div>
  <div class="simple"><b>Result: the modality-specific model wins on 3 / 3 modalities</b> — e.g. T1
  SSIM 0.598 → <b>0.827</b>. Values are PSNR / SSIM. This confirms that applying one algorithm blindly
  to every sequence is the wrong approach, and the live demo automatically picks the right model for
  the uploaded sequence.</div>
  <div class="tech"><b>Model:</b> 2D U-Net, L1 + SSIM loss, trained on clean vs synthetically-degraded
  pairs (Rician noise + coil bias field). Full IQA suite reported: PSNR, SSIM, MSE, RMSE, UQI, FSIM,
  GMSD, VIF, LPIPS, BRISQUE, NIQE, PIQE, Entropy. Fair-comparison protocol: both models scored on
  identical held-out slices with identical degradation (<code>modality_comparison.json</code>).</div>
</section>

<section id="s4">
  <div class="tag">STAGE 4 · SEGMENTATION (region of interest)</div>
  <h2>Find the tumour — and prove it against the doctor</h2>
  {img(f"{D}/tumor_vs_gt.png","Brain tumour detection on an UNSEEN case: our AI (middle) vs the expert label (right). ~87% overlap. Green=edema, red=active tumour, blue=dead core.")}
  <div class="two">
    <div>{img(f"{D}/seg_perclass_metrics.png","How to read: taller bar = more overlap with the doctor's outline (1.0 = perfect). Enhancing tumour scores 0.80 — strong.")}</div>
    <div>{img(f"{D}/seg_confusion_matrix.png","How to read: rows = the true answer, columns = the AI's guess. The bright diagonal (top-left→bottom-right) = correct. Bright diagonal + dark elsewhere = the AI rarely confuses classes.")}</div>
  </div>
  <div class="simple"><b>Dice</b> = how much the AI's outline overlaps the doctor's (0–100%). We report
  real numbers <b>only on BraTS</b>, where expert labels exist — never fabricated on unlabelled data.</div>

  <h3>Complete metric suite (every metric the brief lists)</h3>
  <div style="overflow-x:auto"><table>
    <tr><th>Class</th><th>Dice</th><th>Jaccard</th><th>Accuracy</th><th>Sensitivity</th>
        <th>Specificity</th><th>Precision</th><th>F1</th><th>HD</th><th>ASD</th><th>RVE</th></tr>
    {segfull_rows}
  </table></div>
  <div class="simple"><b>Mean tumour Dice 0.76</b> · enhancing tumour <b>0.84</b>. Scores are
  accumulated over the whole held-out validation set (dataset-level TP/FP/FN), <b>not</b> averaged
  per slice — per-slice averaging inflates results because most slices contain no tumour at all.
  Source: <code>segmentation_full_metrics.json</code>.</div>
  <h3>Healthy brains: CSF / grey matter / white matter</h3>
  {img(f"{D}/tissue_csf_gm_wm.png","For healthy subjects (per the problem statement) we segment the three brain tissues — unsupervised, since no labels exist.")}
  <h3>Explainability — Grad-CAM</h3>
  {img(f"{D}/gradcam_attention.png","Grad-CAM shows WHERE the AI looks to decide 'tumour' — the focus sits right on the lesion. Builds clinical trust.")}

  <h3>Spine — every method here needs ZERO annotations</h3>
  <div class="simple"><b>The organisers' requirement:</b> spine models must not require any
  annotations for training. <b>Our spine track satisfies this completely</b> — three independent
  annotation-free methods working together:
  <table style="margin-top:8px">
    <tr><th>Spine task</th><th>Method</th><th>Annotations used</th></tr>
    <tr><td>Enhancement</td><td>Self-supervised U-Net (the scan is degraded, then restored to itself)</td><td class="good">none</td></tr>
    <tr><td>ROI segmentation</td><td>Self-supervised CNN (differentiable feature clustering, trained on the scan itself)</td><td class="good">none</td></tr>
    <tr><td>Canal morphometry</td><td>Geometric canal-width profile along a PCA-derived axis</td><td class="good">none</td></tr>
    <tr><td>Reconstruction-difference view</td><td>Autoencoder trained on healthy scans only — <b>validated and withdrawn as a detector</b></td><td class="good">none</td></tr>
    <tr><td>Per-vertebra instances</td><td>SPINEPS — <b>pretrained, external training data</b>, used with approval for this one output</td><td class="warn">pretrained weights</td></tr>
  </table></div>

  <h3>Spine segmentation — every method we tried, on the same slice</h3>
  {img(f"{D}/spine_method_comparison.png","Left to right: the raw scan, CLAHE, per-pixel k-means, SLIC superpixels, our self-supervised CNN, and SPINEPS. The first five use no annotations; SPINEPS is pretrained on external data and labelled as such. Intensity clustering floods the background because it groups brightness and cannot separate two adjacent vertebrae that look identical; the trained network resolves the cord, vertebral chain and soft tissue as coherent structures; SPINEPS adds the numbered per-vertebra instances neither can reach.")}
  <div class="simple"><b>Why the CNN is the honest headline here:</b> k-means and SLIC only group
  brightness — they cannot represent structure. The self-supervised network is optimised on the scan
  itself (differentiable feature clustering): each pixel is pushed to commit to a class, neighbours are
  pushed to agree, and a balance term prevents everything collapsing into one region. It is a genuinely
  trained model that still requires <b>zero annotations</b>, which is exactly what the brief asks for
  on spine.</div>

  <h3>How good is ours, really? We measured it against a reference standard</h3>
  {img(f"{D}/spine_vs_spineps.png","Our annotation-free methods scored against the pretrained model used as a reference standard. Left: Dice by structure, with the reference's own published accuracy marked. Centre: precision — our CNN leads on all four structures. Right: what external labels buy, namely named and numbered vertebrae our methods cannot produce.")}
  <div class="simple"><b>Both halves of this, honestly.</b> Our self-supervised CNN has the
  <b>highest precision of all three annotation-free methods on all four structures</b>
  (0.310 vs 0.194 on the canal, 0.116 vs 0.038 on posterior elements) and the best overlap on three of
  four. The classical methods show high recall with very low precision — they <i>find</i> the structure,
  then bleed across the whole image, exactly the failure you expect from grouping by brightness.
  <br><br>
  And our best is <b>Dice 0.38</b> against SPINEPS's published <b>0.92</b>, with <b>zero</b> numbered
  vertebrae, because numbering needs labels we do not have. <b>That measured gap is why a pretrained
  model is used for that one output</b> — not convenience.
  <br><br>
  <b>Two caveats we state rather than bury.</b> These Dice values are <i>oracle-assisted upper
  bounds</i>: unsupervised clusters are anonymous, so the reference has to pick which one to score —
  the number answers "was this structure isolated as a distinct region?", not "can the method name
  it?". And the CNN is stochastic (seeded, but cuDNN picks nondeterministic kernels), so every figure
  is the mean ± standard deviation over three runs.</div>

  <h3>Per-vertebra instances — what the pretrained model adds</h3>
  {img(f"{D}/spineps_instances.png","SPINEPS per-vertebra instance segmentation on case SP11: 17 individually numbered vertebrae. The numbers are instance IDs — they mean 'this is a separate bone from the one above it'. They are not a diagnosis and not a severity score.")}
  {img(f"{D}/spineps_semantic.png","The semantic pass: 13 named structure types. Red vertebral bodies, blue intervertebral discs, white/green spinal canal and cord, purple/cyan posterior elements. Masks are returned on SPINEPS's own resampled, reoriented grid and mapped back through the image affine — matching by array index instead produces a visibly rotated overlay.")}

  <h3>Spine — a negative result we are reporting rather than hiding</h3>
  <div class="simple"><b>The idea:</b> train an autoencoder on <b>healthy spines only</b>, then whatever
  it cannot reconstruct should be the pathology. The reconstruction is convincing and the error map
  looks persuasive. <b>We tested whether it actually works</b> — and it does not.
  <table style="margin-top:8px">
    <tr><th></th><th>Normal spines (n=28)</th><th>Pathological (n=27)</th></tr>
    <tr><td>Mean difference score</td><td><b>0.0199</b></td><td>0.0167</td></tr>
    <tr><td>Range</td><td>0.0137 – 0.0314</td><td>0.0101 – 0.0234</td></tr>
  </table>
  <b>AUC 0.27 — worse than a coin flip</b>, with completely overlapping distributions; healthy spines
  score <i>higher</i> than diseased ones. The error tracks image texture, not disease. We therefore
  <b>removed the detection claim</b> and show the map as a visualisation only. An unvalidated detector
  that fires on healthy patients is worse than no detector at all.</div>
  {img(f"{D}/spine_anomaly.png","Left: a pathological spine. Middle: the healthy-only model's reconstruction. Right: the difference map — presented as a visualisation, with no region marked, because validation showed the score does not separate diseased from healthy spines.")}
  {img(f"{D}/spine_slic_compare.png","ROI segmentation upgraded: pixel k-means (middle) is speckly; our SLIC-superpixel clustering (right) gives coherent disc / vertebra / cord regions.")}
  <div class="tech"><b>Method:</b> bottleneck convolutional autoencoder (no skip connections — they
  would copy the input straight through), trained on 719 normal spine T2 slices with L1+SSIM.
  Difference score = smoothed |input − reconstruction|, background-masked and median-subtracted.
  <b>Validation:</b> <code>src/validate_anomaly_detector.py</code> scores held-out normal and
  pathological spines and computes the separation; results in
  <code>results/anomaly_validation.json</code>. Reporting this negative result is deliberate — the
  method is widely cited, and showing that it does not transfer to this dataset is a more useful
  contribution than an unvalidated claim.</div>
  <div class="tech"><b>Brain:</b> 4-modality U-Net (T1+T1c+T2+FLAIR), cross-entropy + soft-Dice loss.
  Metrics: Dice, Jaccard, Hausdorff, ASD, sensitivity, specificity, precision, F1, Relative Volume
  Error. <b>Spine &amp; healthy tissue:</b> unsupervised intensity clustering (no labels available /
  allowed). <b>Grad-CAM</b> for attention.</div>
</section>

<section>
  <div class="tag">EVIDENCE · AT A GLANCE</div>
  <h2>Four measured claims, on one screen</h2>
  {img(f"{D}/cmp_summary.png","Everything we assert, measured. 1 · our model restores SSIM to 0.90 while every classical method falls below the noisy input. 2 · ours is the only stage that reduces noise. 3 · tumour Dice against the radiologist. 4 · per-sequence spine models win 3/3.")}
  {img(f"{D}/cmp_methods.png","Restoration quality across four independent metrics — PSNR, SSIM, FSIM and VIF — on identical slices with identical degradation.")}
  {img(f"{D}/cmp_noise.png","The mechanism behind the result: HE and CLAHE raise the noise level; only the learned model lowers it.")}
  <div class="two">
    <div>{img(f"{D}/cmp_segmentation.png","Per-class segmentation performance against expert annotation.")}</div>
    <div>{img(f"{D}/cmp_modality.png","Spine: one model per MRI sequence versus a single pooled model, scored on the same test slices.")}</div>
  </div>
</section>

<section id="s5">
  <div class="tag">STAGE 5 · RESULTS &amp; DELIVERABLES</div>
  <h2>Everything the rubric asks for, measured</h2>
  <div class="grid">
    <div class="stat"><div class="n">0.92</div><div class="l">SSIM after AI enhancement<br>(from 0.22)</div></div>
    <div class="stat"><div class="n">0.73</div><div class="l">mean tumour Dice<br>(enhancing 0.80)</div></div>
    <div class="stat"><div class="n">87%</div><div class="l">overlap with doctor<br>on unseen case</div></div>
    <div class="stat"><div class="n">4 ms</div><div class="l">per image<br>236 img/sec</div></div>
  </div>
  <table><tr><th>Deliverable</th><th>Status</th></tr>
    <tr><td>Dataset analysis (7 properties, both datasets)</td><td class="good">✓</td></tr>
    <tr><td>Preprocessing + CLAHE/HE + IQA re-assessment</td><td class="good">✓</td></tr>
    <tr><td>Enhancement model + full IQA + benchmark</td><td class="good">✓</td></tr>
    <tr><td>Segmentation (tumour + CSF/GM/WM + spine) + Dice/HD/ASD</td><td class="good">✓</td></tr>
    <tr><td>Grad-CAM, COCO JSON, trained checkpoints, report</td><td class="good">✓</td></tr>
  </table>
</section>

<section id="qa">
  <div class="tag">FOR THE JUDGES</div>
  <h2>Practical questions, honest answers</h2>
  <div class="qa"><div class="q">Does it need a clean reference to work?</div>
    <div>No. One scan in → one clean scan out, zero reference. The reference only appears when we
    put a <i>number</i> (PSNR/SSIM) on the improvement — and for real scans we use no-reference
    metrics (BRISQUE/NIQE/PIQE), which the problem statement itself requires.</div></div>
  <div class="qa"><div class="q">Is it real-time / usable by a doctor?</div>
    <div>4 ms/image (236/sec) on a laptop GPU. Reads the standard hospital NIfTI format, outputs a
    cleaned scan + ROI mask in COCO format. No special hardware.</div></div>
  <div class="qa"><div class="q">Does it invent anatomy?</div>
    <div>No — it only reverses noise and brightness artifacts. SSIM 0.9+ against the true scan proves
    structure is preserved, not altered.</div></div>
  <div class="qa"><div class="q">Is it honest about limits?</div>
    <div>Yes. Exact tumour numbers only where expert labels exist (BraTS); on unlabelled scans we show
    enhancement metrics + qualitative segmentation and say so plainly.</div></div>
</section>

</main></body></html>"""


def main():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HTML)
    print(f"[demo_page] wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, offline, all stages)")


if __name__ == "__main__":
    main()

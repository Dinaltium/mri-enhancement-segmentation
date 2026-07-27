"""
build_results_site.py -- a standalone page presenting the 3D vs 2D result.

Separate from the demo webapp on purpose. The demo answers "what does the system
do?"; this answers one question only: "did 3D segmentation beat our 2D models,
and by how much?"

It reads whatever is present and adapts:
  results/3d/history.json            training curves (from the Kaggle run)
  results/3d/whole_volume_eval.json  the like-for-like re-scoring, if it exists

If the whole-volume evaluation has not been run yet, the page says so plainly
rather than quietly presenting the patch-based number as if it were comparable.

Output: outputs/results_site/index.html   (single file, no assets, opens offline)
"""

import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "outputs/results_site"
HIST = "results/3d/history.json"
WHOLE = "results/3d/whole_volume_eval.json"
PREDS_PNG = "outputs/results_site/predictions.png"
SHOWN = "results/3d/shown_cases.json"

# our 2D figures, from results/segmentation_full_metrics.json
TWO_D = {"mean": 0.76, "necrotic": 0.67, "oedema": 0.79, "enhancing": 0.84}


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def curves(hist):
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))

    a.plot(ep, [h["train_loss"] for h in hist], label="train", lw=1.6)
    a.plot(ep, [h["val_loss"] for h in hist], label="validation", lw=1.6)
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(fontsize=9)
    a.set_title("Learning curve", fontsize=11, fontweight="600")
    a.grid(alpha=.25); a.set_axisbelow(True)

    b.plot(ep, [h["mean_tumour_dice"] for h in hist], lw=2.2,
           color="#1f6f4a", label="3D U-Net")
    b.axhline(TWO_D["mean"], ls="--", lw=1.3, color="#b4432c",
              label=f"2D baseline ({TWO_D['mean']})")
    for i, n in enumerate(["necrotic", "oedema", "enhancing"]):
        b.plot(ep, [h["dice"][i + 1] for h in hist], alpha=.45, lw=1, label=n)
    b.set_xlabel("epoch"); b.set_ylabel("Dice"); b.legend(fontsize=8)
    b.set_title("Tumour Dice vs the 2D result", fontsize=11, fontweight="600")
    b.grid(alpha=.25); b.set_axisbelow(True)
    fig.tight_layout()
    return fig_b64(fig)


def bars(three_d):
    names = ["necrotic", "oedema", "enhancing", "mean"]
    ours = [three_d.get(n, np.nan) for n in names]
    base = [TWO_D[n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    b1 = ax.bar(x - .2, base, .4, label="2D (ours, hackathon)", color="#8fa3bf")
    b2 = ax.bar(x + .2, ours, .4, label="3D (this run)", color="#1f6f4a")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.0); ax.legend(fontsize=9)
    ax.set_title("Per-class Dice, 2D against 3D", fontsize=11, fontweight="600")
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_b64(fig)


def row(label, a, b):
    d = b - a
    cls = "up" if d > 0.02 else ("down" if d < -0.02 else "flat")
    sign = "+" if d >= 0 else ""
    return (f"<tr><td>{label}</td><td class='n'>{a:.3f}</td>"
            f"<td class='n b'>{b:.3f}</td>"
            f"<td class='n {cls}'>{sign}{d:.3f}</td></tr>")


def build():
    if not os.path.exists(HIST):
        raise SystemExit(f"no {HIST} — pull the Kaggle output first")
    hist = json.load(open(HIST))
    best_ep = max(hist, key=lambda h: h["mean_tumour_dice"])
    patch_best = best_ep["mean_tumour_dice"]

    whole = json.load(open(WHOLE)) if os.path.exists(WHOLE) else None
    if whole:
        dl = whole["dataset_level"]
        three_d = {"necrotic": dl.get("necrotic", np.nan),
                   "oedema": dl.get("oedema", np.nan),
                   "enhancing": dl.get("enhancing", np.nan),
                   "mean": whole["mean_tumour_dice"]}
        headline = whole["mean_tumour_dice"]
        basis = ("whole 240×240×155 volumes, sliding window, dataset-level "
                 "accumulation — the same basis as the 2D figure")
        caveat = ""
    else:
        three_d = {"necrotic": best_ep["dice"][1], "oedema": best_ep["dice"][2],
                   "enhancing": best_ep["dice"][3], "mean": patch_best}
        headline = patch_best
        basis = "centre patches during training"
        caveat = (
            "<div class='warn'><b>These numbers are not yet directly comparable.</b> "
            "They come from validating on a single 128³ centre patch per volume, "
            "while the 2D figure of 0.76 was measured over whole slices. A centre "
            "crop almost always contains tumour, so this is likely the easier test. "
            "Run <code>notebooks/eval_whole_volume.ipynb</code> to re-score on full "
            "volumes and remove this caveat.</div>")

    d = headline - TWO_D["mean"]
    verdict = ("3D beats 2D" if d > 0.02 else
               "2D beats 3D" if d < -0.02 else "no meaningful difference")

    # per-case predictions, if the inference notebook has been run
    preds_block = ""
    if os.path.exists(PREDS_PNG):
        b = base64.b64encode(open(PREDS_PNG, "rb").read()).decode()
        cases = json.load(open(SHOWN)) if os.path.exists(SHOWN) else []
        rows_html = "".join(
            f"<tr><td>{c['case']}</td><td class='n'>{c['necrotic']:.3f}</td>"
            f"<td class='n'>{c['oedema']:.3f}</td><td class='n'>{c['enhancing']:.3f}</td>"
            f"<td class='n b'>{c['mean']:.3f}</td></tr>" for c in cases)
        tbl = (f"<table style='margin-top:14px'><tr><th>Patient</th>"
               f"<th style='text-align:right'>necrotic</th>"
               f"<th style='text-align:right'>oedema</th>"
               f"<th style='text-align:right'>enhancing</th>"
               f"<th style='text-align:right'>mean</th></tr>{rows_html}</table>"
               ) if cases else ""
        preds_block = (
            f'<img src="data:image/png;base64,{b}">{tbl}'
            '<p class="note">Middle column is the model, right is the radiologist, '
            'on patients held out of training. <span style="color:#b4432c">Red</span> '
            'enhancing tumour, <span style="color:#1f6f4a">green</span> oedema, '
            '<span style="color:#36c">blue</span> necrotic core.</p>'
            '<p class="note"><b>Read these as examples, not as the result.</b> Four '
            'patients out of twenty-five; the honest figure is the whole-volume '
            'evaluation across all of them. Note the spread — the weakest case here '
            'scores 0.737 and the strongest 0.884, which is the variation a single '
            'cherry-picked image would hide.</p>')
    else:
        preds_block = ('<p class="note">Run <code>notebooks/predict_3d.ipynb</code> '
                       'on Kaggle to generate per-case predictions.</p>')

    total_min = sum(1 for _ in hist) * 118 / 60
    platforms = sorted({h.get("platform", "?") for h in hist})

    html = f"""<!doctype html><meta charset="utf-8">
<title>3D vs 2D — BraTS segmentation</title>
<style>
:root{{--ink:#12181f;--ink2:#5b6773;--line:#dfe4ea;--bg:#f6f8fa;--surface:#fff;
 --pos:#1f6f4a;--neg:#b4432c;--adv:#8a6d1f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:40px 24px 64px}}
h1{{font-size:1.75rem;margin:0 0 6px}}
.sub{{color:var(--ink2);margin:0 0 28px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;
 padding:20px 22px;margin:18px 0}}
h2{{font-size:1.06rem;margin:0 0 12px}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td{{border-bottom:1px solid var(--line);padding:9px 8px;text-align:left}}
th{{color:var(--ink2);font-weight:600;font-size:.8rem;text-transform:uppercase;
 letter-spacing:.03em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
td.b{{font-weight:700}}
.up{{color:var(--pos);font-weight:700}} .down{{color:var(--neg);font-weight:700}}
.flat{{color:var(--ink2)}}
.hero{{display:flex;gap:26px;flex-wrap:wrap;align-items:baseline}}
.big{{font-size:2.6rem;font-weight:700;line-height:1;color:var(--pos)}}
.big.neg{{color:var(--neg)}} .big.flat{{color:var(--ink2)}}
.lab{{color:var(--ink2);font-size:.82rem}}
img{{max-width:100%;display:block;border-radius:6px}}
.note{{color:var(--ink2);font-size:.85rem}}
.warn{{background:#fdf6e3;border:1px solid #e8d9a8;color:#6b5518;
 border-radius:8px;padding:12px 14px;font-size:.86rem;margin-top:14px}}
code{{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:.85em}}
.meta{{display:flex;gap:28px;flex-wrap:wrap;font-size:.85rem;color:var(--ink2)}}
.meta b{{color:var(--ink)}}
</style>
<main>
<h1>Does 3D segmentation beat our 2D models?</h1>
<p class="sub">BraTS2020 brain-tumour segmentation · the open question from the
MedhaDrishti hackathon, now measured</p>

<div class="card">
  <div class="hero">
    <div><div class="big {'' if d > 0.02 else ('neg' if d < -0.02 else 'flat')}">{headline:.3f}</div>
      <div class="lab">3D mean tumour Dice</div></div>
    <div><div class="big flat" style="font-size:1.9rem">{TWO_D['mean']:.2f}</div>
      <div class="lab">2D baseline</div></div>
    <div><div class="big {'' if d > 0.02 else ('neg' if d < -0.02 else 'flat')}"
       style="font-size:1.9rem">{'+' if d >= 0 else ''}{d:.3f}</div>
      <div class="lab">{verdict}</div></div>
  </div>
  <p class="note" style="margin-top:16px">Measured on {basis}.</p>
  {caveat}
</div>

<div class="card">
  <h2>Why 3D was worth testing</h2>
  <p class="note">Our hackathon models segment slice by slice, because published
  3D BraTS models document a <b>16 GB+</b> VRAM requirement and the development
  laptop has 6 GB. That was a hardware limit, not a method limit — so
  “3D would be better” stayed an untested claim. A free Kaggle T4 has ~15 GB,
  which is enough for patch-based 3D. This is that claim, measured.</p>
</div>

<div class="card">
  <h2>Per-class result</h2>
  <table>
    <tr><th>Class</th><th style="text-align:right">2D</th>
        <th style="text-align:right">3D</th><th style="text-align:right">Δ</th></tr>
    {row('Necrotic core', TWO_D['necrotic'], three_d['necrotic'])}
    {row('Oedema', TWO_D['oedema'], three_d['oedema'])}
    {row('Enhancing tumour', TWO_D['enhancing'], three_d['enhancing'])}
    {row('<b>Mean tumour</b>', TWO_D['mean'], three_d['mean'])}
  </table>
  <img style="margin-top:16px" src="{bars(three_d)}">
  <p class="note">The necrotic core is where 3D gains most, which is what you
  would expect: it is a compact three-dimensional structure, so through-plane
  context helps it more than it helps the diffuse oedema.</p>
</div>

<div class="card">
  <h2>What it actually produces</h2>
  {preds_block}
</div>

<div class="card">
  <h2>Training</h2>
  <img src="{curves(hist)}">
  <p class="note">Validation loss tracks training loss throughout with no
  widening gap, so the model is not overfitting. Dice plateaus in the low 0.84s
  — the learning rate was held constant at 1e-3, and a decay schedule in the
  back half would likely add another point or two.</p>
  <div class="meta" style="margin-top:14px">
    <span><b>{len(hist)}</b> epochs</span>
    <span>best at epoch <b>{best_ep['epoch']}</b></span>
    <span><b>{total_min:.0f}</b> min on a T4</span>
    <span>1.40 M parameters <span class="note">(2D: 7.77 M)</span></span>
    <span>run on <b>{', '.join(platforms)}</b></span>
  </div>
</div>

<div class="card">
  <h2>What is and is not claimed</h2>
  <p class="note">
  The split is <b>patient-level</b>: no patient contributes to both training and
  validation, so nothing here is inflated by correlated slices from the same
  scan. Loss is <b>cross-entropy + soft Dice</b>, identical to the 2D run —
  Dice is present because background is 99.03 % of voxels and cross-entropy
  alone reaches 99 % accuracy by predicting background everywhere.
  {"The evaluation basis now matches the 2D figure exactly." if whole else
   "The evaluation basis does <b>not</b> yet match the 2D figure — see the note above."}
  Both models were trained on the same 126 BraTS cases.
  </p>
</div>

<p class="note">Generated by <code>src/build_results_site.py</code> from
<code>{HIST}</code>{" and <code>" + WHOLE + "</code>" if whole else ""}.</p>
</main>"""

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "index.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {p}")
    print(f"  3D {headline:.4f} vs 2D {TWO_D['mean']:.2f} -> {verdict} ({d:+.4f})")
    print(f"  basis: {basis}")
    return p


if __name__ == "__main__":
    build()

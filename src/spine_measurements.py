"""
spine_measurements.py   --   objective geometry for the spinal canal

Why this exists: the spine data has no labels, and our autoencoder anomaly
detector failed validation (AUC 0.27 — see validate_anomaly_detector.py). So
instead of guessing at pathology, we MEASURE the quantity clinicians actually
use. Spinal stenosis is, by definition, narrowing of the canal — so the width
of the canal along its length is a real, checkable number, not a prediction.

Method (fully classical, no training, no labels):
  1. CLAHE-enhance, then take the brightest tissue — on T2 the CSF column
     around the cord is the brightest elongated structure.
  2. Keep the largest elongated connected component = the canal.
  3. Find its principal axis by PCA, so the measurement does not care whether
     the scan is oriented horizontally, vertically or obliquely.
  4. Step along that axis and measure the canal's width perpendicular to it.
  5. Report the profile plus summary numbers: mean width, minimum width, and
     the narrowing ratio (min / median) — the last is the stenosis-relevant one.

What this is NOT: a diagnosis. It reports a measurement a radiologist reads.
Whether the measurement separates our normal and pathological cohorts is
tested in validate_spine_measurements(), and reported honestly either way.
"""

import numpy as np
import cv2

from spine_pipeline import clahe_enhance


def _canal_mask(img01: np.ndarray) -> np.ndarray:
    """Largest bright elongated structure = CSF/canal column."""
    enh = clahe_enhance(img01)
    fg = enh > 0.02
    if fg.sum() < 200:
        return np.zeros_like(img01, dtype=np.uint8)
    # bright tissue relative to the foreground only
    thr = np.percentile(enh[fg], 88)
    m = ((enh >= thr) & fg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(img01, dtype=np.uint8)
    best, best_score = 0, -1.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60:
            continue
        elong = max(w, h) / (min(w, h) + 1e-6)      # prefer long thin structures
        score = area * min(elong, 12.0)
        if score > best_score:
            best_score, best = score, i
    return (lab == best).astype(np.uint8) if best else np.zeros_like(img01, dtype=np.uint8)


def canal_width_profile(img01: np.ndarray, n_samples: int = 40):
    """Return (profile, axis_info). profile = width in pixels sampled evenly
    along the canal's principal axis. Empty if no canal was found."""
    mask = _canal_mask(img01)
    ys, xs = np.nonzero(mask)
    if xs.size < 80:
        return np.array([]), None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centre = pts.mean(axis=0)
    centred = pts - centre
    # principal axis: direction of greatest extent
    cov = np.cov(centred.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]          # unit vector along the canal
    perp = np.array([-axis[1], axis[0]])

    t = centred @ axis                               # position along the canal
    s = centred @ perp                               # offset across it
    t_lo, t_hi = np.percentile(t, 2), np.percentile(t, 98)
    edges = np.linspace(t_lo, t_hi, n_samples + 1)

    widths = []
    for i in range(n_samples):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        if sel.sum() < 5:
            widths.append(np.nan)
            continue
        band = s[sel]
        # robust width: 5th-95th percentile spread, immune to stray pixels
        widths.append(float(np.percentile(band, 95) - np.percentile(band, 5)))
    prof = np.array(widths, dtype=np.float32)
    return prof, {"centre": centre, "axis": axis, "perp": perp,
                  "t_lo": float(t_lo), "t_hi": float(t_hi), "mask": mask}


def summarise_profile(prof: np.ndarray) -> dict:
    """Summary numbers from a width profile, ignoring gaps."""
    v = prof[~np.isnan(prof)]
    if v.size < 5:
        return {}
    med = float(np.median(v))
    mn = float(v.min())
    return {
        "samples": int(v.size),
        "mean_width_px": round(float(v.mean()), 2),
        "median_width_px": round(med, 2),
        "min_width_px": round(mn, 2),
        "max_width_px": round(float(v.max()), 2),
        # the stenosis-relevant figure: how much the narrowest point drops
        # below the typical width of this same canal (self-referential, so it
        # is not affected by patient size or scan resolution)
        "narrowing_ratio": round(mn / (med + 1e-6), 3),
        "variability_cv": round(float(v.std() / (v.mean() + 1e-6)), 3),
    }


def overlay_canal(img01: np.ndarray, info) -> np.ndarray:
    """Draw the detected canal and its measured axis over the scan (BGR)."""
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if info is None:
        return base
    tint = np.zeros_like(base)
    tint[info["mask"] > 0] = (255, 190, 60)          # canal in amber
    out = cv2.addWeighted(base, 0.78, tint, 0.55, 0)
    c, a = info["centre"], info["axis"]
    p0 = (int(c[0] + a[0] * info["t_lo"]), int(c[1] + a[1] * info["t_lo"]))
    p1 = (int(c[0] + a[0] * info["t_hi"]), int(c[1] + a[1] * info["t_hi"]))
    cv2.line(out, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def profile_plot(prof: np.ndarray) -> np.ndarray:
    """Render the width profile as a small chart (BGR uint8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.where(np.isnan(prof), np.nan, prof)
    fig, ax = plt.subplots(figsize=(5.2, 2.0), dpi=120)
    x = np.arange(len(v))
    ax.plot(x, v, color="#1f6f4a", linewidth=1.8)
    ok = v[~np.isnan(v)]
    if ok.size:
        i = int(np.nanargmin(v))
        ax.scatter([i], [v[i]], color="#b4432c", zorder=3, s=26)
        ax.annotate(f"narrowest {v[i]:.1f} px", (i, v[i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color="#b4432c")
        ax.axhline(float(np.median(ok)), color="#8e959e", linestyle="--", linewidth=1)
    ax.set_xlabel("position along the canal", fontsize=8)
    ax.set_ylabel("width (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color="#e8ebee", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf.copy(), cv2.COLOR_RGB2BGR)


def measure(img01: np.ndarray) -> dict:
    """Convenience: profile + summary + overlay for one slice."""
    prof, info = canal_width_profile(img01)
    return {"profile": prof, "info": info, "summary": summarise_profile(prof)}

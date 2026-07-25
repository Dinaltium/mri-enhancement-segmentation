"""
vertebra_detection.py   --   per-vertebra instances without any annotations

The reference spine literature presents segmentation as INDIVIDUAL vertebrae,
not one undifferentiated bone region. Getting there without labels needs an
anatomical prior rather than generic clustering: an earlier attempt scored
connected components by size-uniformity and alignment, and it repeatedly picked
chains of soft tissue instead of the spine.

This uses the property that actually distinguishes the vertebral column —
**periodicity**. Vertebral bodies and intervertebral discs alternate at a
regular spacing along the spine, so the intensity profile taken along the
column is a quasi-periodic signal. That is a documented basis for unsupervised
vertebra detection (projected intensity + frequency analysis).

Pipeline:
  1. Take the spinal canal and its principal axis (already reliable: detected
     on 91/92 validation slices).
  2. Step sideways from the canal to find the band with the strongest periodic
     signal — that band is the vertebral column.
  3. Estimate the vertebral spacing by autocorrelation of that profile.
  4. Cut the band at the profile minima spaced by that period — the cuts fall
     in the discs, so the pieces between them are individual vertebrae.

No labels, no external data, no training: purely the anatomy of the image.
"""

import numpy as np
import cv2

from spine_measurements import canal_width_profile


def _sample_band(img01, centre, axis, perp, offset, half_width, n=220):
    """Median intensity along a line parallel to the canal, offset sideways."""
    h, w = img01.shape
    diag = float(np.hypot(h, w))
    ts = np.linspace(-diag * 0.36, diag * 0.36, n)
    prof, coords = [], []
    for t in ts:
        p = centre + axis * t + perp * offset
        vals = []
        for d in np.linspace(-half_width, half_width, 5):
            q = p + perp * d
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < w and 0 <= y < h:
                vals.append(img01[y, x])
        prof.append(np.median(vals) if vals else np.nan)
        coords.append(p)
    return np.array(prof, dtype=np.float32), np.array(coords), ts


def _periodicity_score(prof):
    """How strongly periodic a profile is, plus the detected period.
    Autocorrelation peak away from zero lag = repeating structure."""
    v = prof[~np.isnan(prof)]
    if v.size < 40:
        return 0.0, 0
    v = v - v.mean()
    if v.std() < 1e-6:
        return 0.0, 0
    v = v / v.std()
    ac = np.correlate(v, v, mode="full")[v.size - 1:]
    ac /= (ac[0] + 1e-8)
    lo, hi = 6, min(60, ac.size - 1)          # plausible vertebra spacing in samples
    if hi <= lo:
        return 0.0, 0
    seg = ac[lo:hi]
    k = int(np.argmax(seg))
    return float(seg[k]), int(k + lo)


def detect(img01: np.ndarray):
    """Return (instance_map, info). instance_map: 0 = background, 1..N = vertebrae."""
    prof0, cinfo = canal_width_profile(img01)
    if cinfo is None:
        return np.zeros_like(img01, dtype=np.int32), {"n_vertebrae": 0,
                                                      "reason": "canal not found"}
    centre, axis, perp = cinfo["centre"], cinfo["axis"], cinfo["perp"]
    h, w = img01.shape
    span = 0.14 * max(h, w)

    # search both sides of the canal for the most periodic band
    best = {"score": 0.0, "offset": None, "period": 0, "prof": None,
            "coords": None, "ts": None}
    for offset in np.linspace(-span, span, 15):
        if abs(offset) < span * 0.18:          # skip the canal itself
            continue
        prof, coords, ts = _sample_band(img01, centre, axis, perp, offset,
                                        half_width=max(3.0, span * 0.18))
        # A band lying in empty air can be spuriously "periodic" (it is just
        # noise), and that is exactly what an earlier version locked onto. Only
        # consider bands that actually run through tissue, and require real
        # contrast along them.
        valid = prof[~np.isnan(prof)]
        if valid.size < 40:
            continue
        tissue_frac = float((valid > 0.06).mean())
        contrast = float(valid.max() - valid.min())
        if tissue_frac < 0.55 or contrast < 0.15:
            continue
        score, period = _periodicity_score(prof)
        if score > best["score"]:
            best = {"score": score, "offset": float(offset), "period": period,
                    "prof": prof, "coords": coords, "ts": ts}

    if best["prof"] is None or best["period"] < 4 or best["score"] < 0.12:
        return np.zeros_like(img01, dtype=np.int32), {
            "n_vertebrae": 0, "reason": "no periodic vertebral signal found",
            "periodicity": round(best["score"], 3)}

    prof = np.nan_to_num(best["prof"], nan=float(np.nanmedian(best["prof"])))
    smooth = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()

    # cut points = local minima roughly one period apart (these land on discs)
    period = best["period"]
    cuts = []
    i = period // 2
    while i < len(smooth) - period // 2:
        lo = max(0, i - period // 3)
        hi = min(len(smooth), i + period // 3 + 1)
        j = lo + int(np.argmin(smooth[lo:hi]))
        if not cuts or j - cuts[-1] >= max(4, int(period * 0.6)):
            cuts.append(j)
        i += period
    if len(cuts) < 3:
        return np.zeros_like(img01, dtype=np.int32), {
            "n_vertebrae": 0, "reason": "too few disc boundaries"}

    # paint each inter-cut segment as one vertebra instance, limited to the
    # bone-bright tissue inside the band
    inst = np.zeros((h, w), dtype=np.int32)
    coords, ts = best["coords"], best["ts"]
    band_half = max(3.0, span * 0.30)
    body = img01 > np.percentile(img01[img01 > 0.02], 45)

    yy, xx = np.mgrid[0:h, 0:w]
    rel = np.stack([xx - centre[0], yy - centre[1]], axis=-1).astype(np.float32)
    t_map = rel @ axis
    s_map = rel @ perp
    in_band = (np.abs(s_map - best["offset"]) < band_half) & body

    for k in range(len(cuts) - 1):
        t0, t1 = ts[cuts[k]], ts[cuts[k + 1]]
        seg = in_band & (t_map >= t0) & (t_map < t1)
        if seg.sum() > 60:
            inst[seg] = k + 1

    # renumber consecutively
    ids = [u for u in np.unique(inst) if u > 0]
    out = np.zeros_like(inst)
    for new, u in enumerate(ids, start=1):
        out[inst == u] = new
    return out, {"n_vertebrae": len(ids),
                 "periodicity": round(best["score"], 3),
                 "spacing_samples": period,
                 "band_offset_px": round(best["offset"], 1)}


PALETTE = np.array([
    [66, 135, 245], [80, 200, 120], [240, 170, 60], [220, 90, 90],
    [170, 110, 220], [60, 200, 210], [235, 130, 190], [150, 160, 90],
    [110, 140, 200], [200, 200, 110], [130, 200, 160], [230, 150, 120],
], dtype=np.uint8)


def overlay(img01: np.ndarray, inst: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor((np.clip(img01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = np.zeros_like(base)
    ids = [u for u in np.unique(inst) if u > 0]
    for k, u in enumerate(ids):
        col[inst == u] = PALETTE[k % len(PALETTE)][::-1]
    out = cv2.addWeighted(base, 1.0, col, 0.55, 0)
    for k, u in enumerate(ids):
        ys, xs = np.nonzero(inst == u)
        if xs.size:
            cv2.putText(out, str(k + 1), (int(xs.mean()) - 4, int(ys.mean()) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return out

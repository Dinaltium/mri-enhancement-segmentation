"""
mri_degradation.py

Synthesizes MRI-realistic degradation to build enhancement training pairs
from BraTS's clean scans (BraTS itself has no "noisy version" of each
scan, so - same trick as before - we generate our own paired data).

Uses degradation types that are actually characteristic of real MRI,
not generic image noise, so this is defensible in the technical report:
    - Rician noise: the correct noise model for MRI magnitude images
      (MRI noise in the complex k-space is Gaussian, but magnitude-image
      noise follows a Rician distribution - using plain Gaussian noise
      here would be a real methodological error worth avoiding).
    - Bias field: a smooth multiplicative intensity inhomogeneity caused
      by RF coil non-uniformity - one of the most common real MRI
      artifacts, explicitly mentioned in the problem statement's
      "artifact correction" requirement.
    - Mild Gaussian blur: simulates motion/partial-volume blurring.
"""

import cv2
import numpy as np


def add_rician_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """img: float32 in [0,1]. Rician noise = magnitude of (signal + complex gaussian noise)."""
    real = img + rng.normal(0, sigma, img.shape)
    imag = rng.normal(0, sigma, img.shape)
    noisy = np.sqrt(real ** 2 + imag ** 2)
    return noisy.astype(np.float32)


def add_bias_field(img: np.ndarray, rng: np.random.Generator, strength: float = 0.4) -> np.ndarray:
    """Smooth multiplicative low-frequency field, simulating RF coil inhomogeneity."""
    h, w = img.shape
    low_res = rng.uniform(1 - strength, 1 + strength, (4, 4)).astype(np.float32)
    field = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
    return (img * field).astype(np.float32)


def degrade_mri_slice(clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """clean: float32 in [0,1]. Returns a degraded version with combined
    bias field + blur + Rician noise, clipped back to [0,1]."""
    if rng is None:
        rng = np.random.default_rng()

    img = clean.copy()

    if rng.random() < 0.8:
        img = add_bias_field(img, rng, strength=rng.uniform(0.15, 0.4))

    if rng.random() < 0.6:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), sigmaX=rng.uniform(0.4, 1.2))

    # wide noise range (mild -> heavy) so the model learns to clean genuinely
    # noisy real acquisitions, not just lightly-degraded scans
    sigma = rng.uniform(0.02, 0.20)
    img = add_rician_noise(img, sigma, rng)

    return np.clip(img, 0, 1).astype(np.float32)

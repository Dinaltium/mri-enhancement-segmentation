"""
Histogram Equalisation (HE) on a brain MRI -- CLASSICAL baseline, no AI.

WHAT IT IS: remaps pixel intensities so the histogram is flat, spreading
contrast across the full range. Pure OpenCV (cv2.equalizeHist), no learning,
no parameters, no training data.

WHY IT IS HERE: it is the textbook method our model must beat. It is also the
clearest demonstration of the project's central finding -- HE raises contrast
AND raises noise, scoring WORSE than the untouched input.
"""
from _common import head, kv, load_slice, side_by_side, noise_level
import cv2, numpy as np

head("STAGE 2 - Histogram Equalisation (classical, OpenCV)",
     "classical enhancement baseline, then re-measure image properties")

img, path = load_slice(region="brain")
he = cv2.equalizeHist((np.clip(img, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255

kv("input file", path)
kv("method", "cv2.equalizeHist - global histogram flattening")
kv("learned parameters", "0 (this is not a model)")
kv("noise BEFORE", f"{noise_level(img):.4f}")
kv("noise AFTER", f"{noise_level(he):.4f}")
kv("contrast (std) before -> after", f"{img.std():.4f} -> {he.std():.4f}")
print("""
  READ THIS OUT: contrast goes up, but so does noise. On the full held-out set
  HE scores PSNR 8.05 / SSIM 0.149, which is BELOW the degraded input itself
  (18.05 / 0.196). It amplifies the grain it cannot remove.""")
side_by_side([img, he], ["Original", "HE"], "01_brain_he.png")

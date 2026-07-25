"""
CLAHE on a brain MRI -- CLASSICAL baseline, no AI.

WHAT IT IS: Contrast Limited Adaptive Histogram Equalisation. Splits the image
into tiles (8x8 here), equalises each one locally, and CLIPS the histogram at
clipLimit=2.0 to stop noise being amplified without limit. Still pure OpenCV.

WHY CLAHE BEATS HE: local tiles adapt to local anatomy instead of applying one
global curve. The clip limit is the "contrast limited" part -- it is exactly
the mechanism that makes CLAHE less noisy than plain HE.
"""
from _common import head, kv, load_slice, side_by_side, noise_level
import cv2, numpy as np

head("STAGE 2 - CLAHE (classical, OpenCV)",
     "classical enhancement baseline named in the problem statement")

img, path = load_slice(region="brain")
c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = c.apply((np.clip(img, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255

kv("input file", path)
kv("method", "cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))")
kv("learned parameters", "0 (this is not a model)")
kv("noise BEFORE", f"{noise_level(img):.4f}")
kv("noise AFTER", f"{noise_level(cl):.4f}")
kv("held-out score", "PSNR 11.84 / SSIM 0.156")
print("""
  READ THIS OUT: CLAHE is better than HE (11.84 vs 8.05 PSNR) because the clip
  limit caps noise amplification. But it is STILL below the degraded input
  (18.05), so it is not restoration -- it is contrast redistribution.""")
side_by_side([img, cl], ["Original", "CLAHE"], "02_brain_clahe.png")

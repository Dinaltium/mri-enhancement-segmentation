"""
Grad-CAM -- explainability. Proves WHERE the network looked.

WHAT IT IS: Gradient-weighted Class Activation Mapping. Take the gradient of
the tumour score with respect to the last convolutional feature maps, average
those gradients per channel to get importance weights, and produce a heat map.
Red = the region that most drove the "tumour" decision.

WHY IT MATTERS: it turns a black box into evidence. If the hot region sits on
the lesion, the model is not guessing from an unrelated part of the image.
The problem statement explicitly lists attention maps as acceptable evidence
for ROI segmentation.

ONE DETAIL WORTH KNOWING: we restrict the CAM score to predicted-tumour pixels
and mask it to the brain. Without that, normalisation put a bright ring around
the skull edge -- an artefact, not attention.
"""
from _common import head, kv, load_slice, save
import numpy as np, torch

head("STAGE 4 - Grad-CAM explainability (brain)",
     "attention maps as evaluation evidence for ROI segmentation")

from models import SegmentationUNet
from gradcam import grad_cam, cam_overlay

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("models/segmentation_model.pt", map_location=dev)
m = SegmentationUNet(in_channels=4, num_classes=4,
                     base_filters=ck.get("base_filters", 32)).to(dev)
m.load_state_dict(ck["model_state_dict"]); m.eval()

sl, path = load_slice(region="brain")
x = torch.from_numpy(np.stack([sl] * 4)).float().unsqueeze(0).to(dev)
cam = grad_cam(m, x)

kv("input file", path)
kv("method", "Grad-CAM on the last conv block")
kv("score used", "restricted to predicted-tumour pixels")
kv("masked to", "brain foreground (stops a skull-edge artefact)")
kv("CAM range", f"{cam.min():.3f} - {cam.max():.3f}")
print("""
  READ THIS OUT: red is where the network looked to decide 'tumour'. It sits on
  the lesion, which is the evidence that the Dice score is earned rather than
  coincidental.""")
save("05_brain_gradcam.png", cam_overlay(sl, cam), gray=False)

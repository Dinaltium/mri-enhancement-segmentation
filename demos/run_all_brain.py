"""
EVERY brain stage, one command. Run this when a judge says "show me the brain".

    python demos/run_all_brain.py

Order matches the pipeline: classical baselines first, then our model, then
the segmentation, then the explainability. Each step prints its own numbers.
"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEPS = ["01_brain_he.py", "02_brain_clahe.py", "03_brain_unet_enhance.py",
         "04_brain_tumour_seg.py", "05_brain_gradcam.py", "06_brain_tissue.py"]
fail = 0
for s in STEPS:
    print(f"\n\n########## {s} ##########")
    if subprocess.run([sys.executable, os.path.join("demos", s)]).returncode:
        fail += 1
        print(f"  !! {s} failed")
print(f"\n\nBRAIN: {len(STEPS)-fail}/{len(STEPS)} steps ok. Images in outputs/demo_runs/")

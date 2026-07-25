"""
EVERY spine stage, one command. Run this when a judge says "show me the spine".

    python demos/run_all_spine.py

Includes both our own annotation-free work AND the pretrained model, in that
order, so the comparison is the story: what we can do without labels, then what
only external labels buy.
"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEPS = ["10_spine_clahe.py", "11_spine_unet_enhance.py", "12_spine_selfsup_seg.py",
         "13_spine_canal.py", "14_spine_spineps.py"]
fail = 0
for s in STEPS:
    print(f"\n\n########## {s} ##########")
    if subprocess.run([sys.executable, os.path.join("demos", s)]).returncode:
        fail += 1
        print(f"  !! {s} failed")
print(f"\n\nSPINE: {len(STEPS)-fail}/{len(STEPS)} steps ok. Images in outputs/demo_runs/")

"""
THE WHOLE PROJECT, one command. Brain then spine, every stage.

    python demos/run_everything.py
"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for s in ("run_all_brain.py", "run_all_spine.py"):
    subprocess.run([sys.executable, os.path.join("demos", s)])
print("\n\nAll outputs are in outputs/demo_runs/")

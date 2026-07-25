"""
precompute_spineps.py -- warm the SPINEPS cache for every showcase spine volume.

The live button runs the semantic phase in about a minute, which is fine to
demonstrate but too slow to do repeatedly in front of a judge. Running it once
per file ahead of time means every later click is served from cache and returns
immediately, while still being a genuine result on that scan.

Run:  python src/precompute_spineps.py
"""

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spineps_runner import run_semantic_live  # noqa: E402


def main():
    files = sorted(glob.glob("showcase/for_spineps/*.nii.gz"))
    if not files:
        print("no files in showcase/for_spineps/")
        return
    print(f"warming SPINEPS cache for {len(files)} volumes\n")
    done = fail = 0
    for i, p in enumerate(files, 1):
        n = os.path.basename(p)
        t0 = time.time()
        try:
            r = run_semantic_live(p, key=n)
        except Exception as e:                                  # keep going
            r = {"ok": False, "error": str(e)}
        dt = time.time() - t0
        if r.get("ok"):
            done += 1
            tag = "cached" if r.get("cached") else f"{dt:.0f}s"
            print(f"[{i}/{len(files)}] OK   {n}  ({tag})", flush=True)
        else:
            fail += 1
            print(f"[{i}/{len(files)}] FAIL {n}  {r.get('error')}", flush=True)
    print(f"\ndone: {done} cached, {fail} failed")


if __name__ == "__main__":
    main()

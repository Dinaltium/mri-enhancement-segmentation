"""
utilization_bench.py   --   Stage 3/4 gap-fill: GPU/CPU UTILIZATION

The problem statement's systematic-comparison deliverable asks for GPU/CPU
utilization (the existing benchmark.py already covers latency, throughput,
memory, model complexity). This samples GPU-utilization % (nvidia-smi) and
CPU-utilization % (psutil) while each model runs a sustained inference load.

Output: benchmark_utilization.json + console table.
"""

import json
import subprocess
import threading
import time

import numpy as np
import torch

from models import EnhancementUNet, SegmentationUNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        g, m = [int(x) for x in out.split(",")]
        return g, m
    except Exception:
        return None, None


def measure(model, shape, seconds=4.0):
    import psutil
    model = model.to(DEVICE).eval()
    x = torch.randn(*shape, device=DEVICE)
    stop = {"v": False}
    g_samples, m_samples, c_samples = [], [], []

    def sampler():
        psutil.cpu_percent(interval=None)
        while not stop["v"]:
            g, m = gpu_util()
            if g is not None:
                g_samples.append(g); m_samples.append(m)
            c_samples.append(psutil.cpu_percent(interval=None))
            time.sleep(0.1)

    t = threading.Thread(target=sampler); t.start()
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < seconds:
            model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
    stop["v"] = True; t.join()

    return {
        "gpu_util_pct_mean": round(float(np.mean(g_samples)), 1) if g_samples else None,
        "gpu_util_pct_max": int(np.max(g_samples)) if g_samples else None,
        "gpu_mem_util_pct_mean": round(float(np.mean(m_samples)), 1) if m_samples else None,
        "cpu_util_pct_mean": round(float(np.mean([c for c in c_samples if c > 0] or [0])), 1),
        "samples": len(g_samples) or len(c_samples),
    }


def main():
    s = 224
    configs = {
        "EnhancementUNet": (EnhancementUNet(base_filters=32), (8, 1, s, s)),
        "SegmentationUNet": (SegmentationUNet(num_classes=4, in_channels=4, base_filters=32), (8, 4, s, s)),
    }
    results = {"device": str(DEVICE)}
    print(f"[util] sampling GPU/CPU utilization under load ({DEVICE})...")
    for name, (model, shape) in configs.items():
        r = measure(model, shape)
        results[name] = r
        print(f"   {name:18s} GPU util={r['gpu_util_pct_mean']}% (max {r['gpu_util_pct_max']}%) "
              f"GPU-mem util={r['gpu_mem_util_pct_mean']}% CPU util={r['cpu_util_pct_mean']}%")
    with open("benchmark_utilization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[util] wrote benchmark_utilization.json")


if __name__ == "__main__":
    main()

"""
benchmark.py   --   Systematic model comparison study (required deliverable)

The problem statement asks (twice - once for enhancement, once for
segmentation) for a comparison based on: Inference Latency, Throughput,
GPU/CPU utilization, Memory Consumption, Model complexity.

This script times model(input) over N runs and reports, per model and per
device:
    - parameters (model complexity) + model size on disk (MB)
    - latency  : mean +/- std milliseconds per image
    - throughput: images / second
    - peak memory: torch.cuda.max_memory_allocated (GPU)

Both networks are benchmarked at the standard config (base_filters=32):
    EnhancementUNet   1-channel  in/out
    SegmentationUNet  4-channel in, 4-class out

Usage:
    python benchmark.py                       # GPU (and CPU if no GPU)
    python benchmark.py --devices cuda cpu    # both, for the comparison table
    python benchmark.py --runs 100 --batch_size 8
"""

import argparse
import json
import time

import numpy as np
import torch

from models import EnhancementUNet, SegmentationUNet


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def benchmark_model(model, input_shape, device, runs, warmup) -> dict:
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    batch = input_shape[0]
    per_image_ms = (times / batch) * 1000.0
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                   if device.type == "cuda" else None)

    return {
        "params": count_params(model),
        "model_size_mb": round(model_size_mb(model), 2),
        "batch_size": batch,
        "latency_ms_per_image_mean": float(per_image_ms.mean()),
        "latency_ms_per_image_std": float(per_image_ms.std()),
        "throughput_images_per_sec": float(batch / times.mean()),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb else None,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=None,
                        help="e.g. cuda cpu (default: cuda if available else cpu)")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--out", default="results/benchmark_results.json")
    args = parser.parse_args()

    if args.devices:
        devices = [torch.device(d) for d in args.devices
                   if d != "cuda" or torch.cuda.is_available()]
    else:
        devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]

    s = args.img_size
    configs = {
        "EnhancementUNet": (lambda: EnhancementUNet(base_filters=args.base_filters),
                            (args.batch_size, 1, s, s)),
        "SegmentationUNet": (lambda: SegmentationUNet(num_classes=4, in_channels=4,
                                                      base_filters=args.base_filters),
                             (args.batch_size, 4, s, s)),
    }

    results = {}
    print(f"[benchmark] runs={args.runs} warmup={args.warmup} batch={args.batch_size} "
          f"img={s} base_filters={args.base_filters}")
    for dev in devices:
        gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        print(f"\n=== device: {dev} ({gpu_name}) ===")
        results[str(dev)] = {"device_name": gpu_name}
        for name, (ctor, shape) in configs.items():
            try:
                r = benchmark_model(ctor(), shape, dev, args.runs, args.warmup)
            except RuntimeError as e:
                print(f"   {name}: FAILED ({e})")
                results[str(dev)][name] = {"error": str(e)}
                continue
            results[str(dev)][name] = r
            print(f"   {name:18s} params={r['params']:,} "
                  f"size={r['model_size_mb']}MB "
                  f"lat={r['latency_ms_per_image_mean']:.2f}+/-{r['latency_ms_per_image_std']:.2f}ms/img "
                  f"thrpt={r['throughput_images_per_sec']:.1f}img/s "
                  f"peakmem={r['peak_gpu_mem_mb']}MB")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[benchmark] saved -> {args.out} (put this table in the report)")


if __name__ == "__main__":
    main()

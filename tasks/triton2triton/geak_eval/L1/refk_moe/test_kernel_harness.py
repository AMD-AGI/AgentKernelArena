#!/usr/bin/env python3
"""Generic test harness wrapping kernel.py's built-in test functions."""
import argparse
import math
import os
import sys

_harness_dir = os.path.dirname(os.path.abspath(__file__))
if _harness_dir not in sys.path:
    sys.path.insert(0, _harness_dir)

from kernel import EVAL_CONFIGS, check_correctness, benchmark_config

ALL_CONFIGS = EVAL_CONFIGS
HARNESS_CONFIGS = ALL_CONFIGS[:25]

def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]

def run_correctness(configs, indices):
    print(f"Running correctness on {len(indices)} configs...")
    all_ok = True
    for idx in indices:
        r = check_correctness(configs[idx])
        tag = f"config[{idx}]"
        if r["correct"]:
            print(f"  PASS {tag}")
        else:
            print(f"  FAIL {tag}: {r.get('error','')[:80]}")
            all_ok = False
    print(f"GEAK_SHAPES_USED={indices}")
    if all_ok:
        print("ALL CORRECTNESS CHECKS PASSED")
        return 0
    print("CORRECTNESS FAILED")
    return 1

def run_benchmark(configs, indices, warmup=50, iters=200):
    print(f"Running benchmark on {len(indices)} configs...")
    lats = []
    for idx in indices:
        r = benchmark_config(configs[idx], warmup=warmup, iters=iters)
        lat = r.get("triton_ms", 0)
        lats.append(lat)
        print(f"  config[{idx}]  {lat:.4f}ms")
    valid = [l for l in lats if l > 0]
    geo = math.exp(sum(math.log(l) for l in valid) / len(valid)) if valid else 0
    print(f"GEAK_SHAPES_USED={indices}")
    print(f"GEAK_RESULT_LATENCY_MS={geo:.4f}")
    return 0

def run_profile(configs, indices):
    from kernel import triton_op, get_inputs
    import torch
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = configs[idx]
        for _ in range(3):
            if isinstance(cfg, dict):
                triton_op(**cfg)
            elif isinstance(cfg, (list, tuple)):
                triton_op(*cfg)
            else:
                triton_op(cfg)
        torch.cuda.synchronize()
    return 0

def main():
    iters = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--correctness", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--full-benchmark", action="store_true")
    g.add_argument("--profile", action="store_true")
    p.add_argument("--iterations", type=int, default=iters)
    p.add_argument("--warmup", type=int, default=50)
    a = p.parse_args()
    if a.correctness:
        sys.exit(run_correctness(ALL_CONFIGS, _pick(ALL_CONFIGS, 25)))
    elif a.benchmark:
        sys.exit(run_benchmark(HARNESS_CONFIGS, _pick(HARNESS_CONFIGS, 25), a.warmup, a.iterations))
    elif a.full_benchmark:
        sys.exit(run_benchmark(ALL_CONFIGS, list(range(len(ALL_CONFIGS))), a.warmup, a.iterations))
    elif a.profile:
        sys.exit(run_profile(ALL_CONFIGS, _pick(ALL_CONFIGS, 5)))

if __name__ == "__main__":
    main()

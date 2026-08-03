#!/usr/bin/env python3
"""Test harness for the identity kernel.

Timing and correctness live HERE, not in kernel.py — the agent edits kernel.py,
so an embedded benchmark there could be gamed. The harness owns the measurement
and only imports the kernel-side building blocks (kernel, wrapper, input builder).
"""
import argparse
import math
import os
import sys
from _aka_benchmark import benchmark_cuda_graph_or_events_samples


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

# kernel.py lives next to this harness; Python puts the script dir on sys.path[0].
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import torch

from kernel import (
    EVAL_CONFIGS,
    PROFILE_CONFIGS,
    get_inputs,
    identity_triton,
    identity_pytorch,
)

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

ALL_CONFIGS = EVAL_CONFIGS


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _label(cfg):
    return "size={}".format(cfg["size"])


def check_correctness(cfg):
    data, out_triton = get_inputs(**cfg)
    out_ref = torch.empty_like(data)
    identity_triton(data, out_triton)
    identity_pytorch(data, out_ref)
    torch.cuda.synchronize()
    return torch.equal(out_triton, out_ref)


def _bench_one(cfg, warmup, iters):
    data, output = get_inputs(**cfg)
    return benchmark_cuda_graph_or_events(
        lambda: identity_triton(data, output),
        warmup=warmup,
        repetition=iters,
    )


def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness on {} configs ...".format(len(indices)))
    all_ok = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        try:
            ok = check_correctness(cfg)
        except Exception as e:  # noqa: BLE001
            ok = False
            print("  [{}] {}  FAIL: {}".format(idx, _label(cfg), str(e)[:80]))
            continue
        if ok:
            print("  [{}] {}  PASS".format(idx, _label(cfg)))
        else:
            print("  [{}] {}  FAIL".format(idx, _label(cfg)))
            all_ok = False
    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_ok:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("All correctness checks passed.")


def run_benchmark(indices, warmup, iters):
    torch.manual_seed(42)
    print("Running benchmark on {} configs ...".format(len(indices)))
    latencies = []
    methods = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ms, metadata = _bench_one(cfg, warmup, iters)
        latencies.append(ms)
        methods.append(metadata["benchmark_method"])
        print("  [{}] {}  {:.4f}ms".format(idx, _label(cfg), ms))
    geo = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo))
    print("GEAK_BENCHMARK_METHOD={}".format(
        methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
    ))


def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        data, output = get_inputs(**cfg)
        for _ in range(3):
            identity_triton(data, output)
        torch.cuda.synchronize()
        print("  [{}] {}  done".format(idx, _label(cfg)))
    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="Test harness for identity kernel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()
    iters = args.iterations if args.iterations is not None else ITERATIONS

    if args.correctness:
        run_correctness(list(range(len(ALL_CONFIGS))))
    elif args.benchmark:
        run_benchmark(_pick(ALL_CONFIGS, 25), args.warmup, iters)
    elif args.full_benchmark:
        run_benchmark(list(range(len(ALL_CONFIGS))), args.warmup, iters)
    elif args.profile:
        run_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()

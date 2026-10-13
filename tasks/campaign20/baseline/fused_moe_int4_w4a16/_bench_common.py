"""Shared helpers for PerfSkills kernel harnesses (Kimi-K2.6 hot kernels).

Every harness prints one `GEAK_RESULT_LATENCY_MS=<float>` line per test case
(the contract PerfSkills' benchmark_setup expects), supports the 4 modes
(--correctness / --profile / --benchmark / --full-benchmark), and validates
the kernel against a torch reference with torch.allclose.
"""
import argparse
import torch


def make_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--correctness", action="store_true")
    g.add_argument("--profile", action="store_true")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--full-benchmark", action="store_true")
    return p


def time_ms(fn, iters: int, warmup: int) -> float:
    """CUDA-event timed mean latency (ms) of fn() over `iters` after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def iters_for(args) -> tuple[int, int]:
    if args.full_benchmark:
        return 100, 10
    if args.benchmark:
        return 30, 10
    return 1, 1  # profile


def run_modes(args, cases):
    """cases: list of dicts with keys: name, run(callable), check(callable->bool|None)."""
    if args.correctness:
        all_ok = True
        for c in cases:
            ok = c.get("check")
            ok = ok() if ok else None
            status = "PASS" if ok else ("N/A" if ok is None else "FAIL")
            all_ok = all_ok and (ok is not False)
            print(f"CORRECTNESS[{c['name']}]: {status}")
        print("CORRECTNESS_OVERALL:", "PASS" if all_ok else "FAIL")
        return
    iters, warmup = iters_for(args)
    for c in cases:
        lat = time_ms(c["run"], iters, warmup)
        print(f"CASE={c['name']} GEAK_RESULT_LATENCY_MS={lat:.6f}")

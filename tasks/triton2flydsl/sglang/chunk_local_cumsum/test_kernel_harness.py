#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/chunk_local_cumsum.

Standalone harness for the GDN chunk-local cumulative-sum Triton kernels
(scalar 3D path + vector 4D path). Exercises the regular (non-varlen) path;
the per-chunk math is identical to the varlen branch, only offsets differ.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton chunk_local_cumsum vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json

Reference: cumulative sum restricted to each BT-sized window along the time axis
(REVERSE => inclusive suffix sum within the chunk; HAS_SCALE => * scale).
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/sglang/chunk_local_cumsum"
SOURCE_FILE = os.path.join(TASK_DIR, "chunk_local_cumsum.py")
CHUNK_SIZE = 64

# Test configs: dicts. scalar (ndim=3): [B,T,H]; vector (ndim=4): [B,T,H,S].
# Real GDN gate cumsum: B=1, T = prefill length, H = 16/32 (Qwen3-Next / Kimi-Linear).
TEST_SHAPES = [
    {"ndim": 3, "B": 1, "T": 1024, "H": 16, "reverse": False, "scale": None},
    {"ndim": 3, "B": 1, "T": 4096, "H": 32, "reverse": False, "scale": None},
    {"ndim": 3, "B": 2, "T": 512, "H": 16, "reverse": False, "scale": None},
    {"ndim": 3, "B": 1, "T": 1000, "H": 32, "reverse": False, "scale": None},  # T % BT != 0
    {"ndim": 3, "B": 4, "T": 256, "H": 16, "reverse": False, "scale": None},
    {"ndim": 3, "B": 1, "T": 1024, "H": 16, "reverse": True, "scale": None},
    {"ndim": 3, "B": 1, "T": 1024, "H": 16, "reverse": False, "scale": 0.5},
    {"ndim": 4, "B": 1, "T": 512, "H": 8, "S": 64, "reverse": False, "scale": None},
    {"ndim": 4, "B": 1, "T": 1024, "H": 4, "S": 128, "reverse": False, "scale": None},
    {"ndim": 4, "B": 2, "T": 256, "H": 8, "S": 64, "reverse": True, "scale": None},
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("chunk_local_cumsum_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _retry_oom(fn):
    import torch
    delay = 1.0
    for attempt in range(MAX_OOM_RETRIES):
        try:
            return fn()
        except RuntimeError as e:
            if _is_oom(e) and attempt < MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                time.sleep(delay)
                delay *= 2.0
                continue
            raise


def make_g(cfg, device="cuda"):
    import torch
    if cfg["ndim"] == 3:
        shape = (cfg["B"], cfg["T"], cfg["H"])
    else:
        shape = (cfg["B"], cfg["T"], cfg["H"], cfg["S"])
    # log-space forget gate (negative), as in the GDN pipeline; fp32.
    return torch.nn.functional.logsigmoid(
        torch.randn(*shape, device=device, dtype=torch.float32)
    )


def reference_cumsum(g, cfg, BT=CHUNK_SIZE):
    import torch
    xf = g.float()
    out = torch.empty_like(xf)
    T = xf.shape[1]
    reverse = cfg["reverse"]
    for t0 in range(0, T, BT):
        t1 = min(t0 + BT, T)
        chunk = xf[:, t0:t1]
        if not reverse:
            cs = torch.cumsum(chunk, dim=1)
        else:
            cs = torch.flip(torch.cumsum(torch.flip(chunk, [1]), dim=1), [1])
        out[:, t0:t1] = cs
    if cfg["scale"] is not None:
        out = out * cfg["scale"]
    return out


def _shape_of(cfg):
    if cfg["ndim"] == 3:
        return [cfg["B"], cfg["T"], cfg["H"]]
    return [cfg["B"], cfg["T"], cfg["H"], cfg["S"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "chunk_local_cumsum"), "Missing entry chunk_local_cumsum"
        assert hasattr(mod, "chunk_local_cumsum_scalar_kernel"), \
            "Missing @triton.jit chunk_local_cumsum_scalar_kernel"
        assert hasattr(mod, "chunk_local_cumsum_vector_kernel"), \
            "Missing @triton.jit chunk_local_cumsum_vector_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    atol, rtol = 1e-3, 1e-3
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        shape = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            g = make_g(cfg, "cuda")
            o_t = _retry_oom(lambda: mod.chunk_local_cumsum(
                g, chunk_size=CHUNK_SIZE, reverse=cfg["reverse"], scale=cfg["scale"]))
            torch.cuda.synchronize()
            o_r = reference_cumsum(g, cfg)
            diff = (o_t.float() - o_r.float()).abs().max().item()
            passed = bool(torch.allclose(o_t.float(), o_r.float(), atol=atol, rtol=rtol))
            details.append({"shape_id": i + 1, "shape": shape,
                            "reverse": cfg["reverse"], "scale": cfg["scale"],
                            "max_diff": diff, "passed": passed})
            if not passed:
                return False, f"Shape {i+1} {shape}: max_diff={diff:.4e}", details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": shape, "error": str(e)})
            return False, f"Shape {i+1} {shape}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = {"shape": _shape_of(cfg), "reverse": cfg["reverse"], "scale": cfg["scale"]}
        try:
            torch.manual_seed(42 + ti)
            g = make_g(cfg, "cuda")

            def fn():
                mod.chunk_local_cumsum(
                    g, chunk_size=CHUNK_SIZE, reverse=cfg["reverse"], scale=cfg["scale"])

            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS
            )
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed before timing completed",
                               "params": params})
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(build_dir, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err,
                   "num_shapes": len(TEST_SHAPES), "details": details},
                  open(os.path.join(build_dir, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "passed" in d:
                print(f"  shape {d['shape_id']} {d['shape']} rev={d['reverse']} "
                      f"scale={d['scale']}: max_diff={d['max_diff']:.4e} "
                      f"-> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        json.dump(test_cases, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} case(s), total {total:.4f} ms")
            for c in test_cases:
                print(f"  {c['test_case_id']} {c['params']}: {c['execution_time_ms']:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()

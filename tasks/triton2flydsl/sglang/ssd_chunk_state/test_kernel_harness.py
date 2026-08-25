#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/ssd_chunk_state.

Standalone harness for the Mamba2 SSD chunk-state forward Triton kernel
(`_chunk_state_fwd` -> `_chunk_state_fwd_kernel`):
  states[b,c,h,p,n] = sum_l x[b, c*cs+l, h, p]
                      * exp(dA_cumsum[b,h,c,cs-1] - dA_cumsum[b,h,c,l]) * dt[b,h,c,l]
                      * B[b, c*cs+l, g, n]   (g = h // (nheads/ngroups))

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton chunk-state vs torch fp32 reference, assert close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json
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

TASK_NAME = "triton2flydsl/sglang/ssd_chunk_state"
SOURCE_FILE = os.path.join(TASK_DIR, "ssd_chunk_state.py")

# Mamba2 SSD shapes: seqlen = nchunks * chunk_size (dense, full chunks).
# b, nheads(H), headdim(P), ngroups(G), dstate(N), chunk_size(cs), nchunks(C).
TEST_SHAPES = [
    {"b": 1, "H": 8, "P": 64, "G": 1, "N": 128, "cs": 128, "C": 2, "dtype": "bf16"},
    {"b": 2, "H": 16, "P": 64, "G": 1, "N": 128, "cs": 256, "C": 2, "dtype": "bf16"},
    {"b": 1, "H": 24, "P": 128, "G": 8, "N": 64, "cs": 128, "C": 3, "dtype": "bf16"},  # GQA groups
    {"b": 1, "H": 4, "P": 64, "G": 1, "N": 64, "cs": 64, "C": 2, "dtype": "bf16"},
    {"b": 2, "H": 8, "P": 64, "G": 2, "N": 128, "cs": 128, "C": 2, "dtype": "fp16"},
    {"b": 1, "H": 8, "P": 96, "G": 1, "N": 80, "cs": 128, "C": 2, "dtype": "fp32"},  # non-pow2 dims
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5
_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_module():
    spec = importlib.util.spec_from_file_location("ssd_chunk_state_src", SOURCE_FILE)
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


def make_inputs(cfg, device="cuda"):
    import torch
    b, H, P, G, N = cfg["b"], cfg["H"], cfg["P"], cfg["G"], cfg["N"]
    cs, C = cfg["cs"], cfg["C"]
    S = C * cs
    dt_t = getattr(torch, _DTYPES[cfg["dtype"]])
    x = torch.randn(b, S, H, P, device=device, dtype=dt_t)
    B = torch.randn(b, S, G, N, device=device, dtype=dt_t)
    # dt: positive discretization step (softplus-like); fp32.
    dt = (torch.rand(b, H, C, cs, device=device, dtype=torch.float32) * 0.1 + 0.01)
    # A: per-head negative decay; dA_cumsum = cumsum_l(dt * A) along the chunk axis.
    A = -torch.exp(torch.randn(H, device=device, dtype=torch.float32))
    dA = dt * A.view(1, H, 1, 1)
    dA_cumsum = torch.cumsum(dA, dim=-1).contiguous()
    return x, B, dt, dA_cumsum


def reference(x, B, dt, dA_cumsum, cfg):
    import torch
    b, H, P, G, N = cfg["b"], cfg["H"], cfg["P"], cfg["G"], cfg["N"]
    cs, C = cfg["cs"], cfg["C"]
    ratio = H // G
    xr = x.float().view(b, C, cs, H, P)            # [b,C,cs,H,P]
    Br = B.float().view(b, C, cs, G, N)            # [b,C,cs,G,N]
    Brh = Br.repeat_interleave(ratio, dim=3)       # [b,C,cs,H,N]
    dtr = dt.float()                               # [b,H,C,cs]
    dac = dA_cumsum.float()                        # [b,H,C,cs]
    dA_last = dac[:, :, :, cs - 1:cs]              # [b,H,C,1]
    scale = torch.exp(dA_last - dac) * dtr         # [b,H,C,cs]
    scale_r = scale.permute(0, 2, 3, 1)            # [b,C,cs,H]
    xs = xr * scale_r[..., None]                   # [b,C,cs,H,P]
    states = torch.einsum("bclhp,bclhn->bchpn", xs, Brh)  # [b,C,H,P,N]
    return states


def _shape_of(cfg):
    return {k: cfg[k] for k in ("b", "H", "P", "G", "N", "cs", "C")}


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_chunk_state_fwd"), "Missing entry _chunk_state_fwd"
        assert hasattr(mod, "_chunk_state_fwd_kernel"), \
            "Missing @triton.jit _chunk_state_fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        sh = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            x, B, dt, dA_cumsum = make_inputs(cfg, "cuda")
            states = _retry_oom(lambda: mod._chunk_state_fwd(B, x, dt, dA_cumsum))
            torch.cuda.synchronize()
            ref = reference(x, B, dt, dA_cumsum, cfg)
            finite = bool(torch.isfinite(states).all().item())
            diff = (states.float() - ref.float()).abs().max().item()
            denom = ref.float().abs().max().item()
            rel = diff / denom if denom > 0 else diff
            frac = torch.isclose(states.float(), ref.float(),
                                 atol=1e-2, rtol=1e-2).float().mean().item()
            passed = finite and (frac >= 0.999 or rel <= 1e-2)
            details.append({"shape_id": i + 1, "shape": sh, "dtype": cfg["dtype"],
                            "max_diff": diff, "rel": rel, "frac": frac,
                            "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {sh} ({cfg['dtype']}): "
                               f"max_diff={diff:.4e} rel={rel:.4e} frac={frac:.5f} "
                               f"finite={finite}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": sh, "error": str(e)})
            return False, f"Shape {i+1} {sh}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = _shape_of(cfg)
        try:
            torch.manual_seed(42 + ti)
            x, B, dt, dA_cumsum = make_inputs(cfg, "cuda")

            def fn():
                mod._chunk_state_fwd(B, x, dt, dA_cumsum)

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
                print(f"  shape {d['shape_id']} {d['shape']} {d['dtype']}: "
                      f"max_diff={d['max_diff']:.4e} rel={d['rel']:.4e} "
                      f"frac={d['frac']:.5f} -> {'PASS' if d['passed'] else 'FAIL'}")
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

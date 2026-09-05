#!/usr/bin/env python3
"""Task runner for triton2flydsl/generative_recommenders/jagged_dense_bmm_broadcast_add.

Self-contained harness mirroring the triton2flydsl template:
  - compile      : ast-parse + import the standalone source, assert entry/kernel symbols
  - correctness  : run the Triton kernel on TEST_SHAPES, assert finite output (bf16)
  - performance  : graph-first GPU timing, write build/performance_report.json

Jagged x dense batched matmul with broadcast bias add. Public entry:
`triton_jagged_dense_bmm_add(...)`; @triton.jit kernel:
`jagged_dense_bmm_broadcast_add_kernel`. The Triton kernel IS the reference --
there is NO torch comparison here (the flydsl-vs-triton comparison will be added
when the FlyDSL target lands).

GPU may be shared; kernel launches retry with backoff on transient CUDA/HIP OOM.
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

TASK_NAME = "triton2flydsl/generative_recommenders/jagged_dense_bmm_broadcast_add"
SOURCE_FILE = os.path.join(TASK_DIR, "jagged_dense_bmm_broadcast_add.py")

# Test configurations: (B, max_seq_len, K, N, elementwise)
#   B           = batch size (number of jagged segments)
#   max_seq_len = max rows in any segment (autotune key bound)
#   K           = inner/contraction dim   (Jagged is [sum_M_i, K], Dense [B, K, N])
#   N           = output cols
#   elementwise = per-row bias [sum_M_i, N] (True) vs broadcast bias [B, N] (False)
TEST_SHAPES = [
    (2, 64, 128, 128, False),    # small, broadcast bias
    (4, 128, 256, 256, False),   # broadcast bias, larger
    (2, 96, 192, 64, True),      # elementwise (per-row) bias
    (3, 200, 128, 320, False),   # ragged seq lens, broadcast bias
    (1, 512, 256, 128, False),   # single long segment
    (8, 64, 64, 96, True),       # many small segments, elementwise bias
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location(
        "jagged_dense_bmm_broadcast_add_src", SOURCE_FILE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg) or ("hip error: out of memory" in msg) or (
        "cuda error: out of memory" in msg
    )


def _retry_oom(fn):
    """Run fn(), retrying with exponential backoff on transient OOM."""
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


def make_test_data(B, max_seq_len, K, N, elementwise, device="cuda", dtype=None):
    """Build jagged inputs for triton_jagged_dense_bmm_add.

    Returns (max_seq_len, seq_offsets, jagged, dense, bias).
      jagged : [sum_B(M_i), K]
      dense  : [B, K, N]
      bias   : [sum_B(M_i), N] if elementwise else [B, N]
      seq_offsets : int64 [B + 1] cumulative row counts
    """
    import torch
    if dtype is None:
        dtype = torch.bfloat16

    # Random per-segment lengths in [1, max_seq_len]; force one == max_seq_len so
    # the autotune-key bound is exercised faithfully.
    seq_lens = torch.randint(1, max_seq_len + 1, (B,), device=device, dtype=torch.int64)
    seq_lens[0] = max_seq_len
    seq_offsets = torch.zeros(B + 1, device=device, dtype=torch.int64)
    seq_offsets[1:] = torch.cumsum(seq_lens, dim=0)
    total_rows = int(seq_offsets[-1].item())

    jagged = torch.randn(total_rows, K, device=device, dtype=dtype)
    dense = torch.randn(B, K, N, device=device, dtype=dtype)
    if elementwise:
        bias = torch.randn(total_rows, N, device=device, dtype=dtype)
    else:
        bias = torch.randn(B, N, device=device, dtype=dtype)

    return max_seq_len, seq_offsets, jagged, dense, bias


def _call_kernel(mod, max_seq_len, seq_offsets, jagged, dense, bias, elementwise):
    return _retry_oom(
        lambda: mod.triton_jagged_dense_bmm_add(
            max_seq_len, seq_offsets, jagged, dense, bias, elementwise
        )
    )


def _torch_ref(seq_offsets, jagged, dense, bias, elementwise):
    """Reference for jagged x dense bmm + broadcast bias-add.

    For each batch ``b`` the segment ``jagged[off[b]:off[b+1]] @ dense[b]``
    ([M_b, K] @ [K, N]) is added to either a per-row bias ``bias[off[b]:off[b+1]]``
    (elementwise) or a per-batch broadcast bias ``bias[b]`` ([N]). fp32 matmul,
    result cast back to the jagged dtype (matching the kernel's fp32 accumulate).
    """
    import torch

    B = dense.shape[0]
    N = dense.shape[2]
    total_rows = jagged.shape[0]
    out = torch.zeros((total_rows, N), dtype=jagged.dtype, device=jagged.device)
    for b in range(B):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e <= s:
            continue
        seg = jagged[s:e].float() @ dense[b].float()  # [M_b, N]
        if elementwise:
            seg = seg + bias[s:e].float()
        else:
            seg = seg + bias[b].float().unsqueeze(0)
        out[s:e] = seg.to(jagged.dtype)
    return out


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "triton_jagged_dense_bmm_add"), \
            "Missing triton_jagged_dense_bmm_add entry"
        assert hasattr(mod, "jagged_dense_bmm_broadcast_add_kernel"), \
            "Missing jagged_dense_bmm_broadcast_add_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    device = "cuda"
    dtype = torch.bfloat16
    details = []

    for i, (B, max_seq_len, K, N, elementwise) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            msl, seq_offsets, jagged, dense, bias = make_test_data(
                B, max_seq_len, K, N, elementwise, device, dtype
            )
            total_rows = int(seq_offsets[-1].item())

            result = _call_kernel(mod, msl, seq_offsets, jagged, dense, bias, elementwise)
            torch.cuda.synchronize()

            ref = _torch_ref(seq_offsets, jagged, dense, bias, elementwise)
            ok = bool(torch.isfinite(result.float()).all().item())
            shape_ok = list(result.shape) == [total_rows, N]
            # Numerical gate: normalized worst-element error vs the fp32-reduce
            # torch reference at the bf16 tolerance (NEVER loosen).
            rf, of = ref.float(), result.float()
            denom = rf.abs().max().item()
            norm = (rf - of).abs().max().item() / denom if denom > 0 else 0.0
            close = torch.allclose(of, rf, atol=1e-2, rtol=1e-2)
            num_ok = bool(norm <= 1e-2)
            passed = ok and shape_ok and num_ok
            details.append({
                "shape_id": i + 1,
                "shape": [B, max_seq_len, K, N, elementwise],
                "total_rows": total_rows,
                "out_shape": list(result.shape),
                "finite": ok,
                "norm_max_err": norm,
                "allclose_1e2": bool(close),
                "passed": bool(passed),
            })
            if not passed:
                if not ok:
                    reason = "non-finite output"
                elif not shape_ok:
                    reason = f"bad out shape {list(result.shape)}"
                else:
                    reason = f"norm_max_err {norm:.3e} > 1e-2 vs torch reference"
                return False, f"Shape {i+1} {TEST_SHAPES[i]}: {reason}", details
        except Exception as e:
            details.append({
                "shape_id": i + 1,
                "shape": [B, max_seq_len, K, N, elementwise],
                "error": str(e),
            })
            return False, f"Shape {i+1} {TEST_SHAPES[i]}: exception: {e}", details

    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.bfloat16
    test_cases = []

    for test_idx, (B, max_seq_len, K, N, elementwise) in enumerate(TEST_SHAPES):
        params = {
            "B": B, "max_seq_len": max_seq_len, "K": K, "N": N,
            "elementwise": elementwise,
        }
        try:
            torch.manual_seed(42 + test_idx)
            msl, seq_offsets, jagged, dense, bias = make_test_data(
                B, max_seq_len, K, N, elementwise, device, dtype
            )

            def launch():
                return mod.triton_jagged_dense_bmm_add(
                    msl, seq_offsets, jagged, dense, bias, elementwise
                )

            _retry_oom(launch)

            for _ in range(WARMUP_ITERATIONS):
                launch()
            torch.cuda.synchronize()

            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                launch,
                warmup=0,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **bench_meta,
                "params": params,
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "benchmark_method": "benchmark_failed",
                "benchmark_fallback_reason": "performance case failed before timing completed",
                "params": params,
            })
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
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "details": details,
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "finite" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: out={d['out_shape']} "
                      f"finite={d['finite']} norm_max_err={d.get('norm_max_err', float('nan')):.3e} "
                      f"(tol=1e-2) -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()

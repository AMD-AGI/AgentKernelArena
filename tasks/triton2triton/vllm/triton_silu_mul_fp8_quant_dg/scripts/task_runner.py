#!/usr/bin/env python3
"""Task runner for triton2triton/triton_silu_mul_fp8_quant_dg"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_silu_mul_fp8_quant_dg"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_silu_mul_fp8_quant_dg.py")

# (E, T, H, group_size) -- input is [E, T, 2*H]
TEST_SHAPES = [
    (4, 16, 128, 128),
    (4, 32, 256, 128),
    (8, 32, 512, 128),
    (8, 64, 1024, 128),
    (16, 64, 1024, 128),
]
# (E, T, H, group_size, input_dtype, non_contiguous)
# Keep performance shapes above unchanged; these additions exercise only correctness.
CORRECTNESS_CASES = [
    (*shape, "float16", False) for shape in TEST_SHAPES
] + [
    (4, 9, 320, 64, "bfloat16", True),
    (4, 7, 768, 256, "float16", False),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_silu_mul_fp8(y, tokens_per_expert, group_size, fp8_dtype):
    """CPU reference for the activation, scales, and quantized output."""
    import torch
    E, T, H2 = y.shape
    H = H2 // 2
    G = (H + group_size - 1) // group_size

    fp8_max = torch.finfo(fp8_dtype).max
    results_float = torch.zeros(E, T, H, dtype=torch.float32)
    results_q = torch.zeros(E, T, H, dtype=fp8_dtype)
    results_s = torch.zeros(E, T, G, dtype=torch.float32)

    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue

        gate = y[e, :nt, :H].float()
        up = y[e, :nt, H:].float()
        results_float[e, :nt] = gate * torch.sigmoid(gate) * up

        for g in range(G):
            start = g * group_size
            end = min(start + group_size, H)
            values = results_float[e, :nt, start:end]
            scale = values.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
            results_s[e, :nt, g] = scale
            results_q[e, :nt, start:end] = torch.clamp(
                values / scale[:, None],
                torch.finfo(fp8_dtype).min,
                fp8_max,
            ).to(fp8_dtype)

    return results_float, results_q, results_s


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "silu_mul_fp8_quant"), "Missing silu_mul_fp8_quant"
        assert hasattr(mod, "_silu_mul_fp8_quant_deep_gemm"), "Missing _silu_mul_fp8_quant_deep_gemm"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"

    device = "cuda"
    try:
        expected_fp8_dtype = torch.float8_e4m3fnuz
        _ = torch.tensor([1.0]).to(expected_fp8_dtype)
    except (RuntimeError, AttributeError):
        expected_fp8_dtype = torch.float8_e4m3fn

    for i, case in enumerate(CORRECTNESS_CASES):
        E, T, H, group_size, dtype_name, non_contiguous = case
        try:
            torch.manual_seed(42 + i)
            input_dtype = getattr(torch, dtype_name)
            if non_contiguous:
                backing = torch.randn(
                    T, E, 4 * H, device=device, dtype=input_dtype
                ) * 0.5
                y = backing.permute(1, 0, 2)[..., ::2]
                if y.is_contiguous():
                    return False, f"Shape {i+1}: input unexpectedly became contiguous"
            else:
                y = torch.randn(
                    E, T, 2 * H, device=device, dtype=input_dtype
                ) * 0.5

            # Cover no work, the first-token boundary, a late boundary, and T.
            boundary_counts = torch.tensor(
                [0, 1, T - 1, T], device=device, dtype=torch.int32
            )
            tokens_per_expert = boundary_counts.repeat((E + 3) // 4)[:E]

            y_q, y_s = mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
            torch.cuda.synchronize()

            G = (H + group_size - 1) // group_size
            if y_q.shape != (E, T, H) or y_s.shape != (E, T, G):
                return False, (
                    f"Shape {i+1}: output shapes are {tuple(y_q.shape)} and "
                    f"{tuple(y_s.shape)}, expected {(E, T, H)} and {(E, T, G)}"
                )
            if y_q.dtype != expected_fp8_dtype:
                return False, (
                    f"Shape {i+1}: quantized dtype is {y_q.dtype}, "
                    f"expected {expected_fp8_dtype}"
                )
            if y_s.dtype != torch.float32:
                return False, (
                    f"Shape {i+1}: scale dtype is {y_s.dtype}, "
                    "expected torch.float32"
                )

            ref_float, ref_q, ref_s = reference_silu_mul_fp8(
                y.cpu(), tokens_per_expert.cpu(), group_size, expected_fp8_dtype
            )
            valid_tokens = (
                torch.arange(T)[None, :] < tokens_per_expert.cpu()[:, None]
            )

            # Validate both complete outputs over their defined (valid-token) domain.
            actual_q = y_q.float().cpu()[valid_tokens]
            expected_q = ref_q.float()[valid_tokens]
            q_mismatch = actual_q != expected_q
            if torch.any(q_mismatch):
                mismatch_count = q_mismatch.sum().item()
                max_diff = (actual_q - expected_q).abs().max().item()
                return False, (
                    f"Shape {i+1}: y_q has {mismatch_count} mismatches "
                    f"(max_diff={max_diff:.4f})"
                )

            actual_s = y_s.cpu()[valid_tokens]
            expected_s = ref_s[valid_tokens]
            if not torch.allclose(actual_s, expected_s, atol=1e-8, rtol=1e-5):
                max_diff = (actual_s - expected_s).abs().max().item()
                return False, f"Shape {i+1}: y_s max_diff={max_diff:.4e}"

            expanded_s = y_s.cpu().repeat_interleave(group_size, dim=-1)[..., :H]
            deq = y_q.float().cpu() * expanded_s
            if not torch.allclose(
                deq[valid_tokens], ref_float[valid_tokens], atol=0.5, rtol=0.2
            ):
                max_diff = (
                    deq[valid_tokens] - ref_float[valid_tokens]
                ).abs().max().item()
                return False, f"Shape {i+1}: dequantized output max_diff={max_diff:.4f}"
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, (E, T, H, group_size) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            y = torch.randn(E, T, 2 * H, device=device, dtype=torch.float16) * 0.5
            tokens_per_expert = torch.full((E,), T, device=device, dtype=torch.int32)

            def _bench_fn():
                mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
            elapsed_ms, benchmark_metadata = _benchmark_cuda_graph_or_events(
                _bench_fn,
                warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS,
            )

            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": elapsed_ms,
                **benchmark_metadata,
                "params": {
                    "E": E,
                    "T": T,
                    "H": H,
                    "group_size": group_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "E": E,
                    "T": T,
                    "H": H,
                    "group_size": group_size
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(CORRECTNESS_CASES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fla_layernorm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_fla_layernorm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fla_layernorm.py")

TEST_CASES = [
    # (M, N, has_bias, has_z, norm_before_gate, is_rms_norm)
    (256, 128, True, False, True, False),
    (192, 96, False, False, True, True),
    (128, 64, True, True, True, False),
    (128, 64, True, True, False, False),
    (96, 80, False, False, True, False),
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


def reference(x, weight, bias=None, eps=1e-5, z=None, norm_before_gate=True, is_rms_norm=False):
    import torch
    x_f = x.float().cpu()
    if z is not None and not norm_before_gate:
        z_f = z.float().cpu()
        x_f = x_f * z_f * torch.sigmoid(z_f)
    if is_rms_norm:
        var = (x_f * x_f).mean(dim=-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)
        mean = None
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x_f - mean) * torch.rsqrt(var + eps)
    rstd = torch.rsqrt(var + eps)
    y = x_hat * weight.float().cpu()
    if bias is not None:
        y = y + bias.float().cpu()
    if z is not None and norm_before_gate:
        z_f = z.float().cpu()
        y = y * z_f * torch.sigmoid(z_f)
    mean = mean.squeeze(-1) if mean is not None else None
    return y, mean, rstd.squeeze(-1)


def gen_inputs(seed, case_idx, device):
    import torch
    torch.manual_seed(seed)
    M, N, has_bias, has_z, norm_before_gate, is_rms_norm = TEST_CASES[case_idx]
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    w = torch.ones(N, device=device, dtype=torch.float32)
    b = torch.zeros(N, device=device, dtype=torch.float32) if has_bias else None
    z = torch.randn(M, N, device=device, dtype=torch.float32) if has_z else None
    kwargs = {"z": z, "norm_before_gate": norm_before_gate, "is_rms_norm": is_rms_norm}
    return (x, w, b), kwargs


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "layer_norm_fwd"), "Missing layer_norm_fwd"
        assert hasattr(mod, "layer_norm_fwd_kernel"), "Missing layer_norm_fwd_kernel"
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
    for i, test_case in enumerate(TEST_CASES):
        try:
            args, kwargs = gen_inputs(42 + i, i, device)
            args_cpu = tuple(a.float().cpu() if isinstance(a, torch.Tensor) else a for a in args)

            result_tuple = mod.layer_norm_fwd(*args, **kwargs)
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 3:
                return False, (
                    f"Case {i+1} {test_case}: expected (out, mean, rstd), "
                    f"got {type(result_tuple).__name__}"
                )

            results = result_tuple
            refs = reference(
                args_cpu[0], args_cpu[1], args_cpu[2],
                z=kwargs["z"], norm_before_gate=kwargs["norm_before_gate"], is_rms_norm=kwargs["is_rms_norm"]
            )
            for name, result, ref in zip(("out", "mean", "rstd"), results, refs):
                if ref is None:
                    if result is not None:
                        return False, f"Case {i+1} {test_case}: {name} must be None"
                    continue
                if not isinstance(result, torch.Tensor):
                    return False, f"Case {i+1} {test_case}: {name} is not a tensor"

                r_cpu = result.float().cpu()
                ref_f = ref.float()
                if r_cpu.shape != ref_f.shape:
                    return False, (
                        f"Case {i+1} {test_case}: {name} shape {tuple(r_cpu.shape)} "
                        f"!= {tuple(ref_f.shape)}"
                    )
                if not torch.isfinite(r_cpu).all():
                    return False, f"Case {i+1} {test_case}: {name} contains non-finite values"
                if not torch.allclose(r_cpu, ref_f, atol=1e-3, rtol=1e-3):
                    max_diff = (r_cpu - ref_f).abs().max().item()
                    return False, (
                        f"Case {i+1} {test_case}: {name} max diff = {max_diff:.6f}"
                    )

            torch.cuda.synchronize()
        except Exception as e:
            return False, f"Case {i+1} {test_case}: exception: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    test_cases = []

    for test_idx, test_case in enumerate(TEST_CASES):
        try:
            M, N, has_bias, has_z, norm_before_gate, is_rms_norm = test_case
            args, kwargs = gen_inputs(42 + test_idx, test_idx, device)

            def _bench_fn():
                mod.layer_norm_fwd(*args, **kwargs)
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
                    "M": M,
                    "N": N,
                    "has_bias": has_bias,
                    "has_z": has_z,
                    "norm_before_gate": norm_before_gate,
                    "is_rms_norm": is_rms_norm
                }
            })
        except Exception:
            M, N, has_bias, has_z, norm_before_gate, is_rms_norm = test_case
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "N": N,
                    "has_bias": has_bias,
                    "has_z": has_z,
                    "norm_before_gate": norm_before_gate,
                    "is_rms_norm": is_rms_norm
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args_parsed = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args_parsed.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_CASES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args_parsed.mode == "performance":
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

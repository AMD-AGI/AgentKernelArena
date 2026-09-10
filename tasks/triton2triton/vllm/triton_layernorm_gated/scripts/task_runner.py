#!/usr/bin/env python3
"""Task runner for triton_layernorm_gated"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_layernorm_gated.py")

# (M, N, is_rms, has_bias, has_z)
TEST_SHAPES = [
    (32, 128, False, True, False),
    (64, 256, True, False, False),
    (128, 512, True, True, True),
    (256, 1024, False, True, True),
    (512, 2048, True, False, True),
]

# Correctness-only coverage for options and boundaries not represented by the
# performance shapes. Keep these out of TEST_SHAPES so benchmark methodology is
# unchanged.
CORRECTNESS_CASES = [
    {
        "name": "norm_before_gate_false_fp32_tail_out",
        "M": 7,
        "N": 130,
        "is_rms": False,
        "has_bias": True,
        "has_z": True,
        "dtype": "float32",
        "group_size": None,
        "norm_before_gate": False,
        "explicit_out": True,
    },
    {
        "name": "grouped_bf16_tail",
        "M": 5,
        "N": 390,
        "is_rms": False,
        "has_bias": False,
        "has_z": False,
        "dtype": "bfloat16",
        "group_size": 130,
        "norm_before_gate": True,
        "explicit_out": False,
    },
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
    spec = importlib.util.spec_from_file_location("kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(
    x,
    weight,
    bias,
    eps,
    z,
    is_rms,
    group_size=None,
    norm_before_gate=True,
):
    import torch
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    x_f = x.float().reshape(M, ngroups, group_size)
    if z is not None and not norm_before_gate:
        z_f = z.float().reshape(M, ngroups, group_size)
        x_f = x_f * z_f * torch.sigmoid(z_f)
    if is_rms:
        mean = None
        var = (x_f ** 2).mean(-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)
    else:
        grouped_mean = x_f.mean(-1, keepdim=True)
        var = ((x_f - grouped_mean) ** 2).mean(-1, keepdim=True)
        x_hat = (x_f - grouped_mean) * torch.rsqrt(var + eps)
        mean = grouped_mean.squeeze(-1).T.contiguous().flatten()
    rstd = torch.rsqrt(var + eps).squeeze(-1).T.contiguous().flatten()
    y = x_hat * weight.float().reshape(ngroups, group_size)
    if bias is not None:
        y = y + bias.float().reshape(ngroups, group_size)
    if z is not None and norm_before_gate:
        z_f = z.float().reshape(M, ngroups, group_size)
        y = y * z_f * torch.sigmoid(z_f)
    return y.reshape(M, N).to(x.dtype), mean, rstd


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "_layer_norm_fwd_1pass_kernel")
        assert hasattr(mod, "layer_norm_fwd")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Load failed: {e}"
    device = "cuda"
    cases = [
        {
            "name": f"base_{i}",
            "M": M,
            "N": N,
            "is_rms": is_rms,
            "has_bias": has_bias,
            "has_z": has_z,
            "dtype": "float16",
            "group_size": None,
            "norm_before_gate": True,
            "explicit_out": False,
        }
        for i, (M, N, is_rms, has_bias, has_z) in enumerate(TEST_SHAPES)
    ] + CORRECTNESS_CASES
    for i, case in enumerate(cases):
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, case["dtype"])
            x = torch.randn(case["M"], case["N"], device=device, dtype=dtype)
            w = torch.randn(case["N"], device=device, dtype=dtype)
            b = (
                torch.randn(case["N"], device=device, dtype=dtype)
                if case["has_bias"]
                else None
            )
            z = torch.randn_like(x) if case["has_z"] else None
            supplied_out = torch.full_like(x, torch.nan) if case["explicit_out"] else None
            eps = 1e-5
            out, mean, rstd = mod.layer_norm_fwd(
                x,
                w,
                b,
                eps,
                z=z,
                out=supplied_out,
                group_size=case["group_size"],
                norm_before_gate=case["norm_before_gate"],
                is_rms_norm=case["is_rms"],
            )
            ref_out, ref_mean, ref_rstd = reference(
                x,
                w,
                b,
                eps,
                z,
                case["is_rms"],
                group_size=case["group_size"],
                norm_before_gate=case["norm_before_gate"],
            )
            if supplied_out is not None and out.data_ptr() != supplied_out.data_ptr():
                return False, f"{case['name']}: did not return the supplied out buffer"
            if not torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2):
                diff = (out - ref_out).abs().max().item()
                return False, f"{case['name']}: output max diff={diff}"
            if case["is_rms"]:
                if mean is not None:
                    return False, f"{case['name']}: RMSNorm returned a mean buffer"
            else:
                if mean is None:
                    return False, f"{case['name']}: LayerNorm did not return mean"
                if mean.shape != ref_mean.shape or mean.dtype != ref_mean.dtype:
                    return False, f"{case['name']}: mean metadata mismatch"
                if not torch.allclose(mean, ref_mean, atol=1e-5, rtol=1e-4):
                    diff = (mean - ref_mean).abs().max().item()
                    return False, f"{case['name']}: mean max diff={diff}"
            if rstd.shape != ref_rstd.shape or rstd.dtype != ref_rstd.dtype:
                return False, f"{case['name']}: rstd metadata mismatch"
            if not torch.allclose(rstd, ref_rstd, atol=1e-5, rtol=1e-4):
                diff = (rstd - ref_rstd).abs().max().item()
                return False, f"{case['name']}: rstd max diff={diff}"
        except Exception as e:
            return False, f"{case['name']}: {e}"
    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []
    device = "cuda"
    test_cases = []

    for test_idx, (M, N, is_rms, has_bias, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            x = torch.randn(M, N, device=device, dtype=torch.float16)
            w = torch.randn(N, device=device, dtype=torch.float16)
            b = torch.randn(N, device=device, dtype=torch.float16) if has_bias else None
            z = torch.randn(M, N, device=device, dtype=torch.float16) if has_z else None
            def _bench_fn():
                mod.layer_norm_fwd(x, w, b, 1e-5, z=z, is_rms_norm=is_rms)
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
                    "is_rms": is_rms,
                    "has_bias": has_bias,
                    "has_z": has_z
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "N": N,
                    "is_rms": is_rms,
                    "has_bias": has_bias,
                    "has_z": has_z
                }
            })
    return test_cases


def main():
    parser = argparse.ArgumentParser()
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
        report = {"status": "ok" if ok else "fail", "error": err}
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

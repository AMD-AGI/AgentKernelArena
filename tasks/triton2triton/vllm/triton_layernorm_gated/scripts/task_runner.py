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
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract cases.  TEST_SHAPES remains the immutable
# performance/scoring workload.
LAYERNORM_GROUPED_CORRECTNESS_CASES = [
    {
        "name": "grouped_layernorm_gate_before_noncompact_out",
        "M": 3,
        "N": 30,
        "group_size": 10,
        "is_rms": False,
        "has_bias": True,
        "has_z": True,
        "norm_before_gate": False,
        "row_padding": 5,
        "provide_out": True,
    },
    {
        "name": "grouped_rmsnorm_gate_after_noncompact_out",
        "M": 2,
        "N": 42,
        "group_size": 14,
        "is_rms": True,
        "has_bias": False,
        "has_z": True,
        "norm_before_gate": True,
        "row_padding": 7,
        "provide_out": True,
    },
]
LAYERNORM_FEATURE_LIMIT_CASE = {"M": 1, "N": 32769, "group_size": 32769}
LAYERNORM_PADDING_CANARY = 123.0


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
    norm_before_gate=True,
    group_size=None,
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
        var = (x_f ** 2).mean(-1, keepdim=True)
        mean_flat = None
        x_hat = x_f * torch.rsqrt(var + eps)
    else:
        mean = x_f.mean(-1, keepdim=True)
        var = ((x_f - mean) ** 2).mean(-1, keepdim=True)
        mean_flat = mean.squeeze(-1).transpose(0, 1).contiguous().flatten()
        x_hat = (x_f - mean) * torch.rsqrt(var + eps)
    rstd = torch.rsqrt(var + eps)
    rstd_flat = rstd.squeeze(-1).transpose(0, 1).contiguous().flatten()
    y = x_hat * weight.float().reshape(1, ngroups, group_size)
    if bias is not None:
        y = y + bias.float().reshape(1, ngroups, group_size)
    if z is not None and norm_before_gate:
        z_f = z.float().reshape(M, ngroups, group_size)
        y = y * z_f * torch.sigmoid(z_f)
    return y.reshape(M, N).to(x.dtype), mean_flat, rstd_flat


def _make_padded_matrix(torch, M, N, row_padding, *, device, random_values=True):
    storage = torch.full(
        (M, N + row_padding),
        LAYERNORM_PADDING_CANARY,
        device=device,
        dtype=torch.float16,
    )
    logical = storage[:, :N]
    if random_values:
        logical.copy_(torch.randn(M, N, device=device, dtype=torch.float16))
    return logical, storage


def _check_stat_tensor(torch, observed, expected, label):
    if observed is None:
        return f"{label}: missing statistic"
    if observed.shape != expected.shape or observed.dtype != torch.float32:
        return (
            f"{label}: wrong statistic contract "
            f"shape={tuple(observed.shape)} dtype={observed.dtype}"
        )
    if not torch.allclose(observed, expected, atol=1e-3, rtol=1e-3):
        max_diff = (observed - expected).abs().max().item()
        return f"{label}: statistic max diff={max_diff}"
    return None


def _run_layernorm_case(
    torch,
    mod,
    *,
    label,
    M,
    N,
    group_size,
    is_rms,
    has_bias,
    has_z,
    norm_before_gate,
    row_padding,
    provide_out,
    device,
):
    x, x_storage = _make_padded_matrix(
        torch, M, N, row_padding, device=device
    )
    z, z_storage = (
        _make_padded_matrix(torch, M, N, row_padding + 2, device=device)
        if has_z
        else (None, None)
    )
    weight = torch.randn(N, device=device, dtype=torch.float16)
    bias = torch.randn(N, device=device, dtype=torch.float16) if has_bias else None
    out, out_storage = (
        _make_padded_matrix(
            torch,
            M,
            N,
            row_padding + 3,
            device=device,
            random_values=False,
        )
        if provide_out
        else (None, None)
    )
    frozen_x = x.clone()
    frozen_z = z.clone() if z is not None else None
    frozen_weight = weight.clone()
    frozen_bias = bias.clone() if bias is not None else None
    eps = 1e-5
    expected_out, expected_mean, expected_rstd = reference(
        frozen_x,
        frozen_weight,
        frozen_bias,
        eps,
        frozen_z,
        is_rms,
        norm_before_gate=norm_before_gate,
        group_size=group_size,
    )
    result_out, mean, rstd = mod.layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms,
    )
    torch.cuda.synchronize()

    if provide_out and result_out is not out:
        return f"{label}: layer_norm_fwd did not return the caller-provided out object"
    if result_out.shape != x.shape or result_out.dtype != x.dtype:
        return f"{label}: wrong output shape or dtype"
    protected_tensors = [x, weight]
    if z is not None:
        protected_tensors.append(z)
    if bias is not None:
        protected_tensors.append(bias)
    protected_storage_ptrs = {
        tensor.untyped_storage().data_ptr() for tensor in protected_tensors
    }
    out_storage_ptr = result_out.untyped_storage().data_ptr()
    if out_storage_ptr in protected_storage_ptrs:
        return f"{label}: output aliases a protected input"
    statistic_tensors = [("rstd", rstd)]
    if mean is not None:
        statistic_tensors.append(("mean", mean))
    statistic_storage_ptrs = set()
    for statistic_name, statistic in statistic_tensors:
        if statistic is None:
            continue
        statistic_storage_ptr = statistic.untyped_storage().data_ptr()
        if statistic_storage_ptr in protected_storage_ptrs:
            return f"{label}: {statistic_name} aliases a protected input"
        if statistic_storage_ptr == out_storage_ptr:
            return f"{label}: {statistic_name} aliases output"
        if statistic_storage_ptr in statistic_storage_ptrs:
            return f"{label}: statistics alias each other"
        statistic_storage_ptrs.add(statistic_storage_ptr)
    if not torch.allclose(result_out, expected_out, atol=1e-2, rtol=1e-2):
        max_diff = (result_out - expected_out).abs().max().item()
        return f"{label}: output max diff={max_diff}"

    for name, observed, frozen in (
        ("x", x, frozen_x),
        ("weight", weight, frozen_weight),
    ):
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated protected input {name}"
    if z is not None and not torch.equal(z, frozen_z):
        return f"{label}: candidate mutated protected input z"
    if bias is not None and not torch.equal(bias, frozen_bias):
        return f"{label}: candidate mutated protected input bias"

    if row_padding and x.is_contiguous():
        return f"{label}: x fixture is unexpectedly contiguous"
    if row_padding and not torch.all(x_storage[:, N:] == LAYERNORM_PADDING_CANARY):
        return f"{label}: candidate modified x padding"
    if z_storage is not None and not torch.all(
        z_storage[:, N:] == LAYERNORM_PADDING_CANARY
    ):
        return f"{label}: candidate modified z padding"
    if out_storage is not None and not torch.all(
        out_storage[:, N:] == LAYERNORM_PADDING_CANARY
    ):
        return f"{label}: candidate overwrote out padding canary"

    if is_rms:
        if mean is not None:
            return f"{label}: RMSNorm must return mean=None"
    else:
        error = _check_stat_tensor(torch, mean, expected_mean, f"{label} mean")
        if error is not None:
            return error
    return _check_stat_tensor(torch, rstd, expected_rstd, f"{label} rstd")


def _check_feature_limit(torch, mod, *, device):
    case = LAYERNORM_FEATURE_LIMIT_CASE
    x = torch.zeros(case["M"], case["N"], device=device, dtype=torch.float16)
    weight = torch.ones(case["N"], device=device, dtype=torch.float16)
    try:
        mod.layer_norm_fwd(
            x,
            weight,
            None,
            1e-5,
            group_size=case["group_size"],
            is_rms_norm=True,
        )
    except RuntimeError as error:
        if "feature" not in str(error).lower():
            return f"feature-limit error is not controlled/descriptive: {error}"
        return None
    return "feature group larger than 64 KiB did not raise a controlled RuntimeError"


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
    for i, (M, N, is_rms, has_bias, has_z) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            error = _run_layernorm_case(
                torch,
                mod,
                label=f"Shape {i}",
                M=M,
                N=N,
                group_size=N,
                is_rms=is_rms,
                has_bias=has_bias,
                has_z=has_z,
                norm_before_gate=True,
                row_padding=0,
                provide_out=False,
                device=device,
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"Shape {i}: {e}"

    for case_index, case in enumerate(LAYERNORM_GROUPED_CORRECTNESS_CASES):
        try:
            torch.manual_seed(142 + case_index)
            error = _run_layernorm_case(
                torch,
                mod,
                label=case["name"],
                M=case["M"],
                N=case["N"],
                group_size=case["group_size"],
                is_rms=case["is_rms"],
                has_bias=case["has_bias"],
                has_z=case["has_z"],
                norm_before_gate=case["norm_before_gate"],
                row_padding=case["row_padding"],
                provide_out=case["provide_out"],
                device=device,
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"{case['name']}: {e}"

    try:
        error = _check_feature_limit(torch, mod, device=device)
        if error is not None:
            return False, error
    except Exception as e:
        return False, f"feature-limit case: {e}"
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
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": (
                len(TEST_SHAPES) + len(LAYERNORM_GROUPED_CORRECTNESS_CASES) + 1
            ),
        }
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

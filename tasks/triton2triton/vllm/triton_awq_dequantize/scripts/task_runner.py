#!/usr/bin/env python3
"""Task runner for triton2triton/triton_awq_dequantize"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_awq_dequantize"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_awq_dequantize.py")

# Test configurations: (K, N_packed, group_size)
# qweight shape: [K, N_packed], scales shape: [K//G, N_packed*8], zeros shape: [K//G, N_packed]
TEST_SHAPES = [
    (64, 8, 32),
    (128, 16, 32),
    (128, 16, 64),
    (256, 32, 128),
    (256, 32, 64),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

AWQ_ATOL = 1e-2
AWQ_RTOL = 1e-2
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
AWQ_TARGET_PARAMETER_NAMES = (
    "qweight_ptr",
    "scales_ptr",
    "zeros_ptr",
    "group_size",
    "result_ptr",
    "num_cols",
    "num_rows",
)
AWQ_TARGET_METADATA_NAMES = {"BLOCK_SIZE_X", "BLOCK_SIZE_Y"}
AWQ_TRITON_LAUNCH_CONTROL_NAMES = {
    "grid",
    "warmup",
    "num_warps",
    "num_stages",
    "num_ctas",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
}
AWQ_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS = {
    "num_warps",
    "num_stages",
    "num_ctas",
    "matrix_instr_nonkdim",
    "kpack",
}

_TRUSTED_TORCH_PATHS = (
    "Tensor",
    "allclose",
    "equal",
    "isfinite",
    "manual_seed",
    "randint",
    "randn",
    "zeros",
    "cuda.Event",
    "cuda.synchronize",
)

# Correctness-only cases expand the public contract without changing the
# immutable TEST_SHAPES performance workload above. Every logical input is a
# C-contiguous view at a nonzero offset inside a guarded backing allocation.
CORRECTNESS_CASES = [
    {
        "name": "group32_signed_x_tail",
        "K": 96,
        "N_packed": 5,
        "group_size": 32,
        "block_size_x": 64,
        "block_size_y": 16,
    },
    {
        "name": "group64_signed_x_tail",
        "K": 192,
        "N_packed": 7,
        "group_size": 64,
        "block_size_x": 8,
        "block_size_y": 32,
    },
    {
        "name": "group128_signed_x_tail",
        "K": 256,
        "N_packed": 9,
        "group_size": 128,
        "block_size_x": 16,
        "block_size_y": 64,
    },
    {
        "name": "per_tensor_non_power_of_two_k",
        "K": 48,
        "N_packed": 3,
        "group_size": "K",
        "block_size_x": 8,
        "block_size_y": 16,
    },
]

# Signed int32 values whose unsigned forms exercise all eight packed nibbles.
SIGNED_89ABCDEF = -1985229329
SIGNED_87654321 = -2023406815


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


def _resolve_attribute(root, dotted_name):
    value = root
    for component in dotted_name.split("."):
        value = getattr(value, component)
    return value


def _evaluator_value_token(value):
    if isinstance(value, (str, int, float, bool, type(None), tuple, list, dict, set)):
        return ("value", repr(value))
    return (
        "identity",
        value,
        getattr(value, "__code__", None),
        repr(getattr(value, "__defaults__", None)),
        repr(getattr(value, "__kwdefaults__", None)),
    )


def _capture_evaluator_integrity(torch):
    runner_globals = {
        name: _evaluator_value_token(value)
        for name, value in globals().items()
        if not name.startswith("__")
    }
    torch_attributes = {
        path: _resolve_attribute(torch, path) for path in _TRUSTED_TORCH_PATHS
    }
    return runner_globals, torch_attributes


def _verify_evaluator_integrity(torch, snapshot):
    runner_globals, torch_attributes = snapshot
    current_names = {name for name in globals() if not name.startswith("__")}
    if current_names != set(runner_globals):
        return "candidate import or execution changed evaluator globals"
    for name, expected in runner_globals.items():
        if _evaluator_value_token(globals()[name]) != expected:
            return f"candidate import or execution changed evaluator global {name}"
    for path, expected in torch_attributes.items():
        if _resolve_attribute(torch, path) is not expected:
            return f"candidate import or execution changed trusted torch primitive {path}"
    return None


class _KernelLaunchRecorder:
    """Delegate the real target launch while preserving its resolved metadata."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = []

    @staticmethod
    def _argument(args, kwargs, position, keyword):
        return args[position] if len(args) > position else kwargs.get(keyword)

    def _record(self, grid, args, kwargs):
        record = {
            "grid": tuple(grid) if grid is not None else None,
            "args": tuple(args),
            "kwargs": dict(kwargs),
            "block_size_x": kwargs.get("BLOCK_SIZE_X"),
            "block_size_y": kwargs.get("BLOCK_SIZE_Y"),
            "qweight": self._argument(args, kwargs, 0, "qweight_ptr"),
            "scales": self._argument(args, kwargs, 1, "scales_ptr"),
            "zeros": self._argument(args, kwargs, 2, "zeros_ptr"),
            "result": self._argument(args, kwargs, 4, "result_ptr"),
        }
        bound_tensors = tuple(
            record[name] for name in ("qweight", "scales", "zeros", "result")
        )
        record["data_ptrs"] = tuple(
            tensor.data_ptr() if hasattr(tensor, "data_ptr") else None
            for tensor in bound_tensors
        )
        record["storage_ptrs"] = tuple(
            tensor.untyped_storage().data_ptr()
            if hasattr(tensor, "untyped_storage")
            else None
            for tensor in bound_tensors
        )
        self.launches.append(record)

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record_and_launch(*args, **kwargs):
            resolved_grid = grid(kwargs) if callable(grid) else grid
            self._record(resolved_grid, args, kwargs)
            record = self.launches[-1]
            record["result"].fill_(float("nan"))
            returned = launch(*args, **kwargs)
            record["post_launch_snapshot"] = record["result"].clone()
            return returned

        return record_and_launch

    def run(self, *args, **kwargs):
        self._record(kwargs.get("grid"), args, kwargs)
        record = self.launches[-1]
        record["result"].fill_(float("nan"))
        returned = self.kernel.run(*args, **kwargs)
        record["post_launch_snapshot"] = record["result"].clone()
        return returned

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def _unpack_awq_word(packed_value):
    """Unpack signed int32 through explicit unsigned two's-complement semantics."""
    unsigned_value = int(packed_value) & 0xFFFFFFFF
    return tuple(
        (unsigned_value >> (nibble_index * 4)) & 0xF
        for nibble_index in AWQ_ORDER
    )


def reference_awq_dequantize(qweight, scales, zeros, group_size):
    """CPU reference: unpack 4-bit AWQ weights and dequantize."""
    import torch
    K, N_packed = qweight.shape
    N = N_packed * 8

    result = torch.zeros((K, N), dtype=torch.float32, device="cpu")
    qweight_cpu = qweight.cpu().to(torch.int32)
    zeros_cpu = zeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()

    for row in range(K):
        group_idx = row // group_size
        for col_packed in range(N_packed):
            weight_values = _unpack_awq_word(
                qweight_cpu[row, col_packed].item()
            )
            zero_values = _unpack_awq_word(
                zeros_cpu[group_idx, col_packed].item()
            )
            for bit_idx, (weight_val, zero_val) in enumerate(
                zip(weight_values, zero_values)
            ):
                out_col = col_packed * 8 + bit_idx
                scale_val = scales_cpu[group_idx, out_col].item()
                result[row, out_col] = (weight_val - zero_val) * scale_val

    return result.to(scales.dtype)


def _make_guarded_contiguous_matrix(
    torch,
    rows,
    columns,
    *,
    dtype,
    device,
    guard_elements,
):
    """Create a C-contiguous logical matrix inside initialized guard storage."""
    logical_elements = rows * columns
    storage_elements = logical_elements + 2 * guard_elements
    if dtype == torch.int32:
        storage = torch.randint(
            0,
            2**31,
            (storage_elements,),
            device=device,
            dtype=dtype,
        )
    elif dtype == torch.float16:
        storage = (
            torch.randn(storage_elements, device=device, dtype=dtype).abs()
            * 0.1
            + 0.01
        )
    else:
        raise ValueError(f"unsupported guarded dtype: {dtype}")
    logical = storage[
        guard_elements : guard_elements + logical_elements
    ].view(rows, columns)
    assert logical.is_contiguous()
    assert logical.storage_offset() == guard_elements
    return logical, storage


def _inject_signed_packed_words(qweight, zeros):
    """Install noncorresponding signed words in weights and zero points."""
    qweight[0, 0] = -(2**31)
    qweight[1, 1] = -1
    qweight[2, 2] = SIGNED_89ABCDEF
    qweight[-1, -1] = 0x13579BDF

    zeros[0, 0] = 0x01234567
    zeros[0, 1] = SIGNED_87654321
    zeros[0, 2] = -(2**31)
    zeros[-1, -1] = -1


def _make_awq_inputs(torch, case, device, seed):
    K = case["K"]
    N_packed = case["N_packed"]
    group_size = K if case["group_size"] == "K" else case["group_size"]
    assert group_size in (32, 64, 128) or group_size == K
    assert K % group_size == 0
    if "block_size_y" in case:
        assert group_size % case["block_size_y"] == 0
    num_groups = K // group_size

    torch.manual_seed(seed)
    qweight, qweight_storage = _make_guarded_contiguous_matrix(
        torch,
        K,
        N_packed,
        dtype=torch.int32,
        device=device,
        guard_elements=13,
    )
    scales, scales_storage = _make_guarded_contiguous_matrix(
        torch,
        num_groups,
        N_packed * 8,
        dtype=torch.float16,
        device=device,
        guard_elements=17,
    )
    zeros, zeros_storage = _make_guarded_contiguous_matrix(
        torch,
        num_groups,
        N_packed,
        dtype=torch.int32,
        device=device,
        guard_elements=19,
    )
    _inject_signed_packed_words(qweight, zeros)
    return (
        qweight,
        scales,
        zeros,
        group_size,
        (
            ("qweight", qweight_storage),
            ("scales", scales_storage),
            ("zeros", zeros_storage),
        ),
    )


def _validate_awq_output(
    torch,
    result,
    ref,
    expected_shape,
    expected_device,
    protected_storage_ptrs,
    label,
    invocation,
):
    prefix = f"{label}, invocation {invocation}"
    if not isinstance(result, torch.Tensor):
        return f"{prefix}: result is not a torch.Tensor"
    if tuple(result.shape) != tuple(expected_shape):
        return f"{prefix}: result shape {tuple(result.shape)} != {tuple(expected_shape)}"
    if result.dtype != torch.float16:
        return f"{prefix}: result dtype {result.dtype} != torch.float16"
    if result.device != expected_device:
        return f"{prefix}: result device {result.device} != {expected_device}"
    if not result.is_contiguous():
        return f"{prefix}: result is not C-contiguous"
    if result.untyped_storage().data_ptr() in protected_storage_ptrs:
        return f"{prefix}: result aliases an input or input backing allocation"
    if not bool(torch.isfinite(result).all().item()):
        return f"{prefix}: result contains a non-finite value"
    if not torch.allclose(result, ref, atol=AWQ_ATOL, rtol=AWQ_RTOL):
        max_diff = (result - ref).abs().max().item()
        return f"{prefix}: max diff = {max_diff:.6f}"
    return None


def _validate_awq_launch_batch(
    torch,
    launches,
    expected_grid,
    block_size_x,
    block_size_y,
    group_size,
    qweight,
    scales,
    zeros,
    result,
    label,
    invocation,
):
    prefix = f"{label}, invocation {invocation}"
    if len(launches) != 1:
        return f"{prefix}: expected exactly one target kernel launch, observed {len(launches)}"
    launch = launches[0]
    positional = launch.get("args")
    keyword = launch.get("kwargs")
    if not isinstance(positional, tuple) or not isinstance(keyword, dict):
        return f"{prefix}: launch operands were not captured"
    if len(positional) > len(AWQ_TARGET_PARAMETER_NAMES):
        return (
            f"{prefix}: target launch has {len(positional)} positional operands; "
            f"expected at most {len(AWQ_TARGET_PARAMETER_NAMES)}"
        )
    positionally_bound = set(AWQ_TARGET_PARAMETER_NAMES[: len(positional)])
    duplicate_bindings = positionally_bound.intersection(keyword)
    if duplicate_bindings:
        return (
            f"{prefix}: duplicate target operand bindings "
            f"{sorted(duplicate_bindings)}"
        )
    allowed_keywords = (
        set(AWQ_TARGET_PARAMETER_NAMES)
        | AWQ_TARGET_METADATA_NAMES
        | AWQ_TRITON_LAUNCH_CONTROL_NAMES
    )
    unexpected_keywords = set(keyword).difference(allowed_keywords)
    if unexpected_keywords:
        return (
            f"{prefix}: target launch has unexpected keyword operands "
            f"{sorted(unexpected_keywords)}"
        )
    launch_control_error = _validate_awq_launch_controls(
        keyword, launch.get("grid"), prefix
    )
    if launch_control_error is not None:
        return launch_control_error

    def operand(index):
        if index < len(positional):
            return positional[index]
        return keyword.get(AWQ_TARGET_PARAMETER_NAMES[index])

    expected_metadata = {
        "grid": expected_grid,
        "block_size_x": block_size_x,
        "block_size_y": block_size_y,
    }
    for key, expected in expected_metadata.items():
        observed = launch.get(key)
        if key != "grid" and type(observed) is not int:
            return f"{prefix}: target launch {key} must be an integer"
        if observed != expected:
            return f"{prefix}: expected {key}={expected}, observed {launch.get(key)}"

    expected_bindings = {
        "qweight": qweight,
        "scales": scales,
        "zeros": zeros,
        "result": result,
    }
    for binding_name, expected_tensor in expected_bindings.items():
        if launch.get(binding_name) is not expected_tensor:
            return (
                f"{prefix}: target launch {binding_name} was not bound to "
                "the evaluated tensor"
            )
    for index, binding_name in ((0, "qweight"), (1, "scales"), (2, "zeros"), (4, "result")):
        if operand(index) is not expected_bindings[binding_name]:
            return f"{prefix}: target operand {binding_name} was not bound exactly once"
    dimensions = (operand(3), operand(5), operand(6))
    expected_dimensions = (group_size, qweight.shape[1], qweight.shape[0])
    if (
        any(type(value) is not int for value in dimensions)
        or dimensions != expected_dimensions
    ):
        return (
            f"{prefix}: target dimensions {dimensions} != "
            f"{expected_dimensions}"
        )
    bound_tensors = tuple(
        expected_bindings[name]
        for name in ("qweight", "scales", "zeros", "result")
    )
    expected_data_ptrs = tuple(tensor.data_ptr() for tensor in bound_tensors)
    expected_storage_ptrs = tuple(
        tensor.untyped_storage().data_ptr() for tensor in bound_tensors
    )
    if launch.get("data_ptrs") != expected_data_ptrs:
        return f"{prefix}: target launch data-pointer binding mismatch"
    if launch.get("storage_ptrs") != expected_storage_ptrs:
        return f"{prefix}: target launch storage binding mismatch"
    return None


def _validate_awq_launch_controls(keyword, resolved_grid, prefix):
    for name in AWQ_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS:
        if name not in keyword:
            continue
        value = keyword[name]
        if type(value) is not int or value <= 0:
            return f"{prefix}: launch control {name} must be a positive integer"
    if "waves_per_eu" in keyword:
        value = keyword["waves_per_eu"]
        if type(value) is not int or value < 0:
            return f"{prefix}: launch control waves_per_eu must be a nonnegative integer"
    if "warmup" in keyword and type(keyword["warmup"]) is not bool:
        return f"{prefix}: launch control warmup must be a boolean"
    if "grid" in keyword:
        value = keyword["grid"]
        if isinstance(value, int) and not isinstance(value, bool):
            value = (value,)
        elif isinstance(value, (tuple, list)):
            value = tuple(value)
        else:
            return f"{prefix}: launch control grid must be an integer sequence"
        if (
            not value
            or any(type(dimension) is not int or dimension <= 0 for dimension in value)
            or value != resolved_grid
        ):
            return f"{prefix}: launch control grid does not match the resolved grid"
    return None


def _validate_awq_target_snapshot(torch, snapshot, result, label, invocation):
    prefix = f"{label}, invocation {invocation}"
    if snapshot.untyped_storage().data_ptr() == result.untyped_storage().data_ptr():
        return f"{prefix}: target snapshot reused the returned output storage"
    if not torch.equal(snapshot, result):
        return (
            f"{prefix}: immediate target snapshot did not match the returned output; "
            "precomputed, detached, dummy, or post-target repair paths are invalid"
        )
    return None


def _check_awq_case(torch, mod, case, device, seed, integrity_guard):
    label = case["name"]
    K = case["K"]
    N_packed = case["N_packed"]
    block_size_x = case["block_size_x"]
    block_size_y = case["block_size_y"]
    qweight, scales, zeros, group_size, protected_inputs = _make_awq_inputs(
        torch, case, device, seed
    )
    frozen_backing = tuple(
        (name, storage, storage.clone())
        for name, storage in protected_inputs
    )
    protected_storage_ptrs = {
        storage.untyped_storage().data_ptr()
        for _, storage in protected_inputs
    }
    ref = reference_awq_dequantize(qweight, scales, zeros, group_size).to(device)
    if not bool(torch.isfinite(ref).all().item()):
        return False, f"{label}: reference contains a non-finite value"

    expected_grid = (
        (N_packed + block_size_x - 1) // block_size_x,
        (K + block_size_y - 1) // block_size_y,
    )
    original_kernel = mod.awq_dequantize_kernel
    recorder = _KernelLaunchRecorder(original_kernel)
    mod.awq_dequantize_kernel = recorder
    results = []
    try:
        for invocation in range(1, 3):
            launch_start = len(recorder.launches)
            result = mod.awq_dequantize_triton(
                qweight,
                scales,
                zeros,
                block_size_x=block_size_x,
                block_size_y=block_size_y,
            )
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return False, integrity_error
            torch.cuda.synchronize()
            launches = recorder.launches[launch_start:]
            launch_error = _validate_awq_launch_batch(
                torch,
                launches,
                expected_grid,
                block_size_x,
                block_size_y,
                group_size,
                qweight,
                scales,
                zeros,
                result,
                label,
                invocation,
            )
            if launch_error is not None:
                return False, launch_error
            snapshot_error = _validate_awq_target_snapshot(
                torch,
                recorder.launches[launch_start]["post_launch_snapshot"],
                result,
                label,
                invocation,
            )
            if snapshot_error is not None:
                return False, snapshot_error
            error = _validate_awq_output(
                torch,
                result,
                ref,
                (K, N_packed * 8),
                qweight.device,
                protected_storage_ptrs,
                label,
                invocation,
            )
            if error:
                return False, error
            results.append(result)
    except Exception as error:
        return False, f"{label}: exception: {error}"
    finally:
        mod.awq_dequantize_kernel = original_kernel

    if results[0].untyped_storage().data_ptr() == results[1].untyped_storage().data_ptr():
        return False, f"{label}: repeated calls did not return fresh storage"
    if not torch.equal(results[0], results[1]):
        return False, f"{label}: repeated calls were not exactly deterministic"
    for name, storage, frozen in frozen_backing:
        if not torch.equal(storage, frozen):
            return False, f"{label}: candidate mutated {name} backing storage"
    return True, None


def _check_contract_rejections(torch, mod, device, integrity_guard):
    base_case = {
        "K": 64,
        "N_packed": 3,
        "group_size": 32,
    }
    qweight, scales, zeros, _, _ = _make_awq_inputs(
        torch, base_case, device, 991
    )
    noncontiguous_qweight = torch.empty(
        (64, 6), device=device, dtype=torch.int32
    )[:, ::2]
    noncontiguous_qweight.copy_(qweight)
    unsupported_group_qweight = torch.randint(
        0, 2**31, (96, 3), device=device, dtype=torch.int32
    )
    unsupported_group_scales = (
        torch.randn((2, 24), device=device, dtype=torch.float16).abs()
        * 0.1
        + 0.01
    )
    unsupported_group_zeros = torch.randint(
        0, 2**31, (2, 3), device=device, dtype=torch.int32
    )
    nondivisible_qweight = torch.randint(
        0, 2**31, (65, 3), device=device, dtype=torch.int32
    )
    nondivisible_scales = (
        torch.randn((2, 24), device=device, dtype=torch.float16).abs()
        * 0.1
        + 0.01
    )
    nondivisible_zeros = torch.randint(
        0, 2**31, (2, 3), device=device, dtype=torch.int32
    )
    invalid_calls = (
        ("int64 qweight", lambda: mod.awq_dequantize_triton(qweight.to(torch.int64), scales, zeros)),
        ("float32 scales", lambda: mod.awq_dequantize_triton(qweight, scales.float(), zeros)),
        ("int64 zeros", lambda: mod.awq_dequantize_triton(qweight, scales, zeros.to(torch.int64))),
        ("noncontiguous qweight", lambda: mod.awq_dequantize_triton(noncontiguous_qweight, scales, zeros)),
        ("mismatched scales", lambda: mod.awq_dequantize_triton(qweight, scales[:, :-1].clone(), zeros)),
        ("mismatched zeros", lambda: mod.awq_dequantize_triton(qweight, scales, zeros[:, :-1].clone())),
        (
            "unsupported group_size",
            lambda: mod.awq_dequantize_triton(
                unsupported_group_qweight,
                unsupported_group_scales,
                unsupported_group_zeros,
            ),
        ),
        (
            "nondivisible scales rows",
            lambda: mod.awq_dequantize_triton(
                nondivisible_qweight,
                nondivisible_scales,
                nondivisible_zeros,
            ),
        ),
        ("mixed devices", lambda: mod.awq_dequantize_triton(qweight.cpu(), scales, zeros)),
        ("non-power-of-two block_size_x", lambda: mod.awq_dequantize_triton(qweight, scales, zeros, block_size_x=3)),
        ("non-divisor block_size_y", lambda: mod.awq_dequantize_triton(qweight, scales, zeros, block_size_y=64)),
    )

    original_kernel = mod.awq_dequantize_kernel
    recorder = _KernelLaunchRecorder(original_kernel)
    mod.awq_dequantize_kernel = recorder
    try:
        for label, invoke in invalid_calls:
            launch_start = len(recorder.launches)
            try:
                invoke()
            except (TypeError, ValueError):
                pass
            except Exception as error:
                return False, f"rejection {label}: unexpected exception: {error}"
            else:
                return False, f"rejection {label}: invalid input was accepted"
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return False, integrity_error
            if len(recorder.launches) != launch_start:
                return False, f"rejection {label}: target kernel was launched"
    finally:
        mod.awq_dequantize_kernel = original_kernel
    return True, None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "awq_dequantize_triton"), "Missing awq_dequantize_triton"
        assert hasattr(mod, "awq_dequantize_kernel"), "Missing awq_dequantize_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    capture_integrity = _capture_evaluator_integrity
    verify_integrity = _verify_evaluator_integrity
    integrity_snapshot = capture_integrity(torch)

    def integrity_guard():
        return verify_integrity(torch, integrity_snapshot)

    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}"
    integrity_error = integrity_guard()
    if integrity_error is not None:
        return False, integrity_error

    device = "cuda"

    for index, (K, N_packed, group_size) in enumerate(TEST_SHAPES):
        case = {
            "name": f"legacy_shape_{index + 1}",
            "K": K,
            "N_packed": N_packed,
            "group_size": group_size,
            "block_size_x": 32,
            "block_size_y": 32,
        }
        ok, error = _check_awq_case(
            torch, mod, case, device, 42 + index, integrity_guard
        )
        if not ok:
            return False, error

    for index, case in enumerate(CORRECTNESS_CASES):
        ok, error = _check_awq_case(
            torch, mod, case, device, 142 + index, integrity_guard
        )
        if not ok:
            return False, error

    ok, error = _check_contract_rejections(torch, mod, device, integrity_guard)
    if not ok:
        return False, error

    return True, None


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (K, N_packed, group_size) in enumerate(TEST_SHAPES):
        try:
            N = N_packed * 8
            num_groups = K // group_size

            torch.manual_seed(0)
            qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            zeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)

            def _bench_fn():
                mod.awq_dequantize_triton(qweight, scales, zeros)
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
                    "K": K,
                    "N_packed": N_packed,
                    "group_size": group_size
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "K": K,
                    "N_packed": N_packed,
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
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_CASES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
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

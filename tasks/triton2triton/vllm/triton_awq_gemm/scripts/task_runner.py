#!/usr/bin/env python3
"""Task runner for triton2triton/triton_awq_gemm"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_awq_gemm"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_awq_gemm.py")

# Test configs: (M, K, N_packed, group_size, split_k)
TEST_SHAPES = [
    (32, 64, 8, 32, 1),
    (64, 128, 16, 32, 1),
    (32, 128, 16, 64, 1),
    (64, 256, 32, 128, 1),
    (128, 256, 32, 64, 1),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Public numerical contract from config.yaml. Keep these literals shared by
# every correctness path so the evaluator cannot drift from the task prompt.
AWQ_ATOL = 1e-2
AWQ_RTOL = 1e-2
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
AWQ_TARGET_PARAMETER_NAMES = (
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "zeros_ptr",
    "scales_ptr",
    "M",
    "N",
    "K",
    "group_size",
)
AWQ_TARGET_METADATA_NAMES = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "SPLIT_K",
}
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

# Signed int32 values whose unsigned representations are 0x89ABCDEF and
# 0x87654321. They deliberately exercise every packed nibble, including bit 31.
SIGNED_89ABCDEF = -1985229329
SIGNED_87654321 = -2023406815

# Non-scoring public-contract cases. TEST_SHAPES above is the immutable
# performance workload and must never be extended with these cases. Every split
# lane performs at least two BLOCK_SIZE_K=32 rounds in these baseline cases.
# group_size="K" denotes the supported one-group branch.
CORRECTNESS_CASES = [
    {
        "name": "split1_group32_row_padded_high_dynamic",
        "M": 3,
        "K": 96,
        "N_packed": 5,
        "group_size": 32,
        "split_k": 1,
        "layout": "row_padded",
        "profile": "high_dynamic",
        "input_padding": 7,
        "qweight_padding": 3,
        "scales_padding": 5,
        "qzeros_padding": 2,
        "block_sizes": (16, 64, 16),
    },
    {
        "name": "split2_group64_inner_strided_high_dynamic",
        "M": 5,
        "K": 192,
        "N_packed": 7,
        "group_size": 64,
        "split_k": 2,
        "layout": "inner_strided",
        "profile": "high_dynamic",
        "input_padding": 11,
        "qweight_padding": 5,
        "scales_padding": 7,
        "qzeros_padding": 3,
    },
    {
        "name": "split4_group128_transposed_high_dynamic",
        "M": 7,
        "K": 256,
        "N_packed": 9,
        "group_size": 128,
        "split_k": 4,
        "layout": "transposed",
        "profile": "high_dynamic",
        "input_padding": 13,
        "qweight_padding": 7,
        "scales_padding": 11,
        "qzeros_padding": 5,
    },
    {
        "name": "split8_group32_row_padded_low_cancellation",
        "M": 3,
        "K": 512,
        "N_packed": 5,
        "group_size": 32,
        "split_k": 8,
        "layout": "row_padded",
        "profile": "low_cancellation",
        "input_padding": 5,
        "qweight_padding": 3,
        "scales_padding": 3,
        "qzeros_padding": 2,
    },
    {
        "name": "split16_group64_transposed_low_cancellation",
        "M": 3,
        "K": 1024,
        "N_packed": 7,
        "group_size": 64,
        "split_k": 16,
        "layout": "transposed",
        "profile": "low_cancellation",
        "input_padding": 7,
        "qweight_padding": 5,
        "scales_padding": 5,
        "qzeros_padding": 3,
    },
    {
        "name": "split32_per_tensor_inner_strided_low_cancellation",
        "M": 2,
        "K": 2048,
        "N_packed": 3,
        "group_size": "K",
        "split_k": 32,
        "layout": "inner_strided",
        "profile": "low_cancellation",
        "input_padding": 5,
        "qweight_padding": 2,
        "scales_padding": 3,
        "qzeros_padding": 2,
    },
]


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
    """Delegate launches while recording the target kernel's split metadata."""

    def __init__(self, kernel, *, torch=None, capture_output=False):
        self.kernel = kernel
        self.torch = torch
        self.capture_output = capture_output
        self.launches = []

    @staticmethod
    def _resolve_grid(grid, kwargs):
        if callable(grid):
            grid = grid(dict(kwargs))
        if grid is None:
            return None
        return tuple(grid)

    def _record(self, grid, args, kwargs):
        record = {
            "grid": self._resolve_grid(grid, kwargs),
            "args": tuple(args),
            "kwargs": dict(kwargs),
            "split_k": kwargs.get("SPLIT_K"),
            "block_m": kwargs.get("BLOCK_SIZE_M"),
            "block_n": kwargs.get("BLOCK_SIZE_N"),
            "block_k": kwargs.get("BLOCK_SIZE_K"),
        }
        self.launches.append(record)
        return record

    @staticmethod
    def _operand(args, kwargs, index):
        if index < len(args):
            return args[index]
        return kwargs.get(AWQ_TARGET_PARAMETER_NAMES[index])

    def _prepare_capture(self, record, args, kwargs):
        if not self.capture_output:
            return
        if self.torch is None:
            raise RuntimeError("AWQ launch capture requires torch")
        inputs = tuple(
            self._operand(args, kwargs, index) for index in (0, 1, 4, 3)
        )
        output = self._operand(args, kwargs, 2)
        if not all(isinstance(value, self.torch.Tensor) for value in inputs):
            raise RuntimeError("AWQ input capture requires four tensor operands")
        if not isinstance(output, self.torch.Tensor):
            raise RuntimeError("AWQ output capture requires a tensor C operand")

        # Freeze the values actually offered to the target before it runs. The
        # candidate may materialize a valid contiguous copy, so value/metadata
        # binding is intentional; retaining mutable argument references alone
        # would let a wrapper repair a dummy operand after the launch.
        record["inputs_before_launch"] = tuple(value.clone() for value in inputs)
        self.torch.cuda.synchronize()
        output.fill_(float("nan"))

    def _finish_output_capture(self, record, args, kwargs):
        if not self.capture_output:
            return
        self.torch.cuda.synchronize()
        output = self._operand(args, kwargs, 2)
        record["output_after_launch"] = output.clone()

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record_and_launch(*args, **kwargs):
            record = self._record(grid, args, kwargs)
            self._prepare_capture(record, args, kwargs)
            result = launch(*args, **kwargs)
            self._finish_output_capture(record, args, kwargs)
            return result

        return record_and_launch

    def run(self, *args, **kwargs):
        record = self._record(kwargs.get("grid"), args, kwargs)
        self._prepare_capture(record, args, kwargs)
        result = self.kernel.run(*args, **kwargs)
        self._finish_output_capture(record, args, kwargs)
        return result

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def _unpack_awq_word(packed_value):
    """Unpack one signed int32 as an unsigned AWQ two's-complement word."""
    unsigned_value = int(packed_value) & 0xFFFFFFFF
    return tuple(
        (unsigned_value >> (nibble_index * 4)) & 0xF
        for nibble_index in AWQ_ORDER
    )


def reference_awq_gemm(input_tensor, qweight, scales, qzeros, group_size):
    """CPU reference: dequantize then matmul."""
    import torch
    K, N_packed = qweight.shape
    N = N_packed * 8
    M = input_tensor.shape[0]

    # Dequantize weights
    dequant = torch.zeros((K, N), dtype=torch.float32, device="cpu")
    qweight_cpu = qweight.cpu().to(torch.int32)
    qzeros_cpu = qzeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()

    for row in range(K):
        group_idx = row // group_size
        for col_packed in range(N_packed):
            weight_values = _unpack_awq_word(
                qweight_cpu[row, col_packed].item()
            )
            zero_values = _unpack_awq_word(
                qzeros_cpu[group_idx, col_packed].item()
            )
            for bit_idx, (weight_val, zero_val) in enumerate(
                zip(weight_values, zero_values)
            ):
                out_col = col_packed * 8 + bit_idx
                scale_val = scales_cpu[group_idx, out_col].item()
                dequant[row, out_col] = (weight_val - zero_val) * scale_val

    input_cpu = input_tensor.cpu().float()
    result = input_cpu @ dequant
    return result.to(scales.dtype)


def _guarded_storage_shape(rows, columns, padding, layout):
    if layout == "row_padded":
        return rows + 2, columns + 2 * padding
    if layout == "inner_strided":
        return rows + 2, 2 * columns + 2 * padding
    if layout == "transposed":
        return columns + 2 * padding, rows + 2
    raise ValueError(f"unsupported guarded layout: {layout}")


def _guarded_logical_view(storage, rows, columns, padding, layout):
    if layout == "row_padded":
        logical = storage[1 : rows + 1, padding : padding + columns]
    elif layout == "inner_strided":
        logical = storage[
            1 : rows + 1,
            padding : padding + 2 * columns : 2,
        ]
    elif layout == "transposed":
        logical = storage[
            padding : padding + columns,
            1 : rows + 1,
        ].T
    else:
        raise ValueError(f"unsupported guarded layout: {layout}")

    assert tuple(logical.shape) == (rows, columns)
    assert logical.storage_offset() > 0
    assert all(stride > 0 for stride in logical.stride())
    assert not logical.is_contiguous()
    return logical


def _make_guarded_float_matrix(
    torch,
    rows,
    columns,
    padding,
    *,
    device,
    layout,
    profile,
    role,
):
    """Return one finite logical float16 matrix and its guarded allocation."""
    storage = torch.randn(
        _guarded_storage_shape(rows, columns, padding, layout),
        device=device,
        dtype=torch.float16,
    )
    logical = _guarded_logical_view(
        storage, rows, columns, padding, layout
    )

    if profile == "high_dynamic":
        if role == "scales":
            logical.abs_().mul_(0.1).add_(0.01)
    elif profile == "low_cancellation":
        row_ids = torch.arange(rows, device=device)[:, None]
        column_ids = torch.arange(columns, device=device)[None, :]
        if role == "input":
            signs = ((row_ids + column_ids) % 2) * 2 - 1
            magnitudes = 0.125 + (column_ids % 3) * 0.0625
            logical.copy_((signs * magnitudes).to(torch.float16))
        elif role == "scales":
            magnitudes = 2.0**-7 + (column_ids % 3) * 2.0**-8
            logical.copy_(magnitudes.expand(rows, columns).to(torch.float16))
        else:
            raise ValueError(f"unsupported float role: {role}")
    else:
        raise ValueError(f"unsupported data profile: {profile}")

    assert bool(torch.isfinite(logical).all().item())
    if role == "scales":
        assert bool((logical > 0).all().item())
    return logical, storage


def _make_guarded_int32_matrix(
    torch,
    rows,
    columns,
    padding,
    *,
    device,
    layout,
):
    """Return a positive-stride packed int32 view and guarded allocation."""
    storage = torch.randint(
        0,
        2**31,
        _guarded_storage_shape(rows, columns, padding, layout),
        device=device,
        dtype=torch.int32,
    )
    logical = _guarded_logical_view(
        storage, rows, columns, padding, layout
    )
    return logical, storage


def _inject_signed_packed_words(qweight, qzeros):
    """Install noncorresponding signed words in both packed input families."""
    qweight[0, 0] = -(2**31)
    qweight[1, 1] = -1
    qweight[2, 2] = SIGNED_89ABCDEF

    # Deliberately differ from qweight at every corresponding location so a
    # signed weight word never disappears through weight-zero cancellation.
    qzeros[0, 0] = 0x01234567
    qzeros[0, 1] = SIGNED_87654321
    qzeros[0, 2] = -(2**31)
    if qzeros.shape[0] > 1:
        qzeros[-1, -1] = -1
        qweight[-1, -1] = 0x13579BDF


def _make_awq_correctness_inputs(torch, case, device):
    M = case["M"]
    K = case["K"]
    N_packed = case["N_packed"]
    N = N_packed * 8
    group_size = K if case["group_size"] == "K" else case["group_size"]
    assert group_size in (32, 64, 128) or group_size == K
    assert K % group_size == 0
    assert K % (32 * case["split_k"]) == 0
    assert K // (32 * case["split_k"]) >= 2
    num_groups = K // group_size
    common = {
        "device": device,
        "layout": case["layout"],
    }

    input_tensor, input_storage = _make_guarded_float_matrix(
        torch,
        M,
        K,
        case["input_padding"],
        profile=case["profile"],
        role="input",
        **common,
    )
    qweight, qweight_storage = _make_guarded_int32_matrix(
        torch,
        K,
        N_packed,
        case["qweight_padding"],
        **common,
    )
    scales, scales_storage = _make_guarded_float_matrix(
        torch,
        num_groups,
        N,
        case["scales_padding"],
        profile=case["profile"],
        role="scales",
        **common,
    )
    qzeros, qzeros_storage = _make_guarded_int32_matrix(
        torch,
        num_groups,
        N_packed,
        case["qzeros_padding"],
        **common,
    )
    _inject_signed_packed_words(qweight, qzeros)
    return (
        input_tensor,
        qweight,
        scales,
        qzeros,
        group_size,
        (
            ("input", input_storage),
            ("qweight", qweight_storage),
            ("scales", scales_storage),
            ("qzeros", qzeros_storage),
        ),
    )


def _validate_awq_output(
    torch,
    result,
    ref,
    expected_shape,
    expected_dtype,
    expected_device,
    protected_storage_ptrs,
    label,
    invocation,
):
    prefix = f"{label}, invocation {invocation}"
    if tuple(result.shape) != expected_shape:
        return f"{prefix}: wrong output shape {tuple(result.shape)}, expected {expected_shape}"
    if result.dtype != expected_dtype:
        return f"{prefix}: wrong output dtype {result.dtype}, expected {expected_dtype}"
    if result.device != expected_device:
        return f"{prefix}: wrong output device {result.device}, expected {expected_device}"
    if result.untyped_storage().data_ptr() in protected_storage_ptrs:
        return f"{prefix}: output aliases an input or its backing storage"
    if not bool(torch.isfinite(result).all().item()):
        return f"{prefix}: output contains a non-finite value"
    if not torch.allclose(
        result.float(),
        ref.float(),
        atol=AWQ_ATOL,
        rtol=AWQ_RTOL,
    ):
        max_diff = (result.float() - ref.float()).abs().max().item()
        return f"{prefix}: max diff = {max_diff:.6f}"
    return None


def _validate_awq_launch_batch(
    torch,
    launches,
    M,
    N,
    K,
    group_size,
    split_k,
    block_sizes,
    expected_inputs,
    result,
    protected_storage_ptrs,
    label,
    invocation,
):
    prefix = f"{label}, invocation {invocation}"
    if len(launches) != 1:
        return f"{prefix}: expected one target-kernel launch, observed {len(launches)}"
    launch = launches[0]
    block_m, block_n, block_k = block_sizes
    expected_grid = (
        ((M + block_m - 1) // block_m) * ((N + block_n - 1) // block_n),
        split_k,
    )
    expected_metadata = {
        "split_k": split_k,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
    }
    for key, expected in expected_metadata.items():
        observed = launch.get(key)
        if type(observed) is not int or observed != expected:
            return (
                f"{prefix}: requested {key}={expected} was not attested; "
                f"observed={observed}"
            )
    if launch.get("grid") != expected_grid:
        return (
            f"{prefix}: expected resolved grid {expected_grid}, "
            f"observed {launch.get('grid')}"
        )
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

    observed_inputs = launch.get("inputs_before_launch")
    if not isinstance(observed_inputs, tuple) or len(observed_inputs) != 4:
        return f"{prefix}: target inputs were not frozen at the launch boundary"
    for input_name, observed, expected in zip(
        ("input", "qweight", "scales", "qzeros"),
        observed_inputs,
        expected_inputs,
    ):
        if not isinstance(observed, torch.Tensor):
            return f"{prefix}: target {input_name} operand is not a tensor"
        if (
            tuple(observed.shape) != tuple(expected.shape)
            or observed.dtype != expected.dtype
            or observed.device != expected.device
            or not torch.equal(observed, expected)
        ):
            return f"{prefix}: target {input_name} operand is not bound to this case"

    dimensions = (operand(5), operand(6), operand(7), operand(8))
    if (
        any(type(value) is not int for value in dimensions)
        or dimensions != (M, N, K, group_size)
    ):
        return (
            f"{prefix}: target dimensions {dimensions} != "
            f"{(M, N, K, group_size)}"
        )

    target_output = operand(2)
    if not isinstance(target_output, torch.Tensor):
        return f"{prefix}: target output operand is not a tensor"
    if (
        tuple(target_output.shape[-2:]) != (M, N)
        or target_output.dtype != torch.float32
        or target_output.device != result.device
        or target_output.untyped_storage().data_ptr() in protected_storage_ptrs
    ):
        return f"{prefix}: target output operand violates the output binding"
    output_after_launch = launch.get("output_after_launch")
    if not isinstance(output_after_launch, torch.Tensor):
        return f"{prefix}: target output was not captured at the launch boundary"
    if target_output.ndim == 3 and target_output.shape[0] == split_k:
        bound_output = output_after_launch.sum(0).to(result.dtype)
    elif target_output.ndim == 2:
        bound_output = output_after_launch
    else:
        return f"{prefix}: target output cannot be reduced to the returned result"
    if not torch.equal(bound_output, result):
        return f"{prefix}: returned result is not derived from the target output"
    return None


def _validate_awq_launch_controls(keyword, resolved_grid, prefix):
    """Reject runtime data hidden in otherwise legitimate Triton controls."""
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
        if isinstance(value, int):
            value = (value,)
        elif isinstance(value, (tuple, list)):
            value = tuple(value)
        else:
            return f"{prefix}: launch control grid must be an integer sequence"
        if (
            not value
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in value
            )
            or value != resolved_grid
        ):
            return f"{prefix}: launch control grid does not match the resolved grid"
    return None


def _check_awq_case(
    torch,
    mod,
    input_tensor,
    qweight,
    scales,
    qzeros,
    group_size,
    split_k,
    label,
    integrity_guard,
    *,
    backing_storages=(),
    block_sizes=(32, 32, 32),
):
    protected_inputs = (
        ("input", input_tensor, input_tensor.clone()),
        ("qweight", qweight, qweight.clone()),
        ("scales", scales, scales.clone()),
        ("qzeros", qzeros, qzeros.clone()),
    )
    frozen_backing = [
        (name, storage, storage.clone()) for name, storage in backing_storages
    ]
    ref = reference_awq_gemm(
        protected_inputs[0][2],
        protected_inputs[1][2],
        protected_inputs[2][2],
        protected_inputs[3][2],
        group_size,
    ).to(input_tensor.device)
    if not bool(torch.isfinite(ref).all().item()):
        return f"{label}: evaluator reference contains a non-finite value"

    original_kernel = mod.awq_gemm_kernel
    recorder = _KernelLaunchRecorder(
        original_kernel, torch=torch, capture_output=True
    )
    mod.awq_gemm_kernel = recorder
    results = []
    launch_batches = []
    try:
        for _invocation in range(2):
            first_launch = len(recorder.launches)
            results.append(
                mod.awq_gemm_triton(
                    input_tensor,
                    qweight,
                    scales,
                    qzeros,
                    split_k,
                    block_size_m=block_sizes[0],
                    block_size_n=block_sizes[1],
                    block_size_k=block_sizes[2],
                )
            )
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return integrity_error
            torch.cuda.synchronize()
            launch_batches.append(recorder.launches[first_launch:])
    finally:
        mod.awq_gemm_kernel = original_kernel

    protected_storage_ptrs = {
        tensor.untyped_storage().data_ptr()
        for _, tensor, _ in protected_inputs
    }
    protected_storage_ptrs.update(
        storage.untyped_storage().data_ptr() for _, storage in backing_storages
    )
    for invocation, (launches, result) in enumerate(
        zip(launch_batches, results), start=1
    ):
        launch_error = _validate_awq_launch_batch(
            torch,
            launches,
            input_tensor.shape[0],
            qweight.shape[1] * 8,
            qweight.shape[0],
            group_size,
            split_k,
            block_sizes,
            (input_tensor, qweight, scales, qzeros),
            result,
            protected_storage_ptrs,
            label,
            invocation,
        )
        if launch_error is not None:
            return launch_error

    expected_shape = (input_tensor.shape[0], qweight.shape[1] * 8)
    for invocation, result in enumerate(results, start=1):
        error = _validate_awq_output(
            torch,
            result,
            ref,
            expected_shape,
            scales.dtype,
            input_tensor.device,
            protected_storage_ptrs,
            label,
            invocation,
        )
        if error is not None:
            return error

    first_storage = results[0].untyped_storage().data_ptr()
    second_storage = results[1].untyped_storage().data_ptr()
    if first_storage == second_storage:
        return f"{label}: repeated calls did not return fresh storage"
    if not torch.equal(results[0], results[1]):
        max_repeat_diff = (
            (results[0].float() - results[1].float()).abs().max().item()
        )
        return f"{label}: repeated calls are nondeterministic; max diff = {max_repeat_diff:.6f}"

    for name, observed, frozen in protected_inputs:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated protected input {name}"
    for name, observed, frozen in frozen_backing:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate wrote outside logical {name} view"
    return None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "awq_gemm_triton"), "Missing awq_gemm_triton"
        assert hasattr(mod, "awq_gemm_kernel"), "Missing awq_gemm_kernel"
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
    dtype = torch.float16

    for i, (M, K, N_packed, group_size, split_k) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            N = N_packed * 8
            num_groups = K // group_size

            input_tensor = torch.randn(M, K, device=device, dtype=dtype)
            qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            qzeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)

            error = _check_awq_case(
                torch,
                mod,
                input_tensor,
                qweight,
                scales,
                qzeros,
                group_size,
                split_k,
                f"Shape {i+1} (M={M}, K={K}, N_packed={N_packed}, G={group_size})",
                integrity_guard,
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, (
                f"Shape {i+1} (M={M}, K={K}, N_packed={N_packed}, G={group_size}): "
                f"exception: {e}"
            )

    for case_index, case in enumerate(CORRECTNESS_CASES):
        name = case["name"]
        try:
            torch.manual_seed(142 + case_index)
            (
                input_tensor,
                qweight,
                scales,
                qzeros,
                group_size,
                backing_storages,
            ) = _make_awq_correctness_inputs(torch, case, device)
            error = _check_awq_case(
                torch,
                mod,
                input_tensor,
                qweight,
                scales,
                qzeros,
                group_size,
                case["split_k"],
                f"Contract case {name}",
                integrity_guard,
                backing_storages=backing_storages,
                block_sizes=case.get("block_sizes", (32, 32, 32)),
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"Contract case {name}: exception: {e}"

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

    for test_idx, (M, K, N_packed, group_size, split_k) in enumerate(TEST_SHAPES):
        try:
            N = N_packed * 8
            num_groups = K // group_size

            torch.manual_seed(0)
            input_tensor = torch.randn(M, K, device=device, dtype=dtype)
            qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            qzeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)

            def _bench_fn():
                mod.awq_gemm_triton(input_tensor, qweight, scales, qzeros, split_k)
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
                    "K": K,
                    "N_packed": N_packed,
                    "group_size": group_size,
                    "split_k": split_k
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "K": K,
                    "N_packed": N_packed,
                    "group_size": group_size,
                    "split_k": split_k
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
            "schema": "aka.awq-gemm-correctness/v1",
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "num_contract_cases": len(CORRECTNESS_CASES),
            "contract_cases": [case["name"] for case in CORRECTNESS_CASES],
            "performance_shape_seeds": [42 + index for index in range(len(TEST_SHAPES))],
            "contract_case_seeds": [142 + index for index in range(len(CORRECTNESS_CASES))],
            "atol": AWQ_ATOL,
            "rtol": AWQ_RTOL,
            "repeated_invocations_per_case": 2,
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

#!/usr/bin/env python3
"""Task runner for triton2triton/triton_matmul_persistent"""
import sys
import os
import json
import argparse
import importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_matmul_persistent"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_matmul_persistent.py")

# Test configurations: (M, N, K)
TEST_SHAPES = [
    (128, 128, 64),
    (256, 512, 128),
    (512, 256, 256),
    (1024, 1024, 512),
    (64, 2048, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

MATMUL_ATOL = 1e-2
MATMUL_RTOL = 1e-2
SIGNED_INT32_MAX = 2**31 - 1
MAX_CORRECTNESS_STORAGE_BYTES = 512 * 1024 * 1024
MATMUL_TRITON_LAUNCH_CONTROL_NAMES = {
    "grid",
    "warmup",
    "num_warps",
    "num_stages",
    "num_ctas",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
}
MATMUL_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS = {
    "num_warps",
    "num_stages",
    "num_ctas",
    "matrix_instr_nonkdim",
    "kpack",
}

_TRUSTED_TORCH_PATHS = (
    "Tensor",
    "allclose",
    "empty",
    "equal",
    "isfinite",
    "manual_seed",
    "mm",
    "randn",
    "cuda.Event",
    "cuda.get_device_properties",
    "cuda.synchronize",
)

# Extra public-contract coverage. These cases never enter run_performance.
# The largest case is deliberately bounded while still launching more tiles
# than an MI355X has compute units; >2^31 indexing is allocation-free below.
CORRECTNESS_CASES = [
    {
        "name": "float16_tail_bias_guarded",
        "M": 131,
        "N": 259,
        "K": 67,
        "dtype": "float16",
        "a_layout": "row_padded",
        "b_layout": "transposed",
        "bias_layout": "inner_strided",
        "padding": 5,
        "requires_partial_group": True,
        "requires_persistent": False,
    },
    {
        "name": "bfloat16_persistent_partial_group",
        "M": 2049,
        "N": 8193,
        "K": 33,
        "dtype": "bfloat16",
        "a_layout": "inner_strided",
        "b_layout": "row_padded",
        "bias_layout": None,
        "padding": 3,
        "requires_partial_group": True,
        "requires_persistent": True,
    },
    {
        "name": "float16_transposed_strided_bias",
        "M": 257,
        "N": 385,
        "K": 33,
        "dtype": "float16",
        "a_layout": "transposed",
        "b_layout": "inner_strided",
        "bias_layout": "contiguous",
        "padding": 7,
        "requires_partial_group": True,
        "requires_persistent": False,
    },
]

VIRTUAL_INDEX_CASES = [
    {
        "name": "a_small_view_crosses_int32",
        "a_shape": (2, 3),
        "a_stride": (2**31, 1),
        "b_shape": (3, 2),
        "b_stride": (2, 1),
        "expected": (True, False, False),
    },
    {
        "name": "b_small_view_crosses_int32",
        "a_shape": (2, 3),
        "a_stride": (3, 1),
        "b_shape": (3, 2),
        "b_stride": (2**30, 1),
        "expected": (False, True, False),
    },
    {
        "name": "signed_int32_boundary_is_safe",
        "a_shape": (2, 2),
        "a_stride": (SIGNED_INT32_MAX - 1, 1),
        "b_shape": (2, 2),
        "b_stride": (2, 1),
        "expected": (False, False, False),
    },
    {
        "name": "contiguous_output_crosses_int32",
        "a_shape": (65536, 1),
        "a_stride": (1, 1),
        "b_shape": (1, 32769),
        "b_stride": (32769, 1),
        "expected": (False, False, True),
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
    """Delegate target launches while retaining resolved launch evidence."""

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
        if isinstance(grid, int):
            return (grid,)
        return tuple(grid)

    def _record(self, grid, args, kwargs):
        record = {
            "grid": self._resolve_grid(grid, kwargs),
            "args": tuple(args),
            "kwargs": dict(kwargs),
        }
        self.launches.append(record)
        return record

    def _prepare_output_capture(self, args):
        if not self.capture_output:
            return
        if self.torch is None or len(args) < 3:
            raise RuntimeError("output capture requires torch and a C argument")
        # Poison every output element immediately before the target launch.
        # A dummy launch followed by wrapper-side computation therefore cannot
        # masquerade as target-kernel engagement.
        args[2].fill_(float("nan"))

    def _finish_output_capture(self, record, args):
        if not self.capture_output:
            return
        self.torch.cuda.synchronize()
        record["output_after_launch"] = args[2].clone()

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record_and_launch(*args, **kwargs):
            record = self._record(grid, args, kwargs)
            self._prepare_output_capture(args)
            result = launch(*args, **kwargs)
            self._finish_output_capture(record, args)
            return result

        return record_and_launch

    def run(self, *args, **kwargs):
        record = self._record(kwargs.get("grid"), args, kwargs)
        self._prepare_output_capture(args)
        result = self.kernel.run(*args, **kwargs)
        self._finish_output_capture(record, args)
        return result

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _max_relative_element_offset(shape, strides):
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same rank")
    if any(int(size) <= 0 for size in shape):
        raise ValueError("tensor dimensions must be positive")
    if any(int(stride) <= 0 for stride in strides):
        raise ValueError("tensor strides must be positive")
    return sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(shape, strides)
    )


def _requires_int64_index(tensor):
    return _max_relative_element_offset(
        tensor.shape, tensor.stride()
    ) > SIGNED_INT32_MAX


def _guarded_storage_shape(rows, columns, padding, layout):
    if layout == "contiguous":
        return rows, columns
    if layout == "row_padded":
        return rows + 2, columns + 2 * padding
    if layout == "inner_strided":
        return rows + 2, 2 * columns + 2 * padding
    if layout == "transposed":
        return columns + 2 * padding, rows + 2
    raise ValueError(f"unsupported matrix layout: {layout}")


def _guarded_logical_view(storage, rows, columns, padding, layout):
    if layout == "contiguous":
        logical = storage
    elif layout == "row_padded":
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
        raise ValueError(f"unsupported matrix layout: {layout}")
    assert tuple(logical.shape) == (rows, columns)
    assert all(stride > 0 for stride in logical.stride())
    if layout != "contiguous":
        assert logical.storage_offset() > 0
        assert not logical.is_contiguous()
    return logical


def _make_guarded_matrix(
    torch,
    rows,
    columns,
    *,
    device,
    dtype,
    padding,
    layout,
    value_scale=1.0,
):
    storage = torch.randn(
        _guarded_storage_shape(rows, columns, padding, layout),
        device=device,
        dtype=dtype,
    )
    if value_scale != 1.0:
        storage.mul_(value_scale)
    logical = _guarded_logical_view(
        storage, rows, columns, padding, layout
    )
    assert bool(torch.isfinite(logical).all().item())
    return logical, storage


def _make_guarded_bias(
    torch,
    length,
    *,
    device,
    dtype,
    padding,
    layout,
    value_scale=1.0,
):
    if layout == "contiguous":
        storage = torch.randn(length, device=device, dtype=dtype)
        logical = storage
    elif layout == "inner_strided":
        storage = torch.randn(
            2 * length + 2 * padding,
            device=device,
            dtype=dtype,
        )
        logical = storage[padding : padding + 2 * length : 2]
        assert logical.storage_offset() > 0
        assert not logical.is_contiguous()
    else:
        raise ValueError(f"unsupported bias layout: {layout}")
    if value_scale != 1.0:
        storage.mul_(value_scale)
    assert tuple(logical.shape) == (length,)
    assert logical.stride(0) > 0
    assert bool(torch.isfinite(logical).all().item())
    return logical, storage


def _matrix_storage_elements(rows, columns, padding, layout):
    storage_shape = _guarded_storage_shape(
        rows, columns, padding, layout
    )
    return storage_shape[0] * storage_shape[1]


def _estimate_case_storage_bytes(case):
    item_size = {
        "float16": 2,
        "bfloat16": 2,
    }[case["dtype"]]
    M, N, K = case["M"], case["N"], case["K"]
    padding = case["padding"]
    backing_elements = _matrix_storage_elements(
        M, K, padding, case["a_layout"]
    ) + _matrix_storage_elements(
        K, N, padding, case["b_layout"]
    )
    if case["bias_layout"] == "contiguous":
        backing_elements += N
    elif case["bias_layout"] == "inner_strided":
        backing_elements += 2 * N + 2 * padding

    # Backings, frozen backing copies, logical snapshots, two live results,
    # the float32 reference, and conservative temporary headroom.
    input_bytes = 3 * backing_elements * item_size
    output_bytes = M * N * (4 * item_size + 16)
    return input_bytes + output_bytes


def _make_contract_case_inputs(torch, case, device):
    estimated_bytes = _estimate_case_storage_bytes(case)
    if estimated_bytes > MAX_CORRECTNESS_STORAGE_BYTES:
        raise RuntimeError(
            f"{case['name']}: estimated correctness storage "
            f"{estimated_bytes} exceeds {MAX_CORRECTNESS_STORAGE_BYTES}"
        )
    dtype = getattr(torch, case["dtype"])
    common = {
        "device": device,
        "dtype": dtype,
        "padding": case["padding"],
        "value_scale": 1.0,
    }
    a, a_storage = _make_guarded_matrix(
        torch,
        case["M"],
        case["K"],
        layout=case["a_layout"],
        **common,
    )
    b, b_storage = _make_guarded_matrix(
        torch,
        case["K"],
        case["N"],
        layout=case["b_layout"],
        **common,
    )
    bias = None
    backing = [("a", a_storage), ("b", b_storage)]
    if case["bias_layout"] is not None:
        bias, bias_storage = _make_guarded_bias(
            torch,
            case["N"],
            layout=case["bias_layout"],
            **common,
        )
        backing.append(("bias", bias_storage))
    return a, b, bias, tuple(backing)


def _grouped_tile_coordinate(
    tile_id, num_pid_m, num_pid_n, group_size_m
):
    num_pid_in_group = group_size_m * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * group_size_m
    current_group_size = min(
        num_pid_m - first_pid_m, group_size_m
    )
    if current_group_size <= 0:
        raise ValueError("tile id is outside the grouped schedule")
    pid_m = first_pid_m + tile_id % current_group_size
    pid_n = (tile_id % num_pid_in_group) // current_group_size
    return pid_m, pid_n


def _validate_grouped_schedule(num_pid_m, num_pid_n, group_size_m):
    coordinates = [
        _grouped_tile_coordinate(
            tile_id, num_pid_m, num_pid_n, group_size_m
        )
        for tile_id in range(num_pid_m * num_pid_n)
    ]
    expected = {
        (pid_m, pid_n)
        for pid_m in range(num_pid_m)
        for pid_n in range(num_pid_n)
    }
    if len(coordinates) != len(set(coordinates)):
        return "grouped tile schedule is not injective"
    if set(coordinates) != expected:
        return "grouped tile schedule does not cover every output tile"
    return None


def _validate_matmul_launch(
    torch,
    launch,
    a,
    b,
    bias,
    result,
    num_sms,
    label,
    invocation,
    *,
    requires_partial_group,
    requires_persistent,
):
    prefix = f"{label}, invocation {invocation}"
    args = launch["args"]
    kwargs = launch["kwargs"]
    if len(args) != 13:
        return (
            f"{prefix}: target launch exposed {len(args)} positional arguments; "
            "expected exactly 13"
        )
    allowed_keywords = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_SMS",
        "A_LARGE",
        "B_LARGE",
        "C_LARGE",
        "HAS_BIAS",
    } | MATMUL_TRITON_LAUNCH_CONTROL_NAMES
    unexpected_keywords = set(kwargs).difference(allowed_keywords)
    if unexpected_keywords:
        return (
            f"{prefix}: target launch has unexpected keyword operands "
            f"{sorted(unexpected_keywords)}"
        )
    launch_control_error = _validate_matmul_launch_controls(
        kwargs, launch.get("grid"), prefix
    )
    if launch_control_error is not None:
        return launch_control_error
    if args[0] is not a or args[1] is not b:
        return f"{prefix}: target launch did not consume the evaluator inputs directly"
    if args[2] is not result:
        return f"{prefix}: returned tensor is not the target kernel's C argument"
    output_after_launch = launch.get("output_after_launch")
    if output_after_launch is None:
        return f"{prefix}: target-kernel output capture is missing"
    if not torch.equal(output_after_launch, result):
        return (
            f"{prefix}: returned output was modified after the target launch "
            "or the target launch did not populate C"
        )

    metadata_names = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_SMS",
    )
    metadata = {}
    for name in metadata_names:
        value = kwargs.get(name)
        if type(value) is not int or value <= 0:
            return f"{prefix}: missing positive integer launch metadata {name}"
        metadata[name] = value
    if metadata["NUM_SMS"] != num_sms:
        return (
            f"{prefix}: NUM_SMS={metadata['NUM_SMS']} does not match "
            f"device value {num_sms}"
        )

    M, K = a.shape
    _, N = b.shape
    block_m = metadata["BLOCK_SIZE_M"]
    block_n = metadata["BLOCK_SIZE_N"]
    num_pid_m = _ceil_div(M, block_m)
    num_pid_n = _ceil_div(N, block_n)
    num_tiles = num_pid_m * num_pid_n
    expected_grid = (min(num_sms, num_tiles),)
    if launch["grid"] != expected_grid:
        return (
            f"{prefix}: expected resolved persistent grid {expected_grid}, "
            f"observed {launch['grid']}"
        )
    if requires_persistent and num_tiles <= num_sms:
        return (
            f"{prefix}: correctness case did not exercise persistent reuse "
            f"({num_tiles} tiles, {num_sms} SMs)"
        )

    group_size = metadata["GROUP_SIZE_M"]
    if requires_partial_group and (
        group_size == 1 or num_pid_m % group_size == 0
    ):
        return (
            f"{prefix}: launch did not exercise a partial final M group "
            f"(num_pid_m={num_pid_m}, GROUP_SIZE_M={group_size})"
        )
    schedule_error = _validate_grouped_schedule(
        num_pid_m, num_pid_n, group_size
    )
    if schedule_error is not None:
        return f"{prefix}: {schedule_error}"

    expected_dimensions = (M, N, K)
    observed_dimensions = tuple(args[4:7])
    if (
        any(type(value) is not int for value in observed_dimensions)
        or observed_dimensions != expected_dimensions
    ):
        return (
            f"{prefix}: launch dimensions {observed_dimensions} != "
            f"{expected_dimensions}"
        )
    expected_strides = (
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        result.stride(0),
        result.stride(1),
    )
    observed_strides = tuple(args[7:13])
    if (
        any(type(value) is not int for value in observed_strides)
        or observed_strides != expected_strides
    ):
        return (
            f"{prefix}: launch strides {observed_strides} != "
            f"{expected_strides}"
        )

    expected_flags = {
        "A_LARGE": _requires_int64_index(a),
        "B_LARGE": _requires_int64_index(b),
        "C_LARGE": _requires_int64_index(result),
        "HAS_BIAS": bias is not None,
    }
    for name, expected in expected_flags.items():
        if kwargs.get(name) is not expected:
            return (
                f"{prefix}: {name}={kwargs.get(name)!r}, "
                f"expected {expected!r}"
            )

    launched_bias = args[3]
    if bias is None:
        if launched_bias is not None:
            return f"{prefix}: disabled bias was not forwarded as None"
    else:
        if launched_bias is None:
            return f"{prefix}: enabled bias was not forwarded"
        if tuple(launched_bias.shape) != tuple(bias.shape):
            return f"{prefix}: target launch received the wrong bias shape"
        if launched_bias.dtype != bias.dtype or launched_bias.device != bias.device:
            return f"{prefix}: target launch received incompatible bias metadata"
        if launched_bias.stride(0) != 1 or not launched_bias.is_contiguous():
            return f"{prefix}: target launch requires a materialized unit-stride bias"
        if not torch.equal(launched_bias, bias):
            return f"{prefix}: materialized bias changed logical values"
    return None


def _validate_matmul_launch_controls(kwargs, resolved_grid, prefix):
    """Permit tuning scalars without permitting hidden runtime operands."""
    for name in MATMUL_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS:
        if name not in kwargs:
            continue
        value = kwargs[name]
        if type(value) is not int or value <= 0:
            return f"{prefix}: launch control {name} must be a positive integer"
    if "waves_per_eu" in kwargs:
        value = kwargs["waves_per_eu"]
        if type(value) is not int or value < 0:
            return f"{prefix}: launch control waves_per_eu must be a nonnegative integer"
    if "warmup" in kwargs and type(kwargs["warmup"]) is not bool:
        return f"{prefix}: launch control warmup must be a boolean"
    if "grid" in kwargs:
        value = kwargs["grid"]
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


def _validate_matmul_launch_batch(
    torch,
    launches,
    a,
    b,
    bias,
    result,
    num_sms,
    label,
    invocation,
    *,
    requires_partial_group,
    requires_persistent,
):
    if len(launches) != 1:
        return (
            f"{label}, invocation {invocation}: expected one target-kernel "
            f"launch, observed {len(launches)}"
        )
    return _validate_matmul_launch(
        torch,
        launches[0],
        a,
        b,
        bias,
        result,
        num_sms,
        label,
        invocation,
        requires_partial_group=requires_partial_group,
        requires_persistent=requires_persistent,
    )


def _validate_matmul_output(
    torch,
    result,
    ref,
    a,
    protected_storage_ptrs,
    label,
    invocation,
):
    prefix = f"{label}, invocation {invocation}"
    if not isinstance(result, torch.Tensor):
        return f"{prefix}: result is not a torch.Tensor"
    if tuple(result.shape) != tuple(ref.shape):
        return f"{prefix}: wrong output shape {tuple(result.shape)}"
    if result.dtype != a.dtype:
        return f"{prefix}: wrong output dtype {result.dtype}, expected {a.dtype}"
    if result.device != a.device:
        return f"{prefix}: wrong output device {result.device}, expected {a.device}"
    if not result.is_contiguous():
        return f"{prefix}: output is not contiguous"
    if result.untyped_storage().data_ptr() in protected_storage_ptrs:
        return f"{prefix}: output aliases an input or backing allocation"
    if not bool(torch.isfinite(result).all().item()):
        return f"{prefix}: output contains a non-finite value"
    if not torch.allclose(
        result.float(), ref.float(), atol=MATMUL_ATOL, rtol=MATMUL_RTOL
    ):
        max_diff = (result.float() - ref.float()).abs().max().item()
        return f"{prefix}: max diff = {max_diff:.6f}"
    return None


def _check_matmul_case(
    torch,
    mod,
    a,
    b,
    bias,
    label,
    integrity_guard,
    *,
    backing_storages=(),
    requires_partial_group=False,
    requires_persistent=False,
):
    protected_inputs = [
        ("a", a, a.clone()),
        ("b", b, b.clone()),
    ]
    if bias is not None:
        protected_inputs.append(("bias", bias, bias.clone()))
    frozen_backing = [
        (name, storage, storage.clone())
        for name, storage in backing_storages
    ]
    ref = torch.mm(protected_inputs[0][2].float(), protected_inputs[1][2].float())
    if bias is not None:
        ref = ref + protected_inputs[2][2].float()
    ref = ref.to(a.dtype)
    if not bool(torch.isfinite(ref).all().item()):
        return f"{label}: evaluator reference contains a non-finite value"

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    original_kernel = mod.matmul_kernel_persistent
    recorder = _KernelLaunchRecorder(
        original_kernel, torch=torch, capture_output=True
    )
    mod.matmul_kernel_persistent = recorder
    results = []
    launch_batches = []
    try:
        for _invocation in range(2):
            launch_start = len(recorder.launches)
            results.append(mod.matmul_persistent(a, b, bias))
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return integrity_error
            torch.cuda.synchronize()
            launch_batches.append(recorder.launches[launch_start:])
    finally:
        mod.matmul_kernel_persistent = original_kernel

    protected_storage_ptrs = {
        tensor.untyped_storage().data_ptr()
        for _, tensor, _ in protected_inputs
    }
    protected_storage_ptrs.update(
        storage.untyped_storage().data_ptr()
        for _, storage in backing_storages
    )
    for invocation, (result, launches) in enumerate(
        zip(results, launch_batches), start=1
    ):
        launch_error = _validate_matmul_launch_batch(
            torch,
            launches,
            a,
            b,
            bias,
            result,
            num_sms,
            label,
            invocation,
            requires_partial_group=requires_partial_group,
            requires_persistent=requires_persistent,
        )
        if launch_error is not None:
            return launch_error
        output_error = _validate_matmul_output(
            torch,
            result,
            ref,
            a,
            protected_storage_ptrs,
            label,
            invocation,
        )
        if output_error is not None:
            return output_error

    first_storage = results[0].untyped_storage().data_ptr()
    second_storage = results[1].untyped_storage().data_ptr()
    if first_storage == second_storage:
        return f"{label}: repeated calls did not return fresh storage"
    if not torch.equal(results[0], results[1]):
        max_repeat_diff = (
            results[0].float() - results[1].float()
        ).abs().max().item()
        return (
            f"{label}: repeated calls are nondeterministic; "
            f"max diff = {max_repeat_diff:.6f}"
        )
    for name, observed, frozen in protected_inputs:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated protected input {name}"
    for name, observed, frozen in frozen_backing:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated backing allocation {name}"
    return None


def _contiguous_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


class _VirtualTensor:
    """Allocation-free tensor metadata accepted by the baseline wrapper."""

    def __init__(self, shape, strides, dtype, device="cuda:0"):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self.dtype = dtype
        self.device = device
        self.is_cuda = True

    def dim(self):
        return len(self.shape)

    def stride(self, dimension=None):
        if dimension is None:
            return self._strides
        return self._strides[dimension]

    def is_contiguous(self):
        return self._strides == _contiguous_strides(self.shape)

    def contiguous(self):
        return _VirtualTensor(
            self.shape,
            _contiguous_strides(self.shape),
            self.dtype,
            self.device,
        )


class _VirtualDeviceProperties:
    multi_processor_count = 304


class _VirtualCuda:
    @staticmethod
    def get_device_properties(_device):
        return _VirtualDeviceProperties()


class _VirtualTorch:
    """Torch facade whose empty operation records metadata but allocates nothing."""

    def __init__(self, torch):
        self.float16 = torch.float16
        self.bfloat16 = torch.bfloat16
        self.float32 = torch.float32
        self.cuda = _VirtualCuda()
        self.empty_requests = []

    def empty(self, shape, *, device, dtype):
        tensor = _VirtualTensor(
            shape, _contiguous_strides(shape), dtype, device
        )
        self.empty_requests.append(tensor)
        return tensor


class _VirtualKernel:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            resolved_grid = _KernelLaunchRecorder._resolve_grid(grid, kwargs)
            self.launches.append(
                {
                    "grid": resolved_grid,
                    "args": tuple(args),
                    "kwargs": dict(kwargs),
                }
            )

        return launch


def _run_virtual_large_index_proof(mod, torch, integrity_guard):
    virtual_torch = _VirtualTorch(torch)
    virtual_kernel = _VirtualKernel()
    original_torch = mod.torch
    original_kernel = mod.matmul_kernel_persistent
    receipts = []
    try:
        mod.torch = virtual_torch
        mod.matmul_kernel_persistent = virtual_kernel
        for case in VIRTUAL_INDEX_CASES:
            launch_start = len(virtual_kernel.launches)
            a = _VirtualTensor(
                case["a_shape"],
                case["a_stride"],
                torch.float16,
            )
            b = _VirtualTensor(
                case["b_shape"],
                case["b_stride"],
                torch.float16,
            )
            result = mod.matmul_persistent(a, b)
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return receipts, integrity_error
            launches = virtual_kernel.launches[launch_start:]
            if len(launches) != 1:
                return receipts, (
                    f"{case['name']}: expected one virtual target launch, "
                    f"observed {len(launches)}"
                )
            launch = launches[0]
            observed = tuple(
                launch["kwargs"].get(name)
                for name in ("A_LARGE", "B_LARGE", "C_LARGE")
            )
            if observed != tuple(case["expected"]):
                return receipts, (
                    f"{case['name']}: large-index flags {observed} != "
                    f"{tuple(case['expected'])}"
                )
            if tuple(result.shape) != (
                case["a_shape"][0],
                case["b_shape"][1],
            ):
                return receipts, f"{case['name']}: wrong virtual output shape"
            block_m = launch["kwargs"].get("BLOCK_SIZE_M")
            block_n = launch["kwargs"].get("BLOCK_SIZE_N")
            num_sms = launch["kwargs"].get("NUM_SMS")
            if not all(
                isinstance(value, int) and value > 0
                for value in (block_m, block_n, num_sms)
            ):
                return receipts, f"{case['name']}: incomplete virtual metadata"
            expected_grid = (
                min(
                    num_sms,
                    _ceil_div(case["a_shape"][0], block_m)
                    * _ceil_div(case["b_shape"][1], block_n),
                ),
            )
            if launch["grid"] != expected_grid:
                return receipts, (
                    f"{case['name']}: virtual grid {launch['grid']} != "
                    f"{expected_grid}"
                )
            receipts.append(
                {
                    "name": case["name"],
                    "a_max_offset": _max_relative_element_offset(
                        case["a_shape"], case["a_stride"]
                    ),
                    "b_max_offset": _max_relative_element_offset(
                        case["b_shape"], case["b_stride"]
                    ),
                    "c_max_offset": _max_relative_element_offset(
                        result.shape, result.stride()
                    ),
                    "flags": observed,
                    "grid": launch["grid"],
                    "allocated_bytes": 0,
                }
            )
    except Exception as error:
        return receipts, f"virtual large-index proof exception: {error}"
    finally:
        mod.torch = original_torch
        mod.matmul_kernel_persistent = original_kernel
    if len(virtual_torch.empty_requests) != len(VIRTUAL_INDEX_CASES):
        return receipts, "virtual proof did not produce one output per case"
    return receipts, None


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "matmul_persistent"), "Missing matmul_persistent"
        assert hasattr(mod, "matmul_kernel_persistent"), "Missing matmul_kernel_persistent"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness(*, include_details=False):
    import torch
    capture_integrity = _capture_evaluator_integrity
    verify_integrity = _verify_evaluator_integrity
    integrity_snapshot = capture_integrity(torch)

    def integrity_guard():
        return verify_integrity(torch, integrity_snapshot)

    details = {
        "virtual_index_receipts": [],
        "completed_scoring_cases": 0,
        "completed_contract_cases": 0,
    }

    def finish(ok, error):
        if include_details:
            return ok, error, details
        return ok, error

    try:
        mod = load_module()
    except Exception as e:
        return finish(False, f"Failed to load module: {e}")
    integrity_error = integrity_guard()
    if integrity_error is not None:
        return finish(False, integrity_error)

    virtual_receipts, virtual_error = _run_virtual_large_index_proof(
        mod, torch, integrity_guard
    )
    details["virtual_index_receipts"] = virtual_receipts
    if virtual_error is not None:
        return finish(False, virtual_error)

    device = "cuda"
    dtype = torch.float16

    for i, (M, N, K) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)

            error = _check_matmul_case(
                torch,
                mod,
                a,
                b,
                None,
                f"Shape {i + 1} (M={M}, N={N}, K={K})",
                integrity_guard,
            )
            if error is not None:
                return finish(False, error)
            details["completed_scoring_cases"] += 1
        except Exception as e:
            return finish(
                False,
                f"Shape {i + 1} (M={M}, N={N}, K={K}): exception: {e}",
            )

    for case_index, case in enumerate(CORRECTNESS_CASES):
        try:
            torch.manual_seed(142 + case_index)
            a, b, bias, backing_storages = _make_contract_case_inputs(
                torch, case, device
            )
            error = _check_matmul_case(
                torch,
                mod,
                a,
                b,
                bias,
                f"Contract case {case['name']}",
                integrity_guard,
                backing_storages=backing_storages,
                requires_partial_group=case["requires_partial_group"],
                requires_persistent=case["requires_persistent"],
            )
            if error is not None:
                return finish(False, error)
            details["completed_contract_cases"] += 1
        except Exception as e:
            return finish(
                False,
                f"Contract case {case['name']}: exception: {e}",
            )

    return finish(True, None)


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (M, N, K) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(0)
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)

            # Warmup
            def _bench_fn():
                mod.matmul_persistent(a, b)
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
                    "K": K
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "M": M,
                    "N": N,
                    "K": K
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
        ok, err, details = run_correctness(include_details=True)
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES),
            "num_contract_cases": len(CORRECTNESS_CASES),
            "num_virtual_index_cases": len(VIRTUAL_INDEX_CASES),
            "max_estimated_storage_bytes": max(
                _estimate_case_storage_bytes(case)
                for case in CORRECTNESS_CASES
            ),
            "virtual_index_allocated_bytes": 0,
            "target_engagement": "poison_then_post_launch_snapshot",
            **details,
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

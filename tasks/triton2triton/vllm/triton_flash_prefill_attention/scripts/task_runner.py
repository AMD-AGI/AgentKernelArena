#!/usr/bin/env python3
"""Task runner for triton2triton/triton_flash_prefill_attention"""
import sys
import os
import json
import argparse
import time
import importlib.util
import math

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/triton_flash_prefill_attention"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_flash_prefill_attention.py")

# Test configurations: (batch_size, seq_len, num_heads, num_kv_heads, head_dim)
TEST_SHAPES = [
    (2, 128, 8, 8, 64),     # small, MHA
    (4, 256, 16, 4, 64),    # medium, GQA
    (2, 512, 32, 8, 128),   # large, GQA
    (1, 1024, 16, 16, 64),  # long seq, MHA
    (8, 64, 8, 1, 64),      # batched, MQA
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract coverage. Performance continues to use only
# TEST_SHAPES and the byte-frozen run_performance implementation below.
FLASH_CORRECTNESS_CASES = [
    {
        "name": "ragged_mha_causal_d32",
        "sequence_lengths": (1, 7, 33),
        "num_heads": 6,
        "num_kv_heads": 6,
        "head_dim": 32,
        "is_causal": True,
        "softmax_scale": None,
        "sliding_window_q": None,
        "sliding_window_k": None,
    },
    {
        "name": "ragged_gqa_full_d64",
        "sequence_lengths": (3, 19, 65),
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "is_causal": False,
        "softmax_scale": None,
        "sliding_window_q": 0,
        "sliding_window_k": 0,
    },
    {
        "name": "ragged_mqa_asymmetric_window_d96",
        "sequence_lengths": (5, 37, 129),
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_dim": 96,
        "is_causal": False,
        "softmax_scale": 0.173,
        "sliding_window_q": 17,
        "sliding_window_k": 9,
    },
    {
        "name": "ragged_gqa_bidirectional_window_d128",
        "sequence_lengths": (2, 63, 257),
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
        "is_causal": False,
        "softmax_scale": None,
        "sliding_window_q": 31,
        "sliding_window_k": 47,
    },
    {
        "name": "long_gqa_causal_left_window_d64",
        "sequence_lengths": (513,),
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "is_causal": True,
        "softmax_scale": None,
        "sliding_window_q": 127,
        "sliding_window_k": None,
    },
    {
        "name": "ragged_mqa_forward_window_only_d32",
        "sequence_lengths": (11, 128),
        "num_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 32,
        "is_causal": False,
        "softmax_scale": None,
        "sliding_window_q": 0,
        "sliding_window_k": 15,
    },
]

FLASH_ATOL = 1e-2
FLASH_RTOL = 1e-2
FLASH_RCP_LN2 = 1.4426950408889634
FLASH_TARGET_PARAMETER_NAMES = (
    "Q",
    "K",
    "V",
    "sm_scale",
    "B_Start_Loc",
    "B_Seqlen",
    "Out",
    "stride_qbs",
    "stride_qh",
    "stride_kbs",
    "stride_kh",
    "stride_vbs",
    "stride_vh",
    "stride_obs",
    "stride_oh",
)
FLASH_TARGET_METADATA_NAMES = {
    "kv_group_num",
    "BLOCK_M",
    "BLOCK_DMODEL",
    "BLOCK_N",
    "IS_CAUSAL",
    "SLIDING_WINDOW_Q",
    "SLIDING_WINDOW_K",
    "Lk",
}
FLASH_TRITON_LAUNCH_CONTROL_NAMES = {
    "grid",
    "warmup",
    "num_warps",
    "num_stages",
    "num_ctas",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
}
FLASH_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS = {
    "num_warps",
    "num_stages",
    "num_ctas",
    "matrix_instr_nonkdim",
    "kpack",
}

_TRUSTED_TORCH_PATHS = (
    "Tensor",
    "allclose",
    "arange",
    "equal",
    "full_like",
    "isfinite",
    "manual_seed",
    "randn",
    "softmax",
    "tensor",
    "zeros_like",
    "cuda.Event",
    "cuda.synchronize",
)


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
    """Dynamically load the source module."""
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
    """Delegate target launches while recording their exact public bindings."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = []

    @staticmethod
    def _resolve_grid(grid, kwargs):
        if callable(grid):
            grid = grid(dict(kwargs))
        return tuple(grid) if grid is not None else None

    @staticmethod
    def _argument(args, kwargs, position, keyword):
        return args[position] if len(args) > position else kwargs.get(keyword)

    def _record(self, grid, args, kwargs, launch_style):
        record = {
            "grid": self._resolve_grid(grid, kwargs),
            "q": self._argument(args, kwargs, 0, "Q"),
            "k": self._argument(args, kwargs, 1, "K"),
            "v": self._argument(args, kwargs, 2, "V"),
            "sm_scale": self._argument(args, kwargs, 3, "sm_scale"),
            "b_start_loc": self._argument(args, kwargs, 4, "B_Start_Loc"),
            "b_seq_len": self._argument(args, kwargs, 5, "B_Seqlen"),
            "out": self._argument(args, kwargs, 6, "Out"),
            "strides": tuple(
                self._argument(args, kwargs, position, keyword)
                for position, keyword in (
                    (7, "stride_qbs"),
                    (8, "stride_qh"),
                    (9, "stride_kbs"),
                    (10, "stride_kh"),
                    (11, "stride_vbs"),
                    (12, "stride_vh"),
                    (13, "stride_obs"),
                    (14, "stride_oh"),
                )
            ),
            "kv_group_num": kwargs.get("kv_group_num"),
            "block_m": kwargs.get("BLOCK_M"),
            "block_dmodel": kwargs.get("BLOCK_DMODEL"),
            "block_n": kwargs.get("BLOCK_N"),
            "is_causal": kwargs.get("IS_CAUSAL"),
            "sliding_window_q": kwargs.get("SLIDING_WINDOW_Q"),
            "sliding_window_k": kwargs.get("SLIDING_WINDOW_K"),
            "head_dim": kwargs.get("Lk"),
            "launch_style": launch_style,
            "raw_args": tuple(args),
            "raw_kwargs": dict(kwargs),
        }
        bound_tensors = tuple(
            record[name]
            for name in ("q", "k", "v", "b_start_loc", "b_seq_len", "out")
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
            self._record(grid, args, kwargs, "bracket")
            record = self.launches[-1]
            record["out"].fill_(float("nan"))
            returned = launch(*args, **kwargs)
            record["post_launch_snapshot"] = record["out"].clone()
            return returned

        return record_and_launch

    def run(self, *args, **kwargs):
        self._record(kwargs.get("grid"), args, kwargs, "run")
        record = self.launches[-1]
        record["out"].fill_(float("nan"))
        returned = self.kernel.run(*args, **kwargs)
        record["post_launch_snapshot"] = record["out"].clone()
        return returned

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def reference_attention(
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    is_causal=True,
    softmax_scale=None,
    sliding_window_q=None,
    sliding_window_k=None,
):
    """
    CPU/PyTorch reference for variable-length packed flash attention.

    q, k, v: [total_tokens, num_heads, head_dim]
    b_start_loc: [batch]
    b_seq_len: [batch]
    Returns: output [total_tokens, num_heads, head_dim]
    """
    import torch
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads

    out = torch.zeros_like(q)
    sm_scale = (
        1.0 / (head_dim ** 0.5)
        if softmax_scale is None
        else softmax_scale
    )

    for b in range(len(b_seq_len)):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()

        for h in range(num_heads):
            kv_h = h // kv_group_num
            q_b = q[start:start + seq_len, h, :]  # [S, D]
            k_b = k[start:start + seq_len, kv_h, :]  # [S, D]
            v_b = v[start:start + seq_len, kv_h, :]  # [S, D]

            # [S, S]
            scores = (q_b.float() @ k_b.float().T) * sm_scale

            query_pos = torch.arange(seq_len, device=scores.device)[:, None]
            key_pos = torch.arange(seq_len, device=scores.device)[None, :]
            mask = torch.zeros_like(scores, dtype=torch.bool)

            if is_causal:
                mask |= key_pos > query_pos
            if sliding_window_q is not None and sliding_window_q > 0:
                mask |= query_pos - key_pos > sliding_window_q
            if sliding_window_k is not None and sliding_window_k > 0:
                mask |= key_pos - query_pos > sliding_window_k

            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            output = attn @ v_b.float()
            out[start:start + seq_len, h, :] = output.to(q.dtype)

    return out


def _uniform_correctness_cases():
    """Preserve the five original uniform correctness workloads."""
    for index, (batch, seq_len, heads, kv_heads, head_dim) in enumerate(
        TEST_SHAPES, start=1
    ):
        yield {
            "name": f"uniform_scoring_shape_{index}",
            "sequence_lengths": (seq_len,) * batch,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "is_causal": True,
            "softmax_scale": None,
            "sliding_window_q": None,
            "sliding_window_k": None,
        }


def _start_locations(sequence_lengths):
    starts = []
    current = 0
    for length in sequence_lengths:
        starts.append(current)
        current += length
    return starts


def _is_positive_power_of_two(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value & (value - 1) == 0
    )


def _validate_flash_launch_batch(
    launches,
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    max_input_len,
    case,
    invocation,
):
    prefix = f"{case['name']}, invocation {invocation}"
    if len(launches) != 1:
        return f"{prefix}: expected one target-kernel launch, observed {len(launches)}"
    launch = launches[0]
    positional = launch.get("raw_args")
    keyword = launch.get("raw_kwargs")
    if not isinstance(positional, tuple) or not isinstance(keyword, dict):
        return f"{prefix}: launch operands were not captured"
    if len(positional) > len(FLASH_TARGET_PARAMETER_NAMES):
        return (
            f"{prefix}: target launch has {len(positional)} positional operands; "
            f"expected at most {len(FLASH_TARGET_PARAMETER_NAMES)}"
        )
    positionally_bound = set(FLASH_TARGET_PARAMETER_NAMES[: len(positional)])
    duplicate_bindings = positionally_bound.intersection(keyword)
    if duplicate_bindings:
        return (
            f"{prefix}: duplicate target operand bindings "
            f"{sorted(duplicate_bindings)}"
        )
    allowed_keywords = (
        set(FLASH_TARGET_PARAMETER_NAMES)
        | FLASH_TARGET_METADATA_NAMES
        | FLASH_TRITON_LAUNCH_CONTROL_NAMES
    )
    unexpected_keywords = set(keyword).difference(allowed_keywords)
    if unexpected_keywords:
        return (
            f"{prefix}: target launch has unexpected keyword operands "
            f"{sorted(unexpected_keywords)}"
        )
    launch_control_error = _validate_flash_launch_controls(
        keyword, launch.get("grid"), prefix
    )
    if launch_control_error is not None:
        return launch_control_error

    def operand(index):
        if index < len(positional):
            return positional[index]
        return keyword.get(FLASH_TARGET_PARAMETER_NAMES[index])

    expected_bindings = {
        "q": q,
        "k": k,
        "v": v,
        "b_start_loc": b_start_loc,
        "b_seq_len": b_seq_len,
        "out": o,
    }
    for binding_name, expected_tensor in expected_bindings.items():
        if launch.get(binding_name) is not expected_tensor:
            return (
                f"{prefix}: target launch {binding_name} was not bound to "
                "the evaluated tensor"
            )
    for index, binding_name in ((0, "q"), (1, "k"), (2, "v"), (4, "b_start_loc"), (5, "b_seq_len"), (6, "out")):
        if operand(index) is not expected_bindings[binding_name]:
            return f"{prefix}: target operand {binding_name} was not bound exactly once"
    bound_tensors = tuple(
        expected_bindings[name]
        for name in ("q", "k", "v", "b_start_loc", "b_seq_len", "out")
    )
    expected_data_ptrs = tuple(tensor.data_ptr() for tensor in bound_tensors)
    expected_storage_ptrs = tuple(
        tensor.untyped_storage().data_ptr() for tensor in bound_tensors
    )
    if launch.get("data_ptrs") != expected_data_ptrs:
        return f"{prefix}: target launch data-pointer binding mismatch"
    if launch.get("storage_ptrs") != expected_storage_ptrs:
        return f"{prefix}: target launch storage binding mismatch"

    block_m = launch.get("block_m")
    block_n = launch.get("block_n")
    block_dmodel = launch.get("block_dmodel")
    if not _is_positive_power_of_two(block_m):
        return f"{prefix}: BLOCK_M is not a positive power of two"
    if not _is_positive_power_of_two(block_n):
        return f"{prefix}: BLOCK_N is not a positive power of two"
    if (
        not _is_positive_power_of_two(block_dmodel)
        or block_dmodel < case["head_dim"]
    ):
        return f"{prefix}: BLOCK_DMODEL does not cover the head dimension"
    expected_grid = (
        len(case["sequence_lengths"]),
        case["num_heads"],
        (max_input_len + block_m - 1) // block_m,
    )
    if launch.get("grid") != expected_grid:
        return (
            f"{prefix}: expected resolved grid {expected_grid}, "
            f"observed {launch.get('grid')}"
        )

    expected_metadata = {
        "kv_group_num": case["num_heads"] // case["num_kv_heads"],
        "is_causal": case["is_causal"],
        "sliding_window_q": case["sliding_window_q"] or 0,
        "sliding_window_k": case["sliding_window_k"] or 0,
        "head_dim": case["head_dim"],
    }
    for key, expected in expected_metadata.items():
        observed = launch.get(key)
        expected_type = bool if key == "is_causal" else int
        if type(observed) is not expected_type:
            return f"{prefix}: target launch {key} must be a {expected_type.__name__}"
        if observed != expected:
            return (
                f"{prefix}: expected {key}={expected}, "
                f"observed {observed}"
            )
    expected_strides = (
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
    )
    if launch.get("strides") != expected_strides:
        return f"{prefix}: target launch stride metadata mismatch"
    runtime_strides = tuple(operand(index) for index in range(7, 15))
    if (
        any(type(value) is not int for value in runtime_strides)
        or runtime_strides != expected_strides
    ):
        return f"{prefix}: target runtime stride operands do not match the evaluated tensors"

    expected_scale = (
        1.0 / (case["head_dim"] ** 0.5)
        if case["softmax_scale"] is None
        else case["softmax_scale"]
    ) * FLASH_RCP_LN2
    try:
        scale_matches = math.isclose(
            float(launch.get("sm_scale")),
            expected_scale,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        scale_matches = False
    if not scale_matches:
        return f"{prefix}: target launch softmax scale mismatch"
    try:
        operand_scale_matches = math.isclose(
            float(operand(3)), expected_scale, rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        operand_scale_matches = False
    if not operand_scale_matches:
        return f"{prefix}: target runtime softmax-scale operand mismatch"
    return None


def _validate_flash_launch_controls(keyword, resolved_grid, prefix):
    for name in FLASH_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS:
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


def _validate_flash_target_snapshot(torch, snapshot, observed_output, label):
    if snapshot.untyped_storage().data_ptr() == (
        observed_output.untyped_storage().data_ptr()
    ):
        return f"{label}: target snapshot reused the wrapper output storage"
    if not torch.equal(snapshot, observed_output):
        return (
            f"{label}: immediate target snapshot did not match the wrapper output; "
            "precomputed, detached, dummy, or post-target repair paths are invalid"
        )
    return None


def _check_flash_case(torch, mod, case, seed, integrity_guard):
    name = case["name"]
    sequence_lengths = case["sequence_lengths"]
    total_tokens = sum(sequence_lengths)
    heads = case["num_heads"]
    kv_heads = case["num_kv_heads"]
    head_dim = case["head_dim"]
    device = "cuda"
    dtype = torch.float16

    torch.manual_seed(seed)
    q = torch.randn(total_tokens, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, kv_heads, head_dim, device=device, dtype=dtype)
    o = torch.full_like(q, float("nan"))
    b_start_loc = torch.tensor(
        _start_locations(sequence_lengths), device=device, dtype=torch.int32
    )
    b_seq_len = torch.tensor(
        sequence_lengths, device=device, dtype=torch.int32
    )

    protected_inputs = (
        ("q", q, q.clone()),
        ("k", k, k.clone()),
        ("v", v, v.clone()),
        ("b_start_loc", b_start_loc, b_start_loc.clone()),
        ("b_seq_len", b_seq_len, b_seq_len.clone()),
    )
    ref = reference_attention(
        protected_inputs[0][2],
        protected_inputs[1][2],
        protected_inputs[2][2],
        protected_inputs[3][2],
        protected_inputs[4][2],
        is_causal=case["is_causal"],
        softmax_scale=case["softmax_scale"],
        sliding_window_q=case["sliding_window_q"],
        sliding_window_k=case["sliding_window_k"],
    )
    if not bool(torch.isfinite(ref).all().item()):
        return f"{name}: evaluator reference contains a non-finite value"

    observed_outputs = []
    max_input_len = max(sequence_lengths)
    original_kernel = mod._fwd_kernel
    recorder = _KernelLaunchRecorder(original_kernel)
    mod._fwd_kernel = recorder
    try:
        for invocation in range(1, 3):
            first_launch = len(recorder.launches)
            o.fill_(float("nan"))
            returned = mod.context_attention_fwd(
                q,
                k,
                v,
                o,
                b_start_loc,
                b_seq_len,
                max_input_len=max_input_len,
                is_causal=case["is_causal"],
                softmax_scale=case["softmax_scale"],
                sliding_window_q=case["sliding_window_q"],
                sliding_window_k=case["sliding_window_k"],
            )
            integrity_error = integrity_guard()
            if integrity_error is not None:
                return integrity_error
            torch.cuda.synchronize()
            launch_error = _validate_flash_launch_batch(
                recorder.launches[first_launch:],
                q,
                k,
                v,
                o,
                b_start_loc,
                b_seq_len,
                max_input_len,
                case,
                invocation,
            )
            if launch_error is not None:
                return launch_error
            if returned is not None:
                return f"{name}, invocation {invocation}: expected a None return"
            if not bool(torch.isfinite(o).all().item()):
                return f"{name}, invocation {invocation}: output is not finite"
            snapshot_error = _validate_flash_target_snapshot(
                torch,
                recorder.launches[first_launch]["post_launch_snapshot"],
                o,
                f"{name}, invocation {invocation}",
            )
            if snapshot_error is not None:
                return snapshot_error
            observed_outputs.append(o.clone())
    finally:
        mod._fwd_kernel = original_kernel

    for input_name, observed, frozen in protected_inputs:
        if not torch.equal(observed, frozen):
            return f"{name}: candidate mutated protected {input_name}"

    if not torch.equal(observed_outputs[0], observed_outputs[1]):
        repeat_diff = (
            (observed_outputs[0].float() - observed_outputs[1].float())
            .abs()
            .max()
            .item()
        )
        return f"{name}: repeated calls are nondeterministic; max diff={repeat_diff:.6f}"

    if not torch.allclose(
        observed_outputs[0], ref, atol=FLASH_ATOL, rtol=FLASH_RTOL
    ):
        max_diff = (observed_outputs[0].float() - ref.float()).abs().max().item()
        return f"{name}: max diff={max_diff:.6f}"
    return None


def run_compile():
    """Check that the source file is valid Python and imports succeed."""
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "context_attention_fwd"), "Missing context_attention_fwd"
        assert hasattr(mod, "_fwd_kernel"), "Missing _fwd_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    """Run correctness checks against PyTorch reference."""
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

    correctness_cases = [*_uniform_correctness_cases(), *FLASH_CORRECTNESS_CASES]
    for case_index, case in enumerate(correctness_cases):
        try:
            error = _check_flash_case(
                torch, mod, case, seed=42 + case_index, integrity_guard=integrity_guard
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"{case['name']}: exception: {e}"

    return True, None


def run_performance():
    """Measure kernel execution time."""
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    device = "cuda"
    dtype = torch.float16
    test_cases = []

    for test_idx, (bs, seq_len, nh, nkv, hd) in enumerate(TEST_SHAPES):
        try:
            total_tokens = bs * seq_len
            torch.manual_seed(42 + test_idx)
            q = torch.randn(total_tokens, nh, hd, device=device, dtype=dtype)
            k = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            v = torch.randn(total_tokens, nkv, hd, device=device, dtype=dtype)
            o = torch.zeros_like(q)

            b_seq_len = torch.full((bs,), seq_len, device=device, dtype=torch.int32)
            b_start_loc = torch.zeros(bs, device=device, dtype=torch.int32)
            for j in range(bs):
                b_start_loc[j] = j * seq_len

            # Warmup
            def _bench_fn():
                mod.context_attention_fwd(
                    q, k, v, o, b_start_loc, b_seq_len,
                    max_input_len=seq_len, is_causal=True,
                )
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
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "num_heads": nh,
                    "num_kv_heads": nkv,
                    "head_dim": hd
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
            "num_shapes": len(TEST_SHAPES) + len(FLASH_CORRECTNESS_CASES),
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

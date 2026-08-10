#!/usr/bin/env python3
"""Task runner for triton2triton/triton_topk_log_softmax"""
import sys, os, json, argparse, importlib.util
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_topk_log_softmax"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_topk_log_softmax.py")
TOPK_TARGET_PARAMETER_NAMES = (
    "output_ptr",
    "logits_ptr",
    "logits_stride",
    "topk_ids_ptr",
    "topk",
    "vocab_size",
)
TOPK_TARGET_METADATA_NAMES = {"BLOCK_SIZE", "PADDED_TOPK"}
TOPK_TRITON_LAUNCH_CONTROL_NAMES = {
    "grid",
    "warmup",
    "num_warps",
    "num_stages",
    "num_ctas",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
}
TOPK_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS = {
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
    "isnan",
    "log_softmax",
    "manual_seed",
    "randint",
    "randn",
    "tensor",
    "cuda.Event",
    "cuda.synchronize",
)

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
    """Delegate the monolithic target launch and record its tensor bindings."""

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

    def _record(self, grid, args, kwargs):
        record = {
            "grid": self._resolve_grid(grid, kwargs),
            "args": tuple(args),
            "kwargs": dict(kwargs),
            "output": self._argument(args, kwargs, 0, "output_ptr"),
            "logits": self._argument(args, kwargs, 1, "logits_ptr"),
            "logits_stride": self._argument(args, kwargs, 2, "logits_stride"),
            "token_ids": self._argument(args, kwargs, 3, "topk_ids_ptr"),
            "topk": self._argument(args, kwargs, 4, "topk"),
            "vocab_size": self._argument(args, kwargs, 5, "vocab_size"),
            "block_size": kwargs.get("BLOCK_SIZE"),
            "padded_topk": kwargs.get("PADDED_TOPK"),
        }
        bound_tensors = (record["output"], record["logits"], record["token_ids"])
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
            self._record(grid, args, kwargs)
            record = self.launches[-1]
            record["output"].fill_(float("nan"))
            returned = launch(*args, **kwargs)
            record["post_launch_snapshot"] = record["output"].clone()
            return returned

        return record_and_launch

    def run(self, *args, **kwargs):
        self._record(kwargs.get("grid"), args, kwargs)
        record = self.launches[-1]
        record["output"].fill_(float("nan"))
        returned = self.kernel.run(*args, **kwargs)
        record["post_launch_snapshot"] = record["output"].clone()
        return returned

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f: source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "compute_token_logprobs"), "Missing compute_token_logprobs"
        assert hasattr(mod, "_topk_log_softmax_kernel"), "Missing _topk_log_softmax_kernel"
        return True, None
    except Exception as e:
        return False, str(e)


TEST_SHAPES = [
    (4, 256, 3),   # (batch, vocab, num_tokens)
    (8, 1024, 5),
    (16, 4096, 10),
    (32, 8192, 20),
    (64, 32768, 10),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Non-scoring public-contract coverage. Keep TEST_SHAPES reserved for performance.
CORRECTNESS_CASES = [
    {
        "name": "padded_stride_fp16_nonpower_duplicates",
        "batch": 3,
        "vocab": 1003,
        "num_tokens": 7,
        "logits_dtype": "float16",
        "token_ids_dtype": "int64",
        "row_padding": 29,
    },
    {
        "name": "large_vocab_batch1_bfloat16_int32_duplicates",
        "batch": 1,
        "vocab": 131071,
        "num_tokens": 8,
        "logits_dtype": "bfloat16",
        "token_ids_dtype": "int32",
        "row_padding": 17,
    },
    {
        "name": "negative_inf_in_every_row",
        "batch": 3,
        "vocab": 4099,
        "num_tokens": 8,
        "logits_dtype": "float32",
        "token_ids_dtype": "int64",
        "row_padding": 13,
        "data": "negative_inf_each_row",
    },
    {
        "name": "all_negative_inf_small_row_matches_reference",
        "batch": 2,
        "vocab": 4099,
        "num_tokens": 8,
        "logits_dtype": "float16",
        "token_ids_dtype": "int64",
        "row_padding": 23,
        "data": "all_negative_inf_row",
        "all_negative_inf_row": 0,
        "expected_nan_rows": (0,),
    },
    {
        "name": "full_negative_inf_chunk_with_finite_neighbors",
        "batch": 2,
        "vocab": 8209,
        "num_tokens": 8,
        "logits_dtype": "float16",
        "token_ids_dtype": "int32",
        "row_padding": 31,
        "data": "negative_inf_full_chunk",
        "negative_inf_chunk_start": 1024,
        "negative_inf_chunk_size": 1024,
    },
    {
        "name": "all_negative_inf_row_matches_reference",
        "batch": 2,
        "vocab": 8209,
        "num_tokens": 8,
        "logits_dtype": "bfloat16",
        "token_ids_dtype": "int64",
        "row_padding": 37,
        "data": "all_negative_inf_row",
        "all_negative_inf_row": 0,
        "expected_nan_rows": (0,),
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


def _make_topk_correctness_inputs(torch, case, device):
    batch = case["batch"]
    vocab = case["vocab"]
    num_tokens = case["num_tokens"]
    row_padding = case["row_padding"]
    logits_dtype = getattr(torch, case["logits_dtype"])
    token_ids_dtype = getattr(torch, case["token_ids_dtype"])

    logits_storage = torch.randn(
        batch, vocab + row_padding, device=device, dtype=logits_dtype
    )
    logits = logits_storage[:, :vocab]
    if batch > 1:
        assert not logits.is_contiguous()
    assert logits.stride(0) == vocab + row_padding

    data = case.get("data", "random_finite")
    if data == "negative_inf_each_row":
        logits[:, 3] = float("-inf")
        logits[:, vocab - 1] = float("-inf")
    elif data == "negative_inf_full_chunk":
        chunk_start = case["negative_inf_chunk_start"]
        chunk_size = case["negative_inf_chunk_size"]
        logits[:, chunk_start : chunk_start + chunk_size] = float("-inf")
    elif data == "all_negative_inf_row":
        all_negative_inf_row = case["all_negative_inf_row"]
        logits[all_negative_inf_row, :] = float("-inf")
        for row in range(batch):
            if row != all_negative_inf_row:
                logits[row, 3] = float("-inf")
                logits[row, vocab - 1] = float("-inf")
    elif data != "random_finite":
        raise ValueError(f"unknown top-k correctness data pattern: {data}")

    if data == "negative_inf_full_chunk":
        chunk_start = case["negative_inf_chunk_start"]
        chunk_size = case["negative_inf_chunk_size"]
        duplicate_unsorted_ids = [
            chunk_start,
            3,
            chunk_start + chunk_size - 1,
            3,
            0,
            vocab - 7,
            chunk_start + 11,
            0,
        ]
    else:
        duplicate_unsorted_ids = [
            vocab - 1,
            3,
            vocab // 2,
            3,
            0,
            vocab - 7,
            11,
            0,
        ]
    token_ids = torch.tensor(
        duplicate_unsorted_ids[:num_tokens],
        device=device,
        dtype=token_ids_dtype,
    ).repeat(batch, 1)
    return logits, token_ids, logits_storage


def _is_positive_power_of_two(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value & (value - 1) == 0
    )


def _validate_topk_launch_batch(
    torch,
    launches,
    logits,
    token_ids,
    result,
    label,
):
    if len(launches) != 1:
        return f"{label}: expected one target-kernel launch, observed {len(launches)}"
    launch = launches[0]
    positional = launch.get("args")
    keyword = launch.get("kwargs")
    if not isinstance(positional, tuple) or not isinstance(keyword, dict):
        return f"{label}: launch operands were not captured"
    if len(positional) > len(TOPK_TARGET_PARAMETER_NAMES):
        return (
            f"{label}: target launch has {len(positional)} positional operands; "
            f"expected at most {len(TOPK_TARGET_PARAMETER_NAMES)}"
        )
    positionally_bound = set(TOPK_TARGET_PARAMETER_NAMES[: len(positional)])
    duplicate_bindings = positionally_bound.intersection(keyword)
    if duplicate_bindings:
        return (
            f"{label}: duplicate target operand bindings "
            f"{sorted(duplicate_bindings)}"
        )
    allowed_keywords = (
        set(TOPK_TARGET_PARAMETER_NAMES)
        | TOPK_TARGET_METADATA_NAMES
        | TOPK_TRITON_LAUNCH_CONTROL_NAMES
    )
    unexpected_keywords = set(keyword).difference(allowed_keywords)
    if unexpected_keywords:
        return (
            f"{label}: target launch has unexpected keyword operands "
            f"{sorted(unexpected_keywords)}"
        )
    launch_control_error = _validate_topk_launch_controls(
        keyword, launch.get("grid"), label
    )
    if launch_control_error is not None:
        return launch_control_error

    def operand(index):
        if index < len(positional):
            return positional[index]
        return keyword.get(TOPK_TARGET_PARAMETER_NAMES[index])

    if launch.get("output") is not result:
        return f"{label}: target output was not bound to the returned tensor"
    if launch.get("logits") is not logits:
        return f"{label}: target logits were not bound to the evaluated tensor"
    if operand(0) is not result or operand(1) is not logits:
        return f"{label}: target output/logits operands were not bound exactly once"

    converted_ids = launch.get("token_ids")
    if not isinstance(converted_ids, torch.Tensor):
        return f"{label}: target token IDs are not a tensor"
    if converted_ids.dtype != torch.int64:
        return f"{label}: target token IDs were not converted to int64"
    if converted_ids.device != token_ids.device:
        return f"{label}: target token IDs are on the wrong device"
    if tuple(converted_ids.shape) != tuple(token_ids.shape):
        return f"{label}: target token ID shape mismatch"
    if not torch.equal(converted_ids, token_ids.to(torch.int64)):
        return f"{label}: target token IDs do not match the evaluated IDs"
    if operand(3) is not converted_ids:
        return f"{label}: target token-ID operand was not bound exactly once"

    bound_tensors = (result, logits, converted_ids)
    expected_data_ptrs = tuple(tensor.data_ptr() for tensor in bound_tensors)
    expected_storage_ptrs = tuple(
        tensor.untyped_storage().data_ptr() for tensor in bound_tensors
    )
    if launch.get("data_ptrs") != expected_data_ptrs:
        return f"{label}: target launch data-pointer binding mismatch"
    if launch.get("storage_ptrs") != expected_storage_ptrs:
        return f"{label}: target launch storage binding mismatch"

    batch, vocab_size = logits.shape
    num_tokens = token_ids.shape[1]
    expected_metadata = {
        "grid": (batch,),
        "logits_stride": logits.stride(0),
        "topk": num_tokens,
        "vocab_size": vocab_size,
    }
    for key, expected in expected_metadata.items():
        observed = launch.get(key)
        if key != "grid" and type(observed) is not int:
            return f"{label}: target launch {key} must be an integer"
        if observed != expected:
            return (
                f"{label}: expected {key}={expected}, "
                f"observed {launch.get(key)}"
            )
    runtime_metadata = (operand(2), operand(4), operand(5))
    expected_runtime_metadata = (
        logits.stride(0),
        num_tokens,
        vocab_size,
    )
    if (
        any(type(value) is not int for value in runtime_metadata)
        or runtime_metadata != expected_runtime_metadata
    ):
        return (
            f"{label}: target runtime metadata {runtime_metadata} != "
            f"{expected_runtime_metadata}"
        )
    if not _is_positive_power_of_two(launch.get("block_size")):
        return f"{label}: BLOCK_SIZE is not a positive power of two"
    padded_topk = launch.get("padded_topk")
    if not _is_positive_power_of_two(padded_topk) or padded_topk < num_tokens:
        return f"{label}: PADDED_TOPK does not cover all requested IDs"
    return None


def _validate_topk_launch_controls(keyword, resolved_grid, label):
    for name in TOPK_TRITON_POSITIVE_INTEGER_LAUNCH_CONTROLS:
        if name not in keyword:
            continue
        value = keyword[name]
        if type(value) is not int or value <= 0:
            return f"{label}: launch control {name} must be a positive integer"
    if "waves_per_eu" in keyword:
        value = keyword["waves_per_eu"]
        if type(value) is not int or value < 0:
            return f"{label}: launch control waves_per_eu must be a nonnegative integer"
    if "warmup" in keyword and type(keyword["warmup"]) is not bool:
        return f"{label}: launch control warmup must be a boolean"
    if "grid" in keyword:
        value = keyword["grid"]
        if isinstance(value, int) and not isinstance(value, bool):
            value = (value,)
        elif isinstance(value, (tuple, list)):
            value = tuple(value)
        else:
            return f"{label}: launch control grid must be an integer sequence"
        if (
            not value
            or any(type(dimension) is not int or dimension <= 0 for dimension in value)
            or value != resolved_grid
        ):
            return f"{label}: launch control grid does not match the resolved grid"
    return None


def _validate_topk_target_snapshot(torch, snapshot, result, label):
    if snapshot.untyped_storage().data_ptr() == result.untyped_storage().data_ptr():
        return f"{label}: target snapshot reused the returned output storage"
    if not torch.allclose(
        snapshot,
        result,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    ):
        return (
            f"{label}: immediate target snapshot did not match the returned output; "
            "precomputed, detached, dummy, or post-target repair paths are invalid"
        )
    return None


def _invoke_topk_with_attestation(
    torch, mod, logits, token_ids, label, integrity_guard
):
    original_kernel = mod._topk_log_softmax_kernel
    recorder = _KernelLaunchRecorder(original_kernel)
    mod._topk_log_softmax_kernel = recorder
    try:
        result = mod.compute_token_logprobs(logits, token_ids)
        integrity_error = integrity_guard()
        if integrity_error is not None:
            return result, integrity_error
        torch.cuda.synchronize()
    finally:
        mod._topk_log_softmax_kernel = original_kernel
    error = _validate_topk_launch_batch(
        torch,
        recorder.launches,
        logits,
        token_ids,
        result,
        label,
    )
    if error is None:
        error = _validate_topk_target_snapshot(
            torch,
            recorder.launches[0]["post_launch_snapshot"],
            result,
            label,
        )
    return result, error


def _validate_topk_repeated_outputs(torch, results, label):
    if len(results) != 2:
        return f"{label}: evaluator did not retain exactly two results"
    first_storage = results[0].untyped_storage().data_ptr()
    second_storage = results[1].untyped_storage().data_ptr()
    if first_storage == second_storage:
        return f"{label}: repeated calls did not return fresh storage"
    if not torch.allclose(
        results[0], results[1], atol=0.0, rtol=0.0, equal_nan=True
    ):
        return f"{label}: repeated calls were not exactly deterministic"
    return None


def _check_topk_case(
    torch,
    mod,
    logits,
    token_ids,
    ref,
    label,
    integrity_guard,
    *,
    backing_storages=(),
    expected_nan_rows=(),
):
    frozen_inputs = (
        ("logits", logits, logits.clone()),
        ("token_ids", token_ids, token_ids.clone()),
    )
    frozen_backing = tuple(
        (name, storage, storage.clone()) for name, storage in backing_storages
    )
    protected_storage_ptrs = {
        tensor.untyped_storage().data_ptr() for _, tensor, _ in frozen_inputs
    }
    protected_storage_ptrs.update(
        storage.untyped_storage().data_ptr() for _, storage in backing_storages
    )
    expected_shape = (logits.shape[0], token_ids.shape[1])
    expected_nan_rows = set(expected_nan_rows)
    results = []
    for invocation in range(1, 3):
        result, launch_error = _invoke_topk_with_attestation(
            torch,
            mod,
            logits,
            token_ids,
            f"{label}, invocation {invocation}",
            integrity_guard,
        )
        if launch_error is not None:
            return launch_error
        if not isinstance(result, torch.Tensor):
            return f"{label}, invocation {invocation}: result is not a torch.Tensor"
        if tuple(result.shape) != expected_shape or result.dtype != torch.float32:
            return (
                f"{label}, invocation {invocation}: expected shape/dtype "
                f"{expected_shape}/{torch.float32}, got {tuple(result.shape)}/{result.dtype}"
            )
        if result.device != logits.device:
            return f"{label}, invocation {invocation}: result is on the wrong device"
        if result.untyped_storage().data_ptr() in protected_storage_ptrs:
            return f"{label}, invocation {invocation}: output aliases an input"
        for row in range(logits.shape[0]):
            if row in expected_nan_rows:
                if not torch.isnan(result[row]).all():
                    return f"{label}, invocation {invocation}: row {row} must be all NaN"
            elif torch.isnan(result[row]).any():
                return f"{label}, invocation {invocation}: unexpected NaN in row {row}"
        if not torch.allclose(
            result, ref, atol=1e-2, rtol=1e-2, equal_nan=True
        ):
            return f"{label}, invocation {invocation}: reference mismatch"
        results.append(result)

    repeat_error = _validate_topk_repeated_outputs(torch, results, label)
    if repeat_error is not None:
        return repeat_error
    for name, observed, frozen in frozen_inputs:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated protected input {name}"
    for name, observed, frozen in frozen_backing:
        if not torch.equal(observed, frozen):
            return f"{label}: candidate mutated backing allocation {name}"
    return None


def run_correctness():
    import torch
    capture_integrity = _capture_evaluator_integrity
    verify_integrity = _verify_evaluator_integrity
    integrity_snapshot = capture_integrity(torch)

    def integrity_guard():
        return verify_integrity(torch, integrity_snapshot)

    try: mod = load_module()
    except Exception as e: return False, f"Failed to load module: {e}"
    integrity_error = integrity_guard()
    if integrity_error is not None:
        return False, integrity_error
    device = "cuda"
    for i, (batch, vocab, ntok) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + i)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch, ntok), dtype=torch.int64, device=device)
            # CPU ref: log_softmax then gather
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ref = log_probs.gather(1, token_ids)
            error = _check_topk_case(
                torch,
                mod,
                logits,
                token_ids,
                ref,
                f"Shape {i+1}",
                integrity_guard,
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"Shape {i+1}: exception: {e}"

    for i, case in enumerate(CORRECTNESS_CASES):
        name = case["name"]
        try:
            torch.manual_seed(142 + i)
            logits, token_ids, logits_storage = _make_topk_correctness_inputs(
                torch, case, device
            )
            ref = torch.log_softmax(logits.float(), dim=-1).gather(
                1, token_ids.to(torch.int64)
            )
            error = _check_topk_case(
                torch,
                mod,
                logits,
                token_ids,
                ref,
                f"Contract case {name}",
                integrity_guard,
                backing_storages=(("logits", logits_storage),),
                expected_nan_rows=case.get("expected_nan_rows", ()),
            )
            if error is not None:
                return False, error
        except Exception as e:
            return False, f"Contract case {name}: exception: {e}"
    return True, None

def run_performance():
    import torch
    try: mod = load_module()
    except Exception: return []
    device = "cuda"
    test_cases = []

    for test_idx, (batch, vocab, ntok) in enumerate(TEST_SHAPES):
        try:
            torch.manual_seed(42 + test_idx)
            logits = torch.randn(batch, vocab, device=device, dtype=torch.float32)
            token_ids = torch.randint(0, vocab, (batch, ntok), dtype=torch.int64, device=device)
            def _bench_fn():
                mod.compute_token_logprobs(logits, token_ids)
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
                    "batch": batch,
                    "vocab": vocab,
                    "num_tokens": ntok
                }
            })
        except Exception:
            test_cases.append({
                "test_case_id": f"perf{test_idx + 1}",
                "execution_time_ms": -1.0,
                "params": {
                    "batch": batch,
                    "vocab": vocab,
                    "num_tokens": ntok
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
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "num_shapes": len(TEST_SHAPES) + len(CORRECTNESS_CASES),
        }
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f: json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f: json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)

if __name__ == "__main__": main()

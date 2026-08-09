import ast
import hashlib
import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks/triton2triton/vllm/triton_awq_dequantize"
RUNNER = TASK_ROOT / "scripts/task_runner.py"
SOURCE = TASK_ROOT / "source/triton_awq_dequantize.py"
CONFIG = TASK_ROOT / "config.yaml"
README = TASK_ROOT / "README.md"

EXPECTED_PERFORMANCE_SHAPES = [
    (64, 8, 32),
    (128, 16, 32),
    (128, 16, 64),
    (256, 32, 128),
    (256, 32, 64),
]
EXPECTED_PERFORMANCE_SHA256 = (
    "4b23d831f1fbddfc045e96d5b3f9f6748e4e603e3364ee0dded9c4886afd33b8"
)
EXPECTED_KERNEL_SHA256 = (
    "2025f9a9005fdc0a8321bbd5eeacca5cf49a51f07ea7e5cb474bb70dbcfbb3e2"
)


def _parse(path):
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _assignments(tree):
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _functions(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _referenced_names(function):
    return {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "awq_dequantize_contract_runner_for_cpu_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def test_awq_dequantize_performance_runner_and_kernel_are_byte_frozen():
    runner_source, runner_tree = _parse(RUNNER)
    source_source, source_tree = _parse(SOURCE)
    assignments = _assignments(runner_tree)
    runner_functions = _functions(runner_tree)
    source_functions = _functions(source_tree)

    assert ast.literal_eval(assignments["TEST_SHAPES"]) == EXPECTED_PERFORMANCE_SHAPES
    assert ast.literal_eval(assignments["WARMUP_ITERATIONS"]) == 10
    assert ast.literal_eval(assignments["BENCHMARK_ITERATIONS"]) == 100

    performance = runner_functions["run_performance"]
    performance_source = ast.get_source_segment(runner_source, performance)
    assert performance_source is not None
    assert hashlib.sha256(performance_source.encode()).hexdigest() == (
        EXPECTED_PERFORMANCE_SHA256
    )
    assert "CORRECTNESS_CASES" not in _referenced_names(performance)

    kernel = source_functions["awq_dequantize_kernel"]
    kernel_source = ast.get_source_segment(source_source, kernel)
    assert kernel_source is not None
    assert hashlib.sha256(kernel_source.encode()).hexdigest() == EXPECTED_KERNEL_SHA256

    benchmark_calls = [
        node
        for node in ast.walk(performance)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_benchmark_cuda_graph_or_events"
    ]
    assert len(benchmark_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in benchmark_calls[0].keywords}
    assert isinstance(keywords["warmup"], ast.Name)
    assert keywords["warmup"].id == "WARMUP_ITERATIONS"
    assert isinstance(keywords["repetition"], ast.Name)
    assert keywords["repetition"].id == "BENCHMARK_ITERATIONS"


def test_awq_dequantize_correctness_cases_cover_groups_tiles_and_storage_guards():
    _, tree = _parse(RUNNER)
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["CORRECTNESS_CASES"])

    assert ast.literal_eval(assignments["AWQ_ATOL"]) == 1e-2
    assert ast.literal_eval(assignments["AWQ_RTOL"]) == 1e-2
    assert ast.literal_eval(assignments["AWQ_ORDER"]) == (0, 4, 1, 5, 2, 6, 3, 7)
    assert {case["group_size"] for case in cases} == {32, 64, 128, "K"}
    assert {case["block_size_x"] for case in cases} == {8, 16, 64}
    assert {case["block_size_y"] for case in cases} == {16, 32, 64}
    assert all(case["N_packed"] % case["block_size_x"] for case in cases)
    assert all(case["K"] > 0 and case["N_packed"] > 0 for case in cases)
    assert all(
        (case["K"] if case["group_size"] == "K" else case["group_size"])
        % case["block_size_y"]
        == 0
        for case in cases
    )
    assert any(
        case["group_size"] == "K" and case["K"] & (case["K"] - 1)
        for case in cases
    )

    maker = ast.unparse(functions["_make_guarded_contiguous_matrix"])
    assert "storage_elements = logical_elements + 2 * guard_elements" in maker
    assert "storage_offset() == guard_elements" in maker
    assert "logical.is_contiguous()" in maker
    assert "CORRECTNESS_CASES" in _referenced_names(functions["run_correctness"])
    assert "CORRECTNESS_CASES" not in _referenced_names(functions["run_performance"])


def test_awq_dequantize_signed_words_have_explicit_unsigned_semantics():
    runner = _load_runner_module()

    assert runner._unpack_awq_word(-(2**31)) == (0, 0, 0, 0, 0, 0, 0, 8)
    assert runner._unpack_awq_word(-1) == (15,) * 8
    assert runner._unpack_awq_word(runner.SIGNED_89ABCDEF) == (
        15,
        11,
        14,
        10,
        13,
        9,
        12,
        8,
    )
    assert runner._unpack_awq_word(runner.SIGNED_87654321) == (
        1,
        5,
        2,
        6,
        3,
        7,
        4,
        8,
    )

    _, tree = _parse(RUNNER)
    injector = ast.unparse(_functions(tree)["_inject_signed_packed_words"])
    assert "qweight[0, 0] = -2 ** 31" in injector
    assert "qweight[1, 1] = -1" in injector
    assert "qweight[2, 2] = SIGNED_89ABCDEF" in injector
    assert "zeros[0, 0] = 19088743" in injector
    assert "zeros[0, 1] = SIGNED_87654321" in injector
    assert "zeros[0, 2] = -2 ** 31" in injector


def test_awq_dequantize_launch_recorder_resolves_grid_and_run_paths():
    runner = _load_runner_module()

    class FakeStorage:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

    class FakeTensor:
        def __init__(self, data_pointer, storage_pointer, value=None):
            self.data_pointer = data_pointer
            self.storage = FakeStorage(storage_pointer)
            self.value = value

        def data_ptr(self):
            return self.data_pointer

        def untyped_storage(self):
            return self.storage

        def fill_(self, value):
            self.value = value
            return self

        def clone(self):
            return FakeTensor(
                self.data_pointer + 10000,
                self.storage.pointer + 10000,
                self.value,
            )

    class FakeTorch:
        @staticmethod
        def equal(left, right):
            return left.value == right.value

    class FakeKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append(("bracket", grid, args, kwargs))
                return "bracket-result"

            return launch

        def run(self, *args, **kwargs):
            self.calls.append(("run", args, kwargs))
            return "run-result"

    kernel = FakeKernel()
    recorder = runner._KernelLaunchRecorder(kernel)
    qweight = FakeTensor(101, 1001)
    scales = FakeTensor(102, 1002)
    zeros = FakeTensor(103, 1003)
    output = FakeTensor(104, 1004)
    qweight.shape = (64, 9)
    result = recorder[lambda meta: (9 // meta["BLOCK_SIZE_X"], 4)](
        qweight,
        scales,
        zeros,
        32,
        output,
        9,
        64,
        BLOCK_SIZE_X=3,
        BLOCK_SIZE_Y=16,
    )
    assert result == "bracket-result"
    expected_first = {
        "grid": (3, 4),
        "block_size_x": 3,
        "block_size_y": 16,
        "qweight": qweight,
        "scales": scales,
        "zeros": zeros,
        "result": output,
        "data_ptrs": (101, 102, 103, 104),
        "storage_ptrs": (1001, 1002, 1003, 1004),
    }
    assert {
        key: recorder.launches[0][key] for key in expected_first
    } == expected_first
    assert recorder.launches[0]["post_launch_snapshot"].value != (
        recorder.launches[0]["post_launch_snapshot"].value
    )
    assert recorder.run(
        qweight,
        scales,
        zeros,
        32,
        output,
        9,
        64,
        grid=(5, 6),
        BLOCK_SIZE_X=8,
        BLOCK_SIZE_Y=32,
    ) == "run-result"
    expected_last = {
        "grid": (5, 6),
        "block_size_x": 8,
        "block_size_y": 32,
        "qweight": qweight,
        "scales": scales,
        "zeros": zeros,
        "result": output,
        "data_ptrs": (101, 102, 103, 104),
        "storage_ptrs": (1001, 1002, 1003, 1004),
    }
    assert {
        key: recorder.launches[-1][key] for key in expected_last
    } == expected_last

    validate = runner._validate_awq_launch_batch
    args = ((3, 4), 3, 16, 32, qweight, scales, zeros, output, "case", 1)
    assert validate(FakeTorch, recorder.launches[:1], *args) is None
    assert "expected exactly one" in validate(FakeTorch, [], *args)
    assert "expected exactly one" in validate(
        FakeTorch, recorder.launches[:1] * 2, *args
    )
    assert "qweight was not bound" in validate(
        FakeTorch, [{**recorder.launches[0], "qweight": object()}], *args
    )
    assert "data-pointer binding mismatch" in validate(
        FakeTorch,
        [{**recorder.launches[0], "data_ptrs": (0, 0, 0, 0)}],
        *args,
    )
    assert "positional operands" in validate(
        FakeTorch,
        [{**recorder.launches[0], "args": recorder.launches[0]["args"] + (object(),)}],
        *args,
    )
    assert "unexpected keyword operands" in validate(
        FakeTorch,
        [{**recorder.launches[0], "kwargs": {**recorder.launches[0]["kwargs"], "hidden": object()}}],
        *args,
    )
    wrong_dimensions = list(recorder.launches[0]["args"])
    wrong_dimensions[3] = 64
    assert "target dimensions" in validate(
        FakeTorch,
        [{**recorder.launches[0], "args": tuple(wrong_dimensions)}],
        *args,
    )
    assert "num_warps must be a positive integer" in validate(
        FakeTorch,
        [{**recorder.launches[0], "kwargs": {**recorder.launches[0]["kwargs"], "num_warps": True}}],
        *args,
    )

    observed = FakeTensor(204, 2004, "correct")
    correct_snapshot = FakeTensor(304, 3004, "correct")
    dummy_snapshot = FakeTensor(404, 4004, float("nan"))
    validate_snapshot = runner._validate_awq_target_snapshot
    assert validate_snapshot(
        FakeTorch, correct_snapshot, observed, "case", 1
    ) is None
    assert "dummy" in validate_snapshot(
        FakeTorch, dummy_snapshot, observed, "case", 1
    )


def test_awq_dequantize_checker_attests_launch_alias_determinism_and_rejection():
    _, tree = _parse(RUNNER)
    functions = _functions(tree)
    checker = ast.unparse(functions["_check_awq_case"])
    launch_validator = ast.unparse(functions["_validate_awq_launch_batch"])
    validator = ast.unparse(functions["_validate_awq_output"])
    rejections = ast.unparse(functions["_check_contract_rejections"])

    for phrase in (
        "original_kernel = mod.awq_dequantize_kernel",
        "mod.awq_dequantize_kernel = recorder",
        "for invocation in range(1, 3)",
        "_validate_awq_launch_batch",
        "post_launch_snapshot",
        "_validate_awq_target_snapshot",
        "repeated calls did not return fresh storage",
        "torch.equal(results[0], results[1])",
        "candidate mutated {name} backing storage",
        "mod.awq_dequantize_kernel = original_kernel",
    ):
        assert phrase in checker

    for phrase in (
        "expected exactly one target kernel launch",
        "expected_bindings",
        "was not bound to the evaluated tensor",
        "data-pointer binding mismatch",
        "storage binding mismatch",
    ):
        assert phrase in launch_validator

    for phrase in (
        "result.dtype != torch.float16",
        "result.device != expected_device",
        "result.is_contiguous()",
        "result.untyped_storage().data_ptr() in protected_storage_ptrs",
        "torch.isfinite(result)",
        "atol=AWQ_ATOL",
        "rtol=AWQ_RTOL",
    ):
        assert phrase in validator

    for phrase in (
        "int64 qweight",
        "float32 scales",
        "int64 zeros",
        "noncontiguous qweight",
        "mismatched scales",
        "mismatched zeros",
        "unsupported group_size",
        "nondivisible scales rows",
        "mixed devices",
        "non-power-of-two block_size_x",
        "non-divisor block_size_y",
        "target kernel was launched",
    ):
        assert phrase in rejections


def test_awq_dequantize_wrapper_validates_without_copy_and_forwards_tiles():
    source, tree = _parse(SOURCE)
    assignments = _assignments(tree)
    functions = _functions(tree)
    validator = ast.unparse(functions["_validate_awq_dequantize_contract"])
    wrapper_node = functions["awq_dequantize_triton"]
    wrapper = ast.get_source_segment(source, wrapper_node)
    assert wrapper is not None

    assert ast.literal_eval(assignments["AWQ_TRITON_SUPPORTED_GROUP_SIZES"]) == (
        32,
        64,
        128,
    )
    assert ast.literal_eval(assignments["AWQ_TRITON_MAX_BLOCK_SIZE"]) == 128
    for phrase in (
        "qweight.dtype != torch.int32",
        "zeros.dtype != torch.int32",
        "scales.dtype != torch.float16",
        "qweight.device == scales.device == zeros.device",
        "tensor.is_contiguous()",
        "K % num_groups != 0",
        "group_size != K",
        "scales.shape != (num_groups, num_cols * 8)",
        "zeros.shape != (num_groups, num_cols)",
        "value & value - 1",
        "group_size % block_size_y != 0",
    ):
        assert phrase in validator

    assert "block_size_x: int = 32" in wrapper
    assert "block_size_y: int = 32" in wrapper
    assert "BLOCK_SIZE_X=block_size_x" in wrapper
    assert "BLOCK_SIZE_Y=block_size_y" in wrapper
    assert ".contiguous(" not in wrapper
    assert ".clone(" not in wrapper
    assert ".copy_(" not in wrapper
    assert ".to(" not in wrapper


def test_awq_dequantize_documentation_matches_the_enforced_contract():
    config = CONFIG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    for phrase in (
        "C-contiguous",
        "unsigned two's-complement",
        "group_size` is exactly 32",
        "rejected rather than copied",
        "fresh C-contiguous",
        "Repeated calls with identical inputs are deterministic",
        "performance command remain unchanged",
    ):
        assert phrase in config

    for phrase in (
        "nonzero offset",
        "hidden contiguous copy",
        "unsigned two's-complement",
        "block_size_y` divides the group size",
        "frozen. Correctness-only cases",
        "baseline GPU kernel body is unchanged",
        "no GPU correctness or speed claim",
    ):
        assert phrase in readme

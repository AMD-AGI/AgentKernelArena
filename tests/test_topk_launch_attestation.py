import ast
import hashlib
import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks/triton2triton/vllm/triton_topk_log_softmax"
RUNNER = TASK_ROOT / "scripts/task_runner.py"
SOURCE = TASK_ROOT / "source/triton_topk_log_softmax.py"
CONFIG = TASK_ROOT / "config.yaml"
README = TASK_ROOT / "README.md"

EXPECTED_PERFORMANCE_SHA256 = (
    "b333c359ccba4b524c025bc07229b775f0f105484533dd3b12f74ff68e57e526"
)
EXPECTED_SOURCE_SHA256 = (
    "181d6ce2e536410426bd501c3c3d195bf61424e1a1693ed24e2ce1378e71f5d3"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "topk_launch_attestation_runner_for_cpu_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def _functions(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return source, {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class FakeStorage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeTensor:
    def __init__(
        self,
        data_pointer,
        storage_pointer,
        shape,
        strides,
        dtype,
        values,
        device="cuda:0",
    ):
        self._data_pointer = data_pointer
        self._storage = FakeStorage(storage_pointer)
        self.shape = shape
        self._strides = strides
        self.dtype = dtype
        self.values = values
        self.device = device

    def data_ptr(self):
        return self._data_pointer

    def untyped_storage(self):
        return self._storage

    def stride(self, dimension):
        return self._strides[dimension]

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        return FakeTensor(
            self._data_pointer + 10000,
            self._storage.pointer + 10000,
            self.shape,
            self._strides,
            dtype,
            self.values,
            self.device,
        )

    def fill_(self, value):
        self.values = value
        return self

    def clone(self):
        return FakeTensor(
            self._data_pointer + 10000,
            self._storage.pointer + 10000,
            self.shape,
            self._strides,
            self.dtype,
            self.values,
            self.device,
        )


class FakeTorch:
    Tensor = FakeTensor
    int64 = "int64"

    @staticmethod
    def equal(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.values == right.values
        )

    @staticmethod
    def allclose(left, right, **_kwargs):
        return left.values == right.values


class FakeKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return (args, kwargs)

        return launch


def test_topk_launch_attestation_accepts_bound_baseline_and_rejects_bypasses():
    runner = _load_runner_module()
    logits = FakeTensor(
        101, 1001, (3, 1003), (1032, 1), "float16", "logits"
    )
    caller_ids = FakeTensor(
        102, 1002, (3, 7), (7, 1), "int32", ((5, 3, 5, 0, 9, 2, 1),) * 3
    )
    converted_ids = FakeTensor(
        202, 2002, (3, 7), (7, 1), "int64", caller_ids.values
    )
    output = FakeTensor(
        103, 1003, (3, 7), (7, 1), "float32", "output"
    )
    recorder = runner._KernelLaunchRecorder(FakeKernel())
    recorder[(3,)](
        output,
        logits,
        1032,
        converted_ids,
        7,
        1003,
        BLOCK_SIZE=1024,
        PADDED_TOPK=8,
    )
    launch = recorder.launches[0]
    validate = runner._validate_topk_launch_batch
    args = (FakeTorch, logits, caller_ids, output, "fake topk")

    assert validate(FakeTorch, [launch], logits, caller_ids, output, "fake topk") is None
    assert "expected one target-kernel launch" in validate(FakeTorch, [], *args[1:])
    assert "expected one target-kernel launch" in validate(
        FakeTorch, [launch, launch], *args[1:]
    )
    assert "launch operands were not captured" in validate(
        FakeTorch, [{"grid": (3,)}], *args[1:]
    )

    other_output = FakeTensor(
        303, 3003, (3, 7), (7, 1), "float32", "output"
    )
    assert "target output was not bound" in validate(
        FakeTorch, [{**launch, "output": other_output}], *args[1:]
    )
    assert "target logits were not bound" in validate(
        FakeTorch, [{**launch, "logits": object()}], *args[1:]
    )
    wrong_ids = FakeTensor(
        204, 2004, (3, 7), (7, 1), "int64", "wrong values"
    )
    assert "do not match" in validate(
        FakeTorch, [{**launch, "token_ids": wrong_ids}], *args[1:]
    )
    assert "data-pointer binding mismatch" in validate(
        FakeTorch, [{**launch, "data_ptrs": (0, 0, 0)}], *args[1:]
    )
    assert "expected topk=7" in validate(
        FakeTorch, [{**launch, "topk": 6}], *args[1:]
    )
    assert "positional operands" in validate(
        FakeTorch,
        [{**launch, "args": launch["args"] + (object(),)}],
        *args[1:],
    )
    assert "unexpected keyword operands" in validate(
        FakeTorch,
        [{**launch, "kwargs": {**launch["kwargs"], "hidden": object()}}],
        *args[1:],
    )
    assert "duplicate target operand bindings" in validate(
        FakeTorch,
        [{**launch, "kwargs": {**launch["kwargs"], "output_ptr": output}}],
        *args[1:],
    )
    assert "num_warps must be a positive integer" in validate(
        FakeTorch,
        [{**launch, "kwargs": {**launch["kwargs"], "num_warps": True}}],
        *args[1:],
    )

    correct_result = FakeTensor(
        403, 4003, (3, 7), (7, 1), "float32", "correct"
    )
    correct_snapshot = FakeTensor(
        503, 5003, (3, 7), (7, 1), "float32", "correct"
    )
    dummy_snapshot = FakeTensor(
        603, 6003, (3, 7), (7, 1), "float32", float("nan")
    )
    validate_snapshot = runner._validate_topk_target_snapshot
    assert validate_snapshot(
        FakeTorch, correct_snapshot, correct_result, "case"
    ) is None
    assert "dummy" in validate_snapshot(
        FakeTorch, dummy_snapshot, correct_result, "case"
    )


def test_topk_correctness_uses_finally_restored_attested_target_only():
    source, functions = _functions(RUNNER)
    invoke = ast.unparse(functions["_invoke_topk_with_attestation"])
    correctness = ast.unparse(functions["run_correctness"])
    checker = ast.unparse(functions["_check_topk_case"])
    performance = ast.unparse(functions["run_performance"])

    for phrase in (
        "original_kernel = mod._topk_log_softmax_kernel",
        "mod._topk_log_softmax_kernel = recorder",
        "mod._topk_log_softmax_kernel = original_kernel",
        "_validate_topk_launch_batch",
        "post_launch_snapshot",
        "_validate_topk_target_snapshot",
    ):
        assert phrase in invoke
    assert "_check_topk_case" in correctness
    assert "_invoke_topk_with_attestation" in checker
    assert "for invocation in range(1, 3)" in checker
    assert "_validate_topk_repeated_outputs" in checker
    assert "_invoke_topk_with_attestation" not in performance

    performance_segment = ast.get_source_segment(
        source, functions["run_performance"]
    )
    assert performance_segment is not None
    assert hashlib.sha256(performance_segment.encode()).hexdigest() == (
        EXPECTED_PERFORMANCE_SHA256
    )
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256


def test_topk_repeated_outputs_require_fresh_storage_and_exact_values():
    runner = _load_runner_module()
    first = FakeTensor(1, 101, (1, 2), (2, 1), "float32", (1.0, 2.0))
    second = FakeTensor(2, 102, (1, 2), (2, 1), "float32", (1.0, 2.0))

    assert runner._validate_topk_repeated_outputs(
        FakeTorch, [first, second], "case"
    ) is None
    assert "fresh storage" in runner._validate_topk_repeated_outputs(
        FakeTorch,
        [first, FakeTensor(3, 101, (1, 2), (2, 1), "float32", (1.0, 2.0))],
        "case",
    )
    assert "exactly deterministic" in runner._validate_topk_repeated_outputs(
        FakeTorch,
        [first, FakeTensor(4, 104, (1, 2), (2, 1), "float32", (1.0, 3.0))],
        "case",
    )


def test_topk_launch_contract_is_documented_without_changing_scoring():
    config = CONFIG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    for phrase in (
        "exactly one monolithic",
        "int64-converted token IDs",
        "returned output tensors",
        "leading stride, batch grid",
    ):
        assert phrase in config
    for phrase in (
        "exactly one target launch",
        "storage pointers",
        "detached dummy",
        "baseline source remain unchanged",
        "no target-GPU correctness or speed claim",
    ):
        assert phrase in readme

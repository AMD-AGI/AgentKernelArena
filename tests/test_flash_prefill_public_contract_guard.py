import ast
import hashlib
import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT / "tasks/triton2triton/vllm/triton_flash_prefill_attention"
)
RUNNER = TASK_ROOT / "scripts/task_runner.py"
SOURCE = TASK_ROOT / "source/triton_flash_prefill_attention.py"
CONFIG = TASK_ROOT / "config.yaml"
README = TASK_ROOT / "README.md"

EXPECTED_PERFORMANCE_SHAPES = [
    (2, 128, 8, 8, 64),
    (4, 256, 16, 4, 64),
    (2, 512, 32, 8, 128),
    (1, 1024, 16, 16, 64),
    (8, 64, 8, 1, 64),
]
EXPECTED_PERFORMANCE_SHA256 = (
    "4e28215317bba4045a4b132ee9466f9b441a7cf6f0394b34f6b56223fdfada3f"
)
EXPECTED_BASELINE_SOURCE_SHA256 = (
    "7f0a141fd36716f848ab95b4be9df3655b5b6da11e07654466076b701c9d3b5b"
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
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
    }


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "flash_prefill_contract_runner_for_cpu_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def test_flash_performance_runner_and_complete_baseline_source_are_byte_frozen():
    runner_source, runner_tree = _parse(RUNNER)
    assignments = _assignments(runner_tree)
    functions = _functions(runner_tree)

    assert ast.literal_eval(assignments["TEST_SHAPES"]) == (
        EXPECTED_PERFORMANCE_SHAPES
    )
    assert ast.literal_eval(assignments["WARMUP_ITERATIONS"]) == 10
    assert ast.literal_eval(assignments["BENCHMARK_ITERATIONS"]) == 100

    performance = functions["run_performance"]
    performance_source = ast.get_source_segment(runner_source, performance)
    assert performance_source is not None
    assert hashlib.sha256(performance_source.encode()).hexdigest() == (
        EXPECTED_PERFORMANCE_SHA256
    )
    assert "FLASH_CORRECTNESS_CASES" not in _referenced_names(performance)

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

    perf_ids = [
        ast.unparse(node)
        for node in ast.walk(performance)
        if isinstance(node, ast.JoinedStr)
        and ast.unparse(node) == "f'perf{test_idx + 1}'"
    ]
    assert len(perf_ids) == 2

    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
        EXPECTED_BASELINE_SOURCE_SHA256
    )


def test_flash_non_scoring_cases_cover_ragged_mha_gqa_mqa_and_window_modes():
    _, tree = _parse(RUNNER)
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["FLASH_CORRECTNESS_CASES"])

    assert ast.literal_eval(assignments["FLASH_ATOL"]) == 1e-2
    assert ast.literal_eval(assignments["FLASH_RTOL"]) == 1e-2
    assert {case["head_dim"] for case in cases} == {32, 64, 96, 128}
    assert {case["is_causal"] for case in cases} == {False, True}
    assert any(
        case["num_heads"] == case["num_kv_heads"] for case in cases
    )
    assert any(case["num_kv_heads"] == 1 for case in cases)
    assert any(
        1 < case["num_kv_heads"] < case["num_heads"] for case in cases
    )
    assert all(
        case["num_heads"] % case["num_kv_heads"] == 0 for case in cases
    )
    assert all(
        len(set(case["sequence_lengths"])) > 1
        for case in cases
        if len(case["sequence_lengths"]) > 1
    )
    assert {129, 257, 513} <= {
        length for case in cases for length in case["sequence_lengths"]
    }
    assert any(
        case["sliding_window_q"]
        and case["sliding_window_k"]
        and not case["is_causal"]
        for case in cases
    )
    assert any(
        case["sliding_window_q"] and case["is_causal"] for case in cases
    )
    assert any(
        not case["sliding_window_q"] and case["sliding_window_k"]
        for case in cases
    )
    assert any(case["softmax_scale"] not in (None, 0) for case in cases)
    assert "FLASH_CORRECTNESS_CASES" in _referenced_names(
        functions["run_correctness"]
    )
    assert "FLASH_CORRECTNESS_CASES" not in _referenced_names(
        functions["run_performance"]
    )


def test_flash_preserves_original_uniform_correctness_workloads():
    runner = _load_runner_module()
    cases = list(runner._uniform_correctness_cases())

    assert [
        (
            len(case["sequence_lengths"]),
            case["sequence_lengths"][0],
            case["num_heads"],
            case["num_kv_heads"],
            case["head_dim"],
        )
        for case in cases
    ] == EXPECTED_PERFORMANCE_SHAPES
    assert all(case["is_causal"] for case in cases)
    assert all(case["sliding_window_q"] is None for case in cases)
    assert all(case["sliding_window_k"] is None for case in cases)
    assert runner._start_locations((3, 19, 65)) == [0, 3, 22]


def test_flash_reference_models_independent_windows_and_float32_accumulation():
    _, tree = _parse(RUNNER)
    reference_source = ast.unparse(_functions(tree)["reference_attention"])

    for phrase in (
        "import torch",
        "q_b.float() @ k_b.float().T",
        "key_pos > query_pos",
        "query_pos - key_pos > sliding_window_q",
        "key_pos - query_pos > sliding_window_k",
        "scores.masked_fill(mask, float('-inf'))",
        "torch.softmax(scores, dim=-1)",
        "attn @ v_b.float()",
    ):
        assert phrase in reference_source


def test_flash_checker_enforces_return_mutation_finite_repeat_and_tolerance():
    _, tree = _parse(RUNNER)
    functions = _functions(tree)
    checker_source = ast.unparse(functions["_check_flash_case"])

    for phrase in (
        "protected_inputs",
        "original_kernel = mod._fwd_kernel",
        "mod._fwd_kernel = recorder",
        "for invocation in range(1, 3)",
        "_validate_flash_launch_batch",
        "post_launch_snapshot",
        "_validate_flash_target_snapshot",
        "o.fill_(float('nan'))",
        "returned is not None",
        "torch.isfinite(o)",
        "candidate mutated protected",
        "torch.equal(observed_outputs[0], observed_outputs[1])",
        "repeated calls are nondeterministic",
        "atol=FLASH_ATOL",
        "rtol=FLASH_RTOL",
        "mod._fwd_kernel = original_kernel",
    ):
        assert phrase in checker_source

    call = next(
        node
        for node in ast.walk(functions["_check_flash_case"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "context_attention_fwd"
    )
    keyword_names = {keyword.arg for keyword in call.keywords}
    assert {
        "max_input_len",
        "is_causal",
        "softmax_scale",
        "sliding_window_q",
        "sliding_window_k",
    } <= keyword_names


def test_flash_launch_attestation_rejects_no_dummy_wrong_and_extra_launches():
    runner = _load_runner_module()

    class FakeStorage:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

    class FakeTensor:
        def __init__(self, data_pointer, storage_pointer, strides, value=None):
            self._data_pointer = data_pointer
            self._storage = FakeStorage(storage_pointer)
            self._strides = strides
            self.value = value

        def data_ptr(self):
            return self._data_pointer

        def untyped_storage(self):
            return self._storage

        def stride(self, dimension):
            return self._strides[dimension]

        def fill_(self, value):
            self.value = value
            return self

        def clone(self):
            return FakeTensor(
                self._data_pointer + 10000,
                self._storage.pointer + 10000,
                self._strides,
                self.value,
            )

    class FakeTorch:
        @staticmethod
        def equal(left, right):
            return left.value == right.value

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return (args, kwargs)

            return launch

    q = FakeTensor(101, 1001, (512, 64))
    k = FakeTensor(102, 1002, (128, 64))
    v = FakeTensor(103, 1003, (128, 64))
    starts = FakeTensor(104, 1004, (1,))
    lengths = FakeTensor(105, 1005, (1,))
    out = FakeTensor(106, 1006, (512, 64))
    case = {
        "name": "fake_flash",
        "sequence_lengths": (3, 19, 65),
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "is_causal": False,
        "softmax_scale": None,
        "sliding_window_q": 0,
        "sliding_window_k": 0,
    }
    scale = (1.0 / (64**0.5)) * runner.FLASH_RCP_LN2
    recorder = runner._KernelLaunchRecorder(FakeKernel())
    recorder[(3, 8, 1)](
        q,
        k,
        v,
        scale,
        starts,
        lengths,
        out,
        512,
        64,
        128,
        64,
        128,
        64,
        512,
        64,
        kv_group_num=4,
        BLOCK_M=128,
        BLOCK_DMODEL=64,
        BLOCK_N=128,
        IS_CAUSAL=False,
        SLIDING_WINDOW_Q=0,
        SLIDING_WINDOW_K=0,
        Lk=64,
    )
    launch = recorder.launches[0]
    validate = runner._validate_flash_launch_batch
    args = (q, k, v, out, starts, lengths, 65, case, 1)
    assert validate([launch], *args) is None
    assert "expected one target-kernel launch" in validate([], *args)
    assert "expected one target-kernel launch" in validate(
        [launch, launch], *args
    )
    assert "launch operands were not captured" in validate(
        [{"grid": (3, 8, 1)}], *args
    )

    other_out = FakeTensor(206, 2006, (512, 64))
    wrong_binding = {**launch, "out": other_out}
    assert "out was not bound" in validate([wrong_binding], *args)
    wrong_grid = {**launch, "grid": (3, 8, 2)}
    assert "expected resolved grid" in validate([wrong_grid], *args)
    wrong_storage = {**launch, "storage_ptrs": (0, 0, 0, 0, 0, 0)}
    assert "storage binding mismatch" in validate([wrong_storage], *args)
    wrong_semantics = {**launch, "sliding_window_q": 17}
    assert "sliding_window_q=0" in validate([wrong_semantics], *args)
    assert "positional operands" in validate(
        [{**launch, "raw_args": launch["raw_args"] + (object(),)}], *args
    )
    assert "unexpected keyword operands" in validate(
        [{**launch, "raw_kwargs": {**launch["raw_kwargs"], "hidden": object()}}],
        *args,
    )
    assert "duplicate target operand bindings" in validate(
        [{**launch, "raw_kwargs": {**launch["raw_kwargs"], "Q": q}}], *args
    )
    assert "kv_group_num must be a int" in validate(
        [{**launch, "kv_group_num": 4.0}], *args
    )
    assert "num_warps must be a positive integer" in validate(
        [{**launch, "raw_kwargs": {**launch["raw_kwargs"], "num_warps": True}}],
        *args,
    )

    observed = FakeTensor(306, 3006, (512, 64), "correct")
    correct_snapshot = FakeTensor(406, 4006, (512, 64), "correct")
    dummy_snapshot = FakeTensor(506, 5006, (512, 64), float("nan"))
    validate_snapshot = runner._validate_flash_target_snapshot
    assert validate_snapshot(FakeTorch, correct_snapshot, observed, "case") is None
    assert "dummy" in validate_snapshot(
        FakeTorch, dummy_snapshot, observed, "case"
    )


def test_flash_wrapper_signature_config_and_readme_agree():
    _, source_tree = _parse(SOURCE)
    wrapper = _functions(source_tree)["context_attention_fwd"]
    assert [argument.arg for argument in wrapper.args.args] == [
        "q",
        "k",
        "v",
        "o",
        "b_start_loc",
        "b_seq_len",
        "max_input_len",
        "is_causal",
        "softmax_scale",
        "sliding_window_q",
        "sliding_window_k",
    ]

    config = CONFIG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for phrase in (
        "variable-length packed sequences",
        "Grouped-query attention (GQA) and multi-query attention (MQA)",
        "Causal and non-causal masking",
        "positive int32 sequence lengths",
        "head dimensions are 32, 64, 96, and 128",
        "`query_pos-key_pos <= window_q`",
        "`key_pos-query_pos <= window_k`",
        "value must be `None`",
        "exactly one `_fwd_kernel` launch",
        "`TEST_SHAPES`,",
        "warmup=10, repetitions=100",
    ):
        assert phrase in config
    assert "arbitrary sequence lengths and head dimensions" not in config

    for phrase in (
        "exact exclusive prefix sum",
        "independent backward and forward windows",
        "float32 PyTorch reference",
        "detached dummy launch",
        "complete baseline source file",
        "does not transplant a candidate optimization",
        "CPU structural tests alone make no GPU correctness or speed claim",
    ):
        assert phrase in readme

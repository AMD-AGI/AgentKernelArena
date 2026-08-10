import ast
import hashlib
import importlib.util
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks/triton2triton/vllm/triton_awq_gemm"
RUNNER = TASK_ROOT / "scripts/task_runner.py"
SOURCE = TASK_ROOT / "source/triton_awq_gemm.py"
CONFIG = TASK_ROOT / "config.yaml"
README = TASK_ROOT / "README.md"

EXPECTED_PERFORMANCE_SHAPES = [
    (32, 64, 8, 32, 1),
    (64, 128, 16, 32, 1),
    (32, 128, 16, 64, 1),
    (64, 256, 32, 128, 1),
    (128, 256, 32, 64, 1),
]
EXPECTED_PERFORMANCE_SHA256 = (
    "6a9e7adb6503ef29f2936eed9ccf4956d5eb64441ffcfbd526bcfbef1de87abe"
)
EXPECTED_KERNEL_SHA256 = (
    "c574470d4e2a2b5ea78d29f19fd36f3e878543b8b9183864027d83e38eb51e8a"
)
EXPECTED_WRAPPER_SHA256 = (
    "315d4acb767ca633647eed0a57b652206dfbed2a18214f49b83633a6bae07e74"
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
        "awq_gemm_contract_runner_for_cpu_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def test_awq_gemm_performance_runner_is_frozen_and_kernel_uses_fp32_accumulation():
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

    kernel_source = ast.get_source_segment(
        source_source, source_functions["awq_gemm_kernel"]
    )
    assert kernel_source is not None
    assert hashlib.sha256(kernel_source.encode()).hexdigest() == (
        EXPECTED_KERNEL_SHA256
    )
    kernel_ast = ast.unparse(source_functions["awq_gemm_kernel"])
    assert "accumulator_dtype = tl.float32" in kernel_ast
    assert "tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)" in kernel_ast
    assert "c = accumulator.to(c_ptr.type.element_ty)" in kernel_ast
    wrapper_ast = ast.unparse(source_functions["awq_gemm_triton"])
    wrapper_source = ast.get_source_segment(
        source_source, source_functions["awq_gemm_triton"]
    )
    assert wrapper_source is not None
    assert hashlib.sha256(wrapper_source.encode()).hexdigest() == (
        EXPECTED_WRAPPER_SHA256
    )
    assert "dtype=torch.float32" in wrapper_ast
    assert "result.sum(0).to(scales.dtype)" in wrapper_ast

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
        node
        for node in ast.walk(performance)
        if isinstance(node, ast.JoinedStr)
        and ast.unparse(node) == "f'perf{test_idx + 1}'"
    ]
    assert len(perf_ids) == 2


def test_awq_gemm_existing_correctness_shapes_keep_high_dynamic_inputs():
    _, tree = _parse(RUNNER)
    functions = _functions(tree)
    correctness_source = ast.unparse(functions["run_correctness"])

    assert "torch.randn(M, K, device=device, dtype=dtype)" in correctness_source
    assert (
        "torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01"
        in correctness_source
    )
    assert "input_tensor = torch.randn(M, K, device=device, dtype=dtype) * 0.125" not in (
        correctness_source
    )
    assert "_check_awq_case" in correctness_source


def test_awq_gemm_contract_cases_cover_every_split_with_active_rounds():
    _, tree = _parse(RUNNER)
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["CORRECTNESS_CASES"])

    assert ast.literal_eval(assignments["AWQ_ATOL"]) == 1e-2
    assert ast.literal_eval(assignments["AWQ_RTOL"]) == 1e-2
    assert ast.literal_eval(assignments["AWQ_ORDER"]) == (0, 4, 1, 5, 2, 6, 3, 7)
    assert {case["split_k"] for case in cases} == {1, 2, 4, 8, 16, 32}
    assert {case["group_size"] for case in cases} == {32, 64, 128, "K"}
    assert {case["layout"] for case in cases} == {
        "row_padded",
        "inner_strided",
        "transposed",
    }
    assert {case["profile"] for case in cases} == {
        "high_dynamic",
        "low_cancellation",
    }
    assert all(case["K"] % (32 * case["split_k"]) == 0 for case in cases)
    assert all(case["K"] // (32 * case["split_k"]) >= 2 for case in cases)
    assert all(case["M"] % 32 != 0 for case in cases)
    assert all((case["N_packed"] * 8) % 32 != 0 for case in cases)
    assert all(
        case[padding] > 0
        for case in cases
        for padding in (
            "input_padding",
            "qweight_padding",
            "scales_padding",
            "qzeros_padding",
        )
    )

    boundary = next(case for case in cases if case["split_k"] == 32)
    assert boundary["group_size"] == "K"
    assert boundary["K"] == 2048
    tuned = next(case for case in cases if "block_sizes" in case)
    assert tuned["block_sizes"] == (16, 64, 16)
    assert "CORRECTNESS_CASES" in _referenced_names(functions["run_correctness"])
    assert "CORRECTNESS_CASES" not in _referenced_names(functions["run_performance"])


def test_awq_gemm_signed_words_use_explicit_unsigned_nibble_semantics():
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
    assert "qzeros[0, 0] = 19088743" in injector
    assert "qzeros[0, 1] = SIGNED_87654321" in injector
    assert "qzeros[0, 2] = -2 ** 31" in injector


def test_awq_gemm_checker_attests_launch_storage_finiteness_and_determinism():
    _, tree = _parse(RUNNER)
    functions = _functions(tree)
    checker_source = ast.unparse(functions["_check_awq_case"])
    validator_source = ast.unparse(functions["_validate_awq_output"])
    recorder_source = ast.unparse(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_KernelLaunchRecorder"
        )
    )

    for phrase in (
        "original_kernel = mod.awq_gemm_kernel",
        "mod.awq_gemm_kernel = recorder",
        "for _invocation in range(2)",
        "mod.awq_gemm_kernel = original_kernel",
        "_validate_awq_launch_batch",
        "repeated calls did not return fresh storage",
        "torch.equal(results[0], results[1])",
        "torch.isfinite(ref)",
        "protected_inputs",
        "frozen_backing",
        "candidate mutated protected input",
        "candidate wrote outside logical",
    ):
        assert phrase in checker_source

    for phrase in (
        "result.dtype != expected_dtype",
        "result.device != expected_device",
        "result.untyped_storage().data_ptr()",
        "output aliases an input or its backing storage",
        "torch.isfinite(result)",
        "atol=AWQ_ATOL",
        "rtol=AWQ_RTOL",
    ):
        assert phrase in validator_source

    assert "kwargs.get('SPLIT_K')" in recorder_source
    assert "self._resolve_grid(grid, kwargs)" in recorder_source
    assert "record['inputs_before_launch']" in recorder_source
    assert "output.fill_(float('nan'))" in recorder_source
    assert "record['output_after_launch'] = output.clone()" in recorder_source
    assert "self.kernel.run(*args, **kwargs)" in recorder_source
    assert "getattr(self.kernel, name)" in recorder_source
    assert isinstance(functions["_check_awq_case"].body[-1], ast.Return)
    assert any(
        isinstance(node, ast.Try) and node.finalbody
        for node in ast.walk(functions["_check_awq_case"])
    )


def test_awq_gemm_launch_attestation_rejects_wrong_or_extra_launches():
    runner = _load_runner_module()

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return (args, kwargs)

            return launch

    recorder = runner._KernelLaunchRecorder(FakeKernel())
    grid = lambda meta: (
        2 * ((40 + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"]),
        meta["SPLIT_K"],
    )
    launch = recorder[grid]
    launch(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        SPLIT_K=2,
    )
    expected = {
        "grid": (2, 2),
        "args": (),
        "kwargs": {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "SPLIT_K": 2,
        },
        "split_k": 2,
        "block_m": 16,
        "block_n": 64,
        "block_k": 32,
    }
    assert recorder.launches == [expected]
    # Full tensor binding is exercised on GPU; these CPU-only cases prove that
    # wrong/extra launches fail before any tensor operation is attempted.
    common = (None, 32, 40, 64, 32, 2, (16, 64, 32), (), None, set(), "case", 1)
    assert "expected one target-kernel launch" in runner._validate_awq_launch_batch(
        common[0], recorder.launches * 2, *common[1:]
    )
    wrong_grid = [{**expected, "grid": (2, 1)}]
    assert "expected resolved grid" in runner._validate_awq_launch_batch(
        common[0], wrong_grid, *common[1:]
    )
    wrong_meta = [{**expected, "block_m": 32}]
    assert "block_m=16 was not attested" in runner._validate_awq_launch_batch(
        common[0], wrong_meta, *common[1:]
    )


def test_awq_gemm_launch_attestation_binds_operands_and_output():
    runner = _load_runner_module()

    class Storage:
        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

    class Tensor:
        def __init__(self, shape, value, pointer, *, dtype="dtype", device="gpu"):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.value = value
            self.dtype = dtype
            self.device = device
            self._storage = Storage(pointer)
            self.reduced = None

        def untyped_storage(self):
            return self._storage

        def sum(self, dimension):
            assert dimension == 0
            return self.reduced

        def to(self, dtype):
            assert dtype == self.dtype
            return self

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def equal(left, right):
            return left.value == right.value

    FakeTorch.Tensor = Tensor

    input_tensor = Tensor((32, 64), "input", 1)
    qweight = Tensor((64, 5), "qweight", 2, dtype="int32")
    scales = Tensor((2, 40), "scales", 3)
    qzeros = Tensor((2, 5), "qzeros", 4, dtype="int32")
    partial = Tensor((2, 32, 40), "partial", 5, dtype="float32")
    result = Tensor((32, 40), "result", 6)
    partial.reduced = result
    launch = {
        "grid": (2, 2),
        "args": (
            input_tensor,
            qweight,
            partial,
            qzeros,
            scales,
            32,
            40,
            64,
            32,
        ),
        "kwargs": {},
        "inputs_before_launch": (input_tensor, qweight, scales, qzeros),
        "output_after_launch": partial,
        "split_k": 2,
        "block_m": 16,
        "block_n": 64,
        "block_k": 32,
    }
    arguments = (
        FakeTorch,
        [launch],
        32,
        40,
        64,
        32,
        2,
        (16, 64, 32),
        (input_tensor, qweight, scales, qzeros),
        result,
        {1, 2, 3, 4},
        "case",
        1,
    )
    assert runner._validate_awq_launch_batch(*arguments) is None

    wrong_input = Tensor((32, 64), "unrelated", 7)
    wrong_launch = {
        **launch,
        "inputs_before_launch": (wrong_input, qweight, scales, qzeros),
    }
    assert "input operand is not bound" in runner._validate_awq_launch_batch(
        FakeTorch, [wrong_launch], *arguments[2:]
    )

    # A wrapper-side precompute cannot be smuggled into a dummy target as a
    # hidden tenth tensor argument or an unreviewed keyword operand.
    hidden_result = Tensor((32, 40), "precomputed", 9)
    extra_positional = {
        **launch,
        "args": (*launch["args"], hidden_result),
    }
    assert "10 positional operands" in runner._validate_awq_launch_batch(
        FakeTorch, [extra_positional], *arguments[2:]
    )
    extra_keyword = {
        **launch,
        "kwargs": {"precomputed_result": hidden_result},
    }
    assert "unexpected keyword operands" in runner._validate_awq_launch_batch(
        FakeTorch, [extra_keyword], *arguments[2:]
    )

    valid_controls = {
        **launch,
        "kwargs": {"num_warps": 4, "num_stages": 2, "waves_per_eu": 0},
    }
    assert runner._validate_awq_launch_batch(
        FakeTorch, [valid_controls], *arguments[2:]
    ) is None
    tensor_control = {
        **launch,
        "kwargs": {"num_warps": hidden_result},
    }
    assert "num_warps must be a positive integer" in (
        runner._validate_awq_launch_batch(
            FakeTorch, [tensor_control], *arguments[2:]
        )
    )
    negative_control = {
        **launch,
        "kwargs": {"waves_per_eu": -1},
    }
    assert "waves_per_eu must be a nonnegative integer" in (
        runner._validate_awq_launch_batch(
            FakeTorch, [negative_control], *arguments[2:]
        )
    )

    partial.reduced = Tensor((32, 40), "unrelated", 8)
    assert "not derived from the target output" in runner._validate_awq_launch_batch(
        *arguments
    )


def test_awq_gemm_recorder_poison_and_snapshot_bracket_target_output():
    runner = _load_runner_module()
    events = []

    class Tensor:
        def __init__(self, name, value="initial"):
            self.name = name
            self.value = value

        def fill_(self, value):
            assert value != value
            self.value = "poisoned"
            events.append("poison")

        def clone(self):
            events.append(f"clone:{self.name}")
            return Tensor(self.name, self.value)

    class Cuda:
        @staticmethod
        def synchronize():
            events.append("synchronize")

    class Torch:
        cuda = Cuda()

    Torch.Tensor = Tensor

    class Kernel:
        def __getitem__(self, _grid):
            def launch(*args, **_kwargs):
                assert args[2].value == "poisoned"
                args[2].value = "kernel-output"
                events.append("launch")

            return launch

    input_tensor = Tensor("input")
    qweight = Tensor("qweight")
    output = Tensor("output")
    qzeros = Tensor("qzeros")
    scales = Tensor("scales")
    recorder = runner._KernelLaunchRecorder(
        Kernel(), torch=Torch(), capture_output=True
    )
    recorder[(1,)](
        input_tensor,
        qweight,
        output,
        qzeros,
        scales,
        SPLIT_K=1,
    )
    assert events == [
        "clone:input",
        "clone:qweight",
        "clone:scales",
        "clone:qzeros",
        "synchronize",
        "poison",
        "launch",
        "synchronize",
        "clone:output",
    ]
    assert recorder.launches[0]["output_after_launch"].value == "kernel-output"
    assert [
        tensor.name for tensor in recorder.launches[0]["inputs_before_launch"]
    ] == ["input", "qweight", "scales", "qzeros"]


def test_awq_gemm_guarded_layouts_and_baseline_wrapper_are_explicit():
    _, runner_tree = _parse(RUNNER)
    _, source_tree = _parse(SOURCE)
    runner_functions = _functions(runner_tree)
    source_functions = _functions(source_tree)

    view_source = ast.unparse(runner_functions["_guarded_logical_view"])
    assert "layout == 'row_padded'" in view_source
    assert "layout == 'inner_strided'" in view_source
    assert "layout == 'transposed'" in view_source
    assert "storage_offset() > 0" in view_source
    assert "not logical.is_contiguous()" in view_source
    assert "all((stride > 0 for stride in logical.stride()))" in view_source

    wrapper = source_functions["awq_gemm_triton"]
    assert [argument.arg for argument in wrapper.args.args] == [
        "input",
        "qweight",
        "scales",
        "qzeros",
        "split_k_iters",
        "block_size_m",
        "block_size_n",
        "block_size_k",
    ]
    assert [ast.literal_eval(default) for default in wrapper.args.defaults] == [32, 32, 32]
    wrapper_source = ast.unparse(wrapper)
    assert "K % group_size == 0" in wrapper_source

    guarded_tensors = set()
    for node in wrapper.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Attribute)
            and node.test.operand.func.attr == "is_contiguous"
            and isinstance(node.test.operand.func.value, ast.Name)
        ):
            continue
        name = node.test.operand.func.value.id
        assert len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
        assignment = node.body[0]
        assert isinstance(assignment.targets[0], ast.Name)
        assert assignment.targets[0].id == name
        assert isinstance(assignment.value, ast.Call)
        assert isinstance(assignment.value.func, ast.Attribute)
        assert assignment.value.func.attr == "contiguous"
        guarded_tensors.add(name)
    assert guarded_tensors == {"input", "qweight", "scales", "qzeros"}


def test_awq_gemm_config_and_module_readme_match_the_enforced_contract():
    config = CONFIG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for phrase in (
        "`M`, `K`, and `N` are",
        "positive finite float16",
        "powers of two from 1 through 32",
        "positive-stride noncompact two-dimensional views",
        "padded row-major, inner-strided, and transposed layouts",
        "fresh deterministic `[M,N]` tensor",
        "must not alias another result",
        "Output must be finite",
        "`atol=1e-2, rtol=1e-2`",
        "`TEST_SHAPES`, `perf1`...`perf5`, warmup=10, repetitions=100",
    ):
        assert phrase in config
    assert "atol=1e-1" not in config
    assert "rtol=1e-1" not in config

    for phrase in (
        "Packed negative int32 words",
        "fresh deterministic",
        "run_performance",
        "source-level baseline change",
        "bounded scalar values",
        "Old timing receipts cannot be reused",
        "CPU structural tests alone make no GPU correctness or speed claim",
    ):
        assert phrase in readme

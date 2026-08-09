import ast
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT
    / "tasks"
    / "triton2triton"
    / "vllm"
    / "triton_matmul_persistent"
)
RUNNER = TASK_ROOT / "scripts" / "task_runner.py"
SOURCE = TASK_ROOT / "source" / "triton_matmul_persistent.py"
CONFIG = TASK_ROOT / "config.yaml"
README = TASK_ROOT / "README.md"

EXPECTED_PERFORMANCE_SHAPES = [
    (128, 128, 64),
    (256, 512, 128),
    (512, 256, 256),
    (1024, 1024, 512),
    (64, 2048, 128),
]
EXPECTED_PERFORMANCE_SHA256 = (
    "4ced4c81c3970c565390b9ff5335b7b3b8556445c339fce7caab34f7b5d340a7"
)
EXPECTED_PERFORMANCE_HELPER_SHA256 = (
    "b1d6c48a7d2318d4b38a05f79dd88ec54f07f776bd7a1af894729ef9bdd7b66e"
)
EXPECTED_COMPUTE_PID_SHA256 = (
    "54418b8c01531fe795dfb0ac5759e543a5106b60386d3e0bf834b50b5710dca7"
)
EXPECTED_KERNEL_SHA256 = (
    "b2501dc669d612ef81b9529aa0dc8ad1b248ec189bebc98e3fda8225719969c9"
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


def _classes(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def _referenced_names(function):
    return {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name)
    }


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "matmul_persistent_contract_runner_for_cpu_test", RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


class _DType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _load_source_with_cpu_stubs():
    """Import the real wrapper without importing torch or Triton packages."""

    torch_stub = types.ModuleType("torch")

    class Tensor:
        pass

    torch_stub.Tensor = Tensor
    torch_stub.float16 = _DType("float16")
    torch_stub.bfloat16 = _DType("bfloat16")
    torch_stub.float32 = _DType("float32")

    triton_stub = types.ModuleType("triton")
    triton_stub.__path__ = []
    triton_stub.jit = lambda function: function
    triton_stub.cdiv = lambda value, divisor: (
        value + divisor - 1
    ) // divisor
    language_stub = types.ModuleType("triton.language")
    language_stub.constexpr = object()
    triton_stub.language = language_stub

    prior = {
        name: sys.modules.get(name)
        for name in ("torch", "triton", "triton.language")
    }
    try:
        sys.modules["torch"] = torch_stub
        sys.modules["triton"] = triton_stub
        sys.modules["triton.language"] = language_stub
        spec = importlib.util.spec_from_file_location(
            "matmul_persistent_source_for_cpu_test", SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in prior.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, torch_stub


def test_matmul_scoring_kernel_and_shared_helper_are_byte_frozen():
    runner_source, runner_tree = _parse(RUNNER)
    source_source, source_tree = _parse(SOURCE)
    assignments = _assignments(runner_tree)
    runner_functions = _functions(runner_tree)
    source_functions = _functions(source_tree)

    assert ast.literal_eval(assignments["TEST_SHAPES"]) == (
        EXPECTED_PERFORMANCE_SHAPES
    )
    assert ast.literal_eval(assignments["WARMUP_ITERATIONS"]) == 10
    assert ast.literal_eval(assignments["BENCHMARK_ITERATIONS"]) == 100

    performance = ast.get_source_segment(
        runner_source, runner_functions["run_performance"]
    )
    assert performance is not None
    assert hashlib.sha256(performance.encode()).hexdigest() == (
        EXPECTED_PERFORMANCE_SHA256
    )
    assert "CORRECTNESS_CASES" not in _referenced_names(
        runner_functions["run_performance"]
    )
    assert "VIRTUAL_INDEX_CASES" not in _referenced_names(
        runner_functions["run_performance"]
    )

    helper_start = runner_source.index("# >>> AKA-GENERATED:")
    helper_end = runner_source.index("# <<< AKA-GENERATED <<<") + len(
        "# <<< AKA-GENERATED <<<"
    )
    helper = runner_source[helper_start:helper_end]
    assert hashlib.sha256(helper.encode()).hexdigest() == (
        EXPECTED_PERFORMANCE_HELPER_SHA256
    )

    for function_name, expected_digest in (
        ("_compute_pid", EXPECTED_COMPUTE_PID_SHA256),
        ("matmul_kernel_persistent", EXPECTED_KERNEL_SHA256),
    ):
        function_source = ast.get_source_segment(
            source_source, source_functions[function_name]
        )
        assert function_source is not None
        assert hashlib.sha256(function_source.encode()).hexdigest() == (
            expected_digest
        )

    benchmark_calls = [
        node
        for node in ast.walk(runner_functions["run_performance"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_benchmark_cuda_graph_or_events"
    ]
    assert len(benchmark_calls) == 1
    keywords = {
        keyword.arg: keyword.value
        for keyword in benchmark_calls[0].keywords
    }
    assert ast.unparse(keywords["warmup"]) == "WARMUP_ITERATIONS"
    assert ast.unparse(keywords["repetition"]) == "BENCHMARK_ITERATIONS"
    assert runner_source.count('f"perf{test_idx + 1}"') == 2


def test_matmul_bounded_cases_cover_dtype_stride_bias_and_persistence():
    runner = _load_runner_module()
    cases = runner.CORRECTNESS_CASES

    assert {case["dtype"] for case in cases} == {
        "float16",
        "bfloat16",
    }
    _, runner_tree = _parse(RUNNER)
    estimator = ast.unparse(_functions(runner_tree)["_estimate_case_storage_bytes"])
    case_builder = ast.unparse(_functions(runner_tree)["_make_contract_case_inputs"])
    assert "'float32': 4" not in estimator
    assert "dtype == torch.float32" not in case_builder
    matrix_layouts = {
        case[key]
        for case in cases
        for key in ("a_layout", "b_layout")
    }
    assert matrix_layouts == {
        "row_padded",
        "inner_strided",
        "transposed",
    }
    assert {case["bias_layout"] for case in cases} == {
        None,
        "contiguous",
        "inner_strided",
    }
    assert all(case["M"] > 0 and case["N"] > 0 and case["K"] > 0 for case in cases)
    assert all(case["padding"] > 0 for case in cases)
    assert all(case["requires_partial_group"] for case in cases)

    blocks = {
        "float16": (128, 256),
        "bfloat16": (128, 128),
    }
    for case in cases:
        block_m, block_n = blocks[case["dtype"]]
        num_pid_m = runner._ceil_div(case["M"], block_m)
        num_pid_n = runner._ceil_div(case["N"], block_n)
        assert num_pid_m % 8 != 0
        assert runner._validate_grouped_schedule(
            num_pid_m, num_pid_n, 8
        ) is None
        estimated = runner._estimate_case_storage_bytes(case)
        assert estimated <= runner.MAX_CORRECTNESS_STORAGE_BYTES
        assert estimated < 2 * 1024**3

    persistent = next(
        case for case in cases if case["requires_persistent"]
    )
    block_m, block_n = blocks[persistent["dtype"]]
    num_tiles = runner._ceil_div(
        persistent["M"], block_m
    ) * runner._ceil_div(persistent["N"], block_n)
    assert num_tiles == 1105
    assert num_tiles > 304

    virtual = {case["name"]: case for case in runner.VIRTUAL_INDEX_CASES}
    assert virtual["a_small_view_crosses_int32"]["a_shape"] == (2, 3)
    assert virtual["b_small_view_crosses_int32"]["b_shape"] == (3, 2)
    assert virtual["signed_int32_boundary_is_safe"]["expected"] == (
        False,
        False,
        False,
    )
    c_case = virtual["contiguous_output_crosses_int32"]
    c_elements = c_case["a_shape"][0] * c_case["b_shape"][1]
    assert c_elements > 2**31
    assert all(
        runner._estimate_case_storage_bytes(case) < 2 * 1024**3
        for case in cases
    )


def test_real_wrapper_virtual_index_proof_is_allocation_free_and_flag_exact():
    runner = _load_runner_module()
    source_module, torch_stub = _load_source_with_cpu_stubs()

    receipts, error = runner._run_virtual_large_index_proof(
        source_module, torch_stub, lambda: None
    )
    assert error is None
    assert len(receipts) == len(runner.VIRTUAL_INDEX_CASES) == 4
    by_name = {receipt["name"]: receipt for receipt in receipts}
    assert by_name["a_small_view_crosses_int32"]["flags"] == (
        True,
        False,
        False,
    )
    assert by_name["b_small_view_crosses_int32"]["flags"] == (
        False,
        True,
        False,
    )
    assert by_name["signed_int32_boundary_is_safe"]["a_max_offset"] == (
        2**31 - 1
    )
    assert by_name["signed_int32_boundary_is_safe"]["flags"] == (
        False,
        False,
        False,
    )
    assert by_name["contiguous_output_crosses_int32"]["c_max_offset"] > (
        2**31 - 1
    )
    assert by_name["contiguous_output_crosses_int32"]["flags"] == (
        False,
        False,
        True,
    )
    assert all(receipt["allocated_bytes"] == 0 for receipt in receipts)
    assert all(len(receipt["grid"]) == 1 for receipt in receipts)

    assert source_module._max_relative_element_offset(
        (2, 2), (2**31 - 2, 1)
    ) == 2**31 - 1
    assert source_module._max_relative_element_offset(
        (2, 2), (2**31 - 1, 1)
    ) == 2**31


def test_real_wrapper_materializes_bias_and_rejects_invalid_metadata():
    runner = _load_runner_module()
    source_module, torch_stub = _load_source_with_cpu_stubs()
    virtual_torch = runner._VirtualTorch(torch_stub)
    virtual_kernel = runner._VirtualKernel()
    source_module.torch = virtual_torch
    source_module.matmul_kernel_persistent = virtual_kernel

    a = runner._VirtualTensor(
        (2, 3), (5, 1), torch_stub.float16
    )
    b = runner._VirtualTensor(
        (3, 5), (7, 1), torch_stub.float16
    )
    bias = runner._VirtualTensor(
        (5,), (2,), torch_stub.float16
    )
    result = source_module.matmul_persistent(a, b, bias)
    assert tuple(result.shape) == (2, 5)
    assert len(virtual_kernel.launches) == 1
    launch = virtual_kernel.launches[0]
    launched_bias = launch["args"][3]
    assert launched_bias is not bias
    assert launched_bias.stride() == (1,)
    assert launched_bias.is_contiguous()
    assert launch["kwargs"]["HAS_BIAS"] is True
    assert tuple(launch["args"][7:11]) == (5, 1, 7, 1)

    def rejected(call):
        launch_count = len(virtual_kernel.launches)
        try:
            call()
        except (AssertionError, ValueError):
            assert len(virtual_kernel.launches) == launch_count
            return
        raise AssertionError("invalid wrapper metadata was accepted")

    rejected(
        lambda: source_module.matmul_persistent(
            runner._VirtualTensor((3,), (1,), torch_stub.float16), b
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            runner._VirtualTensor((2, 4), (4, 1), torch_stub.float16), b
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            a,
            runner._VirtualTensor((3, 5), (5, 1), torch_stub.float32),
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            a,
            runner._VirtualTensor(
                (3, 5), (5, 1), torch_stub.float16, "cuda:1"
            ),
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            runner._VirtualTensor((2, 3), (0, 1), torch_stub.float16),
            b,
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            a,
            b,
            runner._VirtualTensor((4,), (1,), torch_stub.float16),
        )
    )
    rejected(
        lambda: source_module.matmul_persistent(
            a,
            b,
            runner._VirtualTensor((5,), (0,), torch_stub.float16),
        )
    )
    non_cuda = runner._VirtualTensor(
        (2, 3), (3, 1), torch_stub.float16
    )
    non_cuda.is_cuda = False
    rejected(lambda: source_module.matmul_persistent(non_cuda, b))
    unsupported = _DType("float64")
    rejected(
        lambda: source_module.matmul_persistent(
            runner._VirtualTensor((2, 3), (3, 1), unsupported),
            runner._VirtualTensor((3, 5), (5, 1), unsupported),
        )
    )


class _MetaTensor:
    def __init__(
        self,
        shape,
        strides,
        *,
        dtype="float16",
        device="cuda:0",
        payload="values",
    ):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self.dtype = dtype
        self.device = device
        self.payload = payload

    def stride(self, dimension=None):
        if dimension is None:
            return self._strides
        return self._strides[dimension]

    def is_contiguous(self):
        stride = 1
        expected = []
        for size in reversed(self.shape):
            expected.append(stride)
            stride *= size
        return self._strides == tuple(reversed(expected))

    def clone(self):
        return _MetaTensor(
            self.shape,
            self._strides,
            dtype=self.dtype,
            device=self.device,
            payload=self.payload,
        )


class _EqualityTorch:
    @staticmethod
    def equal(left, right):
        return (
            tuple(left.shape) == tuple(right.shape)
            and left.dtype == right.dtype
            and left.device == right.device
            and left.payload == right.payload
        )


def _valid_fake_launch(runner):
    M, N, K = 131, 259, 67
    a = _MetaTensor((M, K), (K + 5, 1), payload="a")
    b = _MetaTensor((K, N), (1, K + 7), payload="b")
    result = _MetaTensor((M, N), (N, 1), payload="result")
    bias = _MetaTensor((N,), (2,), payload="bias")
    launched_bias = _MetaTensor((N,), (1,), payload="bias")
    args = (
        a,
        b,
        result,
        launched_bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        result.stride(0),
        result.stride(1),
    )
    launch = {
        "grid": (4,),
        "args": args,
        "kwargs": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "NUM_SMS": 304,
            "A_LARGE": False,
            "B_LARGE": False,
            "C_LARGE": False,
            "HAS_BIAS": True,
        },
        "output_after_launch": result.clone(),
    }
    return a, b, bias, result, launch


def _validate_fake_batch(runner, launches, a, b, bias, result):
    return runner._validate_matmul_launch_batch(
        _EqualityTorch,
        launches,
        a,
        b,
        bias,
        result,
        304,
        "fake",
        1,
        requires_partial_group=True,
        requires_persistent=False,
    )


def test_launch_attestation_rejects_extra_wrong_and_dummy_launches():
    runner = _load_runner_module()
    a, b, bias, result, valid = _valid_fake_launch(runner)

    assert _validate_fake_batch(
        runner, [valid], a, b, bias, result
    ) is None
    assert "expected one target-kernel launch" in _validate_fake_batch(
        runner, [], a, b, bias, result
    )
    assert "expected one target-kernel launch" in _validate_fake_batch(
        runner, [valid, valid], a, b, bias, result
    )

    hidden_result = _MetaTensor(result.shape, result.stride(), payload="hidden")
    extra_positional = {
        **valid,
        "args": (*valid["args"], hidden_result),
    }
    assert "expected exactly 13" in _validate_fake_batch(
        runner, [extra_positional], a, b, bias, result
    )
    extra_keyword = {
        **valid,
        "kwargs": {**valid["kwargs"], "precomputed_result": hidden_result},
    }
    assert "unexpected keyword operands" in _validate_fake_batch(
        runner, [extra_keyword], a, b, bias, result
    )
    valid_controls = {
        **valid,
        "kwargs": {
            **valid["kwargs"],
            "num_warps": 8,
            "num_stages": 3,
            "waves_per_eu": 0,
        },
    }
    assert _validate_fake_batch(
        runner, [valid_controls], a, b, bias, result
    ) is None
    tensor_control = {
        **valid,
        "kwargs": {**valid["kwargs"], "num_warps": hidden_result},
    }
    assert "num_warps must be a positive integer" in _validate_fake_batch(
        runner, [tensor_control], a, b, bias, result
    )

    wrong_grid = {**valid, "grid": (3,)}
    assert "expected resolved persistent grid" in _validate_fake_batch(
        runner, [wrong_grid], a, b, bias, result
    )
    wrong_flag = {
        **valid,
        "kwargs": {**valid["kwargs"], "A_LARGE": True},
    }
    assert "A_LARGE=True" in _validate_fake_batch(
        runner, [wrong_flag], a, b, bias, result
    )
    wrong_c = _MetaTensor(result.shape, result.stride(), payload="other")
    wrong_target = {
        **valid,
        "args": valid["args"][:2] + (wrong_c,) + valid["args"][3:],
    }
    assert "not the target kernel's C argument" in _validate_fake_batch(
        runner, [wrong_target], a, b, bias, result
    )
    dummy = {
        **valid,
        "output_after_launch": _MetaTensor(
            result.shape, result.stride(), payload="poison"
        ),
    }
    assert "modified after the target launch" in _validate_fake_batch(
        runner, [dummy], a, b, bias, result
    )
    missing_capture = dict(valid)
    missing_capture.pop("output_after_launch")
    assert "output capture is missing" in _validate_fake_batch(
        runner, [missing_capture], a, b, bias, result
    )
    nonunit_bias = _MetaTensor(bias.shape, (2,), payload="bias")
    wrong_bias = {
        **valid,
        "args": valid["args"][:3]
        + (nonunit_bias,)
        + valid["args"][4:],
    }
    assert "materialized unit-stride bias" in _validate_fake_batch(
        runner, [wrong_bias], a, b, bias, result
    )


def test_output_capture_poisons_c_before_target_and_catches_dummy_kernel():
    runner = _load_runner_module()

    class CaptureTensor:
        def __init__(self, value=0.0):
            self.value = value

        def fill_(self, value):
            self.value = value
            return self

        def clone(self):
            return CaptureTensor(self.value)

    class FakeCuda:
        synchronizations = 0

        @classmethod
        def synchronize(cls):
            cls.synchronizations += 1

    class FakeTorch:
        cuda = FakeCuda

    class DummyKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                return None

            return launch

    c = CaptureTensor(123.0)
    recorder = runner._KernelLaunchRecorder(
        DummyKernel(), torch=FakeTorch, capture_output=True
    )
    recorder[(1,)]("a", "b", c)
    assert len(recorder.launches) == 1
    captured = recorder.launches[0]["output_after_launch"]
    assert math.isnan(captured.value)
    assert FakeCuda.synchronizations == 1
    c.value = 42.0
    assert captured.value != c.value


def test_checker_attests_reference_order_storage_aliasing_and_determinism():
    _, tree = _parse(RUNNER)
    functions = _functions(tree)
    classes = _classes(tree)
    checker = ast.unparse(functions["_check_matmul_case"])
    launch_validator = ast.unparse(functions["_validate_matmul_launch"])
    output_validator = ast.unparse(functions["_validate_matmul_output"])
    recorder = ast.unparse(classes["_KernelLaunchRecorder"])

    for phrase in (
        "torch.mm(protected_inputs[0][2].float(), protected_inputs[1][2].float())",
        "ref = ref + protected_inputs[2][2].float()",
        "ref = ref.to(a.dtype)",
        "for _invocation in range(2)",
        "frozen_backing",
        "protected_storage_ptrs",
        "repeated calls did not return fresh storage",
        "torch.equal(results[0], results[1])",
        "candidate mutated protected input",
        "candidate mutated backing allocation",
        "capture_output=True",
    ):
        assert phrase in checker

    for phrase in (
        "args[0] is not a or args[1] is not b",
        "args[2] is not result",
        "output_after_launch",
        "expected resolved persistent grid",
        "GROUP_SIZE_M",
        "_validate_grouped_schedule",
        "tuple(args[7:13])",
        "A_LARGE",
        "B_LARGE",
        "C_LARGE",
        "HAS_BIAS",
        "materialized unit-stride bias",
    ):
        assert phrase in launch_validator

    for phrase in (
        "result.is_contiguous()",
        "result.untyped_storage().data_ptr()",
        "output aliases an input or backing allocation",
        "torch.isfinite(result)",
        "atol=MATMUL_ATOL",
        "rtol=MATMUL_RTOL",
    ):
        assert phrase in output_validator

    assert "args[2].fill_(float('nan'))" in recorder
    assert "record['output_after_launch'] = args[2].clone()" in recorder
    assert "self.kernel.run(*args, **kwargs)" in recorder
    assert "getattr(self.kernel, name)" in recorder
    assert any(
        isinstance(node, ast.Try) and node.finalbody
        for node in ast.walk(functions["_check_matmul_case"])
    )


def test_source_wrapper_and_kernel_have_explicit_index_and_bias_contracts():
    source, tree = _parse(SOURCE)
    functions = _functions(tree)
    wrapper = ast.unparse(functions["matmul_persistent"])
    kernel = ast.unparse(functions["matmul_kernel_persistent"])
    offset_helper = ast.unparse(functions["_max_relative_element_offset"])

    for phrase in (
        "a.dim() == 2 and b.dim() == 2",
        "M > 0 and N > 0 and (K > 0)",
        "a.dtype == b.dtype",
        "a.device == b.device",
        "a.is_cuda and b.is_cuda",
        "all((stride > 0 for stride in a.stride()))",
        "all((stride > 0 for stride in b.stride()))",
        "bias.shape[0] == N",
        "bias.dtype == a.dtype",
        "bias.device == a.device",
        "bias.is_cuda",
        "bias.stride(0) > 0",
        "bias = bias.contiguous()",
        "get_device_properties(a.device)",
        "A_LARGE=_requires_int64_index(a)",
        "B_LARGE=_requires_int64_index(b)",
        "C_LARGE=_requires_int64_index(c)",
    ):
        assert phrase in wrapper

    assert "tensor.numel" not in wrapper
    assert "zip(shape, strides)" in offset_helper
    assert "(int(size) - 1) * int(stride)" in offset_helper
    for phrase in (
        "if A_LARGE:",
        "offs_am = offs_am.to(tl.int64)",
        "if B_LARGE:",
        "offs_bn = offs_bn.to(tl.int64)",
        "if A_LARGE or B_LARGE:",
        "tl.arange(0, BLOCK_SIZE_K).to(tl.int64)",
        "if C_LARGE:",
        "offs_cm = offs_cm.to(tl.int64)",
        "offs_cn = offs_cn.to(tl.int64)",
    ):
        assert phrase in kernel

    original_kernel = ast.get_source_segment(
        source, functions["matmul_kernel_persistent"]
    )
    assert hashlib.sha256(original_kernel.encode()).hexdigest() == (
        EXPECTED_KERNEL_SHA256
    )


def test_config_readme_and_generated_artifacts_match_public_contract():
    config = CONFIG.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    combined = config + "\n" + readme

    for phrase in (
        "float16 and bfloat16",
        "Float32 is rejected",
        "positive non-unit stride",
        "maximum relative element offset",
        "fresh contiguous deterministic",
        "one-dimensional persistent grid",
        "grouped tile",
        "backing storage",
        "virtual",
        "2 GiB",
        "run_performance",
        "hidden tensor or object",
    ):
        assert phrase in combined
    assert "allocation-free virtual tensors" in readme
    assert "source audit" in readme
    assert "CPU structural tests" in readme
    assert "matched\nperformance" in readme
    assert not list((TASK_ROOT / "build").glob("*.json"))

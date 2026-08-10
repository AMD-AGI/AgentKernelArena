import ast
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks/triton2triton/vllm"

RUNNERS = {
    "batched": TASK_ROOT / "triton_batched_moe/scripts/task_runner.py",
    "fused": TASK_ROOT / "triton_fused_moe/scripts/task_runner.py",
    "layernorm": TASK_ROOT / "triton_layernorm_gated/scripts/task_runner.py",
}

EXPECTED_PERFORMANCE_SHAPES = {
    "batched": [
        (4, 16, 64, 64),
        (8, 32, 128, 128),
        (8, 64, 256, 256),
        (16, 64, 512, 512),
        (8, 128, 1024, 512),
    ],
    "fused": [
        (16, 64, 4, 64, 2),
        (32, 128, 8, 128, 2),
        (64, 256, 8, 256, 2),
        (128, 512, 16, 512, 2),
        (256, 1024, 8, 1024, 2),
    ],
    "layernorm": [
        (32, 128, False, True, False),
        (64, 256, True, False, False),
        (128, 512, True, True, True),
        (256, 1024, False, True, True),
        (512, 2048, True, False, True),
    ],
}

EXPECTED_PERFORMANCE_SHA256 = {
    "batched": "91ef2484a036c60d95d7fda96150240b2ccccfed2f9fbac49f34846f1ab557c2",
    "fused": "85684837a07e78dfdedba79a29a3cf541510b8d7d4e24aec5b40e6c21881c26a",
    "layernorm": "025f40a4bf0df265aa8cdb75380b539088c2509feb6507d5f3c34f2f5a4ee23f",
}


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _assignments(tree):
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            values[target.id] = node.value
    return values


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


def test_performance_contract_literals_and_ids_remain_unchanged():
    for task, runner in RUNNERS.items():
        tree = _parse(runner)
        assignments = _assignments(tree)
        functions = _functions(tree)
        assert ast.literal_eval(assignments["TEST_SHAPES"]) == (
            EXPECTED_PERFORMANCE_SHAPES[task]
        )
        assert ast.literal_eval(assignments["WARMUP_ITERATIONS"]) == 10
        assert ast.literal_eval(assignments["BENCHMARK_ITERATIONS"]) == 100

        performance = functions["run_performance"]
        source = runner.read_text(encoding="utf-8")
        performance_source = ast.get_source_segment(source, performance)
        assert performance_source is not None
        assert hashlib.sha256(performance_source.encode()).hexdigest() == (
            EXPECTED_PERFORMANCE_SHA256[task]
        )
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
            and node.values
            and isinstance(node.values[0], ast.Constant)
            and node.values[0].value == "perf"
        ]
        assert len(perf_ids) == 2
        for perf_id in perf_ids:
            assert ast.unparse(perf_id) == "f'perf{test_idx + 1}'"


def test_batched_moe_adds_zero_nonmultiple_and_noncompact_correctness_only():
    tree = _parse(RUNNERS["batched"])
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["BATCHED_MOE_CORRECTNESS_CASES"])

    assert [case["shape"] for case in cases] == [
        (3, 7, 33, 35),
        (3, 17, 70, 73),
    ]
    assert any(all(count == 0 for count in case["expert_num_tokens"]) for case in cases)
    assert any(0 in case["expert_num_tokens"] and max(case["expert_num_tokens"]) > 0 for case in cases)
    assert any(
        case["a_last_dim_padding"] > 0 and case["b_last_dim_padding"] > 0
        for case in cases
    )
    assert "BATCHED_MOE_CORRECTNESS_CASES" in _referenced_names(
        functions["run_correctness"]
    )
    assert "BATCHED_MOE_CORRECTNESS_CASES" not in _referenced_names(
        functions["run_performance"]
    )
    case_checker = ast.unparse(functions["_check_batched_moe_case"])
    assert "result.dtype != A.dtype" in case_checker
    assert "result.untyped_storage().data_ptr()" in case_checker
    assert "candidate wrote outside logical" in case_checker


def test_fused_moe_valid_domain_branches_zero_m_and_resource_gate_are_non_scoring():
    tree = _parse(RUNNERS["fused"])
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["FUSED_MOE_VALID_ID_CASES"])
    resource = ast.literal_eval(assignments["FUSED_MOE_RESOURCE_CASE"])
    peak_expression = assignments["FUSED_MOE_MAX_EXTRA_PEAK_BYTES"]
    peak_limit = ast.literal_eval(peak_expression)

    assert any(case["M"] == 0 for case in cases)
    assert {case["mul_routed_weight"] for case in cases} == {False, True}
    assert {case["provide_weights"] for case in cases} == {False, True}
    for case in cases:
        flattened = [expert for row in case["topk_ids"] for expert in row]
        assert all(0 <= expert < case["E"] for expert in flattened)
    assert resource == {"M": 8192, "K": 16, "E": 128, "N": 16, "topk": 8}
    rejected_scratch_bytes = (
        resource["E"]
        * 2 ** ((resource["M"] * resource["topk"] - 1).bit_length())
        * 8
    )
    assert rejected_scratch_bytes == 64 * 1024 * 1024
    assert peak_limit == 20 * 1024 * 1024
    assert rejected_scratch_bytes > peak_limit
    assert peak_limit < 2 * 1024 * 1024 * 1024

    resource_gate = functions["_run_fused_moe_resource_gate"]
    nested_cuda_calls = {
        node.func.attr
        for node in ast.walk(resource_gate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and node.func.value.attr == "cuda"
    }
    assert {"reset_peak_memory_stats", "max_memory_allocated"} <= nested_cuda_calls
    resource_gate_source = ast.unparse(resource_gate)
    assert "fixture_allocated" in resource_gate_source
    assert "retained_after_warmup" in resource_gate_source
    assert "_run_fused_moe_resource_gate" in {
        node.func.id
        for node in ast.walk(functions["run_correctness"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "FUSED_MOE_RESOURCE_CASE" not in _referenced_names(
        functions["run_performance"]
    )


def test_layernorm_covers_grouping_gate_order_out_stats_strides_and_feature_limit():
    tree = _parse(RUNNERS["layernorm"])
    assignments = _assignments(tree)
    functions = _functions(tree)
    cases = ast.literal_eval(assignments["LAYERNORM_GROUPED_CORRECTNESS_CASES"])
    feature_case = ast.literal_eval(assignments["LAYERNORM_FEATURE_LIMIT_CASE"])

    assert {case["is_rms"] for case in cases} == {False, True}
    assert {case["norm_before_gate"] for case in cases} == {False, True}
    assert all(case["N"] % case["group_size"] == 0 for case in cases)
    assert all(case["group_size"] < case["N"] for case in cases)
    assert all(case["row_padding"] > 0 for case in cases)
    assert all(case["provide_out"] for case in cases)
    assert feature_case == {"M": 1, "N": 32769, "group_size": 32769}

    case_runner = functions["_run_layernorm_case"]
    layernorm_calls = [
        node
        for node in ast.walk(case_runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "layer_norm_fwd"
    ]
    assert len(layernorm_calls) == 1
    call_keywords = {keyword.arg for keyword in layernorm_calls[0].keywords}
    assert {"out", "group_size", "norm_before_gate", "is_rms_norm"} <= call_keywords
    assert {"mean", "rstd"} <= _referenced_names(case_runner)
    assert any(
        isinstance(node, ast.IsNot)
        for node in ast.walk(case_runner)
    )
    assert "LAYERNORM_PADDING_CANARY" in _referenced_names(case_runner)
    case_runner_source = ast.unparse(case_runner)
    assert "protected_storage_ptrs" in case_runner_source
    assert "statistic_storage_ptrs" in case_runner_source
    assert "statistics alias each other" in case_runner_source

    feature_guard = functions["_check_feature_limit"]
    assert any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "RuntimeError"
        for node in ast.walk(feature_guard)
    )
    assert "LAYERNORM_GROUPED_CORRECTNESS_CASES" not in _referenced_names(
        functions["run_performance"]
    )


def test_task_configs_state_the_public_contract_boundaries():
    batched_config = (TASK_ROOT / "triton_batched_moe/config.yaml").read_text()
    fused_config = (TASK_ROOT / "triton_fused_moe/config.yaml").read_text()
    layernorm_config = (TASK_ROOT / "triton_layernorm_gated/config.yaml").read_text()

    for phrase in ("zero-token experts", "noncompact logical views", "scoring are unchanged"):
        assert phrase in batched_config
    for phrase in (
        "`[0, E)`",
        "Invalid expert IDs are outside this task contract",
        "`M=0`",
        "no 2 GiB address probe",
    ):
        assert phrase in fused_config
    for phrase in (
        "grouped LayerNorm",
        "exact `out` object",
        "mean/rstd",
        "controlled `RuntimeError`",
        "no 2 GiB address probe",
    ):
        assert phrase in layernorm_config

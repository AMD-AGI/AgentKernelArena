"""Production-size variants keep their original task and replace only what their README says they replace."""
import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from src.task_spec import load_task_spec, required_gpu_arches

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
HIP_VARIANTS = {
    "hip2hip/SimpleMatmulModule_large": "hip2hip/SimpleMatmulModule",
    "hip2hip/SoftmaxModule_large": "hip2hip/SoftmaxModule",
}
VLLM_VARIANTS = {
    "triton2triton/vllm_large/triton_awq_gemm": "triton2triton/vllm/triton_awq_gemm",
    "triton2triton/vllm_large/triton_w8a8_block_scaled_mm": "triton2triton/vllm/triton_w8a8_block_scaled_mm",
}
VARIANTS = {**HIP_VARIANTS, **VLLM_VARIANTS}
CONFIGS_BLOCK = re.compile(r"    configs = \[\n.*?\n    \]\n", re.DOTALL)
CASE_TABLE = re.compile(r"# Test configs:[^\n]*\nTEST_SHAPES = \[\n(?:    [^\n]*\n)+\]\n")
REFERENCES = {"triton_awq_gemm": "reference_awq_gemm",
              "triton_w8a8_block_scaled_mm": "reference_w8a8_block_scaled_mm"}
AWQ_RANDOM_WEIGHTS = (
    "            qweight = torch.randint(0, 2**31, (K, N_packed), device=device, dtype=torch.int32)\n"
    "            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01\n"
    "            qzeros = torch.randint(0, 2**31, (num_groups, N_packed), device=device, dtype=torch.int32)\n")
AWQ_QUANTIZED_WEIGHTS = "            qweight, scales, qzeros = make_awq_weights(K, N, group_size, device)\n"
AWQ_GATE = "    torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-1)\n"
AWQ_NORMWISE_GATE = AWQ_GATE + (
    "    error = (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)\n"
    "    if error > 5e-3:\n"
    "        raise AssertionError(f'AWQ GEMM normwise relative error {error.item():.2e} exceeds 5e-3')\n")


def files(task: str) -> set[str]:
    root = TASKS / task
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def text(task: str, relative: str) -> str:
    return (TASKS / task / relative).read_text()


def changed(variant: str, original: str) -> set[str]:
    return {relative for relative in files(variant)
            if (TASKS / variant / relative).read_bytes() != (TASKS / original / relative).read_bytes()}


def literal_rows(block: str) -> list:
    return [ast.literal_eval(line.strip().rstrip(",")) for line in block.splitlines()[1:-1]]


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_keeps_its_originals_files_and_config_and_runs_only_on_gfx942(variant):
    original = VARIANTS[variant]
    assert files(variant) == files(original)
    spec = load_task_spec(TASKS / variant / "config.yaml", task_id=variant).to_mapping()
    assert required_gpu_arches(spec.get("platform_support")) == ("gfx942",)
    config = yaml.safe_load(text(variant, "config.yaml"))
    assert config.pop("platform_support") == {"required_arch": "gfx942"}
    assert config == yaml.safe_load(text(original, "config.yaml"))
    readme = " ".join(text(variant, "README.md").split())
    kind, language = ("production-size", "HIP") if variant in HIP_VARIANTS else ("LLM-shape", "Triton")
    assert f"This is the {kind} variant of `{original}`." in readme
    assert f"Compute the operator in {language} kernels the candidate defines and launches." in readme
    assert "The task is validated on gfx942 only." in readme


@pytest.mark.parametrize("variant", HIP_VARIANTS)
def test_hip_variant_replaces_only_the_case_list(variant):
    original = HIP_VARIANTS[variant]
    modules = {relative for relative in files(variant)
               if relative.startswith(("pytorch_code_module/", "pytorch_code_functional/"))}
    assert changed(variant, original) == {"README.md", "config.yaml", "workload.json", *modules}
    cases = []
    for relative in sorted(modules):
        source = text(variant, relative)
        assert CONFIGS_BLOCK.sub("", source) == CONFIGS_BLOCK.sub("", text(original, relative))
        cases.append(literal_rows(CONFIGS_BLOCK.search(source).group(0)))
    assert all(configs == cases[0] for configs in cases)
    manifest, before = (json.loads(text(task, "workload.json")) for task in (variant, original))
    assert {k: v for k, v in manifest.items() if k != "cases"} == {k: v for k, v in before.items() if k != "cases"}
    assert [[entry["shape"] for entry in case["params"]["inputs"]] for case in manifest["cases"]] == \
        [[list(shape) for shape in case] for case in cases[0]]
    dtypes = {entry["dtype"] for case in manifest["cases"] for entry in case["params"]["inputs"]}
    assert dtypes == {entry["dtype"] for case in before["cases"] for entry in case["params"]["inputs"]}
    assert [(case["params"]["model_init_seed"], case["params"]["correctness_seed"]) for case in manifest["cases"]] \
        == [(0, 1337 + index) for index in range(len(manifest["cases"]))]


def without_cases_and_reference(source: str, reference: str) -> str:
    source, tables = CASE_TABLE.subn("", source, count=1)
    source, references = re.subn(rf"^def {reference}\(.*?(?=^def run_compile)", "", source, flags=re.S | re.M)
    assert tables == references == 1
    return source


@pytest.mark.parametrize("variant", VLLM_VARIANTS)
def test_vllm_variant_replaces_cases_references_and_only_its_stated_gates(variant):
    original = VLLM_VARIANTS[variant]
    name = variant.rsplit("/", 1)[1]
    awq = name == "triton_awq_gemm"
    stated = {"README.md", "config.yaml", "workloads.json", "scripts/task_runner.py"}
    assert changed(variant, original) == stated | ({"_arena_checks.py"} if awq else set())
    runner = text(variant, "scripts/task_runner.py")
    before = without_cases_and_reference(text(original, "scripts/task_runner.py"), REFERENCES[name])
    if awq:
        assert before.count(AWQ_RANDOM_WEIGHTS) == 2
        before = before.replace(AWQ_RANDOM_WEIGHTS, AWQ_QUANTIZED_WEIGHTS)
        checks = text(original, "_arena_checks.py")
        assert checks.count("scales.mul_(.5).add_(.015625)") == checks.count(AWQ_GATE) == 1
        assert text(variant, "_arena_checks.py") == checks.replace(
            "scales.mul_(.5).add_(.015625)", "scales.mul_(.75)").replace(AWQ_GATE, AWQ_NORMWISE_GATE)
    assert without_cases_and_reference(runner, REFERENCES[name]) == before
    table = literal_rows(CASE_TABLE.search(runner).group(0).split("\n", 1)[1])
    manifest, prior = (json.loads(text(task, "workloads.json")) for task in (variant, original))
    kept = {k: v for k, v in prior.items() if k not in ("cases", "input_table", "migration")}
    assert {k: v for k, v in manifest.items() if k not in ("cases", "input_table")} == kept
    assert manifest["input_table"] == [list(row) for row in table]
    template = prior["cases"][0]
    for index, case in enumerate(manifest["cases"]):
        assert case["params"]["configuration"] == list(table[index])
        assert case["dtype"] == template["dtype"] and case["checks"] == template["checks"]

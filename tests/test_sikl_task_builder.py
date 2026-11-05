"""SIKL import fidelity, schema-v3 emission and QA acceptance gates; no model or GPU needed."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from agents.sikl_task_builder.bundle import ImportProblem, inspect_bundle
from agents.sikl_task_builder.config import Config
from agents.sikl_task_builder.execution import run_process
from agents.sikl_task_builder.materialize import (
    authoring_editable, boundary_kinds, check_contract, materialize_task, task_digest, task_files,
)
from agents.sikl_task_builder.orchestrator import install_task, run
from agents.sikl_task_builder import validation
from src.task_contract import execution_config, load_task_contract, render_workspace

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "SIKL-task/tiny_gemm"
SIZES = (1, 2, 16, 64, 256, 1024)
BASELINE_MAIN = "import triton\nfrom .helper import matmul\n\n\ndef run(a, b):\n    return matmul(a, b)\n"
BASELINE_HELPER = "def matmul(a, b):\n    return a @ b.T\n"
REFERENCE = "import torch\n\n\ndef run(a, b):\n    return a.double() @ b.double().T\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(bundle, rows):
    (bundle / "workloads/gemm.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def rows_for(sizes):
    return [{"definition": "tiny_gemm", "workload": {
        "axes": {"m": m}, "uuid": f"case-{m}",
        "inputs": {"a": {"type": "random"}, "b": {"type": "random"}}},
        "solution": None, "evaluation": None} for m in sizes]


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle with spaces"
    definition = {
        "name": "tiny_gemm", "op_type": "gemm", "description": "out = a @ b^T.",
        "axes": {"m": {"type": "var"}, "n": {"type": "const", "value": 8},
                 "k": {"type": "const", "value": 16}},
        "inputs": {"a": {"shape": ["m", "k"], "dtype": "float32"},
                   "b": {"shape": ["n", "k"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["m", "n"], "dtype": "float32"}},
        "constraints": ["m <= 4096"],
    }
    write_json(root / "definitions/gemm.json", definition)
    (root / "workloads").mkdir(parents=True)
    write_rows(root, rows_for(SIZES))
    target = {"language": "triton", "destination_passing_style": False,
              "entry_point": "main.py::run", "target_hardware": ["MI355X"]}
    write_json(root / "solutions/baseline/gemm.json", {
        "name": "baseline", "definition": "tiny_gemm", "license": "Apache-2.0", "spec": target,
        "sources": [{"path": "main.py", "content": BASELINE_MAIN},
                    {"path": "helper.py", "content": BASELINE_HELPER}]})
    write_json(root / "solutions/reference/gemm.json", {
        "name": "reference", "definition": "tiny_gemm", "spec": {**target, "language": "python"},
        "sources": [{"path": "main.py", "content": REFERENCE}]})
    return root


def draft_dir(tmp_path, name="tiny_gemm"):
    return tmp_path / "drafts" / name


@pytest.fixture
def emitted(bundle, tmp_path):
    task = inspect_bundle(bundle)[0]
    cfg = Config(str(bundle))
    draft = draft_dir(tmp_path)
    materialize_task(task, cfg, draft)
    return task, cfg, draft


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def edit_yaml(path, change):
    value = yaml.safe_load(path.read_text())
    change(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def run_script(script, cwd):
    return subprocess.run([sys.executable, "-c", script], cwd=cwd,
                          env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)


def rendered(draft, tmp_path, language="triton", name="workspace"):
    workspace = tmp_path / name
    shutil.copytree(draft, workspace)
    render_workspace(draft, workspace, task_id=TASK_ID, target_language=language)
    return workspace


def test_links_cases_without_executing_source(bundle):
    solution = bundle / "solutions/reference/gemm.json"
    data = json.loads(solution.read_text())
    data["sources"][0]["content"] = "raise RuntimeError('must not import')\ndef run(a, b): return a\n"
    write_json(solution, data)
    tasks = inspect_bundle(bundle)
    assert len(tasks) == 1
    assert [r["workload"]["uuid"] for r in tasks[0].rows] == [f"case-{m}" for m in SIZES]
    assert tasks[0].reference == data


@pytest.mark.parametrize("problem", ["duplicate_uuid", "mixed_definition", "unknown_axis", "constraint",
                                     "input_descriptor", "shape_expression", "duplicate_key"])
def test_rejects_cases_it_cannot_preserve(bundle, problem):
    workload = bundle / "workloads/gemm.jsonl"
    rows = [json.loads(line) for line in workload.read_text().splitlines()]
    definition = json.loads((bundle / "definitions/gemm.json").read_text())
    if problem == "duplicate_uuid":
        rows[1]["workload"]["uuid"] = "case-1"
    elif problem == "mixed_definition":
        rows[1]["definition"] = "other"
    elif problem == "unknown_axis":
        rows[1]["workload"]["axes"]["extra"] = 1
    elif problem == "constraint":
        definition["constraints"] = ["m % 2 == 0"]
    elif problem == "shape_expression":
        definition["inputs"]["a"]["shape"] = ["m", "k*2"]
    elif problem == "input_descriptor":
        rows[0]["workload"]["inputs"]["a"] = {"type": "file", "path": "a.bin"}
    if problem == "duplicate_key":
        workload.write_text(workload.read_text().replace('"m": 1}', '"m": 1, "m": 2}'))
    else:
        write_rows(bundle, rows)
    write_json(bundle / "definitions/gemm.json", definition)
    with pytest.raises(ImportProblem):
        inspect_bundle(bundle)


def test_ambiguous_solution_requires_selection(bundle):
    path = bundle / "solutions/baseline/gemm.json"
    other = json.loads(path.read_text())
    other["name"] = "alternative"
    write_json(path.with_name("other.json"), other)
    with pytest.raises(ImportProblem, match="choose exactly one"):
        inspect_bundle(bundle)
    assert inspect_bundle(bundle, {"tiny_gemm": {"baseline": "baseline"}})[0].baseline["name"] == "baseline"


@pytest.mark.parametrize("source_path", ["../escape.py", "/absolute.py", "./main.py", "sub/../main.py", "a\\b.py"])
def test_rejects_path_traversal(bundle, source_path):
    path = bundle / "solutions/baseline/gemm.json"
    data = json.loads(path.read_text())
    data["sources"][0]["path"] = source_path
    write_json(path, data)
    with pytest.raises(ImportProblem):
        inspect_bundle(bundle)


def test_rejects_symlinks(bundle):
    (bundle / "linked").symlink_to(bundle / "definitions/gemm.json")
    with pytest.raises(ImportProblem, match="symbolic"):
        inspect_bundle(bundle)


def test_emission_is_a_valid_language_neutral_v3_package(emitted):
    task, cfg, draft = emitted
    assert check_contract(task, cfg, draft)["ok"]
    files = sorted(p.relative_to(draft).as_posix() for p in draft.rglob("*") if p.is_file())
    assert files == ["README.md", "baseline/helper.py", "baseline/kernel.py", "baseline/main.py",
                     "reference.py", "task.yaml", "workloads.json"]
    assert (draft / "baseline/main.py").read_text() == BASELINE_MAIN
    assert (draft / "baseline/helper.py").read_text() == BASELINE_HELPER
    assert (draft / "reference.py").read_text() == REFERENCE
    raw = yaml.safe_load((draft / "task.yaml").read_text())
    assert raw["schema_version"] == 3
    assert raw["operator"] == {"name": "tiny_gemm", "category": "gemm"}
    assert raw["interface"]["axes"]["m"] == {"kind": "var", "min": 1, "max": 1024}
    assert raw["interface"]["inputs"]["a"]["distribution"] == {"kind": "normal", "mean": 0.0, "std": 1.0}
    assert raw["reference"] == {"file": "reference.py", "symbol": "run"}
    assert raw["baseline"]["language"] == "triton" and raw["baseline"]["entry"] == "baseline/kernel.py"
    provenance = raw["baseline"]["provenance"]
    assert provenance["revision"] == f"sha256:{task.digest}" and provenance["license"] == "Apache-2.0"
    assert raw["compare"] == {"out": {"bound": "baseline_relative"}}
    assert raw["platform_support"] == {"required_arch": ["gfx950"]}
    assert "language" not in json.dumps(raw["interface"])  # the run selects the candidate language
    contract = load_task_contract(draft / "task.yaml", task_id=TASK_ID)
    workloads = contract.workloads["cases"]
    rows = {case["id"]: case for case in workloads if case["source"].startswith("SIKL")}
    assert sorted(rows) == sorted(f"case-{m}" for m in SIZES)
    assert all(rows[f"case-{m}"]["axes"] == {"m": m} for m in SIZES)
    assert all(case["checks"] == ["correctness", "performance"] for case in rows.values())
    controls = [case for case in workloads if case["id"].startswith("boundary_")]
    assert {case["axes"]["m"] for case in controls} == {3, 17}
    assert all(case["checks"] == ["correctness"] for case in controls)
    for language in ("triton", "gluon", "hip", "flydsl"):
        config = execution_config(draft / "task.yaml", task_id=TASK_ID, target_language=language)
        state = config["candidate"]["initial_state"]
        assert state == ("implemented" if language == "triton" else "unimplemented")
        assert config["contract"]["baseline_language"] == "triton"


def test_rendered_workspace_runs_reference_inputs_and_baseline_on_cpu(emitted, tmp_path):
    _, _, draft = emitted
    workspace = rendered(draft, tmp_path)
    assert (workspace / "solution/kernel.py").read_text() == (draft / "baseline/kernel.py").read_text()
    result = subprocess.run([sys.executable, "-B", "-m", "_aka_eval.evaluate", "validate-task"], cwd=workspace,
                            env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1].split("=", 1)[1])
    assert report["metadata"]["candidate_state"] == "implemented"
    script = '''import importlib, sys, types
from pathlib import Path
import torch
from _aka_eval import values
from _aka_eval.contract import load_contract
# The baseline imports triton; this CPU fixture only stands in for the module.
sys.modules.setdefault("triton", types.ModuleType("triton"))
package = types.ModuleType("baseline_pkg")
package.__path__ = [str(Path("baseline").resolve())]
sys.modules["baseline_pkg"] = package
kernel = importlib.import_module("baseline_pkg.kernel")
contract = load_contract()
for case in contract.cases:
    data = values.make_inputs(contract, case, case.seeds[0], torch.device("cpu"))
    again = values.make_inputs(contract, case, case.seeds[0], torch.device("cpu"))
    assert torch.equal(data.kernel["a"], again.kernel["a"])
    expected = values.reference_outputs(contract, data.logical)["out"]
    assert expected.dtype == torch.float64
    out = torch.full(expected.shape, float("nan"))
    assert kernel.run(**data.kernel, out=out) is None
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)
'''
    result = run_script(script, workspace)
    assert result.returncode == 0, result.stderr


def test_baseline_wrapper_orders_and_checks_returned_outputs(bundle, tmp_path):
    path = bundle / "definitions/gemm.json"
    definition = json.loads(path.read_text())
    definition["outputs"] = {"z_out": {"shape": ["m", "n"], "dtype": "float32"},
                             "a_out": {"shape": ["m", "n"], "dtype": "float32"}}
    write_json(path, definition)
    for role in ("baseline", "reference"):
        solution = bundle / f"solutions/{role}/gemm.json"
        main = ("import triton\ndef run(a, b):\n    return {'a_out': a @ b.T, 'z_out': -(a @ b.T)}\n"
                if role == "baseline" else "def run(a, b):\n    y = a.double() @ b.double().T\n    return -y, y\n")
        edit_json(solution, lambda s: s.update(sources=[{"path": "main.py", "content": main}]))
    task = inspect_bundle(bundle)[0]
    draft = draft_dir(tmp_path)
    materialize_task(task, Config(str(bundle)), draft)
    assert yaml.safe_load((draft / "task.yaml").read_text())["interface"]["outputs"].keys() == {"z_out", "a_out"}
    script = '''import importlib, sys, types
from pathlib import Path
import torch
sys.modules["triton"] = types.ModuleType("triton")
package = types.ModuleType("baseline_pkg")
package.__path__ = [str(Path("baseline").resolve())]
sys.modules["baseline_pkg"] = package
kernel = importlib.import_module("baseline_pkg.kernel")
a, b = torch.ones(2, 16), torch.ones(8, 16)
z, first = torch.empty(2, 8), torch.empty(2, 8)
kernel.run(a=a, b=b, z_out=z, a_out=first)
assert torch.all(z == -16) and torch.all(first == 16)
main = sys.modules["baseline_pkg.main"]
main.run = lambda a, b: (a,)
kernel._sikl_baseline = main.run
try: kernel.run(a=a, b=b, z_out=z, a_out=first)
except ValueError: pass
else: raise AssertionError("a short output tuple was accepted")
'''
    result = run_script(script, draft)
    assert result.returncode == 0, result.stderr


def test_check_contract_enforces_the_authoring_boundary(emitted):
    task, cfg, draft = emitted
    assert any(item.startswith("inputs.py") for item in authoring_editable(task))
    original = {p: p.read_bytes() for p in draft.rglob("*") if p.is_file()}

    def restore():
        for path in [p for p in draft.rglob("*") if p.is_file() and p not in original]:
            path.unlink()
        for path, data in original.items():
            path.write_bytes(data)

    def boundary(extra):
        def change(value):
            value["cases"].append(extra)
        return change

    accepted = [
        lambda: (draft / "README.md").write_text("# Reviewed\n"),
        lambda: edit_json(draft / "workloads.json", boundary(
            {"id": "extra_m5", "axes": {"m": 5}, "checks": ["correctness"], "source": "control"})),
        lambda: edit_yaml(draft / "task.yaml", lambda v: v["interface"]["inputs"]["a"].update(
            distribution={"kind": "uniform", "low": -1.0, "high": 1.0})),
    ]
    for change in accepted:
        change()
        assert check_contract(task, cfg, draft)["ok"], check_contract(task, cfg, draft)["diagnostics"]
        restore()
    rejected = {
        "scored_case_added": lambda: edit_json(draft / "workloads.json", boundary(
            {"id": "extra_m5", "axes": {"m": 5}, "checks": ["correctness", "performance"], "source": "x"})),
        "contract_changed": lambda: edit_json(draft / "workloads.json", lambda v: v["cases"].pop(0)),
        "undeclared_file": lambda: (draft / "helper.py").write_text("x = 1\n"),
        "missing_file": lambda: (draft / "reference.py").unlink(),
    }
    for code, change in rejected.items():
        change()
        diagnostics = check_contract(task, cfg, draft)["diagnostics"]
        assert code in {d["code"] for d in diagnostics}, (code, diagnostics)
        restore()
    for change in (
        lambda v: v["interface"]["inputs"]["a"].update(dtype="bfloat16"),
        lambda v: v["compare"]["out"].update(bound="tolerance", atol=1.0, rtol=1.0),
        lambda v: v["interface"]["axes"]["m"].update(max=4096),
        lambda v: v["interface"].update(generator={"file": "reference.py", "symbol": "run"}),
        lambda v: v["baseline"]["provenance"].update(license="MIT"),
    ):
        edit_yaml(draft / "task.yaml", change)
        assert "contract_changed" in {d["code"] for d in check_contract(task, cfg, draft)["diagnostics"]}
        restore()
    for name in ("baseline/main.py", "baseline/kernel.py", "reference.py"):
        (draft / name).write_text((draft / name).read_text() + "\n# edited\n")
        assert not check_contract(task, cfg, draft)["ok"]
        restore()
    # A generator becomes owned task code only through inputs.py::make_inputs.
    edit_yaml(draft / "task.yaml", lambda v: (v["interface"]["inputs"]["b"].update(
        distribution={"kind": "generator"}), v["interface"].update(generator={"file": "inputs.py",
                                                                              "symbol": "make_inputs"})))
    assert "missing_file" in {d["code"] for d in check_contract(task, cfg, draft)["diagnostics"]}
    (draft / "inputs.py").write_text("import aiter\n\ndef make_inputs(case, inputs, generator, device):\n"
                                     "    return {}\n")
    assert "unvendored_dependency" in {d["code"] for d in check_contract(task, cfg, draft)["diagnostics"]}
    (draft / "inputs.py").write_text("import torch\n\n\ndef make_inputs(case, inputs, generator, device):\n"
                                     "    return {'b': torch.ones(8, 16, device=device)}\n")
    assert check_contract(task, cfg, draft)["ok"]


def test_boundary_controls_are_added_only_when_rows_lack_distinct_ones(bundle, tmp_path):
    def controls(sizes):
        write_rows(bundle, rows_for(sizes))
        cases = json.loads(task_files(inspect_bundle(bundle)[0], Config(str(bundle)))["workloads.json"])["cases"]
        return [case["axes"]["m"] for case in cases if case["id"].startswith("boundary_")]
    assert controls((3, 5, 17, 64, 512, 2048)) == []
    # 3 is both the minimum and the only small non-power-of-two: one more distinct case is needed.
    assert controls((3, 17, 64, 512, 2048)) == [5]
    axes = {"m": {"kind": "var", "min": 3, "max": 2048}}
    assert {"minimum", "non_power_of_two"} <= boundary_kinds({"m": 3}, axes)
    assert "unaligned" in boundary_kinds({"m": 17}, axes)


def test_duplicate_rows_merge_into_one_case(bundle, tmp_path):
    rows = rows_for(SIZES)
    duplicate = json.loads(json.dumps(rows[2]))
    duplicate["workload"]["uuid"] = "again-16"
    write_rows(bundle, rows + [duplicate])
    task = inspect_bundle(bundle)[0]
    cases = json.loads(task_files(task, Config(str(bundle)))["workloads.json"])["cases"]
    merged = [case for case in cases if case["axes"] == {"m": 16}]
    assert len(merged) == 1 and "again-16" in merged[0]["source"]


@pytest.mark.parametrize("sizes", [(1, 2, 512), (64, 128, 256, 1024, 2048)])
def test_rows_that_cannot_meet_the_performance_rule_need_a_spec(bundle, tmp_path, sizes):
    write_rows(bundle, rows_for(sizes))
    task = inspect_bundle(bundle)[0]
    with pytest.raises(ImportProblem) as error:
        materialize_task(task, Config(str(bundle)), draft_dir(tmp_path))
    assert error.value.code == "needs_spec"
    result = run(Config(str(bundle)), repo=tmp_path / "repo", runtime=lambda c: {},
                 generator=lambda *a: pytest.fail("no generator for an unbuildable spec"))
    assert result["tasks"]["tiny_gemm"]["state"] == "needs_spec"


def test_boundary_rule_respects_the_observed_domain(bundle, tmp_path):
    # Both var axes span at most 16 units of their observed step (128 and 8): no
    # in-domain value is an unaligned boundary, however large the sizes are.
    edit_json(bundle / "definitions/gemm.json", lambda d: d["axes"].update(n={"type": "var"}))
    rows = []
    for m, n in ((128, 8), (1024, 8), (2048, 64), (2048, 128), (1024, 128)):
        row = rows_for((m,))[0]
        row["workload"].update(axes={"m": m, "n": n}, uuid=f"case-{m}-{n}")
        rows.append(row)
    write_rows(bundle, rows)
    with pytest.raises(ImportProblem, match="boundary") as error:
        task_files(inspect_bundle(bundle)[0], Config(str(bundle)))
    assert error.value.code == "needs_spec"


@pytest.mark.parametrize("problem,code", [
    ("aiter", "unvendored_dependency"), ("os", "forbidden_dependency"), ("nested", "unsupported_layout"),
    ("kernel", "unsupported_layout"), ("reference_numpy", "reference_dependency"),
    ("reference_files", "unsupported_reference"), ("license", "missing_license"),
    ("license_text", "missing_license_text"), ("language", "unsupported_baseline_language"),
])
def test_unvendorable_sources_fail_closed(bundle, problem, code):
    baseline = bundle / "solutions/baseline/gemm.json"
    reference = bundle / "solutions/reference/gemm.json"
    if problem == "aiter":
        edit_json(baseline, lambda s: s["sources"][1].update(content="import aiter\n" + BASELINE_HELPER))
    elif problem == "os":
        edit_json(baseline, lambda s: s["sources"][1].update(content="import os\n" + BASELINE_HELPER))
    elif problem == "nested":
        edit_json(baseline, lambda s: s["sources"][1].update(path="impl/helper.py"))
    elif problem == "kernel":
        edit_json(baseline, lambda s: (s["sources"][0].update(path="kernel.py"),
                                       s["spec"].update(entry_point="kernel.py::run")))
    elif problem == "reference_numpy":
        edit_json(reference, lambda s: s["sources"][0].update(content="import numpy\n" + REFERENCE))
    elif problem == "reference_files":
        edit_json(reference, lambda s: s["sources"].append({"path": "helper.py", "content": "x = 1\n"}))
    elif problem == "license":
        edit_json(baseline, lambda s: s.pop("license"))
    elif problem == "license_text":
        edit_json(baseline, lambda s: s.update(license="MIT"))
    elif problem == "language":
        edit_json(baseline, lambda s: (s["spec"].update(language="python"), s["sources"][0].update(
            content=BASELINE_MAIN.replace("import triton\n", ""))))
    with pytest.raises(ImportProblem) as error:
        task_files(inspect_bundle(bundle)[0], Config(str(bundle)))
    assert error.value.code == code


def test_license_sources_and_bundle_license_text(bundle, tmp_path):
    edit_json(bundle / "solutions/baseline/gemm.json", lambda s: s.pop("license"))
    files = task_files(inspect_bundle(bundle)[0], Config(str(bundle), license="Apache-2.0"))
    assert "baseline/LICENSE" not in files
    (bundle / "LICENSE").write_text("MIT License text\n")
    files = task_files(inspect_bundle(bundle)[0], Config(str(bundle), license="MIT"))
    assert files["baseline/LICENSE"] == "MIT License text\n"
    assert yaml.safe_load(files["task.yaml"])["baseline"]["provenance"]["license"] == "MIT"


@pytest.mark.parametrize("imports,language", [
    ("import triton\n", "triton"), ("from triton.experimental import gluon\n", "gluon"),
    ("import flydsl\n", "flydsl"),
])
def test_python_baseline_language_comes_from_its_imports(bundle, imports, language):
    edit_json(bundle / "solutions/baseline/gemm.json", lambda s: (
        s["spec"].update(language="python"),
        s["sources"][0].update(content=BASELINE_MAIN.replace("import triton\n", imports))))
    raw = yaml.safe_load(task_files(inspect_bundle(bundle)[0], Config(str(bundle)))["task.yaml"])
    assert raw["baseline"]["language"] == language


def test_scalar_inputs_become_declared_scalars_and_case_values(bundle, tmp_path):
    edit_json(bundle / "definitions/gemm.json", lambda d: d["inputs"].update(
        mode={"shape": None, "dtype": "int32", "description": "Upstream modes: 0, 1, 2."}))
    rows = rows_for(SIZES)
    for index, row in enumerate(rows):
        row["workload"]["inputs"]["mode"] = {"type": "scalar", "value": index % 2 * 2}
    write_rows(bundle, rows)
    task = inspect_bundle(bundle)[0]
    files = task_files(task, Config(str(bundle)))
    raw = yaml.safe_load(files["task.yaml"])
    assert raw["interface"]["inputs"]["mode"] == {"kind": "scalar", "dtype": "int32", "default": 0,
                                                  "description": "Upstream modes: 0, 1, 2."}
    cases = {case["id"]: case for case in json.loads(files["workloads.json"])["cases"]}
    assert [cases[f"case-{m}"]["scalars"] for m in SIZES] == [{"mode": i % 2 * 2} for i in range(len(SIZES))]
    assert "mode=mode" in files["baseline/kernel.py"]


@pytest.fixture
def bundle_v2(bundle):
    definition = json.loads((bundle / "definitions/gemm.json").read_text())
    definition.update(
        schema_version=2,
        reference="def run(a, b):\n    return a.double() @ b.double().T\n",
        initialize=("import torch\nfrom dataclasses import dataclass\n"
                    "@dataclass\nclass Settings:\n    scale: float = 0.5\n"
                    "def run(inputs, seed=0):\n"
                    "    rng = torch.Generator(device=inputs['a'].device).manual_seed(seed)\n"
                    "    for value in inputs.values():\n"
                    "        value.normal_(std=Settings().scale, generator=rng)\n"
                    "    return inputs\n"),
        compare=("import torch\ndef run(actual, expected):\n"
                 "    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)\n"),
    )
    write_json(bundle / "definitions/gemm.json", definition)
    (bundle / "solutions/reference/gemm.json").unlink()
    baseline = json.loads((bundle / "solutions/baseline/gemm.json").read_text())
    baseline["spec"].pop("target_hardware")
    baseline["spec"]["target"] = [{"arch": "gfx950", "hardware_id": "MI355X"}]
    write_json(bundle / "solutions/baseline/gemm.json", baseline)
    return bundle


def test_initialize_callback_owns_the_inputs_and_is_not_editable(bundle_v2, tmp_path):
    task = inspect_bundle(bundle_v2)[0]
    cfg = Config(str(bundle_v2))
    draft = draft_dir(tmp_path)
    info = materialize_task(task, cfg, draft)
    assert not any(item.startswith("inputs.py") for item in info["editable"])
    assert (draft / "sikl_initialize.py").read_text() == task.definition["initialize"]
    assert (draft / "reference.py").read_text() == task.definition["reference"]
    assert task.origins["reference"] == "definitions/gemm.json#reference"
    raw = yaml.safe_load((draft / "task.yaml").read_text())
    assert {d["distribution"]["kind"] for d in raw["interface"]["inputs"].values()} == {"generator"}
    assert raw["interface"]["generator"] == {"file": "inputs.py", "symbol": "make_inputs"}
    assert "compare callback is not carried over" in (draft / "README.md").read_text()
    assert check_contract(task, cfg, draft)["ok"]
    (draft / "inputs.py").write_text((draft / "inputs.py").read_text() + "\n# replace distribution\n")
    assert not check_contract(task, cfg, draft)["ok"]
    (draft / "inputs.py").write_text(task_files(task, cfg)["inputs.py"])
    edit_yaml(draft / "task.yaml", lambda v: v["interface"]["inputs"]["a"].update(
        distribution={"kind": "normal", "mean": 0.0, "std": 1.0}))
    assert not check_contract(task, cfg, draft)["ok"]


def test_initialize_callback_runs_through_the_evaluator_inputs(bundle_v2, tmp_path):
    task = inspect_bundle(bundle_v2)[0]
    draft = draft_dir(tmp_path)
    materialize_task(task, Config(str(bundle_v2)), draft)
    workspace = rendered(draft, tmp_path, language="hip")
    script = '''import torch
from _aka_eval import values
from _aka_eval.contract import load_contract
contract = load_contract()
case = contract.cases[-1]
first = values.make_inputs(contract, case, case.seeds[0], torch.device("cpu"))
again = values.make_inputs(contract, case, case.seeds[0], torch.device("cpu"))
other = values.make_inputs(contract, case, case.seeds[1], torch.device("cpu"))
assert all(torch.equal(first.logical[k], again.logical[k]) for k in ("a", "b"))
assert not torch.equal(first.logical["a"], other.logical["a"])
assert 0.3 < float(first.logical["a"].std()) < 0.7
assert first.logical["a"].shape == (case.axes["m"], 16)
'''
    result = run_script(script, workspace)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("version", [0, 3, True, "2", None])
def test_unknown_schema_rejected(bundle_v2, version):
    edit_json(bundle_v2 / "definitions/gemm.json", lambda d: d.update(schema_version=version))
    with pytest.raises(ImportProblem, match="schema"):
        inspect_bundle(bundle_v2)


@pytest.mark.parametrize("name", ["reference", "initialize", "compare"])
def test_callbacks_need_syntax_and_run_without_importing(bundle_v2, name):
    path = bundle_v2 / "definitions/gemm.json"
    data = json.loads(path.read_text())
    data[name] = "raise RuntimeError('metadata must not execute')\ndef run(*args, **kw): pass\n"
    write_json(path, data)
    assert inspect_bundle(bundle_v2)[0].definition[name] == data[name]
    data[name] = "def no_run(): pass\n"
    write_json(path, data)
    with pytest.raises(ImportProblem, match="run"):
        inspect_bundle(bundle_v2)


def test_v2_embedded_reference_cannot_be_overridden(bundle_v2):
    with pytest.raises(ImportProblem, match="owned by the definition"):
        inspect_bundle(bundle_v2, {"tiny_gemm": {"reference": "other"}})


def test_target_arch_is_authoritative(bundle_v2, tmp_path):
    edit_json(bundle_v2 / "solutions/baseline/gemm.json", lambda s: s["spec"]["target"][0].update(arch="gfx942"))
    task = inspect_bundle(bundle_v2)[0]
    with pytest.raises(ImportProblem, match="does not match"):
        materialize_task(task, Config(str(bundle_v2)), draft_dir(tmp_path))


def _fake_audit(monkeypatch, *, status="PASS", finalized=True, stale=False, tamper=False, foreign=False):
    """Stand in for the GPU subprocess: real materialization, a synthetic QA report."""
    import shutil as shell
    from agents.qa_auditor.report import build_report, write_report
    from src.task_materialization import load_materialization_record
    real_which = shell.which
    monkeypatch.setattr(shell, "which", lambda name, *a, **k: None if name == "cp" else real_which(name, *a, **k))

    def process(argv, cwd, log, timeout, env):
        request = json.loads(Path(argv[-1]).read_text())
        _, _, workspace = validation.prepare_audit_workspace(request)
        package_sha = load_materialization_record(workspace)["task_package_sha256"]
        findings = [] if status == "PASS" else [{
            "id": "QA-CONTROLS", "severity": "blocking", "title": "t", "detail": "the zeros control passes",
            "evidence": {}, "fix": "strengthen inputs"}]
        report = build_report(task_id="other/task" if foreign else request["task_id"], findings=findings,
                              task_package_sha256=package_sha, target_language="triton",
                              request_id="earlier-request" if stale else request["validation_id"])
        if finalized:
            write_report(workspace, report)
        else:
            (workspace / "qa_report.yaml").write_text(yaml.safe_dump(report))
        if tamper:
            (Path(request["package"]) / "README.md").write_text("# auditor tampering\n")
        return {"ok": status == "PASS", "exit_code": 0 if status == "PASS" else 1, "timed_out": False}

    monkeypatch.setattr(validation, "run_process", process)
    monkeypatch.setattr(validation, "runtime_identity", lambda c: {"arch": "gfx950"})


@pytest.mark.parametrize("kwargs,accepted", [
    ({}, True), ({"status": "FAIL"}, False), ({"finalized": False}, False), ({"stale": True}, False),
    ({"tamper": True}, False), ({"foreign": True}, False),
])
def test_validation_requires_a_current_finalized_qa_pass(emitted, tmp_path, monkeypatch, kwargs, accepted):
    _, cfg, draft = emitted
    _fake_audit(monkeypatch, **kwargs)
    result = validation.validate_task(draft, tmp_path / "validation", cfg)
    assert result["ok"] == accepted
    assert result["task_id"] == TASK_ID
    if kwargs == {"status": "FAIL"}:
        assert result["report_complete"] and result["overall_status"] == "FAIL"
        assert result["diagnostics"][0]["check"] == "QA-CONTROLS"
        assert result["diagnostics"][0]["severity"] == "blocking"
    evidence = json.loads((tmp_path / "validation" / result["validation_id"] / "evidence.json").read_text())
    assert evidence["ok"] == accepted


def test_qa_settings_follow_the_validator_section():
    cfg = Config("bundle", validator={"backend": "claude_code", "model": "m", "effort": "high"})
    settings = validation.qa_eval_config(cfg)
    assert settings["agent"]["template"] == "qa_auditor"
    assert (settings["agent"]["backend"], settings["agent"]["model"], settings["agent"]["effort"]) == (
        "claude_code", "m", "high")
    assert validation.qa_eval_config(Config("bundle", validator={"backend": "none"}))["agent"]["backend"] == "none"


def test_install_rejects_stale_evidence_and_conflicts(emitted, tmp_path):
    _, _, draft = emitted
    evidence = {"ok": True, "task_digest": task_digest(draft)}
    output = tmp_path / "tasks"
    destination = install_task(draft, output, "task", evidence)
    assert install_task(draft, output, "task", evidence) == destination
    (draft / "README.md").write_text("# repaired\n")
    with pytest.raises(ImportProblem, match="changed"):
        install_task(draft, output, "task", evidence)
    evidence["task_digest"] = task_digest(draft)
    with pytest.raises(ImportProblem, match="overwrite"):
        install_task(draft, output, "task", evidence)


def test_validation_copies_only_files_that_will_be_installed(emitted, tmp_path):
    _, _, draft = emitted
    (draft / "baseline/__pycache__").mkdir()
    (draft / "baseline/__pycache__/kernel.pyc").write_bytes(b"stale bytecode")
    (draft / "build").mkdir()
    (draft / "build/undeclared.so").write_bytes(b"old build artifact")
    workspace = tmp_path / "clean-validation"
    validation.copy_task_files(draft, workspace)
    installed = install_task(draft, tmp_path / "output", "task", {"ok": True, "task_digest": task_digest(draft)})
    assert task_digest(workspace) == task_digest(installed)
    assert not (workspace / "baseline/__pycache__").exists()
    assert not (workspace / "build").exists()


def test_controller_repairs_revalidates_installs_and_resumes(bundle, tmp_path):
    repo = tmp_path / "repo"
    cfg = Config(str(bundle))
    calls = []

    def generator(config, draft, root, task, feedback, log, timeout):
        calls.append("generate")
        if len(calls) > 1:
            assert not feedback["validation"]["ok"]
            with (draft / "README.md").open("a") as stream:
                stream.write("\nReviewed after the first audit.\n")
        return {"ok": True}

    def validator(draft, artifacts, config, timeout):
        calls.append("validate")
        return {"ok": len(calls) == 4, "task_digest": task_digest(draft), "validation_id": "accepted"}

    fake_runtime = lambda cfg: {"arch": "gfx950"}  # noqa: E731
    result = run(cfg, repo=repo, generator=generator, validator=validator, runtime=fake_runtime)
    assert result["ok"] and calls == ["generate", "validate"] * 2
    installed = repo / "tasks/SIKL-task/tiny_gemm"
    assert (installed / "task.yaml").is_file() and not (installed / "config.yaml").exists()
    resumed = run(cfg, result["run_id"], repo=repo, generator=generator, validator=validator, runtime=fake_runtime)
    assert resumed["ok"] and len(calls) == 4
    cfg.agent_timeout += 1
    with pytest.raises(ValueError, match="Resume refused"):
        run(cfg, result["run_id"], repo=repo, runtime=fake_runtime)


def test_failed_generation_cannot_install_even_with_valid_contract(bundle, tmp_path):
    def validator(*args, **kwargs):
        pytest.fail("Failed generation must not reach acceptance")
    result = run(Config(str(bundle), max_repair_attempts=1), repo=tmp_path / "repo",
                 generator=lambda *a: {"ok": False}, validator=validator, runtime=lambda c: {})
    assert not result["ok"]
    assert result["tasks"]["tiny_gemm"]["attempts"] == 2
    assert not (tmp_path / "repo/tasks/SIKL-task/tiny_gemm").exists()


def test_resume_checks_snapshot_and_keeps_deadline(bundle, tmp_path):
    cfg = Config(str(bundle))
    repo = tmp_path / "repo"

    def crash(*args):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(cfg, repo=repo, generator=crash, runtime=lambda c: {})
    root = next((repo / cfg.artifact_root).iterdir())
    state = json.loads((root / "state.json").read_text())
    state["tasks"]["tiny_gemm"]["deadline_at"] = time.time() - 1
    write_json(root / "state.json", state)
    result = run(cfg, root.name, repo=repo, generator=crash, runtime=lambda c: {})
    assert result["tasks"]["tiny_gemm"]["state"] == "failed"
    (root / "bundle/workloads/gemm.jsonl").write_text("{}")
    with pytest.raises(ImportProblem, match="snapshot"):
        run(cfg, root.name, repo=repo, runtime=lambda c: {})


def test_timeout_stops_process_group(tmp_path):
    marker = tmp_path / "survived"
    child = "import time; from pathlib import Path; time.sleep(0.8); Path(%r).touch()" % str(marker)
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(10)" % child
    result = run_process([sys.executable, "-c", parent], tmp_path, tmp_path / "process.log", 0.2)
    assert not result["ok"] and result["timed_out"]
    time.sleep(0.9)
    assert not marker.exists()


@pytest.mark.parametrize("kwargs", [{"generator": []}, {"max_repair_attempts": -1}, {"output_dir": ""},
                                    {"tasks": "tiny_gemm"}, {"validator": {"backend": "gpt"}},
                                    {"license": ""}])
def test_bad_config_fails_before_agent_execution(kwargs):
    with pytest.raises(ValueError):
        Config("bundle", **kwargs)


@pytest.mark.parametrize("field", ["target_language", "policy"])
def test_language_and_numerical_policy_are_not_builder_options(field):
    with pytest.raises(TypeError):
        Config("bundle", **{field: "triton"})


def test_metadata_cli_does_not_import_torch():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import agents.sikl_task_builder.__main__; assert "torch" not in sys.modules'],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("malformed", ["missing_result", "missing_case", "wrong_role", "failed_manifest"])
def test_diagnostic_tool_rejects_false_success(emitted, tmp_path, monkeypatch, malformed):
    task, cfg, draft = emitted
    monkeypatch.setattr(validation, "runtime_identity", lambda c: {})
    manifest = json.loads(subprocess.run(
        [sys.executable, "-B", "-m", "_aka_eval.evaluate", "validate-task"], cwd=rendered(draft, tmp_path),
        env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True).stdout.splitlines()[-1].split("=", 1)[1])

    def process(argv, cwd, log, timeout, env):
        assert argv[1:4] == ["-B", "-m", "_aka_eval.evaluate"] and Path(cwd).name == "workspace"
        log.parent.mkdir(parents=True, exist_ok=True)
        if argv[4:] == ["validate-task"]:
            value = dict(manifest, status="FAIL") if malformed == "failed_manifest" else manifest
        else:
            cases = [{**row, "status": "PASS"} for row in manifest["cases"]]
            if malformed == "missing_case":
                cases.pop()
            value = {**manifest, "role": "baseline" if malformed == "wrong_role" else "candidate",
                     "action": "correctness", "status": "PASS", "cases": cases, "metadata": {}}
        text = "PASS\n" if malformed == "missing_result" and argv[4:] != ["validate-task"] else (
            "ARENA_EVAL_RESULT=" + json.dumps(value))
        log.write_text(text + "\n")
        passed = value.get("status") == "PASS"
        return {"ok": passed, "exit_code": 0 if passed else 1, "timed_out": False, "log": str(log)}

    monkeypatch.setattr(validation, "run_process", process)
    assert not validation.check_task(draft, tmp_path / "check", cfg, "correctness")["ok"]


def test_deadline_cleans_framework_action_in_separate_session(tmp_path):
    marker = tmp_path / 'escaped-session'
    started = tmp_path / 'started'
    child = (
        'from pathlib import Path; import time; '
        f'Path({str(started)!r}).touch(); time.sleep(1.5); Path({str(marker)!r}).touch()'
    )
    parent = (
        'from src.task_execution import _run_process; from pathlib import Path; import os,sys; '
        f'_run_process((sys.executable, "-c", {child!r}), Path.cwd(), dict(os.environ), 30)'
    )
    result = run_process([sys.executable, '-c', parent], ROOT, tmp_path / 'deadline.log', 0.8)
    assert started.is_file(), (tmp_path / 'deadline.log').read_text()
    assert result['timed_out'] and not result['ok']
    time.sleep(1.6)
    assert not marker.exists(), 'framework action outlived the authoring deadline'


def test_report_identity_follows_output_suite():
    assert Config('bundle', output_dir='tasks/imported/subsuite').arena_task_id('gemm') == 'imported/subsuite/gemm'
    assert Config('bundle', output_dir='tasks/../tasks/imported').arena_task_id('gemm') == 'imported/gemm'
    assert Config('bundle', output_dir=str(ROOT / 'tasks/imported')).arena_task_id('gemm') == 'imported/gemm'


def test_builder_never_depends_on_the_retired_validator():
    for path in (ROOT / "agents/sikl_task_builder").rglob("*.py"):
        assert "task_validator" not in path.read_text(), path

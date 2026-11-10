# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""CPU pre-checks for a schema-v3 task package, before the GPU QA audit.

    python3 -m src.tools.check_task tasks/<baseline_language>/<category>/<operator>/<scenario> [--json PATH]

Checks that the contract loads and its path matches its baseline language,
category and operator, that a workspace renders for every target
language, that the baseline meets the candidate rules of its own language, the
QA auditor's static package/reference/provenance rules, and that inputs and the
reference run on the CPU for the smallest and largest correctness cases. It
cannot replace the GPU audit (kernels, timing, controls).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.task_contract import TARGET_LANGUAGES, load_task_contract, render_workspace  # noqa: E402
from src.task_spec import TaskConfigError  # noqa: E402

_SMOKE = r'''
import json, sys, torch
from _aka_eval import values
from _aka_eval.contract import load_contract
contract = load_contract()
device = torch.device("cpu")
cases = sorted(contract.cases, key=lambda c: sum(c.axes.values()))
report = []
for case in (cases[0], cases[-1]) if len(cases) > 1 else cases:
    if case is cases[-1] and len(cases) > 1 and sum(case.axes.values()) > int(sys.argv[1]):
        report.append({"case": case.case_id, "skipped": "too large for a CPU smoke"})
        continue
    inputs = values.make_inputs(contract, case, case.seeds[0], device)
    outputs = values.reference_outputs(contract, inputs.logical)
    symbols = contract.symbols(case)
    shapes = {}
    for name in contract.checked_tensors:
        declaration = contract.outputs.get(name) or contract.inputs[name]
        expected = contract.shape(declaration, symbols)
        actual = tuple(outputs[name].shape)
        if actual != expected:
            raise SystemExit(f"reference output {name} has shape {actual}, declared {expected}")
        if outputs[name].dtype.is_floating_point and outputs[name].dtype not in (torch.float32, torch.float64):
            raise SystemExit(f"reference output {name} is {outputs[name].dtype}; use float32/float64")
        shapes[name] = list(actual)
    report.append({"case": case.case_id, "outputs": shapes})
    del inputs, outputs  # free the first case before drawing the next one
print("SMOKE=" + json.dumps(report))
'''


def check(task_dir: Path, *, smoke_limit: int) -> dict:
    task_dir = task_dir.resolve()
    task_id = task_dir.relative_to(ROOT / "tasks").as_posix()
    result: dict = {"task": task_id, "errors": [], "warnings": []}
    try:
        contract = load_task_contract(task_dir / "task.yaml", task_id=task_id)
    except (TaskConfigError, OSError, ValueError) as exc:
        result["errors"].append(f"contract: {exc}")
        return result
    if problem := contract.layout_problem():
        result["errors"].append(f"layout: {problem}")
    baseline_language = contract.baseline_language
    with tempfile.TemporaryDirectory(prefix="aka-check-") as scratch:
        for language in TARGET_LANGUAGES:
            workspace = Path(scratch) / language
            shutil.copytree(task_dir, workspace, ignore=shutil.ignore_patterns("__pycache__"))
            try:
                render_workspace(task_dir, workspace, task_id=task_id, target_language=language)
            except (TaskConfigError, OSError, ValueError) as exc:
                result["errors"].append(f"render {language}: {exc}")
        workspace = Path(scratch) / (baseline_language if baseline_language in TARGET_LANGUAGES else "triton")
        if baseline_language in TARGET_LANGUAGES:
            from src.tools.arena_eval.audit import PolicyError, audit_hip_sources, audit_python_sources

            try:
                sources = {p.relative_to(workspace / "solution").as_posix(): p.read_text(encoding="utf-8")
                           for p in (workspace / "solution").rglob("*")
                           if p.is_file() and not (baseline_language == "asm" and p.suffix == ".co")}
                if baseline_language == "hip":
                    audit_hip_sources(sources, "kernel.hip")
                else:
                    audit_python_sources(sources, baseline_language)
            except PolicyError as exc:
                result["errors"].append(f"baseline fails the {baseline_language} candidate rules: {exc}")
            except UnicodeDecodeError as exc:
                result["errors"].append(f"baseline holds a file that is not UTF-8 source text: {exc}")
        else:
            result["warnings"].append(f"{baseline_language} baseline: no candidate-path positive control")
        from agents.qa_auditor.deterministic import check_package, check_provenance, check_reference

        findings: list = []
        check_package(workspace, findings)
        check_reference(workspace, contract.raw, findings)
        check_provenance(workspace, contract.raw, findings)
        for finding in findings:
            result["errors"].append(f"{finding.check_id}: {finding.detail}")
        if not (task_dir / "README.md").is_file():
            result["warnings"].append("README.md is missing")
        from src.tools.arena_eval.values import is_exact_dtype

        interface = contract.interface
        for name, declaration in contract.raw["compare"].items():
            dtype = (interface["outputs"].get(name) or interface["inputs"].get(name) or {}).get("dtype", "")
            if declaration["bound"] == "exact" and is_exact_dtype(dtype):
                continue  # integer, boolean and packed outputs always compare exactly
            if declaration["bound"] != "baseline_relative" or len(declaration) > 1:
                result["warnings"].append(f"compare.{name} departs from the default bound: justify it in README.md")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ARENA_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run([sys.executable, "-B", "-c", _SMOKE, str(smoke_limit)], cwd=workspace, env=env,
                                 capture_output=True, text=True, timeout=1800)
        lines = [line for line in process.stdout.splitlines() if line.startswith("SMOKE=")]
        if process.returncode or not lines:
            result["errors"].append("CPU inputs/reference smoke failed: " + (process.stderr or process.stdout)[-2000:])
        else:
            result["smoke"] = json.loads(lines[-1].split("=", 1)[1])
    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_dirs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--smoke-limit", type=int, default=1 << 16,
                        help="skip the CPU reference for cases whose var-axis sum exceeds this")
    args = parser.parse_args()
    results = [check(path, smoke_limit=args.smoke_limit) for path in args.task_dirs]
    text = json.dumps(results, indent=1)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

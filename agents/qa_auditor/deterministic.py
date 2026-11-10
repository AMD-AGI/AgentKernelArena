# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic QA checks over a materialized task and its captured evidence.

Every check here is reproducible from the task package, the framework's
initial-validation evidence and the GPU control suite. None of them consult a
model. Their findings are the auditor's blocking evidence.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

from .rubric import BLOCKING, MIN_BASELINE_CASE_MS, WARNING, check

PRODUCTION_PACKAGES = frozenset({
    "aiter", "vllm", "sglang", "sgl_kernel", "flash_attn", "flashinfer", "xformers", "deep_gemm",
    "transformer_engine", "apex", "triton_kernels", "lightllm", "tensorrt_llm", "fbgemm_gpu", "mamba_ssm",
    "causal_conv1d", "fla", "liger_kernel", "unsloth", "bitsandbytes", "torchao", "composable_kernel", "ck",
    "hipblaslt", "rocblas",
})
FORBIDDEN_MODULES = frozenset({
    "subprocess", "socket", "urllib", "requests", "http", "ctypes", "importlib", "multiprocessing",
    "threading", "asyncio", "pickle", "shelve", "tempfile", "shutil", "glob", "pathlib", "os", "sys",
})
REFERENCE_MODULES = frozenset({"torch", "math", "typing", "functools", "itertools", "operator",
                               "dataclasses", "__future__", "collections", "numbers", "enum"})
LOCAL_PATH = re.compile(r"(?:/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\)[^\s\"'`)]*")
ABSOLUTE_PATH = re.compile(r"""["'](/(?:sgl-workspace|opt|usr|root|home|workspace|tmp|shared_nfs|data|mnt)\b[^"']*)["']""")
HIP_FORBIDDEN_INCLUDE = re.compile(r'#\s*include\s*[<"]((?:rocblas|hipblas|hipblaslt|miopen|ck|composable_kernel|'
                                   r'rocwmma|rccl|aiter)[^>"]*)[>"]')
MOVING_REVISIONS = frozenset({"main", "master", "head", "latest", "todo", "unknown", "n/a", "none", "tbd"})


@dataclass
class Finding:
    check_id: str
    detail: str
    evidence: dict = field(default_factory=dict)
    severity: str | None = None

    def to_mapping(self) -> dict:
        definition = check(self.check_id)
        return {"id": self.check_id, "severity": self.severity or definition.severity,
                "title": definition.title, "detail": self.detail, "evidence": self.evidence,
                "fix": definition.fix}


def _python_imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.append((node.module, node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "__import__", "eval", "exec", "compile", "open"):
            found.append((f"<{node.func.id}()>", node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in ("ops", "_C") and isinstance(node.value, ast.Name) \
                and node.value.id == "torch":
            found.append((f"torch.{node.attr}", node.lineno))
    return found


def _task_python_files(workspace: Path) -> list[Path]:
    files = []
    for path in sorted(workspace.rglob("*.py")):
        relative = path.relative_to(workspace)
        if relative.parts[0] in ("_aka_eval", "solution") or "__pycache__" in relative.parts or _is_os_junk(path):
            continue
        files.append(path)
    return files


def _is_os_junk(path: Path) -> bool:
    """macOS metadata written by archive tools: AppleDouble ``._*`` forks and ``.DS_Store``."""
    return path.name.startswith("._") or path.name == ".DS_Store"


def _declared_code_objects(workspace: Path) -> dict:
    try:
        raw = yaml.safe_load((workspace / "task.yaml").read_text(encoding="utf-8")) or {}
        declared = raw["baseline"]["provenance"].get("code_objects") or {}
    except (OSError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        return {}
    return declared if isinstance(declared, dict) else {}


def check_package(workspace: Path, findings: list[Finding]) -> None:
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if relative.parts[0] not in ("_aka_eval", "solution") and _is_os_junk(path):
            findings.append(Finding("QA-PACKAGE", f"{relative.as_posix()} is operating-system metadata; delete it",
                                    {"path": relative.as_posix()}))
    for path in _task_python_files(workspace):
        relative = path.relative_to(workspace).as_posix()
        try:
            imports = _python_imports(path)
        except SyntaxError as exc:
            findings.append(Finding("QA-PACKAGE", f"{relative} does not parse: {exc}"))
            continue
        for module, line in imports:
            top = module.split(".")[0]
            if top in PRODUCTION_PACKAGES:
                findings.append(Finding("QA-PACKAGE", f"{relative}:{line} imports {module}; vendor the code instead",
                                        {"path": relative, "line": line}))
            elif top in FORBIDDEN_MODULES or module.startswith("<") or module in ("torch.ops", "torch._C"):
                findings.append(Finding("QA-PACKAGE", f"{relative}:{line} uses {module}",
                                        {"path": relative, "line": line}))
        text = path.read_text(encoding="utf-8")
        for match in ABSOLUTE_PATH.finditer(text):
            findings.append(Finding("QA-PACKAGE", f"{relative} names the absolute path {match.group(1)}",
                                    {"path": relative}))
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if (not path.is_file() or relative.parts[0] in ("_aka_eval", "solution") or relative.as_posix() == "config.yaml"
                or path.suffix not in (".py", ".md", ".yaml", ".yml", ".json", ".hip", ".h", ".hpp", ".cuh", ".txt")):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in LOCAL_PATH.finditer(text):
            findings.append(Finding("QA-PACKAGE", f"{relative.as_posix()} names a local filesystem path "
                                    f"{match.group(0)!r}; describe sources by their public project and path",
                                    {"path": relative.as_posix()}))
            break
    declared = _declared_code_objects(workspace)
    for path in sorted((workspace / "baseline").rglob("*")):
        if path.suffix in (".hip", ".h", ".hpp", ".cuh", ".cpp", ".cu", ".inc"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in HIP_FORBIDDEN_INCLUDE.finditer(text):
                findings.append(Finding("QA-PACKAGE", f"{path.relative_to(workspace)} includes {match.group(1)}; "
                                        "library kernels must be rewritten as task code"))
        elif path.is_file() and path.suffix in (".so", ".co", ".hsaco", ".o", ".a", ".bin"):
            relative = path.relative_to(workspace).as_posix()
            if path.suffix == ".co" and declared.get(relative) == hashlib.sha256(path.read_bytes()).hexdigest():
                continue    # a vendored code object of an asm baseline, pinned by SHA256
            findings.append(Finding("QA-PACKAGE", f"{relative} is a binary; baselines are source, except "
                                    "code objects an asm baseline declares in baseline.provenance.code_objects"))


def check_reference(workspace: Path, contract_raw: dict, findings: list[Finding]) -> None:
    reference = workspace / contract_raw["reference"]["file"]
    local = {p.stem for p in workspace.glob("*.py")}
    try:
        imports = _python_imports(reference)
    except (OSError, SyntaxError) as exc:
        findings.append(Finding("QA-REFERENCE", f"Cannot read the reference: {exc}"))
        return
    for module, line in imports:
        top = module.split(".")[0]
        if top in local and top != reference.stem:
            continue
        if top not in REFERENCE_MODULES or module.startswith("torch.ops") or module.startswith("<"):
            findings.append(Finding("QA-REFERENCE", f"reference imports {module} (line {line}); use PyTorch only",
                                    {"line": line}))


def check_provenance(workspace: Path, contract_raw: dict, findings: list[Finding]) -> None:
    provenance = contract_raw["baseline"]["provenance"]
    revision = str(provenance["revision"]).strip()
    if revision.lower() in MOVING_REVISIONS:
        findings.append(Finding("QA-PROVENANCE", f"revision {revision!r} is not a fixed commit or release"))
    for key in ("source", "license"):
        if str(provenance[key]).strip().lower() in MOVING_REVISIONS:
            findings.append(Finding("QA-PROVENANCE", f"provenance.{key} is a placeholder"))
    license_name = str(provenance["license"]).strip()
    has_license = any((workspace / "baseline").glob("LICENSE*"))
    if license_name not in ("Apache-2.0", "task-original") and not has_license:
        findings.append(Finding("QA-PROVENANCE", f"{license_name} baseline code needs baseline/LICENSE"))


def _actions(context: dict, role: str, action: str) -> list[dict]:
    rows = []
    for record in context.get("actions", []):
        result = record.get("result") or {}
        if result.get("role") == role and result.get("action") == action:
            rows.append(record)
    return rows


def check_initial(context: dict, contract_raw: dict | None, findings: list[Finding]) -> None:
    initial = context.get("initial_validation") or {}
    if not initial.get("accepted"):
        errors = initial.get("errors") or ["initial validation was not accepted"]
        findings.append(Finding("QA-INITIAL", "; ".join(str(e)[:600] for e in errors[:3])))
        return
    if contract_raw is None:
        return
    from src.task_contract import TARGET_LANGUAGES

    if contract_raw["baseline"]["language"] in TARGET_LANGUAGES:
        if initial.get("candidate_checks") != "PASS":
            findings.append(Finding("QA-POSITIVE", f"candidate checks: {initial.get('candidate_checks')}"))
    else:
        findings.append(Finding("QA-POSITIVE-UNAVAILABLE",
                                f"baseline language {contract_raw['baseline']['language']}"))
    for role in ("baseline", "candidate"):
        for record in _actions(context, role, "performance"):
            for case in record["result"].get("cases", []):
                method = case.get("benchmark_method")
                if method != "cuda_graph_cold_cache":
                    findings.append(Finding("QA-TIMING-METHOD", f"{role} {case.get('test_case_id')}: {method}"))
                if role == "baseline":
                    time_ms = case.get("execution_time_ms")
                    if not isinstance(time_ms, (int, float)) or time_ms < MIN_BASELINE_CASE_MS:
                        findings.append(Finding(
                            "QA-TIMING-FLOOR", f"{case.get('test_case_id')}: baseline {time_ms} ms < "
                            f"{MIN_BASELINE_CASE_MS} ms", {"case": case.get("test_case_id"), "ms": time_ms}))


def run_controls(workspace: Path, output: Path, timeout_s: int, findings: list[Finding]) -> dict | None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARENA_")}
    env.update(ARENA_EVAL_PHASE="task_validation", PYTHONDONTWRITEBYTECODE="1")
    python = os.environ.get("AGENT_KERNEL_ARENA_PYTHON") or sys.executable
    try:
        process = subprocess.run([python, "-B", "-m", "_aka_eval.controls", "--json", str(output)],
                                 cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        findings.append(Finding("QA-CONTROLS-RUN", f"control suite exceeded {timeout_s}s"))
        return None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        findings.append(Finding("QA-CONTROLS-RUN", f"control suite wrote no report (exit {process.returncode}): "
                                f"{process.stderr[-1500:]}"))
        return None
    if report.get("status") != "COMPLETE":
        findings.append(Finding("QA-CONTROLS-RUN", str(report.get("error"))[:1500]))
        return report
    for name in report.get("survived", []):
        findings.append(Finding("QA-CONTROLS", f"the {name} control passes correctness", {"control": name}))
    for item in report.get("input_findings", []):
        findings.append(Finding("QA-INPUTS", f"{item['input']} ({item['case']}): {item['issue']}", item))
    for name, dtype in (report.get("reference_dtypes") or {}).items():
        if dtype in ("bfloat16", "float16") or dtype.startswith("float8"):
            findings.append(Finding("QA-REFERENCE", f"reference returns {name} as {dtype}; use float32 or float64"))
    failed = [item for item in report.get("calibration_probe", []) if item.get("passed") is False]
    for name in sorted({item["output"] for item in failed}):
        cases = [item for item in failed if item["output"] == name]
        worst = max(cases, key=lambda item: (item.get("violation_fraction", 0), item.get("peak_bound_multiple", 0)))
        if worst.get("sqnr_db") is not None:
            measure = f"sqnr {worst['sqnr_db']:.4g} dB ({worst['peak_bound_multiple']:.3g}x the allowed noise)"
        else:
            measure = (f"violation fraction {worst['violation_fraction']:.3g}, "
                       f"peak {worst['peak_bound_multiple']:.3g}x, bias z {worst['bias_z_score']:.3g}")
        findings.append(Finding(
            "QA-BOUND-CALIBRATION",
            f"{name}: a float32 PyTorch evaluation of the reference fails the {worst.get('bound', '')} bound in "
            f"{len(cases)} case(s), e.g. {worst['case']}: {measure}",
            {"output": name, "cases": [item["case"] for item in cases]}))
    if report.get("baseline_nondeterministic_cases"):
        findings.append(Finding("QA-NONDETERMINISTIC",
                                f"cases {report['baseline_nondeterministic_cases'][:6]}"))
    return report


def check_warnings(workspace: Path, contract_raw: dict | None, findings: list[Finding]) -> None:
    if not (workspace / "README.md").is_file():
        findings.append(Finding("QA-README", "README.md is missing"))
    if contract_raw is None:
        return
    for name, declaration in contract_raw["compare"].items():
        overrides = {k: v for k, v in declaration.items() if k != "bound"}
        if declaration["bound"] != "baseline_relative" or overrides.get("k", 2) != 2 or \
                "baseline_min_sqnr_db" in overrides or overrides.get("max_violation_fraction", 0) > 0:
            findings.append(Finding("QA-BOUND-OVERRIDE", f"{name}: {declaration}"))
    arch = (os.environ.get("PYTORCH_ROCM_ARCH") or "").split(";")[0]
    claimed = (contract_raw.get("platform_support") or {}).get("required_arch") or []
    claimed = [claimed] if isinstance(claimed, str) else claimed
    unverified = [a for a in claimed if a != arch]
    if unverified:
        findings.append(Finding("QA-UNVERIFIED-ARCH", f"audited on {arch or 'unknown'}; unverified: {unverified}"))


def run_deterministic(workspace: Path, context: dict, state: Path, *, controls_timeout_s: int) -> tuple[list, dict]:
    """All deterministic findings and the raw evidence they came from."""
    findings: list[Finding] = []
    evidence: dict[str, Any] = {}
    config = context.get("task_config") or {}
    contract_raw = None
    if config.get("contract"):
        from src.task_contract import load_task_contract
        from src.task_spec import TaskConfigError

        try:
            contract_raw = load_task_contract(workspace / "task.yaml", task_id=context["task_id"]).raw
        except (TaskConfigError, OSError, ValueError) as exc:
            findings.append(Finding("QA-SPEC", str(exc)))
    check_initial(context, contract_raw, findings)
    if contract_raw is not None:
        check_package(workspace, findings)
        check_reference(workspace, contract_raw, findings)
        check_provenance(workspace, contract_raw, findings)
        if (context.get("initial_validation") or {}).get("accepted"):
            evidence["controls"] = run_controls(workspace, state / "qa_controls.json", controls_timeout_s, findings)
    check_warnings(workspace, contract_raw, findings)
    return findings, evidence


__all__ = ["BLOCKING", "WARNING", "Finding", "run_deterministic"]

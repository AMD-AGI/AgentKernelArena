"""Deterministic schema-v3 task emission from one SIKL definition.

The controller owns the operator interface, the reference, the vendored
baseline and every workload case derived from the bundle rows, including the
performance selection. The generator agent may repair only input
distributions (or inputs.py), README.md, and add correctness-only workload
cases; ``check_contract`` enforces that boundary. See docs/how-to/add-task.md.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import yaml

from agents.qa_auditor.deterministic import (
    FORBIDDEN_MODULES, MOVING_REVISIONS, PRODUCTION_PACKAGES, REFERENCE_MODULES,
)
from src.task_contract import (
    CATEGORIES, DTYPES, FLOAT_DTYPES, MAX_PERFORMANCE_CASES, MIN_PERFORMANCE_CASES, MIN_SIZE_SPAN,
    SCALAR_DTYPES, UNALIGNED_BLOCK, execution_config, load_task_contract, validate_task_contract,
)
from src.task_spec import TaskConfigError

from . import BUILDER_VERSION
from .bundle import ImportProblem, TaskSpec, source_files, solution_targets
from .config import Config

TASK_FILE = "task.yaml"
WORKLOADS = "workloads.json"
REFERENCE = "reference.py"
INPUTS = "inputs.py"
INITIALIZE = "sikl_initialize.py"
README = "README.md"
BASELINE_DIR = "baseline"
BASELINE_ENTRY = f"{BASELINE_DIR}/kernel.py"
GENERATOR_ENTRY = {"file": INPUTS, "symbol": "make_inputs"}
UNIMPLEMENTED_INPUTS = "SIKL-INPUTS-UNIMPLEMENTED"
BOUNDARY_KINDS = ("minimum", "non_power_of_two", "unaligned")
BOUNDARY_SOURCE = "AgentKernelArena boundary control; not a SIKL production row"
_DTYPE_ALIASES = {"float4_e2m1": "float4_e2m1fn_x2"}
_INTEGER_RANGES = {"bool": (0, 2), "int8": (-128, 128), "uint8": (0, 256), "int16": (-1024, 1024),
                   "int32": (0, 1024), "int64": (0, 1024)}
# Substring hints, in priority order (a MoE GEMM is a MoE operator).
_CATEGORY_HINTS = (
    ("moe", "moe"), ("attn", "attention"), ("attention", "attention"), ("mla", "attention"),
    ("gemm", "gemm"), ("matmul", "gemm"), ("bmm", "gemm"), ("gemv", "gemm"), ("norm", "normalization"),
    ("rope", "positional"), ("rotary", "positional"), ("quant", "quantization"), ("topk", "sorting"),
    ("sort", "sorting"), ("sampl", "sampling"), ("embed", "embedding"), ("conv", "convolution"),
    ("cache", "kv_cache"), ("silu", "activation"), ("gelu", "activation"), ("swiglu", "activation"),
    ("activation", "activation"), ("reduc", "reduction"), ("ssm", "state_space"), ("mamba", "state_space"),
)
_FORBIDDEN_CALLS = {"__import__", "eval", "exec", "compile", "open"}


def operator_name(definition: str) -> str:
    text = re.sub(r"\W+", "_", definition).strip("_").lower()
    return text if text and not text[0].isdigit() else "op_" + text


def operator_category(op_type) -> str:
    text = str(op_type or "").lower()
    if text in CATEGORIES:
        return text
    return next((category for hint, category in _CATEGORY_HINTS if hint in text), "other")


def _dtype(name, where: str) -> str:
    value = _DTYPE_ALIASES.get(name, name)
    if value not in DTYPES:
        raise ImportProblem("unsupported_dtype", f"{where}: dtype {name!r} is not a task-contract dtype")
    return value


def _default_distribution(dtype: str) -> dict:
    if dtype in FLOAT_DTYPES:
        return {"kind": "normal", "mean": 0.0, "std": 1.0}
    if dtype in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[dtype]
        return {"kind": "randint", "low": low, "high": high}
    return {"kind": "generator"}


def _imports(source: str, origin: str) -> list[tuple[str, int, int]]:
    """(module, relative level, line), plus ``module.name`` for from-imports, and forbidden calls."""
    try:
        tree = ast.parse(source, filename=origin)
    except SyntaxError as exc:
        raise ImportProblem("invalid_source", f"{origin}: {exc}") from exc
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, 0, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.append((module, node.level, node.lineno))
            found += [(f"{module}.{alias.name}".strip("."), node.level, node.lineno) for alias in node.names]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            found.append((f"<{node.func.id}()>", 0, node.lineno))
        elif (isinstance(node, ast.Attribute) and node.attr in ("ops", "_C")
              and isinstance(node.value, ast.Name) and node.value.id == "torch"):
            found.append((f"torch.{node.attr}", 0, node.lineno))
    return found


def _reject_dependencies(source: str, origin: str, *, allowed: frozenset[str] | None = None,
                         relative: bool = False) -> None:
    for module, level, line in _imports(source, origin):
        if level:
            if not relative:
                raise ImportProblem("unsupported_layout", f"{origin}:{line} uses a relative import")
            if level > 1:
                raise ImportProblem("unsupported_layout", f"{origin}:{line} imports above its flat package")
            continue
        top = module.split(".")[0]
        if top in PRODUCTION_PACKAGES:
            raise ImportProblem("unvendored_dependency",
                                f"{origin}:{line} imports {module}; vendor that code into the bundle")
        if top in FORBIDDEN_MODULES or module.startswith("<") or module in ("torch.ops", "torch._C"):
            raise ImportProblem("forbidden_dependency", f"{origin}:{line} uses {module}")
        if allowed is not None and top not in allowed:
            raise ImportProblem("reference_dependency",
                                f"{origin}:{line} imports {module}; a reference uses PyTorch and the standard library")


def baseline_language(solution: dict, files: dict[str, str]) -> str:
    declared = solution["spec"]["language"]
    if declared in ("triton", "flydsl"):
        return declared
    modules = {module for path, content in files.items() for module, level, _ in _imports(content, path)
               if not level}
    flydsl = any(m.split(".")[0] == "flydsl" for m in modules)
    triton = any(m.split(".")[0] == "triton" for m in modules)
    gluon = any(m.startswith("triton.experimental.gluon") for m in modules)
    if flydsl and not triton:
        return "flydsl"
    if gluon and not flydsl:
        return "gluon"
    if triton and not flydsl:
        return "triton"
    raise ImportProblem("unsupported_baseline_language",
                        f"Cannot resolve the kernel language of the {declared!r} baseline from its imports")


def _vendored_baseline(task: TaskSpec) -> tuple[dict[str, str], str, str, str]:
    files = source_files(task.baseline)
    for path in files:
        if "/" in path or path in ("__init__.py", "kernel.py"):
            raise ImportProblem("unsupported_layout",
                                f"baseline source {path}: a vendored baseline is one flat package beside "
                                "the generated kernel.py")
    for path, content in files.items():
        _reject_dependencies(content, f"baseline/{path}", relative=True)
    entry_file, symbol = task.baseline["spec"]["entry_point"].split("::")
    return files, baseline_language(task.baseline, files), entry_file, symbol


def _reference(task: TaskSpec) -> tuple[str, str]:
    files = source_files(task.reference)
    entry_file, symbol = task.reference["spec"]["entry_point"].split("::")
    if list(files) != [entry_file]:
        raise ImportProblem("unsupported_reference",
                            "The reference must be one self-contained source file; it becomes reference.py")
    content = files[entry_file]
    _reject_dependencies(content, f"reference {entry_file}", allowed=REFERENCE_MODULES)
    return content, symbol


def _license(task: TaskSpec, config: Config) -> str:
    candidates = (task.baseline.get("license"), task.baseline.get("spec", {}).get("license"), config.license)
    value = next((v.strip() for v in candidates if isinstance(v, str) and v.strip()), None)
    if value is None or value.lower() in MOVING_REVISIONS:
        raise ImportProblem("missing_license", "The baseline solution records no license; set `license` in the "
                            "builder configuration to the bundle's actual SPDX license")
    if value not in ("Apache-2.0", "task-original") and not task.license_text:
        raise ImportProblem("missing_license_text", f"A {value} baseline needs the bundle's LICENSE text")
    return value


def _satisfies(definition: dict, dimensions: dict[str, int]) -> bool:
    comparisons = {"<=": int.__le__, "<": int.__lt__, ">=": int.__ge__, ">": int.__gt__,
                   "==": int.__eq__, "!=": int.__ne__}
    for constraint in definition.get("constraints", []):
        left, op, right = re.fullmatch(r"\s*(\w+)\s*(<=|>=|==|!=|<|>)\s*(\w+)\s*", constraint).groups()
        values = [dimensions[v] if v in dimensions else int(v) for v in (left, right)]
        if not comparisons[op](*values):
            return False
    return True


def boundary_kinds(values: dict[str, int], axes: dict) -> set[str]:
    """The workload boundary classes one case covers, as the task contract counts them."""
    kinds = set()
    for name, value in values.items():
        step = axes[name].get("multiple_of", 1)
        units = value // step
        if value == axes[name]["min"]:
            kinds.add("minimum")
        if not (units > 0 and units & (units - 1) == 0):
            kinds.add("non_power_of_two")
        if units > UNALIGNED_BLOCK and units % UNALIGNED_BLOCK:
            kinds.add("unaligned")
    return kinds


def _distinct(groups: dict[str, set[str]]) -> bool:
    def search(index: int, used: frozenset[str]) -> bool:
        if index == len(BOUNDARY_KINDS):
            return True
        return any(search(index + 1, used | {case}) for case in sorted(groups[BOUNDARY_KINDS[index]])
                   if case not in used)
    return search(0, frozenset())


def _candidate_units(kind: str, low: int, high: int) -> list[int]:
    if kind == "minimum":
        return [low]
    if kind == "non_power_of_two":
        values = [u for u in range(max(low, 3), high + 1) if u & (u - 1) and u != low]
        return sorted(values, key=lambda u: (u > UNALIGNED_BLOCK, u))[:64]
    return [u for u in range(UNALIGNED_BLOCK + 1, high + 1) if u % UNALIGNED_BLOCK][:64]


def _case_id(raw: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", raw).strip(".") or "case"
    case_id, suffix = base, 2
    while case_id in used:
        case_id, suffix = f"{base}_{suffix}", suffix + 1
    used.add(case_id)
    return case_id


def _identity(case: dict) -> str:
    return json.dumps([case["axes"], case.get("scalars", {})], sort_keys=True)


def _add_boundary_cases(cases: list[dict], rows: int, axes: dict, var: list[str],
                        definition: dict, consts: dict, used: set[str]) -> int:
    """Add the fewest in-domain correctness-only controls the contract's boundary rule needs."""
    base = min(cases[:rows], key=lambda c: (math.prod(c["axes"].values()), cases.index(c)))
    added = 0
    for _ in range(len(BOUNDARY_KINDS)):
        groups = {kind: {c["id"] for c in cases if kind in boundary_kinds(c["axes"], axes)}
                  for kind in BOUNDARY_KINDS}
        if _distinct(groups):
            return added
        order = ("non_power_of_two", "unaligned", "minimum")
        missing = [kind for kind in order if not groups[kind]] or list(order)
        identities = {_identity(c) for c in cases}
        for kind in missing:
            case = None
            for name in var:
                step = axes[name].get("multiple_of", 1)
                for units in _candidate_units(kind, axes[name]["min"] // step, axes[name]["max"] // step):
                    values = {**base["axes"], name: units * step}
                    candidate = {"axes": values, **({"scalars": base["scalars"]} if "scalars" in base else {})}
                    if (_identity(candidate) not in identities
                            and _satisfies(definition, {**consts, **values})):
                        case = {"id": _case_id(f"boundary_{kind}", used), **candidate,
                                "checks": ["correctness"], "source": BOUNDARY_SOURCE}
                        break
                if case:
                    break
            if case:
                cases.append(case)
                added += 1
                break
        else:
            break
    groups = {kind: {c["id"] for c in cases if kind in boundary_kinds(c["axes"], axes)} for kind in BOUNDARY_KINDS}
    if not _distinct(groups):
        raise ImportProblem("needs_spec", "No in-domain cases satisfy the task contract's boundary rule "
                            f"(minimum, non-power-of-two, not a multiple of {UNALIGNED_BLOCK} units) "
                            "under the definition's constraints and the observed axis ranges")
    return added


def _workloads(task: TaskSpec, axes: dict, var: list[str], scalar_names: list[str]) -> tuple[dict, dict]:
    definition = task.definition
    consts = {k: a["value"] for k, a in axes.items() if a["kind"] == "const"}
    used, cases, merged, by_identity = set(), [], {}, {}
    for index, row in enumerate(task.rows, 1):
        workload = row["workload"]
        case = {"axes": {name: workload["axes"][name] for name in var}}
        if scalar_names:
            case["scalars"] = {name: workload["inputs"][name]["value"] for name in scalar_names}
        identity = _identity(case)
        if identity in by_identity:
            merged.setdefault(by_identity[identity], []).append(workload["uuid"])
            continue
        case = {"id": _case_id(workload["uuid"], used), **case, "checks": ["correctness"],
                "source": f"SIKL {task.origins['workload']} row {index} ({workload['uuid']})"}
        by_identity[identity] = case["id"]
        cases.append(case)
    for case in cases:
        if case["id"] in merged:
            uuids = merged[case["id"]]
            listed = ", ".join(uuids[:8]) + (f" and {len(uuids) - 8} more" if len(uuids) > 8 else "")
            case["source"] += f"; same axes and scalars as rows {listed}"
    sizes = [math.prod(case["axes"].values()) for case in cases]
    if len(cases) < MIN_PERFORMANCE_CASES:
        raise ImportProblem("needs_spec", f"{len(cases)} distinct workload shapes; a task needs at least "
                            f"{MIN_PERFORMANCE_CASES} performance cases from production rows")
    if max(sizes) < MIN_SIZE_SPAN * min(sizes):
        raise ImportProblem("needs_spec", f"Workload var-axis sizes span {min(sizes)}..{max(sizes)}; performance "
                            f"cases must span at least {MIN_SIZE_SPAN}x")
    order = sorted(range(len(cases)), key=lambda i: (sizes[i], i))
    count = min(len(cases), MAX_PERFORMANCE_CASES)
    chosen = {order[round(k * (len(cases) - 1) / (count - 1))] for k in range(count)}
    for index in chosen:
        cases[index]["checks"] = ["correctness", "performance"]
    rows = len(cases)
    added = _add_boundary_cases(cases, rows, axes, var, definition, consts, used)
    summary = {"distinct": rows, "merged": sum(len(v) for v in merged.values()), "performance": count,
               "boundary": added}
    description = (f"Every distinct axis/scalar combination of the SIKL rows in {task.origins['workload']}; "
                   f"{count} performance cases chosen by size rank, the rest correctness-only.")
    if added:
        description += f" {added} correctness-only boundary control(s) added by sikl_task_builder."
    return {"description": description, "cases": cases}, summary


def _interface(task: TaskSpec) -> tuple[dict, list[str], list[str], list[str]]:
    definition, rows = task.definition, task.rows
    axes, var = {}, []
    for name, axis in definition["axes"].items():
        if axis["type"] == "const":
            axes[name] = {"kind": "const", "value": axis["value"]}
        else:
            observed = [row["workload"]["axes"][name] for row in rows]
            step = math.gcd(*observed)
            axes[name] = {"kind": "var", "min": min(observed), "max": max(observed)}
            if step > 1:
                axes[name]["multiple_of"] = step
            var.append(name)
        if isinstance(axis.get("description"), str) and axis["description"].strip():
            axes[name]["description"] = axis["description"]
    inputs, scalars, owned = {}, [], []
    for name, spec in definition["inputs"].items():
        if spec.get("shape") is None:
            dtype = spec["dtype"]
            if dtype not in SCALAR_DTYPES:
                raise ImportProblem("unsupported_scalar", f"input {name}: scalar dtype {dtype!r} is not one of "
                                    f"{SCALAR_DTYPES}")
            inputs[name] = {"kind": "scalar", "dtype": dtype, "default": rows[0]["workload"]["inputs"][name]["value"]}
            scalars.append(name)
            continue
        dtype = _dtype(spec["dtype"], f"input {name}")
        distribution = ({"kind": "generator"} if definition.get("initialize")
                        else _default_distribution(dtype))
        inputs[name] = {"kind": "tensor", "shape": list(spec["shape"]), "dtype": dtype, "distribution": distribution}
        if distribution["kind"] == "generator":
            owned.append(name)
    for declaration, spec in zip(inputs.values(), definition["inputs"].values()):
        if isinstance(spec.get("description"), str) and spec["description"].strip():
            declaration["description"] = spec["description"]
    outputs = {name: {"shape": list(spec["shape"]), "dtype": _dtype(spec["dtype"], f"output {name}")}
               for name, spec in definition["outputs"].items()}
    interface = {"axes": axes, "inputs": inputs, "outputs": outputs}
    if owned:
        interface["generator"] = dict(GENERATOR_ENTRY)
    return interface, var, scalars, owned


def _baseline_wrapper(task: TaskSpec, entry_file: str, symbol: str, language: str) -> str:
    definition = task.definition
    inputs, outputs = list(definition["inputs"]), list(definition["outputs"])
    module = entry_file.removesuffix(".py")
    lines = [f'"""Arena run() interface for the vendored SIKL {language} baseline {task.baseline.get("name")}.',
             "",
             f"Calls {task.baseline['spec']['entry_point']} unchanged and copies the outputs it returns into",
             "the preallocated outputs. Generated by sikl_task_builder.",
             '"""',
             f"from .{module} import {symbol} as _sikl_baseline", "", "",
             f"def run(*, {', '.join(inputs + outputs)}):",
             f"    _result = _sikl_baseline({', '.join(f'{n}={n}' for n in inputs)})",
             "    if isinstance(_result, dict):",
             f"        _result = ({', '.join(f'_result[{n!r}]' for n in outputs)},)",
             "    elif not isinstance(_result, (tuple, list)):",
             "        _result = (_result,)",
             f"    if len(_result) != {len(outputs)}:",
             f"        raise ValueError('The SIKL baseline must return {len(outputs)} output(s)')"]
    lines += [f"    {name}.copy_(_result[{index}])" for index, name in enumerate(outputs)]
    return "\n".join(lines) + "\n"


def _initialize_inputs(definition: dict) -> str:
    rows = []
    for name, spec in definition["inputs"].items():
        if spec.get("shape") is None:
            rows.append(f"    ({name!r}, None, None),")
        else:
            rows.append(f"    ({name!r}, {list(spec['shape'])!r}, torch.{_dtype(spec['dtype'], name)}),")
    return "\n".join([
        '"""Inputs from the SIKL definition\'s initialize callback.',
        "",
        f"{INITIALIZE} holds the callback source unchanged. It fills buffers this wrapper",
        "allocates with the declared shapes and dtypes, in declaration order, using a seed drawn",
        "from the framework generator. Generated by sikl_task_builder.",
        '"""',
        "import torch",
        "",
        f"from {INITIALIZE.removesuffix('.py')} import run as _initialize",
        "",
        "_INPUTS = (", *rows, ")",
        "",
        "",
        "def make_inputs(case, inputs, generator, device):",
        "    axes = case['axes']",
        "    values = {}",
        "    for name, shape, dtype in _INPUTS:",
        "        if shape is None:",
        "            values[name] = inputs[name]",
        "        else:",
        "            values[name] = torch.empty([axes[d] if isinstance(d, str) else d for d in shape],",
        "                                       dtype=dtype, device=device)",
        "    buffers = {name: values[name] for name, shape, _ in _INPUTS if shape is not None}",
        "    seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator, device=device).item())",
        "    if _initialize(values, seed=seed) is not values:",
        "        raise ValueError('initialize must return the original input dictionary')",
        "    for name, buffer in buffers.items():",
        "        if values[name] is not buffer:",
        "            raise ValueError(f'initialize replaced input buffer {name}')",
        "    return buffers",
        "",
    ])


def _inputs_stub(owned: list[str]) -> str:
    return (f'"""Task-owned input generator. {UNIMPLEMENTED_INPUTS}\n\n'
            f"No built-in distribution fits {', '.join(owned)}. Implement make_inputs from the operator's\n"
            "semantics and return exactly those inputs; draw all randomness from ``generator``.\n"
            '"""\n\n\n'
            "def make_inputs(case, inputs, generator, device):\n"
            f"    raise NotImplementedError({UNIMPLEMENTED_INPUTS!r})\n")


def _readme(task: TaskSpec, config: Config, raw: dict, language: str, summary: dict) -> str:
    definition, origins = task.definition, task.origins
    if definition.get("initialize"):
        inputs = (f"Tensor inputs come from the definition's initialize callback ({INITIALIZE}, unchanged), "
                  f"wrapped by {INPUTS}.")
    else:
        inputs = ("The bundle records inputs only as random; their distributions are a synthesis policy, not "
                  "recovered production data. Review them against the operator's semantics.")
    compare = ("The definition's compare callback is not carried over; outputs use the task contract's "
               "bounds (exact for non-floating outputs, baseline_relative otherwise)."
               if definition.get("compare") else
               "Outputs use the task contract's bounds: exact for non-floating outputs, baseline_relative otherwise.")
    return "\n".join([
        f"# {task.task_id}", "", raw["description"], "",
        f"- Operator type `{definition['op_type']}`, category `{raw['operator']['category']}`.",
        f"- Source: SIKL bundle `{Path(config.input_dir).name}`; definition `{origins['definition']}`, "
        f"workloads `{origins['workload']}`, baseline `{origins['baseline']}`, reference `{origins['reference']}`.",
        f"- Baseline: {language} solution `{task.baseline.get('name')}`, vendored unchanged under `baseline/`; "
        "`baseline/kernel.py` adapts its return-value entry point to the in-place `run()` interface.",
        "- Reference: `reference.py` is the bundle's reference source, unchanged.",
        f"- Workloads: {summary['distinct']} distinct rows ({summary['merged']} duplicate rows merged), "
        f"{summary['performance']} performance cases chosen by size rank, {summary['boundary']} boundary "
        "correctness control(s) added by the builder.",
        f"- Inputs: {inputs}", f"- Comparison: {compare}",
        f"- Generated by sikl_task_builder version {BUILDER_VERSION}.", "",
    ])


def task_files(task: TaskSpec, config: Config) -> dict[str, str]:
    from src.preprocessing import _resolve_gfx_arch
    arch = _resolve_gfx_arch(config.target_gpu_model)
    if not arch:
        raise ImportProblem("unsupported_platform", config.target_gpu_model)
    aliases = {arch.lower(), config.target_gpu_model.lower()}
    for role, solution in (("baseline", task.baseline), ("reference", task.reference)):
        targets = solution_targets(solution)
        if not any(t["arch"] == arch if "arch" in t else t["hardware_id"].lower() in aliases for t in targets):
            raise ImportProblem("platform_deferred", f"{role} hardware {targets} does not match {config.target_gpu_model}")
    sources, language, entry_file, symbol = _vendored_baseline(task)
    reference, reference_symbol = _reference(task)
    license_name = _license(task, config)
    interface, var, scalars, owned = _interface(task)
    workloads, summary = _workloads(task, interface["axes"], var, scalars)
    definition = task.definition
    description = (definition.get("description") or "").strip() or (
        f"The {task.task_id} operator ({definition['op_type']}) from a SIKL bundle.")
    raw = {
        "schema_version": 3,
        "description": " ".join(description.split()) + " Inputs are read-only; every output is written in place.",
        "operator": {"name": operator_name(task.task_id), "category": operator_category(definition["op_type"])},
        "interface": interface,
        "reference": {"file": REFERENCE, "symbol": reference_symbol},
        "baseline": {"language": language, "entry": BASELINE_ENTRY, "provenance": {
            "source": f"SIKL bundle {Path(config.input_dir).name}, solution {task.baseline.get('name')}",
            "revision": f"sha256:{task.digest}",
            "path": [task.origins["baseline"], task.origins["definition"]],
            "license": license_name,
            "modifications": "Vendored unchanged; baseline/kernel.py adapts the return-value entry point to "
                             "the in-place run() interface.",
        }},
        "compare": {name: {"bound": "baseline_relative" if spec["dtype"] in FLOAT_DTYPES else "exact"}
                    for name, spec in interface["outputs"].items()},
        "workloads": WORKLOADS,
        "platform_support": {"required_arch": [arch]},
        "timeout_s": max(60, config.command_timeout),
    }
    try:
        validate_task_contract(deepcopy(raw), deepcopy(workloads), task_id=config.arena_task_id(task.task_id))
    except TaskConfigError as exc:
        raise ImportProblem("invalid_task_contract", str(exc)) from exc
    files = {TASK_FILE: yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
             WORKLOADS: json.dumps(workloads, indent=2) + "\n",
             REFERENCE: reference,
             BASELINE_ENTRY: _baseline_wrapper(task, entry_file, symbol, language),
             README: _readme(task, config, raw, language, summary)}
    for path, content in sources.items():
        files[f"{BASELINE_DIR}/{path}"] = content
    if license_name not in ("Apache-2.0", "task-original"):
        files[f"{BASELINE_DIR}/LICENSE"] = task.license_text
    if definition.get("initialize"):
        _reject_dependencies(definition["initialize"], "initialize callback")
        files[INITIALIZE] = definition["initialize"]
        files[INPUTS] = _initialize_inputs(definition)
    elif owned:
        files[INPUTS] = _inputs_stub(owned)
    return files


def authoring_editable(task: TaskSpec) -> list[str]:
    """What the generator agent may change; everything else is the controller's emission."""
    scope = [README, f"{WORKLOADS}: add correctness-only cases (rows-derived cases are fixed)"]
    # A supplied initializer owns the distribution and cannot be repaired by
    # choosing an easier input population after a numerical failure.
    if not task.definition.get("initialize"):
        scope += [INPUTS, f"{TASK_FILE}: interface.inputs.<tensor>.distribution and interface.generator "
                          f"({GENERATOR_ENTRY['file']}::{GENERATOR_ENTRY['symbol']})"]
    return scope


def materialize_task(task: TaskSpec, config: Config, destination: Path) -> dict:
    files = task_files(task, config)
    if destination.exists():
        raise ImportProblem("destination_exists", f"Refusing to replace existing draft: {destination}")
    destination.mkdir(parents=True)
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return {"task_id": task.task_id, "files": sorted(files), "editable": authoring_editable(task)}


def task_tree(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ImportProblem("symlink", f"Symlink in task: {relative}")
        if "__pycache__" in relative.parts or relative.parts[0] == "build":
            continue
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def task_digest(root: Path) -> str:
    from .bundle import fingerprint
    return fingerprint(task_tree(root))


def _masked(raw, initialize: bool):
    value = deepcopy(raw)
    interface = value.get("interface") if isinstance(value, dict) else None
    if not initialize and isinstance(interface, dict):
        interface.pop("generator", None)
        for declaration in (interface.get("inputs") or {}).values():
            if isinstance(declaration, dict) and declaration.get("kind") == "tensor":
                declaration.pop("distribution", None)
    return value


def _check_task_yaml(expected: str, path: Path, initialize: bool, diagnostics: list) -> dict | None:
    try:
        actual = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        diagnostics.append({"code": "invalid_yaml", "path": TASK_FILE, "message": str(exc)})
        return None
    if _masked(actual, initialize) != _masked(yaml.safe_load(expected), initialize):
        diagnostics.append({"code": "contract_changed", "path": TASK_FILE,
                            "message": "Only tensor-input distributions and interface.generator are editable"})
    generator = ((actual or {}).get("interface") or {}).get("generator") if isinstance(actual, dict) else None
    if generator is not None and generator != GENERATOR_ENTRY:
        diagnostics.append({"code": "contract_changed", "path": TASK_FILE,
                            "message": f"interface.generator must be {GENERATOR_ENTRY}"})
    return actual


def _check_workloads(expected: str, path: Path, diagnostics: list) -> None:
    try:
        actual = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        diagnostics.append({"code": "invalid_json", "path": WORKLOADS, "message": str(exc)})
        return
    wanted = json.loads(expected)
    if (not isinstance(actual, dict) or set(actual) != {"description", "cases"}
            or actual["description"] != wanted["description"] or not isinstance(actual["cases"], list)):
        diagnostics.append({"code": "contract_changed", "path": WORKLOADS,
                            "message": "Keep the controller's description and case list"})
        return
    cases = {case.get("id"): case for case in actual["cases"] if isinstance(case, dict)}
    for case in wanted["cases"]:
        if cases.get(case["id"]) != case:
            diagnostics.append({"code": "contract_changed", "path": WORKLOADS,
                                "message": f"Controller case {case['id']} was removed or changed"})
    known = {case["id"] for case in wanted["cases"]}
    for case in actual["cases"]:
        if isinstance(case, dict) and case.get("id") not in known and case.get("checks") != ["correctness"]:
            diagnostics.append({"code": "scored_case_added", "path": WORKLOADS,
                                "message": f"Added case {case.get('id')} must be correctness-only"})


def check_contract(task: TaskSpec, config: Config, draft: Path) -> dict:
    try:
        expected = task_files(task, config)
    except ImportProblem as exc:
        return {"ok": False, "task_id": task.task_id, "task_digest": None,
                "diagnostics": [{"code": exc.code, "message": str(exc)}]}
    initialize = bool(task.definition.get("initialize"))
    editable = {README} if initialize else {README, INPUTS}
    actual = task_tree(draft)
    diagnostics = []
    raw = None
    for relative, content in expected.items():
        path = draft / relative
        if not path.is_file():
            if relative != INPUTS or initialize:
                diagnostics.append({"code": "missing_file", "path": relative})
        elif relative == TASK_FILE:
            raw = _check_task_yaml(content, path, initialize, diagnostics)
        elif relative == WORKLOADS:
            _check_workloads(content, path, diagnostics)
        elif relative not in editable and path.read_text() != content:
            diagnostics.append({"code": "contract_changed", "path": relative})
    for relative in actual:
        if relative not in expected and relative not in editable:
            diagnostics.append({"code": "undeclared_file", "path": relative})
    for path in sorted(draft.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(draft).as_posix()
        try:
            ast.parse(path.read_text(), filename=relative)
        except SyntaxError as exc:
            diagnostics.append({"code": "syntax_error", "path": relative, "message": str(exc)})
            continue
        if relative == INPUTS and not initialize:
            try:
                _reject_dependencies(path.read_text(), INPUTS)
            except ImportProblem as exc:
                diagnostics.append({"code": exc.code, "path": INPUTS, "message": str(exc)})
            if UNIMPLEMENTED_INPUTS in path.read_text():
                diagnostics.append({"code": "unimplemented_inputs", "path": INPUTS,
                                    "message": "Implement make_inputs for the generator-owned inputs"})
    generator = ((raw or {}).get("interface") or {}).get("generator") if isinstance(raw, dict) else None
    if generator is not None and not (draft / INPUTS).is_file():
        diagnostics.append({"code": "missing_file", "path": INPUTS})
    if not diagnostics:
        try:
            task_id = config.arena_task_id(task.task_id)
            load_task_contract(draft / TASK_FILE, task_id=task_id)
            execution_config(draft / TASK_FILE, task_id=task_id, target_language="triton")
        except TaskConfigError as exc:
            diagnostics.append({"code": "invalid_task_contract", "path": TASK_FILE, "message": str(exc)})
    return {"ok": not diagnostics, "task_id": task.task_id,
            "task_digest": task_digest(draft), "diagnostics": diagnostics}

"""Self-contained task action protocol; Arena freezes/separates role workspaces."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent
PROTOCOL = "arena-eval-v1"
PREFIX = "ARENA_EVAL_RESULT="
METHODS = {"cuda_graph", "cuda_event_fallback"}


def local_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a normalized task-relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in (".", "..", "") for part in value.split("/")):
        raise ValueError(f"Invalid task-relative path: {value}")
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Path escapes task workspace: {value}")
    return result


def config():
    import yaml
    with (ROOT / "config.yaml").open() as f:
        data = yaml.safe_load(f)
    if data.get("schema_version") != 2:
        raise ValueError("This runner requires schema_version 2")
    return data


def candidate_files(cfg=None):
    cfg = config() if cfg is None else cfg
    files = []
    for item in cfg["candidate"]["editable"]:
        path = local_path(item if isinstance(item, str) else item["path"])
        if isinstance(item, dict) and item.get("scope") == "tree":
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)
    return files


def candidate_relative_path():
    cfg = config()
    entries = cfg["candidate"].get("entrypoints", [])
    p = local_path(entries[0]["file"]) if entries else candidate_files(cfg)[0]
    return p.relative_to(ROOT).as_posix()


def is_stub(fn):
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    if not body:
        return True
    # Only an unconditional starter is unimplemented; conditional error paths
    # inside a real implementation must not be mistaken for an empty candidate.
    first = body[0]
    return isinstance(first, ast.Pass) or (isinstance(first, ast.Raise) and
        isinstance(first.exc, (ast.Name, ast.Call)) and
        (first.exc.id if isinstance(first.exc, ast.Name) else getattr(first.exc.func, "id", "")) == "NotImplementedError")


def source_state(cfg):
    states = []
    trees = {}
    for entry in cfg["candidate"].get("entrypoints", []):
        path = local_path(entry["file"])
        if not path.is_file():
            states.append(False)
            continue
        if path not in trees:
            trees[path] = ast.parse(path.read_text(), filename=str(path))
        symbol = entry.get("symbol")
        found = next((n for n in trees[path].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == symbol), None)
        states.append(found is not None and not is_stub(found))
    if not states:
        raise ValueError("Task requires explicit candidate entrypoints")
    return "implemented" if any(states) else "unimplemented", states


def check_dependencies(paths, final_language=True):
    """Enforce declared implementation dependencies, including from X import Y."""
    forbidden = {"src", "agents", "model", "test_kernel_harness", "task_runtime", "task_reference", "task_baseline", "reference_controls", "scripts"}
    backend_seen = False
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = {}
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [a.name for a in node.names]
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = [module] + [f"{module}.{a.name}" for a in node.names]
                for a in node.names:
                    aliases[a.asname or a.name] = f"{module}.{a.name}"
            for module in imported:
                parts = set(module.split("."))
                if parts & forbidden:
                    raise ValueError(f"Protected dependency in candidate: {module}")
                if final_language and module.split(".")[0] in {"triton", "cupy", "numba", "aiter", "ctypes", "subprocess"}:
                    raise ValueError(f"Final operator must execute FlyDSL, not {module}")
                backend_seen |= module == "flydsl" or module.startswith("flydsl.")
        def dotted(node):
            if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute): return dotted(node.value) + "." + node.attr
            return ""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            name = dotted(node.func)
            if final_language and (name in {"torch.mm", "torch.bmm", "torch.matmul", "torch.einsum", "torch.softmax", "torch.log_softmax", "torch.layer_norm", "torch.rms_norm"} or name.startswith("torch.nn.functional.")):
                raise ValueError(f"Library operator shortcut in candidate: {name}")
            if name in {"eval", "exec", "__import__", "importlib.import_module", "importlib.util.spec_from_file_location"}:
                raise ValueError(f"Dynamic implementation loading is not allowed: {name}")
    if final_language and not backend_seen:
        raise ValueError("Candidate does not declare a FlyDSL implementation dependency")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manifest():
    path = local_path(config()["evaluation"]["workloads"])
    rows = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))["cases"]
    ids = set()
    for row in rows:
        if row["test_case_id"] in ids: raise ValueError("Duplicate manifest case")
        ids.add(row["test_case_id"])
        if "correctness" not in row["checks"]: raise ValueError("Every case needs correctness coverage")
    if not rows or not any("performance" in r["checks"] for r in rows):
        raise ValueError("Empty correctness/performance manifest")
    json.dumps(rows, allow_nan=False)
    return rows


def require_result_rows(records, expected, mapping):
    if not isinstance(records, list): raise ValueError("Benchmark did not return case evidence")
    by_id = {}
    for record in records:
        case_id = mapping.get(record.get("test_case_id"))
        if case_id is None or case_id in by_id: raise ValueError("Unexpected/duplicate benchmark case")
        latency = record.get("execution_time_ms")
        if type(latency) not in (float, int) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid device latency")
        if record.get("benchmark_method") not in METHODS: raise ValueError("Unsupported timing method")
        by_id[case_id] = record
    if set(by_id) != {r["test_case_id"] for r in expected}: raise ValueError("Incomplete benchmark case coverage")
    result = []
    for case in expected:
        raw = by_id[case["test_case_id"]]
        row = {k: copy.deepcopy(v) for k,v in case.items() if k != "checks"}
        row.update(status="PASS", execution_time_ms=raw["execution_time_ms"], benchmark_method=raw["benchmark_method"])
        # Shape/params in raw legacy rows omit semantic fields in some tasks;
        # the independent protected manifest is authoritative for identity.
        row["metadata"] = {k:v for k,v in raw.items() if k not in {"test_case_id", "shape", "params", "dtype", "execution_time_ms", "benchmark_method"}}
        result.append(row)
    return result


def check_flydsl_execution(fn):
    """Record actual FlyDSL runtime calls outside timed regions.

    This is implementation evidence alongside import restrictions and numerical
    checks, not a security sandbox or a proof against arbitrary hostile Python.
    """
    seen = set()
    prior = sys.getprofile()
    def profile(frame, event, arg):
        if event == "call" and frame.f_globals.get("__name__", "").startswith("flydsl."):
            name = frame.f_code.co_name
            owner = type(frame.f_locals.get("self")).__name__
            if name == "__call__" and any(s in owner for s in ("Jit", "Compiled", "Kernel")):
                seen.add(frame.f_globals["__name__"] + "." + owner)
    sys.setprofile(profile)
    try: fn()
    finally: sys.setprofile(prior)
    if not seen: raise RuntimeError("No FlyDSL kernel runtime invocation observed during correctness")
    return sorted(seen)


def run(argv):
    role, action = ("task", "validate-task") if argv == ["validate-task"] else (argv + [""] * 2)[:2]
    report = {"protocol":PROTOCOL,"role":role,"action":action,"status":"FAIL","cases":[]}
    try:
        if (role,action) not in {("task","validate-task")} | {(r,a) for r in ("baseline","candidate") for a in ("compile","correctness","performance")}:
            raise ValueError("Expected validate-task or baseline|candidate compile|correctness|performance")
        cfg = config(); cases = manifest()
        selected = cases if action == "validate-task" else [r for r in cases if action in r["checks"]]
        report["cases"] = [dict(copy.deepcopy(r), status="FAIL") for r in selected]
        for row in report["cases"]:
            if action != "validate-task": row.pop("checks",None)
        provided = role == "baseline" and cfg["baseline"]["kind"] == "provided"
        state, defined = (None, []) if provided else source_state(cfg)
        if role == "task":
            report["metadata"] = {"candidate_state":state}
            if state != cfg["candidate"]["initial_state"]: raise ValueError("Actual candidate state disagrees with config")
            controls = load_module("arena_reference_controls", ROOT/"scripts/reference_controls.py")
            report["metadata"]["reference_controls"] = controls.run()
        else:
            provided = role == "baseline" and cfg["baseline"]["kind"] == "provided"
            if not provided and (state != "implemented" or not all(defined)):
                raise ValueError("Missing or unimplemented declared candidate entrypoint; no baseline fallback")
            initial_phase = __import__("os").environ.get("ARENA_EVAL_PHASE") == "task_validation"
            final_language = role == "candidate" and not initial_phase or cfg["candidate"].get("initial_language") == "flydsl"
            paths = [local_path(p) for p in cfg["baseline"].get("source_files", [])] if provided else candidate_files(cfg)
            if not provided: check_dependencies(paths, final_language)
            if action == "compile":
                for path in paths:
                    if path.suffix == ".py": compile(path.read_text(), str(path), "exec")
                if not paths: raise ValueError("No source to compile")
                report["metadata"] = {"compile_kind":"python_bytecode", "sources":[str(p.relative_to(ROOT)) for p in paths], "jit":"case-specific GPU compilation is exercised by correctness"}
            else:
                actions = load_module("arena_task_actions",ROOT/"scripts/task_actions.py")
                h = load_module("arena_harness",ROOT/"test_kernel_harness.py")
                # Each task explicitly binds provided baseline or working candidate.
                if hasattr(actions,"select_role"): actions.select_role(h, role, provided)
                if final_language and not provided:
                    # Task binding audits each candidate operator invocation;
                    # reference/baseline launches cannot satisfy this evidence.
                    observed = actions.check(h)
                    if not observed:
                        raise RuntimeError("No FlyDSL candidate operator invocation observed")
                    report["metadata"] = {"flydsl_runtime_invocations":observed,
                                          "flydsl_evidence_scope":"candidate_operator_calls"}
                else: actions.check(h)
                if action == "performance":
                    report["cases"] = require_result_rows(actions.performance(h),selected,actions.PERFORMANCE_IDS)
        report["status"] = "PASS"
        for row in report["cases"]: row["status"] = "PASS"
        json.dumps(report, allow_nan=False)
    except (Exception, SystemExit) as exc:
        report.update(status="FAIL",reason=f"{type(exc).__name__}: {exc}")
        # A caught harness assertion can include a launch/runtime failure. Never
        # label it numerical_mismatch without separately attested numeric evidence.
        for row in report["cases"]: row.update(status="FAIL",reason=report["reason"])
    try:
        encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    except (ValueError,TypeError) as exc:
        report = {"protocol":PROTOCOL,"role":role,"action":action,"status":"FAIL","cases":[],"reason":f"Non-JSON evidence: {exc}"}
        encoded = json.dumps(report, allow_nan=False)
    print(PREFIX + encoded, flush=True)
    return 0 if report["status"] == "PASS" else 1

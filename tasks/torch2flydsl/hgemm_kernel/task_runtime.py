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


def _allowed_aiter_helpers():
    """Read task-owned exceptions; never grant access to an operator namespace."""
    path = local_path("scripts/dependency_policy.json")
    if not path.exists():
        return {}
    policy = json.loads(path.read_text())
    if not isinstance(policy, dict) or set(policy) != {"version", "helpers"} or type(policy["version"]) is not int or policy["version"] != 1:
        raise ValueError("Invalid protected candidate dependency policy")
    if not isinstance(policy["helpers"], list):
        raise ValueError("Dependency policy helpers must be a list")
    allowed = {}
    for item in policy["helpers"]:
        if not isinstance(item, dict) or set(item) != {"symbol", "file", "function"}:
            raise ValueError("Expected an exact helper symbol, candidate file and function")
        symbol, function = item["symbol"], item["function"]
        if (not isinstance(symbol, str) or not symbol.startswith("aiter.")
                or not all(part.isidentifier() for part in symbol.split("."))
                or not isinstance(function, str) or not function.isidentifier()):
            raise ValueError("Invalid candidate helper identity")
        source = local_path(item["file"])
        allowed.setdefault(symbol, set()).add((source, function))
    return allowed


def check_candidate_control_plane(tree):
    """Reject direct module-registry/control-frame access in candidate sources.

    This applies to the normal task-local source closure, not installed FlyDSL
    internals. Ordinary sys metadata stays available. These structural checks
    supplement protected imports; they are not a Python security sandbox.
    """
    blocked = {"modules", "_getframe", "_current_frames", "getprofile", "setprofile",
               "gettrace", "settrace", "exc_info", "exception", "_current_exceptions",
               "monitoring", "__dict__", "__getattribute__"}
    frames = {"f_globals", "f_locals", "f_back", "f_builtins", "gi_frame", "cr_frame",
              "ag_frame", "tb_frame", "__globals__", "__builtins__"}
    dispatch = {"_python_dispatch", "_disable_current_modes", "_DisableTorchDispatch",
                "_DisablePythonDispatcher", "_pop_mode", "_pop_mode_temporarily",
                "_unset_dispatch_mode"}
    # Each name maps to every sys-derived value it may hold, so reassignment
    # cannot hide a registry alias and the fixed point is reached monotonically.
    aliases, modules = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".")[0]
                if item.name in {"sys", "os"}:
                    aliases.setdefault(item.asname or item.name, set()).add(item.name)
                if any(part in dispatch for part in item.name.split(".")):
                    raise ValueError("Candidate cannot disable the evaluator dispatch audit")
                modules.add(item.asname or root)
        elif isinstance(node, ast.ImportFrom) and node.module in {"sys", "os"}:
            for item in node.names:
                if item.name == "*" or (node.module == "sys" and item.name in blocked):
                    raise ValueError("Candidate cannot access the evaluator module registry/control frames")
                value = "sys" if node.module == "os" and item.name == "sys" else node.module + "." + item.name
                aliases.setdefault(item.asname or item.name, set()).add(value)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if any(part in dispatch for part in node.module.split(".")) or \
                    any(item.name in dispatch for item in node.names):
                raise ValueError("Candidate cannot disable the evaluator dispatch audit")
    def names(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id) or {node.id}
        if isinstance(node, ast.Attribute):
            values = {base + "." + node.attr for base in names(node.value)}
            return {"sys" if value == "os.sys" else value for value in values}
        return {""}
    def tracked(value):
        return value in {"sys", "os"} or value.startswith(("sys.", "os."))
    for _ in range(len(list(ast.walk(tree)))):
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                values = {value for value in names(node.value) if tracked(value)}
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and values - aliases.get(target.id, set()):
                        aliases.setdefault(target.id, set()).update(values)
                        changed = True
        if not changed:
            break
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        # A sys object cannot be passed to a helper/container that would hide a
        # later registry lookup. Ordinary name aliases remain statically tracked.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and "sys" in names(node):
            parent = parents.get(node)
            attribute = isinstance(parent, ast.Attribute) and parent.value is node
            alias = ((isinstance(parent, ast.Assign) and parent.value is node and
                      all(isinstance(t, ast.Name) for t in parent.targets)) or
                     (isinstance(parent, ast.AnnAssign) and parent.value is node and
                      isinstance(parent.target, ast.Name)))
            if not (attribute or alias):
                raise ValueError("Candidate cannot pass the evaluator module registry/control frames through a sys object")
        if any(value.startswith("sys.") and value.split(".")[1] in blocked for value in names(node)):
            raise ValueError("Candidate cannot access the evaluator module registry/control frames")
        if isinstance(node, ast.Attribute) and node.attr in frames:
            raise ValueError("Candidate cannot access evaluator frames or function globals")
        if isinstance(node, ast.Attribute) and node.attr in dispatch:
            raise ValueError("Candidate cannot disable the evaluator dispatch audit")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target])
            for target in targets:
                base = target
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if base is not target and isinstance(base, ast.Name) and base.id in modules:
                    raise ValueError("Candidate cannot mutate an imported module's state")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"getattr", "vars"} and node.args and "sys" in names(node.args[0]):
                raise ValueError("Dynamic sys attribute lookup is not a candidate implementation dependency")
            if node.func.id in {"getattr", "setattr", "delattr"} and len(node.args) >= 2 and \
                    isinstance(node.args[1], ast.Constant) and node.args[1].value in frames | dispatch:
                raise ValueError("Candidate cannot access evaluator frames or function globals")
            if node.func.id in {"setattr", "delattr"} and node.args:
                base = node.args[0]
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in modules:
                    raise ValueError("Candidate cannot mutate an imported module's state")


def candidate_source_closure(paths):
    """Collect normal task-local Python imports without importing candidate code.

    Installed libraries are not traversed. This supplements the existing static
    dependency and runtime-call checks; it is not a Python security sandbox.
    """
    root = ROOT.resolve(strict=True)
    sources, queued = [], set()
    protected = {"src", "agents", "model", "test_kernel_harness", "task_runtime",
                 "task_reference", "task_baseline", "reference_controls", "scripts",
                 "_aka_benchmark"}
    reserved = (set(sys.builtin_module_names) | set(sys.stdlib_module_names) |
                {"torch", "flydsl", "triton", "aiter", "numpy", "typing_extensions"})

    def checked(path, directory=False):
        path = Path(path).absolute()
        if not path.is_relative_to(root):
            raise ValueError(f"Candidate source escapes task workspace: {path}")
        current = root
        for part in path.relative_to(root).parts:
            if part in {".", ".."}:
                raise ValueError("Candidate source has non-normalized components")
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Candidate source symlink is not allowed: {current}")
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Candidate source escapes task workspace")
        if not (path.is_dir() if directory else path.is_file() and path.suffix == ".py"):
            raise ValueError(f"Candidate Python source is missing or invalid: {path}")
        return path

    def add(path):
        path = checked(path)
        if path not in queued:
            queued.add(path)
            sources.append(path)

    def resolve(parts, *, optional=False):
        if not parts:
            return
        if set(parts) & protected:
            raise ValueError(f"Protected dependency in candidate: {'.'.join(parts)}")
        current, local = root, False
        for index, name in enumerate(parts):
            directory, module = current / name, current / (name + ".py")
            initializer = directory / "__init__.py"
            package = initializer.exists() or initializer.is_symlink()
            # Python's path finder prefers a regular package, then a same-name
            # module, and only then a namespace-package directory.
            if not package and (module.exists() or module.is_symlink()):
                if index == 0 and name in reserved:
                    raise ValueError(f"Candidate shadows library module: {name}")
                add(module)
                if index != len(parts) - 1 and not optional:
                    raise ValueError(f"Local module is not a package: {'.'.join(parts)}")
                return
            elif directory.exists() or directory.is_symlink():
                checked(directory, directory=True)
                if index == 0 and name in reserved:
                    raise ValueError(f"Candidate shadows library module: {name}")
                local = True
                if package:
                    add(initializer)
                current = directory
            else:
                if local and not optional:
                    raise ValueError(f"Missing task-local import: {'.'.join(parts)}")
                return  # External installed dependency, or a from-import value.

    for path in paths:
        add(path)
    index = 0
    while index < len(sources):
        path = sources[index]
        index += 1
        tree = ast.parse(path.read_text(), filename=str(path))
        package = list(path.relative_to(root).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError("Relative candidate import escapes its task package")
                    base = package[:len(package) - node.level + 1]
                else:
                    base = []
                parts = base + (node.module.split(".") if node.module else [])
                resolve(parts, optional=not node.level)
                for alias in node.names:
                    if alias.name != "*":
                        resolve(parts + alias.name.split("."), optional=True)
    return sources


def check_dependencies(paths, final_language=True):
    """Reject external operator delegation before importing candidate code.

    Final FlyDSL arithmetic cannot be supplied by AITER (including its FlyDSL
    kernels), other backend operators or native launch shortcuts. An existing
    task-owned helper exception permits only an exact named import and direct
    calls in the declared function; it never exposes an AITER module object.
    Initial Triton / provided baseline evaluation keeps its original backend.
    """
    paths = candidate_source_closure(paths)
    forbidden = {"src", "agents", "model", "test_kernel_harness", "task_runtime", "task_reference", "task_baseline", "reference_controls", "scripts", "_aka_benchmark", "inspect", "gc", "builtins", "importlib"}
    external = {"triton", "cupy", "numba", "aiter", "ctypes", "subprocess"}
    loaders = {"eval", "exec", "__import__", "builtins.eval", "builtins.exec", "builtins.__import__",
               "importlib.import_module", "importlib.util.spec_from_file_location"}
    native_dispatch = ("torch.ops", "torch.classes", "torch.utils.cpp_extension")
    helpers = _allowed_aiter_helpers() if final_language else {}
    backend_seen = False
    for path in paths:
        path = Path(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        if final_language:
            check_candidate_control_plane(tree)
        aliases = {}
        helper_bindings = []
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def enclosing_function(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return node
            return None

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
                owner = enclosing_function(node)
                if (final_language and node.level == 0 and module.startswith("aiter.") and owner is not None
                        and all((path.resolve(), owner.name) in helpers.get(f"{module}.{a.name}", set()) for a in node.names)):
                    helper_bindings.extend((a.asname or a.name, owner) for a in node.names)
                    imported = []
            for module in imported:
                parts = set(module.split("."))
                if parts & forbidden:
                    raise ValueError(f"Protected dependency in candidate: {module}")
                if final_language and module.lstrip(".").split(".")[0] in external:
                    raise ValueError(f"Final operator must execute FlyDSL, not {module}")
                if final_language and any(module == prefix or module.startswith(prefix + ".") for prefix in native_dispatch):
                    raise ValueError(f"External native operator dispatch in candidate: {module}")
                backend_seen |= module == "flydsl" or module.startswith("flydsl.")

        # An allowed helper must not become an indirect route to its globals,
        # module, or another operator. Import aliases are supported; exporting
        # the callable, introspecting it, and passing it elsewhere are not.
        for name, owner in helper_bindings:
            for node in ast.walk(owner):
                if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                    raise ValueError("AITER helper binding must remain local to its declared task function")
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name:
                    parent = parents.get(node)
                    if not (isinstance(parent, ast.Call) and parent.func is node and enclosing_function(node) is owner):
                        raise ValueError("AITER helper must be called directly in its declared task function")

        def dotted(node):
            if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute): return dotted(node.value) + "." + node.attr
            return ""

        # Track simple aliases of imported objects before candidate import, so
        # t = torch cannot turn a module mutation into an apparently local write.
        for _ in range(len(list(ast.walk(tree)))):
            changed = False
            for binding in ast.walk(tree):
                if not isinstance(binding, (ast.Assign, ast.AnnAssign)):
                    continue
                value = binding.value
                name = dotted(value)
                if not name or name.split(".")[0] not in {v.split(".")[0] for v in aliases.values()}:
                    continue
                targets = binding.targets if isinstance(binding, ast.Assign) else [binding.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in aliases:
                        aliases[target.id] = name
                        changed = True
            if not changed:
                break
        imported_roots = {name.split(".")[0] for name in aliases.values()}
        for node in ast.walk(tree):
            name = dotted(node)
            if name in {"sys.modules", "sys._getframe", "sys.settrace", "sys.setprofile"}:
                raise ValueError(f"Protected runtime state in candidate: {name}")
            if isinstance(node, ast.Attribute) and node.attr in {
                "__dict__", "__globals__", "__builtins__", "__code__", "__closure__",
                "__subclasses__", "__getattribute__", "f_globals", "f_locals", "f_back",
            }:
                raise ValueError(f"Runtime introspection is not allowed: {node.attr}")
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                base = node.value
                while isinstance(base, (ast.Attribute, ast.Subscript)):
                    base = base.value
                if dotted(base).split(".")[0] in imported_roots:
                    raise ValueError(f"Protected dependency mutation in candidate: {name}")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in {"globals", "locals", "vars", "setattr", "delattr"}:
                    raise ValueError(f"Dynamic runtime introspection is not allowed: {node.id}")
                if node.id == "getattr" and not (
                    isinstance(parents.get(node), ast.Call) and parents[node].func is node
                ):
                    raise ValueError("Dynamic runtime introspection alias is not allowed: getattr")
            if isinstance(node, ast.Call):
                called = dotted(node.func)
                if called in {"globals", "locals", "vars", "setattr", "delattr"}:
                    raise ValueError(f"Dynamic runtime introspection is not allowed: {called}")
                if called == "getattr" and not (
                    len(node.args) in (2, 3) and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "_cf"
                    and dotted(node.args[0]).split(".")[0] not in imported_roots
                ):
                    raise ValueError("Dynamic runtime introspection is not allowed: getattr")
            if final_language and isinstance(getattr(node, "ctx", None), ast.Load):
                if name in loaders:
                    raise ValueError(f"Dynamic implementation loading is not allowed: {name}")
                if any(name == prefix or name.startswith(prefix + ".") for prefix in native_dispatch):
                    raise ValueError(f"External native operator dispatch in candidate: {name}")
            if not isinstance(node, ast.Call): continue
            name = dotted(node.func)
            if final_language and (name in {"torch.mm", "torch.bmm", "torch.matmul", "torch.einsum", "torch.softmax", "torch.log_softmax", "torch.layer_norm", "torch.rms_norm"} or name.startswith("torch.nn.functional.")):
                raise ValueError(f"Library operator shortcut in candidate: {name}")
            if name in loaders:
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
        # Preserve the raw benchmark fields and bind the same measured values
        # under the validator's device-timing evidence key.
        row["metadata"]["device_timing"] = {
            key: copy.deepcopy(raw[key]) for key in (
                "benchmark_method", "benchmark_target_ms", "benchmark_samples",
                "benchmark_max_repeats", "benchmark_warmup",
                "benchmark_effective_repeats", "benchmark_fallback_reason",
                "benchmark_timed_run_kind",
            ) if key in raw
        }
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
                    observed = check_flydsl_execution(lambda: actions.check(h))
                    report["metadata"] = {"flydsl_runtime_invocations":observed}
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

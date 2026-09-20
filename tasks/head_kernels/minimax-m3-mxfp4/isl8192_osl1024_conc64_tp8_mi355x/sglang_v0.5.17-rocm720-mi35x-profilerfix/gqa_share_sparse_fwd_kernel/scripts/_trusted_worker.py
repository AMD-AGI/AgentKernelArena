#!/usr/bin/env python3
"""Initialize trusted measurement primitives before any candidate overlay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import sys


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"protected module alias already names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_declared_modules(task, cfg):
    """Preload task-declared trusted helpers before installing a candidate overlay.

    Aliases for one file deliberately share one object. Worker entrypoints execute
    that object's main(), so attestation covers the functions actually used.
    """
    declared = (cfg.get("headkernel") or {}).get("trusted_worker_modules") or {}
    if not isinstance(declared, dict):
        raise RuntimeError("headkernel.trusted_worker_modules must be a name-to-path mapping")
    modules, files = {}, {}
    reserved = {"torch", "runtime_integrity", "runtime_preflight", "harness_lib",
                "_aka_benchmark", "_trusted_worker", "__main__"}
    loading = [True]
    candidate_roots = [(task / "source").resolve(), (task / "ut/kernel_src").resolve(),
                       (task / "ut/_cand_overlay").resolve()]

    def preload_audit(event, args):
        if not loading[0] or event not in {"open", "exec"}:
            return
        filename = args[0].co_filename if event == "exec" else args[0]
        if not isinstance(filename, (str, bytes)):
            return
        try:
            path = Path(filename.decode() if isinstance(filename, bytes) else filename).resolve()
        except (OSError, ValueError):
            return
        if any(path.is_relative_to(root) for root in candidate_roots):
            raise RuntimeError("trusted helper preload attempted candidate access before attestation")

    if declared:
        sys.addaudithook(preload_audit)
    try:
        for name, relative in declared.items():
            if (not isinstance(name, str) or name in reserved
                    or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name)
                    or not isinstance(relative, str) or Path(relative).is_absolute()):
                raise RuntimeError("invalid trusted worker module declaration")
            path = (task / relative).resolve()
            if (not path.is_relative_to(task) or path.suffix != ".py" or not path.is_file()
                    or path.relative_to(task).parts[0] not in {"ut", "scripts"}
                    or any(part in {"source", "kernel_src", "_cand_overlay"}
                           for part in path.relative_to(task).parts)):
                raise RuntimeError("trusted worker module must be a protected task-local Python file")
            if path in files:
                if name in sys.modules and sys.modules[name] is not files[path]:
                    raise RuntimeError(f"trusted module alias is already in use: {name}")
                sys.modules[name] = files[path]
            else:
                files[path] = load(name, path)
            modules[name] = files[path]
    finally:
        loading[0] = False
    return modules, files


def require_uncached_overlay_modules(overlay):
    """A cached native module would bypass a lazy source-module overlay."""
    manifest_path = overlay / "_overlay_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest.get("modules", []):
        name = entry["module"]
        if name in sys.modules:
            raise RuntimeError("source module loaded before protected overlay: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--script", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--attest-file", action="append", default=[])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    task = Path(args.task_root).resolve()
    script = Path(args.script).resolve()
    if not script.is_relative_to(task):
        raise RuntimeError("trusted worker script must be task-local")
    # No task overlay is on PYTHONPATH at interpreter startup. Import trusted
    # torch/timer/comparator code first, then install monitoring, then overlay.
    import yaml
    runtime = load("runtime_preflight", task / "scripts/runtime_preflight.py")
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    runtime.require_runtime(cfg, phase="environment")
    import torch
    benchmark = load("_aka_benchmark", task / "scripts/_aka_benchmark.py")
    harness = load("harness_lib", task / "ut/harness_lib.py")
    monitor = load("runtime_integrity", task / "scripts/runtime_integrity.py")
    trusted_modules, trusted_files = load_declared_modules(task, cfg)
    if (cfg.get("headkernel") or {}).get("generated_input_revision") and not trusted_files:
        raise RuntimeError("generated_input_revision requires headkernel.trusted_worker_modules")
    if trusted_files and script not in trusted_files:
        raise RuntimeError("worker entrypoint is missing from headkernel.trusted_worker_modules")
    if trusted_modules:
        trusted_modules["harness_lib"] = harness
    if script in trusted_files:
        trusted_modules["__main__"] = sys.modules["__main__"]
    # The complete preflight executes after candidate binding. Its actual
    # functions and imported-module bindings must already be attested too.
    trusted_modules["runtime_preflight"] = runtime
    guard = monitor.RuntimeIntegrity(task, torch, benchmark, harness, args.overlay,
                                     trusted_modules=trusted_modules)
    guard.install()
    try:
        if args.overlay:
            overlay = Path(args.overlay).resolve()
            if not overlay.is_relative_to(task):
                raise RuntimeError("candidate/baseline overlay must stay inside the task")
            sys.path.insert(0, str(overlay))
            sitecustomize = overlay / "sitecustomize.py"
            if not sitecustomize.is_file():
                raise RuntimeError("overlay sitecustomize.py is missing")
            require_uncached_overlay_modules(overlay)
            load("_headkernel_overlay", sitecustomize)
            guard.check()
        # Resolve installed native dependencies and the actual target only
        # after helpers are attested and the selected overlay is active.
        runtime.require_runtime(cfg)
        guard.check()
        sys.path.insert(0, str(script.parent))
        tail = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        sys.argv = [str(script), *tail]
        try:
            if script in trusted_files:
                entrypoint = getattr(trusted_files[script], "main", None)
                if not callable(entrypoint):
                    raise RuntimeError("declared trusted worker entrypoint has no main()")
                guard.check()
                result = entrypoint()
                guard.check()
                if result not in (None, 0):
                    raise SystemExit(result)
            else:
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            traceback = exc.__traceback__
            while traceback:
                if guard._candidate_frame(traceback.tb_frame):
                    raise monitor.IntegrityError("candidate exited before trusted work completed") from exc
                traceback = traceback.tb_next
            if exc.code not in (None, 0):
                raise
    finally:
        guard.close()
    outputs = {}
    for filename in args.attest_file:
        path = Path(filename).resolve()
        if not path.is_relative_to(task):
            raise RuntimeError("attested output must be task-local")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        outputs[str(path)] = digest.hexdigest()
    completion = Path(args.completion).resolve()
    if not completion.is_relative_to(task / "build"):
        raise RuntimeError("worker completion must stay in the task build directory")
    completion.write_text(json.dumps({"status": "complete", "nonce": args.nonce,
                                      "script": str(script), "outputs": outputs}) + "\n")


if __name__ == "__main__":
    main()

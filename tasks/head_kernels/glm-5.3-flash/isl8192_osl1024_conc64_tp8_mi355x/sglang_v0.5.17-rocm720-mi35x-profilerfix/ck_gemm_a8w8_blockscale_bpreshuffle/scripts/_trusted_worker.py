#!/usr/bin/env python3
"""Initialize trusted measurement primitives before any candidate overlay."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
    runtime.require_runtime(yaml.safe_load((task / "config.yaml").read_text()))
    import torch
    benchmark = load("_aka_benchmark", task / "scripts/_aka_benchmark.py")
    harness = load("harness_lib", task / "ut/harness_lib.py")
    monitor = load("runtime_integrity", task / "scripts/runtime_integrity.py")
    guard = monitor.RuntimeIntegrity(task, torch, benchmark, harness, args.overlay)
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
            load("_headkernel_overlay", sitecustomize)
            guard.check()
        sys.path.insert(0, str(script.parent))
        tail = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        sys.argv = [str(script), *tail]
        try:
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

#!/usr/bin/env python3
"""Set frozen AITER dispatch before the common _aka_benchmark runtime imports."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise RuntimeError(f"trusted alias names another file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    dispatch = load("dense_dispatch_contract", HERE.parent / "ut/dispatch_contract.py")
    dispatch.prepare_environment()
    runner = load("_dense_arena_runner", HERE / "task_runner.py")
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())

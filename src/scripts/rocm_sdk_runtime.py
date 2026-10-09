#!/usr/bin/env python3
"""Keep ROCm SDK workloads and the profiler on the same core library tree."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def sdk_core() -> Path:
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or spec.origin is None:
        raise RuntimeError("The selected runtime has no ROCm SDK core package")
    return Path(spec.origin).parent


def runtime_environment(environment: dict[str, str]) -> dict[str, str]:
    core = sdk_core()
    # The expanded developer tree can contain a second copy of COMGR/LLVM.
    # PyTorch's absolute core preload cannot be redirected by LD_LIBRARY_PATH.
    # Starting both the profiler and its workload with core libraries prevents
    # loading that second copy and duplicate LLVM option registration.
    core_lib = str(core / "lib")
    devel_lib = str(core.parent / "_rocm_sdk_devel" / "lib")
    libraries = [core_lib, *(
        path for path in environment.get("LD_LIBRARY_PATH", "").split(":")
        if path and path not in {core_lib, devel_lib}
    )]
    return {
        **environment, "LD_LIBRARY_PATH": ":".join(libraries),
    }


def profiler_command(arguments: list[str], environment: dict[str, str]):
    executable = sdk_core() / "bin" / "rocprofv3"
    if not executable.is_file():
        raise RuntimeError(f"ROCm SDK profiler is missing: {executable}")
    return [str(executable), *arguments], runtime_environment(environment)


def main() -> None:
    if sys.argv[1:] == ["--print-library-path"]:
        print(runtime_environment(dict(os.environ))["LD_LIBRARY_PATH"])
        return
    command, environment = profiler_command(sys.argv[1:], dict(os.environ))
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()

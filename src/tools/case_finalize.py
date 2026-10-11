"""Run the protected task's complete-union finalizer in a fresh CPU process."""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "request", "union", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    task = Path(args.task).resolve(strict=True)
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        os.environ[name] = ""

    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "triton"}:
                raise ImportError("CPU full-case finalizer must not initialize GPU task runtimes")
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and Path(spec.origin).is_absolute():
                if Path(spec.origin).resolve().is_relative_to(task / "source"):
                    raise ImportError("CPU full-case finalizer must not import candidate source")
            return None

    sys.meta_path.insert(0, Guard())
    sys.path.insert(0, str(task / "scripts"))
    namespace = runpy.run_path(str(task / "scripts/task_runner.py"), run_name="_aka_full_case_finalize")
    finalizer = namespace.get("finalize_case_union")
    if not callable(finalizer):
        raise ValueError("protected task must expose finalize_case_union")
    manifest = json.loads((task / "cases.json").read_text())
    request = json.loads(Path(args.request).read_text())
    union = json.loads(Path(args.union).read_text())
    report = finalizer(manifest, request, union)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

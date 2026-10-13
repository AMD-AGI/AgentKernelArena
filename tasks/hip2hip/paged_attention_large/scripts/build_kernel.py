"""Build only the bundled source, with source-bound caches inside build/."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_extension(namespace):
    from torch.utils.cpp_extension import load

    source = ROOT / "src"
    files = sorted(path for path in source.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    identity = f"{digest.hexdigest()[:20]}-{arch}"
    build = ROOT / "build" / identity
    copied_source = build / "src"
    for path in files:
        destination = copied_source / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != path.read_bytes():
            shutil.copyfile(path, destination)
    # Hipify writes only into this disposable build copy, never task sources.
    sources = sorted(str(path) for path in copied_source.rglob("*")
                     if path.suffix in (".cu", ".cpp"))
    if not sources:
        raise RuntimeError("Bundled candidate has no buildable source")
    os.environ["PYTORCH_ROCM_ARCH"] = arch
    os.environ.setdefault("MAX_JOBS", "8")
    hip = (torch.version.hip or "").split(".")
    if len(hip) < 2:
        raise RuntimeError("ROCm version is unavailable")
    flags = ["-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__",
             "-DENABLE_FP8", "-DENABLE_BF16", "-DHIP_FP8_TYPE_FNUZ",
             f"-DTORCH_HIP_VERSION={int(hip[0]) * 100 + int(hip[1])}"]
    load(name=namespace, sources=sources,
         extra_include_paths=[str(copied_source), str(copied_source / "core"),
                              str(copied_source / "include")],
         extra_cflags=flags, extra_cuda_cflags=flags,
         build_directory=str(build), with_cuda=True, is_python_module=False,
         verbose=False)
    return getattr(torch.ops, namespace)


def timing_options():
    from _aka_benchmark import hip_source_graph_capture_policy
    sources = sorted(path for path in (ROOT / "src").rglob("*")
                     if path.suffix in (".cu", ".cpp", ".h", ".hpp", ".cuh"))
    safe, reason = hip_source_graph_capture_policy(*sources)
    if not safe:
        # A candidate cannot silently escape the measured graph via stream 0.
        # The framework owns paired forced-Event selection when needed.
        raise RuntimeError(f"Candidate must preserve graph-safe current-stream launches: {reason}")
    return {}

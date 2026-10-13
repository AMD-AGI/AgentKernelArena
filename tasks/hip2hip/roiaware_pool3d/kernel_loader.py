# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from pathlib import Path
import shutil
import tempfile

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
# PyTorch hipify can write generated *_hip.cpp/HIP files next to its inputs.
# Keep those writes in a fresh task-owned build directory, never frozen sources.
_build_root = ROOT / "build" / "native_sources"
if not _build_root.resolve().is_relative_to(ROOT):
    raise ValueError("Native build directory escapes the task")
_build_root.mkdir(parents=True, exist_ok=True)
for _path in (ROOT / "src").rglob("*"):
    if not _path.resolve().is_relative_to(ROOT):
        raise ValueError("Native source tree escapes the task")
_stage = Path(tempfile.mkdtemp(prefix="compile-", dir=_build_root))
shutil.copytree(ROOT / "src", _stage / "src")
# Retain staged inputs as compilation evidence. Copying the whole src directory
# preserves relative includes; each load snapshots the current candidate bytes.
roiaware_pool3d_ext = load(name='roiaware_pool3d',
    sources=[str(_stage / relative) for relative in ['src/roiaware_pool3d_kernel.hip', 'src/roiaware_pool3d.cpp']],
    verbose=True)

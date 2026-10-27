# SPDX-License-Identifier: MIT
"""Check pinned installed AITER sources and retain its ordinary tuned dispatch."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(*, require_gpu=True):
    identity = json.loads((ROOT / "source_identity.json").read_text())
    # Preserve AITER's normal table merge and hardware selection. No task-local
    # empty table, selected backend, forced solution, or tuning mutation.
    for name in ("AITER_CONFIG_GEMM_BF16", "CU_NUM"):
        if os.environ.get(name) not in (None, ""):
            raise RuntimeError(f"Unbound GEMM dispatch override: {name}")
    if os.environ.get("AITER_TUNE_GEMM") not in (None, "", "0"):
        raise RuntimeError("Scoring must not mutate upstream tuning tables")
    if os.environ.get("GPU_ARCHS") not in (None, "native", "gfx950"):
        raise RuntimeError("GPU_ARCHS differs from the required gfx950 runtime")
    aiter = importlib.import_module("aiter")
    package = Path(aiter.__file__).resolve().parent
    if package.is_relative_to(ROOT / "upstream"):
        raise RuntimeError("Materialized source must not shadow the installed AITER package")
    for relative, expected in identity["aiter_files"].items():
        for path in (ROOT / "upstream" / relative,
                     package / Path(relative).relative_to("aiter")):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"AITER source/table differs from the pin: {relative}")
    expected_tables = set(identity["model_tables"])
    for source_root in (ROOT / "upstream" / "aiter", package):
        tables = {"aiter/" + p.relative_to(source_root).as_posix()
                  for p in (source_root / "configs/model_configs").glob("*bf16_tuned_gemm*.csv")
                  if p.is_file() and "untuned" not in p.name}
        if tables != expected_tables:
            raise RuntimeError("Ordinary AITER BF16 tuned-table source set changed")
    result = {"aiter_revision": identity["revision"],
              "aiter_version": importlib.metadata.version("amd-aiter"),
              "source_identity": hashlib.sha256((ROOT / "source_identity.json").read_bytes()).hexdigest(),
              "runtime_image": os.environ.get("AKA_SCORING_IMAGE_RUNTIME_REF"),
              "source_check_scope": "Selected Python source and complete BF16 dispatch-table set; native qualification is separate."}
    if require_gpu:
        import torch

        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A ROCm gfx950 GPU is required; no CPU execution or silent skip")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = properties.gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            raise RuntimeError(f"Expected gfx950, found {arch}")
        result.update(gpu_arch=arch, gpu_name=properties.name, torch_version=torch.__version__)
    return result


if __name__ == "__main__":
    print(json.dumps(verify_runtime(require_gpu=False), sort_keys=True))

#!/usr/bin/env python3
"""Check materialized and installed upstream source and the target runtime."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(*, require_gpu=True):
    lock = json.loads((ROOT / "source_identity.json").read_text())
    package = Path(importlib.import_module("aiter").__file__).resolve().parent
    for relative, digest in lock["aiter_files"].items():
        for path in (ROOT / "upstream" / relative,
                     package / Path(relative).relative_to("aiter")):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"AITER source/config differs from the declared pin: {relative}")
    import torch
    import triton

    evidence = {
        "aiter_revision": lock["revision"],
        "installed_aiter_version": importlib.metadata.version("amd-aiter"),
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "torch_hip_version": torch.version.hip,
        "scoring_image": os.environ.get("AKA_SCORING_IMAGE_RUNTIME_REF"),
        "checked_source_files": lock["aiter_files"],
        "dispatch": "public fused_recurrent_kda; config=None; shipped gfx950 Gluon KDA table",
    }
    if require_gpu:
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("ROCm gfx950 is required; no CPU or skipped timing")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = props.gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            raise RuntimeError(f"Expected gfx950, found {arch}")
        evidence.update(gpu_arch=arch, gpu_name=props.name)
        from aiter.ops.triton.attention import kda
        from aiter.ops.triton.utils import config_utils

        if kda._ARCH != arch or Path(config_utils.AITER_TRITON_CONFIGS_PATH).resolve() != package / "ops/triton/configs":
            raise RuntimeError("The effective AITER architecture or config root changed")
    return evidence


if __name__ == "__main__":
    print(json.dumps(verify_runtime(require_gpu=False), sort_keys=True))

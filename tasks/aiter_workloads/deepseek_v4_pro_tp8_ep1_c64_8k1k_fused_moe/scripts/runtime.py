#!/usr/bin/env python3
"""Verify declared upstream materialization and the installed baseline source."""
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
    workload = json.loads((ROOT / "workloads.json").read_text())
    binding = workload.get("runtime_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Composed AITER runtime/build identity has not been bound")
    if os.environ.get("AKA_SCORING_IMAGE_RUNTIME_REF") != binding.get("image_ref") or not binding.get("image_ref"):
        raise RuntimeError("Actual scoring image does not match the bound composed runtime")
    record = binding["build_evidence"]
    evidence_path = (ROOT / record["path"]).resolve(strict=True)
    if not evidence_path.is_relative_to(ROOT):
        raise ValueError("Runtime evidence escapes the task workspace")
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("Runtime build evidence content changed")
    for name, value in workload["runtime_environment"].items():
        os.environ[name] = value
    aiter = importlib.import_module("aiter")
    package = Path(aiter.__file__).resolve().parent
    checked = {}
    for relative, expected in lock["aiter_files"].items():
        for label, path in (
            ("materialized", ROOT / "upstream" / relative),
            ("installed", package / Path(relative).relative_to("aiter")),
        ):
            if not path.is_file():
                raise RuntimeError(f"Missing {label} upstream source: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"{label} AITER source differs from pin: {relative}")
        checked[relative] = expected
    result = {
        "upstream_revision": lock["revision"],
        "installed_aiter_version": importlib.metadata.version("aiter"),
        "checked_source_files": checked,
        "scoring_image": binding["image_ref"],
        "runtime_build_evidence": record,
        "scope": "Bound scoring image/build evidence and selected source checks; native qualification remains separate.",
    }
    if require_gpu:
        import torch

        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A compatible ROCm GPU is required; no CPU timing or silent skip")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = properties.gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            raise RuntimeError(f"This task requires gfx950, found {arch}")
        result.update(gpu_arch=arch, gpu_name=properties.name, torch_version=torch.__version__)
    return result


if __name__ == "__main__":
    print(json.dumps(verify_runtime(require_gpu=False), sort_keys=True))

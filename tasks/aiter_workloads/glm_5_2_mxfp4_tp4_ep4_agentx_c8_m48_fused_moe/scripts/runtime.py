#!/usr/bin/env python3
"""Check this task's pinned AITER source and intended qualification runtime."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime(*, require_gpu=True):
    workload = json.loads((ROOT / "workloads.json").read_text())
    identity = json.loads((ROOT / "source_identity.json").read_text())
    reference_image = workload["runtime"]["reference_image"]
    actual_image = os.environ.get("AKA_SCORING_IMAGE_RUNTIME_REF")
    table = ROOT / identity["tuning_table"]
    # The table is part of the protected task's declared baseline configuration.
    previous = os.environ.get("AITER_CONFIG_FMOE")
    if previous and Path(previous).resolve() != table.resolve():
        raise RuntimeError("A conflicting MoE table override is set")
    os.environ["AITER_CONFIG_FMOE"] = str(table)
    for key, value in workload["runtime"]["dispatch_environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"Conflicting dispatch environment: {key}")
    aiter = importlib.import_module("aiter")
    package = Path(aiter.__file__).resolve().parent
    for relative, digest in identity["aiter_files"].items():
        for path in (ROOT / "upstream" / relative, package / Path(relative).relative_to("aiter")):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"AITER source differs from the pin: {relative}")
    if importlib.metadata.version("flydsl") != "0.4.0":
        raise RuntimeError("This qualification target requires FlyDSL 0.4.0")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.__version__.startswith("2.11.0") or not torch.version.hip:
        raise RuntimeError("This qualification target requires PyTorch 2.11.0 ROCm")
    result = {"aiter_revision": identity["revision"], "torch": torch.__version__,
              "hip": torch.version.hip, "flydsl": "0.4.0", "image": actual_image,
              "reference_image": reference_image,
              "tuning_table": identity["tuning_table"],
              "native_validation": "This action must exercise the task's own compiled path."}
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("A ROCm GPU is required; CPU fallback is not a task PASS")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        if properties.gcnArchName.split(":")[0] != "gfx950":
            raise RuntimeError("This task requires gfx950")
        if properties.multi_processor_count != 256:
            raise RuntimeError("The selected tuned table requires 256 CUs")
        result.update(gpu=properties.name, arch=properties.gcnArchName,
                      compute_units=properties.multi_processor_count)
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import resolve_activation_dtype, get_2stage_cfgs, get_padded_M
        resolved = resolve_activation_dtype(QuantType.per_1x32, dtypes.fp4x2,
                                           activation=ActivationType.Silu,
                                           gate_mode="separated", M=48,
                                           hidden_dtype=torch.bfloat16)
        if resolved != dtypes.fp4x2:
            raise RuntimeError("The public call does not resolve the declared MXFP4 activation path")
        result["resolved_activation_dtype"] = str(resolved)
        selected = get_2stage_cfgs(
            get_padded_M(48), 6144, 2048, 65, 9, torch.bfloat16,
            dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32, True,
            ActivationType.Silu, False, 0, 0, is_shuffled=True,
            gate_mode="separated", is_ep=True, opus_weights_shuffled=True,
            input_dtype=torch.bfloat16,
        )
        selection = {"stage1": selected.stage1.keywords.get("kernelName"),
                     "stage2": selected.stage2.keywords.get("kernelName"),
                     "block_m": selected.block_m, "fuse_quant": selected.fuse_quant}
        if selection != workload["runtime"]["expected_dispatch"]:
            raise RuntimeError(f"Declared baseline dispatch differs: {selection}")
        result["resolved_baseline_dispatch"] = selection
    return result


if __name__ == "__main__":
    print(json.dumps(verify_runtime(require_gpu=False), sort_keys=True))

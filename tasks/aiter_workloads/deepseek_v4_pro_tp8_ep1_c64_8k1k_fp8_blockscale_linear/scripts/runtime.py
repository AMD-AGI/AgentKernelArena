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

# Settings read by the pinned files on this call path: the bpreshuffle tuned
# table override (jit/core.py:175, 707), architecture/CU selection
# (chip_info.py:60, 277) and AITER import/JIT/native roots (__init__.py,
# jit/core.py). Absence is meaningful: capture it as JSON null. This is a
# task-local binding, not a scrub of the process environment.
DISPATCH_ENVIRONMENT = (
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    "GPU_ARCHS", "CU_NUM", "AITER_TRITON_ONLY", "AITER_AOT_IMPORT",
    "AITER_REBUILD", "ENABLE_CK", "AITER_DISABLE_KERNARG_PRELOAD",
    "AITER_ENABLE_EXPERIMENTAL", "AITER_META_DIR", "AITER_JIT_DIR",
    "OPUS_GEN_CO_DIR", "CK_DIR", "HIP_KITTENS_DIR", "FLYDSL_RUNTIME_CACHE_DIR",
)
TABLE_OVERRIDES = ("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",)


def verify_dispatch_environment(binding, required):
    """Reject a different effective setup before AITER caches import-time flags."""
    declared = binding.get("dispatch_environment")
    if not isinstance(declared, dict) or set(declared) != set(DISPATCH_ENVIRONMENT):
        raise RuntimeError("Every linear dispatch setting must be bound, including absent settings")
    if not isinstance(required, dict) or any(
        name not in declared or type(value) is not str or declared[name] != value
        for name, value in required.items()
    ):
        raise RuntimeError("Dispatch binding differs from the selected workload environment")
    for name, value in declared.items():
        if value is not None and type(value) is not str:
            raise ValueError(f"Dispatch setting requires a string or null: {name}")
        if os.environ.get(name) != value:
            raise RuntimeError(f"Effective runtime dispatch setting differs: {name}")
    # The pinned image's normal table discovery is the bound carrier. A path
    # override needs a protected table-content binding before it can be admitted.
    if any(declared[name] not in (None, "") for name in TABLE_OVERRIDES):
        raise RuntimeError("An explicit tuned-table override requires a protected content binding; unsupported")
    return declared


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
    environment = verify_dispatch_environment(binding, workload["runtime_environment"])
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
        "installed_aiter_version": importlib.metadata.version("amd-aiter"),
        "checked_source_files": checked,
        "scoring_image": binding["image_ref"],
        "runtime_build_evidence": record,
        "dispatch_environment": environment,
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
        # The caller takes this bpreshuffle path only on HIP >= 7.2 (SGLang fp8_utils.py:68-73).
        if tuple(int(part) for part in torch.version.hip.split(".")[:2]) < (7, 2):
            raise RuntimeError(f"This task requires HIP >= 7.2, found {torch.version.hip}")
        result.update(gpu_arch=arch, gpu_name=properties.name, torch_version=torch.__version__,
                      hip_version=torch.version.hip)
    return result


def verify_baseline_native():
    """Bind the baseline's own declared native artifacts after its actual loads.

    The protected manifest (captured with scripts/native_identity.py at
    qualification) names each required artifact. Extensions are observed in
    sys.modules as AITER's loader left them; device code and tables by declared
    paths under the AITER package root. Nothing is loaded, compiled, discovered
    or looked up through a getter here, and candidate-owned artifacts are
    neither required nor prohibited. An unobservable required artifact stays
    missing, so the check fails closed. File identity is not proof of dispatch.
    """
    import sys
    from native_identity import capture_native_identity, verify_native_identity

    binding = json.loads((ROOT / "workloads.json").read_text()).get("runtime_binding") or {}
    declared = binding.get("native_identity")
    if not isinstance(declared, dict) or not isinstance(declared.get("manifest"), dict):
        raise RuntimeError("Baseline native artifact identity has not been bound")
    record = declared["manifest"]
    path = (ROOT / record["path"]).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("Native identity manifest escapes the task workspace")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != record["sha256"]:
        raise RuntimeError("Native identity manifest content changed")
    expected = json.loads(body)
    roots = {"aiter": str(Path(sys.modules["aiter"].__file__).resolve().parent)}
    observed = capture_native_identity(
        expected["required"],
        python_modules={key: sys.modules.get(name) for key, name in declared.get("python_modules", {}).items()},
        device_code_paths=declared.get("device_code_paths"), table_paths=declared.get("table_paths"),
        roots=roots, max_artifact_bytes=expected["limits"]["max_artifact_bytes"],
        max_total_bytes=expected["limits"]["max_total_bytes"],
    )
    verify_native_identity(expected, observed, roots=roots)
    return observed


if __name__ == "__main__":
    print(json.dumps(verify_runtime(require_gpu=False), sort_keys=True))

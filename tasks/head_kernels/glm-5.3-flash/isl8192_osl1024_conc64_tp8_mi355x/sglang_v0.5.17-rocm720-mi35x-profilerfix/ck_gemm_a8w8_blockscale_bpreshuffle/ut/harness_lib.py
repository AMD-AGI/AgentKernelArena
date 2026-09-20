"""Protected native binding and comparisons for the GLM GEMM operator tasks."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text())
_NATIVE = None
_CANDIDATE = None


def verify_hash(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"frozen native source/dispatch hash mismatch: {Path(path).name}")


def configure_dispatch():
    """Both worker processes use the same protected stock native dispatch table."""
    if META.get("dispatch_file"):
        path = HERE / META["dispatch_file"]
        verify_hash(path, META["dispatch_sha256"])
        os.environ["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"] = str(path)


def native_function():
    global _NATIVE
    if _NATIVE is None:
        configure_dispatch()
        native = importlib.import_module(META["source_module"])
        verify_hash(native.__file__, META["source_sha256"])
        module_name, symbol = META["target_callable"].split(":")
        _NATIVE = getattr(importlib.import_module(module_name), symbol)
        if _NATIVE is not getattr(native, META["symbol"]):
            raise RuntimeError("the installed production alias was already replaced")
        if META.get("dispatch_file"):
            observed = native.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
            if Path(observed).resolve() != (HERE / META["dispatch_file"]).resolve():
                raise RuntimeError("native FP8 dispatcher did not select the protected table")
            # Parse the immutable table before entering any editable frame.
            # Runtime integrity correctly forbids candidates from opening ut/
            # files; their native getter uses this already-populated cache.
            for case in META["cases"]:
                native.get_CKGEMM_config(case["m"], case["n"], case["k"], observed)
    return _NATIVE


def bind_candidate(kind, original, candidate, target_module, implementation_module,
                   loaded_modules=None):
    """Bind the actual serving aliases; never replace CK's native kernel_entry.

    Kept independent of GPU imports so negative-control tests can prove that the
    import-time solMap and package aliases execute the changed function.
    """
    loaded_modules = sys.modules if loaded_modules is None else loaded_modules
    if kind == "bf16":
        if target_module.solMap.get("torch") is not original:
            raise RuntimeError("native solMap['torch'] identity differs from the recorded seam")
        target_module.torch_gemm = candidate
        for key, value in list(target_module.solMap.items()):
            if value is original:
                target_module.solMap[key] = candidate
        if target_module.solMap["torch"] is not candidate:
            raise RuntimeError("candidate did not reach solMap['torch']")
    elif kind == "fp8":
        symbol = "gemm_a8w8_blockscale_bpreshuffle"
        if getattr(target_module, symbol) is not original:
            raise RuntimeError("the AITER package alias was already replaced")
        setattr(target_module, symbol, candidate)
        if getattr(implementation_module, symbol) is original:
            setattr(implementation_module, symbol, candidate)
        # SGLang may already have bound the package-level alias during preflight.
        fp8_utils = loaded_modules.get("sglang.srt.layers.quantization.fp8_utils")
        if fp8_utils is not None and hasattr(fp8_utils, symbol):
            if getattr(fp8_utils, symbol) is not original:
                raise RuntimeError("the SGLang FP8 alias was already replaced")
            setattr(fp8_utils, symbol, candidate)
    else:
        raise ValueError(f"unknown GLM GEMM kind: {kind}")


def candidate_function():
    global _CANDIDATE
    if _CANDIDATE is None:
        original = native_function()
        path = HERE.parent / "source/kernel.py"
        # The FP8 function contains relative imports in non-gfx950 branches.
        name = ("aiter.ops._aka_glm_fp8_candidate" if META["kind"] == "fp8"
                else "_aka_glm_bf16_candidate")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        candidate = getattr(module, META["symbol"])
        if candidate is original:
            raise RuntimeError("candidate resolves to the native callable")
        target = importlib.import_module(META["target_callable"].split(":")[0])
        implementation = importlib.import_module(META["source_module"])
        bind_candidate(META["kind"], original, candidate, target, implementation)
        _CANDIDATE = candidate
    return _CANDIDATE


def build_candidate_overlay(task_dir, meta):
    """Common runner interface: protected case calls bind explicitly on demand.

    No startup hook or copied source alias is needed: reference workers call
    native_function only; candidate workers load source/kernel.py directly.
    """
    task_ut = Path(task_dir).resolve()
    if task_ut != HERE or meta["candidate_bind"]["file"] != "../source/kernel.py":
        raise RuntimeError("unexpected task-local candidate binding")
    if not (HERE.parent / "source/kernel.py").is_file():
        raise RuntimeError("candidate source is missing")
    return None, None


def to_device_like(value, device):
    return value.to(device)


def correct(output, reference, tol):
    """Original upstream elementwise mixed tolerance, including its RMS floor."""
    import torch
    if (not torch.is_tensor(output) or output.shape != reference.shape
            or output.dtype != reference.dtype or output.device != reference.device):
        return False, float("inf")
    observed, expected = output.float(), reference.float()
    atol = tol * expected.pow(2).mean().sqrt().clamp_min(1e-6)
    error = (observed - expected).abs()
    ok = bool((error <= atol + tol * expected.abs()).all())
    return ok, float((error / (expected.abs() + atol)).max().item())

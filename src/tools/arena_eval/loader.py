###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlashInfer-Bench Contributors
#
# Adapted from AKLR (aklr/core/runtime/builder/builder.py and
# aklr/core/jit/hip_extension.py), which adapts FlashInfer-Bench
# (https://github.com/flashinfer-ai/flashinfer-bench), Source:
# flashinfer_bench/compile/{builder,utils}.py and
# flashinfer_bench/compile/builders/{python_builder,triton_builder}.py.
# Modified by Advanced Micro Devices, Inc. for AgentKernelArena.
#
# This file is distributed under the Apache License 2.0.
###############################################################################
"""Make a baseline or candidate implementation callable as ``run(**call)``.

Python-family languages (Triton, Gluon, FlyDSL, TileLang) import their entry
file as a flat package built in memory, so helper files import only relatively.
HIP builds its ``.hip`` translation units into a torch extension whose op
``run`` takes exactly the call keys and writes exactly the declared tensors.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .contract import WORKSPACE, Contract, ContractError, Implementation

PACKAGE_PREFIX = "_aka_impl_"
HIP_SOURCE_SUFFIXES = (".hip",)
HIP_HEADER_SUFFIXES = (".h", ".hpp", ".cuh", ".inc")
_FLAGS = ("-O3",)


class BuildError(RuntimeError):
    """An implementation could not be made callable."""


@dataclass
class Loaded:
    implementation: Implementation
    call: Callable[..., Any]
    package: str
    sources: dict[str, str]
    hip_op: str | None = None


def implementation_sources(implementation: Implementation) -> dict[str, str]:
    """Relative path -> text of every file under the implementation root."""
    root = (WORKSPACE / implementation.root).resolve(strict=True)
    if not root.is_relative_to(WORKSPACE) or not root.is_dir():
        raise BuildError(f"{implementation.role} root {implementation.root} is not a task directory")
    sources = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise BuildError(f"{implementation.role} source {relative} is a symlink")
        if path.is_file():
            try:
                sources[relative.as_posix()] = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                raise BuildError(f"{implementation.role} source {relative} is not UTF-8 text") from exc
    return sources


def _is_documentation(path: str) -> bool:
    """License and documentation files that ship beside a vendored implementation but are never compiled."""
    name = Path(path).name
    if name.endswith(HIP_SOURCE_SUFFIXES + HIP_HEADER_SUFFIXES):
        return False
    return name.upper().startswith(("LICENSE", "NOTICE", "COPYING")) or name.endswith(".md")


def _content_hash(sources: dict[str, str], extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode())
    for path, text in sorted(sources.items()):
        digest.update(path.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()[:16]


def _bind_check(function: Callable, call_keys: tuple[str, ...], label: str) -> None:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{label} has no inspectable signature") from exc
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise BuildError(f"{label} must name each call key explicitly, not *args/**kwargs")
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise BuildError(f"{label} parameters must be keyword arguments")
    names = [p.name for p in signature.parameters.values()]
    if sorted(names) != sorted(call_keys):
        raise BuildError(f"{label} takes {names}, not the call keys {list(call_keys)}")


def _load_python(contract: Contract, implementation: Implementation) -> Loaded:
    sources = implementation_sources(implementation)
    entry = Path(implementation.entry_file).relative_to(implementation.root).as_posix()
    if entry not in sources or not entry.endswith(".py"):
        raise BuildError(f"{implementation.role} entry {implementation.entry_file} is missing")
    for path in sources:
        if path.endswith(".py") and ("/" in path or path == "__init__.py"):
            raise BuildError(f"{implementation.role} Python sources are one flat package; "
                             f"{path} is not allowed")
    if implementation.language == "asm":
        # Assembly packages launch their .s kernels through the evaluator's helper.
        from . import asm

        sys.modules[asm.ALIAS] = asm
    package = f"{PACKAGE_PREFIX}{implementation.role}_{_content_hash(sources)}"
    root = (WORKSPACE / implementation.root).resolve(strict=True)
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root)]
        module.__package__ = package
        sys.modules[package] = module
    try:
        entry_module = importlib.import_module(f"{package}.{entry.removesuffix('.py')}")
    except Exception as exc:
        raise BuildError(f"Cannot import {implementation.entry_file}: {type(exc).__name__}: {exc}") from exc
    function = getattr(entry_module, implementation.symbol, None)
    if not callable(function) or not inspect.isfunction(function):
        raise BuildError(f"{implementation.entry_file} does not define function {implementation.symbol}")
    _bind_check(function, contract.call_keys, f"{implementation.entry_file}::{implementation.symbol}")
    return Loaded(implementation, function, package, sources)


def hip_available() -> bool:
    if torch.version.hip is None:
        return False
    from torch.utils import cpp_extension

    rocm = cpp_extension.ROCM_HOME
    return rocm is not None and Path(rocm, "bin", "hipcc").is_file() and cpp_extension.is_ninja_available()


def _cache_root() -> Path:
    configured = os.environ.get("AKA_KERNEL_CACHE_DIR")
    root = Path(configured) if configured else Path(os.environ.get("TMPDIR", "/tmp")) / "aka-kernel-cache"
    root = root.resolve()
    if root.is_relative_to(WORKSPACE):
        raise BuildError("The kernel build cache must be outside the task workspace")
    return root


def _toolchain() -> str:
    from torch.utils import cpp_extension

    version = Path(cpp_extension.ROCM_HOME or "", ".info", "version")
    return json.dumps({"torch": torch.__version__,
                       "rocm": version.read_text().strip() if version.is_file() else None,
                       "flags": _FLAGS})


def current_arch() -> str:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(properties, "gcnArchName", "") or ""
    return arch.split(":")[0]


def _check_op_schema(label: str, op_schema, contract: Contract) -> None:
    names = [argument.name for argument in op_schema.arguments]
    if sorted(names) != sorted(contract.call_keys):
        raise BuildError(f"{label} takes {names}, not the call keys {list(contract.call_keys)}")
    writes = {argument.name for argument in op_schema.arguments
              if argument.alias_info is not None and argument.alias_info.is_write}
    expected = set(contract.checked_tensors)
    if writes != expected or op_schema.returns:
        raise BuildError(f"{label}: {op_schema} must write exactly {sorted(expected)}, "
                         "each a Tensor(a!), and return ()")


def _load_hip(contract: Contract, implementation: Implementation) -> Loaded:
    if not hip_available():
        raise BuildError("HIP implementations need a ROCm PyTorch with hipcc and ninja")
    sources = implementation_sources(implementation)
    entry = Path(implementation.entry_file).relative_to(implementation.root).as_posix()
    if entry not in sources or not entry.endswith(".hip"):
        raise BuildError(f"{implementation.role} entry {implementation.entry_file} is missing")
    sources = {path: text for path, text in sources.items() if not _is_documentation(path)}
    for path in sources:
        if not path.endswith(HIP_SOURCE_SUFFIXES + HIP_HEADER_SUFFIXES):
            raise BuildError(f"HIP implementation file {path} is neither a .hip source nor a header")
    arch = current_arch()
    if not arch:
        raise BuildError("No GPU architecture to build HIP for")
    package = f"aka_hip_{implementation.role}_{_content_hash(sources, _toolchain() + arch)}"
    build = _cache_root() / "hip" / package
    source_dir = build / "src"
    marker = build / "built"
    library = build / f"{package}.so"
    from torch.utils import cpp_extension

    if marker.is_file() and library.is_file():
        torch.ops.load_library(str(library))
    else:
        for path, text in sources.items():
            target = source_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        previous = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_ROCM_ARCH"] = arch
        try:
            cpp_extension.load(name=package,
                               sources=[str(source_dir / p) for p in sorted(sources) if p.endswith(".hip")],
                               extra_include_paths=[str(source_dir)], extra_cuda_cflags=list(_FLAGS),
                               build_directory=str(build), is_python_module=False, verbose=False)
        except Exception as exc:
            raise BuildError(f"HIP build failed: {type(exc).__name__}: {exc}") from exc
        finally:
            if previous is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = previous
        marker.touch()
    try:
        op = getattr(getattr(torch.ops, package), implementation.symbol).default
    except (AttributeError, RuntimeError) as exc:
        raise BuildError(f"{implementation.entry_file} registers no op {implementation.symbol} in "
                         "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)") from exc
    _check_op_schema(f"{implementation.entry_file}::{implementation.symbol}", op._schema, contract)
    return Loaded(implementation, lambda **call: op(**call), package, sources, f"{package}::{implementation.symbol}")


def load_implementation(contract: Contract, role: str) -> Loaded:
    implementation = contract.implementation(role)
    if implementation.is_python:
        return _load_python(contract, implementation)
    if implementation.language == "hip":
        return _load_hip(contract, implementation)
    raise ContractError(f"No builder for language {implementation.language!r}")


def language_available(language: str) -> tuple[bool, str]:
    """Whether this runtime can build ``language``; the reason when not."""
    modules = {"triton": "triton", "gluon": "triton.experimental.gluon", "flydsl": "flydsl",
               "tilelang": "tilelang"}
    if language == "hip":
        return (True, "") if hip_available() else (False, "hipcc/ninja/ROCm PyTorch not found")
    if language == "asm":
        from .asm import available

        return available()
    if language not in modules:
        return False, f"unknown language {language!r}"
    try:
        importlib.import_module(modules[language])
    except Exception as exc:  # A broken install is as unavailable as a missing one.
        return False, f"cannot import {modules[language]}: {type(exc).__name__}: {exc}"
    return True, ""

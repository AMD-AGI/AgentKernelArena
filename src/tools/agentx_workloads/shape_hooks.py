# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Shape hooks: give Python-launched kernels a recorded op with input shapes in the torch profiler trace.

The census links each GPU kernel to the CPU op that launched it (the trace's ``External id``) and reads
the op's ``Input Dims`` and ``Input type``. A Triton, Gluon or FlyDSL kernel launched straight from a
Python wrapper has no such op, so its census row carries no shapes. A shape hook wraps such a wrapper
so that, while the torch profiler records, each call runs inside a recorded function scope named
``pyhook::<label>`` whose inputs are the call's arguments: with ``record_shapes`` on, the profiler
writes it as a ``cpu_op`` event with ``Input Dims`` / ``Input type``, and the kernels the wrapper
launches carry its ``External id``. The census then reads the scope like any other op.

Properties:

* Off unless enabled: nothing is patched unless ``AKA_SHAPE_HOOKS`` names targets (the capture sets it
  on the server container, which imports this module from ``shape_hooks_site/sitecustomize.py``).
* Cheap: when the profiler is not recording, a hooked call costs one extra Python frame and a flag
  test; the scope (``torch._C._profiler._RecordFunctionFast``, the one TorchInductor puts around its
  Triton launches) is created only while the profiler is on.
* No numerical effect: the wrapper calls the original function with the original arguments.
* CUDA graphs: graph replays run no Python, so they are not hooked; during capture the profiler is
  off and the wrapper only forwards the call. Under ``torch.compile`` tracing the wrapper also forwards.
* Targets are patched in place, in the module that defines them, right after that module executes
  (an import hook), so ``from module import fn`` in other modules picks up the hooked function. A module
  executed again from the same spec (aiter's old-path aliases re-execute the new module) is patched again.
  A target that is already imported when the hooks are installed is patched immediately.

Target syntax (``AKA_SHAPE_HOOKS`` and the capture's ``--shape-hooks``): a comma-separated list of
``module:qualname[=label]`` entries and preset names (``default``); ``none`` or an empty string
disables the hooks. The label defaults to ``<top package>.<function name>``.

This file must stay importable on its own (standard library only at import time; torch is imported
when the hooks are installed), because the server container loads it by path.
"""
from __future__ import annotations

import functools
import importlib.abc
import os
import sys
import threading

ENV = "AKA_SHAPE_HOOKS"
PREFIX = "pyhook::"
SITE_DIR = "shape_hooks_site"  # directory (next to this file) with the server's sitecustomize.py

# Wrappers that launch Triton / Gluon / FlyDSL kernels without a recorded op in the serving image
# (SGLang 0.5.23 ROCm, aiter). Entries whose module a deployment never imports are inert.
PRESETS = {
    "default": [
        # FP8 block-scale GEMM (DeepSeek-V3 / R1 dense layers on gfx950 tuned shapes) and the MLA absorbed BMM
        "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale:gemm_a8w8_blockscale",
        "aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant:"
        "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
        # Gated DeltaNet and causal conv1d of hybrid linear-attention models (Qwen3-Next, Qwen3.5)
        "sglang.kernels.ops.mamba.causal_conv1d_triton:causal_conv1d_fn",
        "sglang.kernels.ops.mamba.causal_conv1d_triton:causal_conv1d_update",
        "aiter.ops.triton.conv.causal_conv1d:causal_conv1d_fn",
        "aiter.ops.triton.conv.causal_conv1d:causal_conv1d_update",
        "sglang.kernels.ops.attention.fla.chunk_delta_h:chunk_gated_delta_rule_fwd_h",
        "sglang.kernels.ops.attention.fla.fused_recurrent:fused_recurrent_gated_delta_rule_fwd",
        "sglang.kernels.ops.attention.fla.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
        "sglang.kernels.ops.attention.fla.fused_recurrent:fused_recurrent_gated_delta_rule_update_fwd",
        "sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent:fused_sigmoid_gating_delta_rule_update",
        # Not gdn_replayssm_spec_decode (the speculative verify update): its q [T, Hk, K] is a layout the
        # gdn_fused_recurrent_decode matcher does not read yet, which would drop rows it now reads by step batch.
        "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode:commit_gdn_replayssm_circular",
        "sglang.kernels.ops.attention.linear.fused_gdn_gating:fused_gdn_gating",
        "sglang.kernels.ops.attention.triton_gdn_fused_proj:fused_qkvzba_split_reshape_cat",
        "sglang.kernels.ops.attention.triton_gdn_fused_proj:fused_qkv_split_gdn_prefill",
        # Triton extend / prefill attention (draft and target extend of several backends) and MLA state merge
        "sglang.kernels.ops.attention.extend_attention:extend_attention_fwd",
        "sglang.kernels.ops.attention.prefill_attention:context_attention_fwd",
        "sglang.kernels.ops.attention.merge_state:merge_state_triton",
        # DeepSeek-V4 sparse MLA, FP4 indexer logits, top-k and split reduce
        "aiter.ops.triton.attention.pa_decode_sparse:pa_decode_sparse",
        "aiter.ops.triton.attention.sparse_mla:sparse_mla_fwd",
        "sglang.kernels.ops.attention.dsv4.fp4_indexer_hip:aiter_fp4_paged_mqa_logits",
        "sglang.kernels.ops.attention.dsv4.fp4_indexer_hip:rowgroup_paged_mqa_logits",
        "sglang.kernels.ops.attention.dsv4.topk:topk_transform_paged_v2",
        "sglang.kernels.ops.attention.sparse_decode_reduce_hip:gfx_sparse_split_reduce",
        "sglang.kernels.ops.attention.fused_qk_norm_rope_store:fused_qk_norm_rope_swa_store",
    ],
}


class Target:
    """One hooked callable: ``module`` defines ``qualname``; its scope is recorded as ``pyhook::<label>``."""

    def __init__(self, module: str, qualname: str, label: str = ""):
        self.module, self.qualname = module, qualname
        self.label = label or f"{module.split('.')[0]}.{qualname.split('.')[-1]}"

    @property
    def op(self) -> str:
        return PREFIX + self.label

    def spec(self) -> str:
        return f"{self.module}:{self.qualname}={self.label}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Target) and (self.module, self.qualname) == (other.module, other.qualname)

    def __hash__(self) -> int:
        return hash((self.module, self.qualname))

    def __repr__(self) -> str:
        return f"Target({self.spec()!r})"


def parse(spec: str | None) -> list[Target]:
    """Targets of a hook specification (see the module docstring); presets expand, duplicates collapse."""
    out: list[Target] = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item or item == "none":
            continue
        if item in PRESETS:
            out += parse(",".join(PRESETS[item]))
            continue
        target, eq, label = item.partition("=")
        module, sep, qualname = target.partition(":")
        if not sep or not module or not qualname or not all(p.isidentifier() for p in module.split(".")) \
                or not all(p.isidentifier() for p in qualname.split(".")) or (eq and not label.strip()):
            raise ValueError(f"shape hook {item!r}: expected module:function[=label] or one of {sorted(PRESETS)}")
        out.append(Target(module, qualname, label.strip()))
    seen, unique = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def canonical(spec: str | None) -> str:
    """The specification as explicit targets (what the server environment receives and a capture records)."""
    return ",".join(t.spec() for t in parse(spec))


def _log(msg: str) -> None:
    print(f"[shape_hooks pid {os.getpid()}] {msg}", file=sys.stderr, flush=True)


def wrap(fn, label: str, record_function=None, profiling=None, compiling=None):
    """``fn`` wrapped so that each call made while the profiler records runs in a ``pyhook::<label>`` scope whose
    inputs are the positional arguments followed by the tensor-valued keyword arguments.

    ``record_function``, ``profiling`` and ``compiling`` default to torch's ``_RecordFunctionFast``, the profiler's
    enabled flag and ``torch.compiler.is_compiling`` (tests pass stand-ins)."""
    if record_function is None or profiling is None or compiling is None:
        import torch
        import torch.autograd.profiler as autograd_profiler
        record_function = record_function or torch._C._profiler._RecordFunctionFast
        profiling = profiling or (lambda: autograd_profiler._is_profiler_enabled)
        compiling = compiling or torch.compiler.is_compiling
        tensor_type = torch.Tensor
    else:
        tensor_type = None
    name = PREFIX + label
    first = [True]

    @functools.wraps(fn)
    def hooked(*args, **kwargs):
        if not profiling() or compiling():
            return fn(*args, **kwargs)
        if first[0]:
            first[0] = False
            _log(f"first recorded call of {name}")
        inputs = list(args)
        if kwargs:
            inputs += [v for v in kwargs.values()
                       if (isinstance(v, tensor_type) if tensor_type is not None else hasattr(v, "shape"))]
        with record_function(name, inputs):
            return fn(*args, **kwargs)

    hooked.__shape_hook__ = name
    return hooked


def _resolve(module, qualname: str):
    """(owner, attribute name, value) of ``qualname`` in ``module``; None when it is missing."""
    owner = module
    parts = qualname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p, None)
        if owner is None:
            return None
    value = getattr(owner, parts[-1], None)
    return None if value is None else (owner, parts[-1], value)


def patch_module(module, targets: list[Target], wrapper=wrap) -> list[str]:
    """Hook ``targets`` defined in ``module``; returns the ops installed."""
    done = []
    for t in targets:
        found = _resolve(module, t.qualname)
        if found is None:
            _log(f"{t.module}:{t.qualname} not found; not hooked")
            continue
        owner, attr, value = found
        if getattr(value, "__shape_hook__", None):
            continue
        if isinstance(value, type) or not callable(value):
            _log(f"{t.module}:{t.qualname} is not a function; not hooked")
            continue
        try:
            raw = vars(owner).get(attr) if isinstance(owner, type) else None
            if isinstance(raw, (staticmethod, classmethod)):  # keep the descriptor kind of a class attribute
                hooked = type(raw)(wrapper(raw.__func__, t.label))
            else:
                hooked = wrapper(value, t.label)
            setattr(owner, attr, hooked)
        except Exception as exc:  # a hook must never break the import of the module it patches
            _log(f"{t.module}:{t.qualname} not hooked: {exc!r}")
            continue
        done.append(t.op)
    if done:
        _log(f"hooked {', '.join(done)} in {module.__name__}")
    return done


class _Finder(importlib.abc.MetaPathFinder):
    """Meta-path finder that patches a target module right after it executes (before any importer sees it)."""

    def __init__(self, by_module: dict[str, list[Target]], wrapper=wrap):
        self.by_module, self.wrapper = by_module, wrapper
        self._local = threading.local()

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.by_module:
            return None
        busy = self._local.__dict__.setdefault("busy", set())
        if fullname in busy:
            return None
        busy.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            busy.discard(fullname)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return spec
        if getattr(loader.exec_module, "__shape_hooks__", False):
            return spec
        original = loader.exec_module
        targets, wrapper = self.by_module[fullname], self.wrapper

        def exec_module(module):
            original(module)
            patch_module(module, targets, wrapper)

        # Stays on the loader: a module can run again from the same spec, and its new functions need the hooks
        # too. aiter's backward-compatibility import paths do that: importing an old path such as
        # aiter.ops.triton.gemm_a8w8_blockscale re-executes the new module from its cached spec and replaces it
        # in sys.modules, so the copy that importers bind would otherwise be unhooked.
        exec_module.__shape_hooks__ = True
        loader.exec_module = exec_module
        return spec


_installed: _Finder | None = None


def install(spec: str | None = None, wrapper=wrap) -> list[Target]:
    """Install the hooks named by ``spec`` (default: ``$AKA_SHAPE_HOOKS``); idempotent. Never raises on a bad
    target at serving time: problems are logged and the server runs unhooked."""
    global _installed
    if _installed is not None:
        return [t for ts in _installed.by_module.values() for t in ts]
    try:
        targets = parse(os.environ.get(ENV, "") if spec is None else spec)
    except ValueError as exc:
        _log(f"{exc}; shape hooks disabled")
        return []
    if not targets:
        return []
    by_module: dict[str, list[Target]] = {}
    for t in targets:
        by_module.setdefault(t.module, []).append(t)
    _installed = _Finder(by_module, wrapper)
    sys.meta_path.insert(0, _installed)
    for name, ts in by_module.items():
        if name in sys.modules:
            patch_module(sys.modules[name], ts, wrapper)
    _log(f"installed {len(targets)} shape hook(s) over {len(by_module)} module(s)")
    return targets


def uninstall() -> None:
    """Remove the import hook (already patched functions stay patched); for tests."""
    global _installed
    if _installed is not None and _installed in sys.meta_path:
        sys.meta_path.remove(_installed)
    _installed = None

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPT-OSS clamped SwiGLU on the block-interleaved gate/up layout with AITER's FlyDSL kernel.

`run` is the launch of aiter.ops.flydsl.moe_kernels.flydsl_swiglu_and_mul_interleaved (its cached builder
`_get_compiled_swiglu` and its `_run_compiled` helper) writing into the caller-provided output.
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir

from .swiglu_and_mul import build_swiglu_and_mul_module


@functools.cache
def _get_compiled_swiglu(inter_dim: int):
    """Compile and cache the fused swiglu_and_mul kernel (interleaved input)."""
    return build_swiglu_and_mul_module(inter_dim)


def _run_compiled(exe, args):
    """JIT-compile on first call, then dispatch via cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # JitFunction.__call__ leaks ir.Context on compilation failure,
        # causing all subsequent JitFunction calls to take a wrong code path
        # (self.func(*args) without CompilationContext → gpu_module_body error).
        # Clean up leaked contexts to isolate failures.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001,S110 - best-effort context cleanup
            pass
        raise


def run(*, x, out):
    inter_dim = out.shape[-1]
    num_rows = x.shape[0]
    _swiglu_fn = _get_compiled_swiglu(inter_dim)
    _run_compiled(_swiglu_fn, (x, out, num_rows, fx.Stream(torch.cuda.current_stream(x.device))))

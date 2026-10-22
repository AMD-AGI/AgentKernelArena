"""Runtime helpers shared by all auto-generated task_runner.py scripts.

Embedded into each task at build time as ``scripts/_runtime.py`` so the runner
has zero engine dependency at execution time. Two responsibilities:

  1. ``build_inputs(test_case, seed)``: deterministic random tensor generation
     from a captured launch signature (shape + dtype + non-tensor scalars).
  2. ``compare(got, expected, dtype)``: dtype-aware allclose with sensible
     tolerances for fp16 / bf16 / fp8.

A small *reference table* lets known kernels (``rms_norm``, ``silu_and_mul``,
``rotary_embedding``, …) compute an analytic expected value purely in PyTorch.
For unknown kernels the runner falls back to a determinism check (run twice
with the same seed → byte-identical output).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------- dtype lookup
_DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
    "float8_e5m2": torch.float8_e5m2,
    "float8_e5m2fnuz": getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2),
    "int64": torch.int64, "long": torch.int64,
    "int32": torch.int32, "int": torch.int32,
    "int16": torch.int16, "int8": torch.int8,
    "uint8": torch.uint8, "uint16": torch.uint16,
    "uint32": torch.uint32, "uint64": torch.uint64,
    "bool": torch.bool,
}


def _dtype(name: str) -> torch.dtype:
    s = name.replace("torch.", "").strip()
    if s in _DTYPE_MAP:
        return _DTYPE_MAP[s]
    return torch.float32


# Argument names that need *structured* values for kernels with semantic
# constraints (typically attention / paged-cache kernels). Random ints would
# crash the kernel because, e.g., ``cu_seqlens_q`` must be a non-decreasing
# prefix-sum starting at 0.
_STRUCTURED_KEYS = {
    # cumulative-seqlen prefix sums for varlen attention
    "cu_seqlens_q", "cu_seqlens_k", "cu_seqlens", "qo_indptr", "kv_indptr",
    "reduce_indptr", "num_kv_splits_indptr", "work_indptr",
    # block / page bookkeeping
    "block_table", "block_tables", "kv_indices", "kv_last_page_lens",
    "kv_last_page_len", "page_indices", "page_indptr",
    # MoE routing
    "topk_ids", "sorted_token_ids", "sorted_expert_ids", "num_valid_ids",
    "expert_ids", "topk_indices", "expert_indptr",
    # general index/slot tensors
    "slot_mapping", "positions", "seq_lens", "context_lens", "query_start_loc",
    "query_lens", "cache_indices", "row_starts", "lengths",
}


def _make_structured(name: str, sig: dict, gen: torch.Generator,
                     device: str = "cuda",
                     ctx: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
    """Generate semantically-valid values for known structured argument names.

    ``ctx`` carries already-resolved scalar args from the same launch (eg.
    ``max_seqlen_q`` from kwargs) so cu_seqlens / block_table sizes can be
    sized against the kernel's own constraints.

    Returns None if we don't have a recipe for ``name``.
    """
    ctx = ctx or {}
    shape = tuple(sig.get("shape", []))
    dtype = _dtype(sig.get("dtype", "int32"))
    nm = name.lower()
    if not shape:
        return None
    n = shape[0] if len(shape) == 1 else shape[-1]
    # Prefix-sum / indptr style: monotonic non-decreasing starting at 0,
    # values bounded by max_seqlen_* if recorded in the same launch.
    if any(k in nm for k in ("cu_seqlens", "indptr", "query_start_loc", "row_starts")):
        if n < 1:
            return torch.zeros(shape, dtype=dtype, device=device)
        # Pick the increment so that each "sequence" has a bounded length and
        # the total fits whatever batched-token tensor was captured for this
        # launch. Defaults: try max_seqlen_q / max_seqlen_k from kwargs first.
        max_len = None
        for k, v in ctx.items():
            kk = k.lower()
            if kk in ("max_seqlen_q", "max_seqlen_k", "max_extend_len",
                      "max_seqlen", "max_context_len") and isinstance(v, int):
                if "_q" in nm and "_q" in kk:
                    max_len = v; break
                if "_k" in nm and "_k" in kk:
                    max_len = v; break
                if max_len is None:
                    max_len = v
        chunk = max(1, min(int(max_len), 64) if max_len else 32)
        # n includes the leading 0 entry, so we have n-1 sequences.
        out = torch.arange(0, n, dtype=dtype, device=device) * chunk
        return out.reshape(shape)
    # seqlen / context_len: small positive ints, bounded by max_seqlen if known.
    if any(k in nm for k in ("seq_lens", "context_lens", "query_lens", "kv_last_page_len", "lengths")):
        max_len = None
        for k, v in ctx.items():
            kk = k.lower()
            if kk.startswith("max_seqlen") and isinstance(v, int):
                max_len = v; break
        v = min(int(max_len), 64) if max_len else 32
        return torch.full(shape, max(1, v), dtype=dtype, device=device)
    # slot_mapping / page_indices / cache_indices: write targets for scattered
    # cache writes. They MUST be unique — duplicate slots make the kernel
    # non-deterministic across runs (concurrent threads racing the same slot).
    # We sample a unique permutation.
    if any(k in nm for k in ("slot_mapping", "page_indices", "cache_indices",
                             "kv_indices")):
        total = 1
        for d in shape:
            total *= d
        # Pool size: at least 4× the number of slots so the permutation has
        # room. Cap at 32k to keep allocation cheap.
        pool = max(total * 4, 256)
        pool = min(pool, 32768)
        perm = torch.randperm(pool, generator=gen, dtype=dtype, device=device)[:total]
        return perm.reshape(shape)
    if any(k in nm for k in ("block_table", "positions")):
        high = 1024
        return torch.randint(0, high, shape, dtype=dtype, device=device, generator=gen)
    # MoE routing — topk_ids/sorted_expert_ids must be valid expert indices.
    # We don't know num_experts here; default to 8 which is common.
    if "expert" in nm or nm == "topk_ids" or nm == "topk_indices":
        return torch.randint(0, 8, shape, dtype=dtype, device=device, generator=gen)
    if nm == "num_valid_ids":
        return torch.tensor([min(shape[0] if shape else 1, 64)] * (shape[0] if shape else 1),
                            dtype=dtype, device=device)
    if "sorted_token_ids" in nm:
        return torch.zeros(shape, dtype=dtype, device=device)
    return None


# ---------------------------------------------------------------- tensor gen
def _make_tensor(sig: dict, gen: torch.Generator, device: str = "cuda",
                 name: str = "",
                 ctx: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    if name and name.lower() in _STRUCTURED_KEYS:
        s = _make_structured(name, sig, gen, device, ctx=ctx)
        if s is not None:
            return s
    shape = tuple(sig.get("shape", []))
    dtype = _dtype(sig.get("dtype", "float32"))
    if dtype.is_floating_point:
        if dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
            getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2),
        ):
            base = (torch.randn(shape, dtype=torch.float32, device=device, generator=gen) * 0.1)
            return base.clamp(-1.0, 1.0).to(dtype)
        return torch.randn(shape, dtype=torch.float32, device=device, generator=gen).to(dtype)
    if dtype == torch.bool:
        return (torch.randint(0, 2, shape, device=device, generator=gen, dtype=torch.int32) > 0)
    # integer tensor — keep values modest so they're plausible indices etc.
    if not shape:
        return torch.tensor(0, dtype=dtype, device=device)
    high = max(2, min(1024, shape[-1] if shape[-1] > 1 else 32))
    # Clamp to dtype range so eg. uint8 doesn't overflow torch.randint's bound
    # check (``to - 1 is out of bounds for unsigned char``).
    if dtype == torch.uint8:
        high = min(high, 256)
    elif dtype == torch.int8:
        high = min(high, 128)
    elif dtype == torch.int16:
        high = min(high, 32768)
    elif dtype == torch.uint16:
        high = min(high, 65536)
    return torch.randint(0, high, shape, dtype=dtype, device=device, generator=gen)


def _decode_opaque(sig: dict) -> Any:
    py_type = (sig.get("py_type") or "").lower()
    rep = sig.get("repr", "")
    if py_type in ("str", "int", "float", "bool", "nonetype"):
        try:
            import ast as _ast
            return _ast.literal_eval(rep)
        except Exception:
            return None
    if py_type == "dtype" and rep.startswith("torch."):
        return _dtype(rep[6:])
    # Captured aiter enums look like ``<QuantType.No: 0>``.  Extracted tasks
    # call pybind functions directly, and those bindings accept the underlying
    # integer rather than the Python enum object from an installed aiter package.
    if "." in rep and rep.startswith("<") and ":" in rep:
        try:
            import re as _re
            m = _re.match(r"<([\w.]+)\.(\w+):\s*(-?\d+)>", rep)
            if m:
                return int(m.group(3))
        except Exception:
            pass
    return None


def _arg_from_sig(sig: dict, gen: torch.Generator, name: str = "",
                  ctx: Optional[Dict[str, Any]] = None) -> Any:
    kind = sig.get("kind", "scalar")
    if kind == "tensor":
        return _make_tensor(sig, gen, name=name, ctx=ctx)
    if kind == "scalar":
        return sig.get("value")
    if kind == "seq":
        return [_arg_from_sig(s, gen, ctx=ctx) for s in sig.get("items", [])]
    if kind == "map":
        return {k: _arg_from_sig(v, gen, name=k, ctx=ctx) for k, v in sig.get("items", {}).items()}
    if kind == "opaque":
        return _decode_opaque(sig)
    return None


def _resolve_scalar_ctx(test_case: dict) -> Dict[str, Any]:
    """First-pass scan: pull any scalar / opaque-scalar values out of the
    launch's args+kwargs so structured tensor generators (cu_seqlens, etc.)
    can size themselves against ``max_seqlen_q`` and friends.
    """
    ctx: Dict[str, Any] = {}
    for entry in (test_case.get("kwargs_sig") or {}).items():
        k, v = entry
        if not isinstance(v, dict):
            continue
        if v.get("kind") == "scalar":
            val = v.get("value")
            if isinstance(val, (int, float, bool, str)):
                ctx[k] = val
        elif v.get("kind") == "opaque":
            dec = _decode_opaque(v)
            if isinstance(dec, (int, float, bool, str)):
                ctx[k] = dec
    return ctx


def build_inputs(test_case: dict, seed: int = 0xC0FFEE) -> Tuple[List[Any], Dict[str, Any]]:
    """Materialize positional + keyword args from a captured launch signature.

    A first pass extracts scalar kwargs (``max_seqlen_q``, etc.) into a context
    dict that the tensor builder consults — so eg. ``cu_seqlens_k`` is sized
    so that its max value matches the captured ``max_seqlen_k``.

    If the test_case carries ``args_names`` (parsed from the op schema), they
    are forwarded to the per-position tensor builder so structured generators
    (slot_mapping → unique perm, cu_seqlens_q → prefix sum) fire even when
    the kernel is called positionally.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA / ROCm device required to build inputs")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ctx = _resolve_scalar_ctx(test_case)
    args_sig = test_case.get("args_sig", [])
    args_names = test_case.get("args_names") or [""] * len(args_sig)
    if len(args_names) < len(args_sig):
        args_names = list(args_names) + [""] * (len(args_sig) - len(args_names))
    args = [
        _arg_from_sig(s, gen, name=args_names[i], ctx=ctx)
        for i, s in enumerate(args_sig)
    ]
    kwargs: Dict[str, Any] = {}
    for k, v in (test_case.get("kwargs_sig") or {}).items():
        kwargs[k] = _arg_from_sig(v, gen, name=k, ctx=ctx)
    return args, kwargs


# ----------------------------------------------- op-specific input repair
# The recorder captures aiter wrappers *positionally* (no arg names), so the
# name-keyed structured generators in ``build_inputs`` never fire for them.
# For paged_attention_ragged that leaves ``kv_indptr`` as random ints (not a
# prefix sum) -> negative/garbage context_len -> the kernel reads out of range
# and produces NaN. We rewrite the KV bookkeeping into a self-consistent paged
# layout. Fully deterministic (no RNG) so the kernel run and the reference run
# observe byte-identical inputs.
def make_consistent_paged_attention_ragged(args: List[Any], kwargs: Dict[str, Any] | None = None) -> None:
    """In-place repair of paged_attention_ragged positional args.

    Layout: 0 out, 1 workspace, 2 query, 3 key_cache, 4 value_cache, 5 scale,
    6 kv_indptr, 7 kv_page_indices, 8 kv_last_page_lens, 9 block_size, ...
    """
    if len(args) < 10:
        return
    query, key_cache = args[2], args[3]
    kv_indptr, kv_page_indices, kv_last_page_lens = args[6], args[7], args[8]
    if not (isinstance(query, torch.Tensor) and isinstance(key_cache, torch.Tensor)
            and isinstance(kv_indptr, torch.Tensor) and isinstance(kv_page_indices, torch.Tensor)):
        return
    num_seqs = int(query.shape[0])
    if kv_indptr.numel() < num_seqs + 1:
        return
    total_pages = int(kv_page_indices.numel())
    num_blocks = int(key_cache.shape[0])
    dev = kv_indptr.device
    try:
        bs = int(args[9])
    except Exception:
        bs = 1
    # Monotonic non-decreasing prefix sum 0..total_pages over num_seqs segments
    # (as even a split as possible) -> every seq gets a valid positive context.
    base = total_pages // num_seqs
    rem = total_pages % num_seqs
    counts = torch.full((num_seqs,), base, dtype=torch.int64, device=dev)
    if rem:
        counts[:rem] += 1
    indptr = torch.zeros(num_seqs + 1, dtype=torch.int64, device=dev)
    indptr[1:] = torch.cumsum(counts, 0)
    kv_indptr[: num_seqs + 1] = indptr.to(kv_indptr.dtype)
    # In-bounds page ids (unique while they fit), deterministic.
    idx = torch.arange(total_pages, dtype=torch.int64, device=dev)
    if num_blocks > 0:
        idx = idx % num_blocks
    kv_page_indices[:] = idx.to(kv_page_indices.dtype)
    # Lift query magnitude so the softmax is non-degenerate. With random Q/K over
    # a long context the attention is near-uniform and the output collapses
    # toward ~0, where the bf16 atol (5e-2) would mask scale/precision errors
    # (e.g. a 1.5x-wrong kernel would still "pass"). A modest query gain peaks the
    # distribution (output ~O(1)) so the reference comparison is discriminative,
    # while the kernel still matches the reference well within tolerance.
    if isinstance(query, torch.Tensor) and query.is_floating_point():
        query.mul_(4.0)
    # block_size==1 passes kv_last_page_lens as NULL (unused); for >1 keep it
    # within [1, block_size] so the last partial page is well-defined.
    if bs > 1 and isinstance(kv_last_page_lens, torch.Tensor) and kv_last_page_lens.numel() >= num_seqs:
        kv_last_page_lens[:num_seqs] = bs


# ---------------------------------------------------------------- aiter wrapper -> pybind normalization
def _dtype_to_aiter_string(value: Any) -> Any:
    if value is None:
        return None
    if value is torch.float16:
        return "fp16"
    if value is torch.bfloat16:
        return "bf16"
    if value is torch.float32:
        return "fp32"
    if isinstance(value, str):
        return value
    return None


def _sanitize_moe_routing(args: List[Any], kwargs: Dict[str, Any]) -> None:
    if len(args) < 8:
        return
    hidden, w1 = args[0], args[1]
    sorted_token_ids, sorted_expert_ids, num_valid_ids = args[3], args[4], args[5]
    if not all(isinstance(x, torch.Tensor) for x in (hidden, w1, sorted_token_ids, sorted_expert_ids, num_valid_ids)):
        return
    try:
        topk = int(args[7])
    except Exception:
        topk = int(kwargs.get("topk", 1) or 1)
    tokens = int(hidden.shape[0])
    experts = int(w1.shape[0])
    valid = max(1, tokens * max(1, topk))
    with torch.no_grad():
        ids = torch.arange(sorted_token_ids.numel(), device=sorted_token_ids.device,
                           dtype=sorted_token_ids.dtype) % valid
        sorted_token_ids.copy_(ids.reshape_as(sorted_token_ids))
        eids = torch.arange(sorted_expert_ids.numel(), device=sorted_expert_ids.device,
                            dtype=sorted_expert_ids.dtype) % max(1, experts)
        sorted_expert_ids.copy_(eids.reshape_as(sorted_expert_ids))
        num_valid_ids.fill_(min(sorted_token_ids.numel(), valid))


def normalize_aiter_call(py_fn_name: str, fc_name: str,
                         args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """Translate captured Python-wrapper arguments to the direct pybind ABI.

    The recorder observes public aiter wrappers such as ``flash_attn_varlen_func``
    and ``ck_moe_stage1_fwd``.  Extracted tasks intentionally call the local
    pybind module directly, so a few wrapper-only keyword names and defaults need
    to be normalized before invocation.
    """
    op = py_fn_name or fc_name
    args = list(args)
    kwargs = dict(kwargs)

    if op in ("ck_moe_stage1_fwd", "ck_moe_stage2_fwd", "ck_moe_stage1", "ck_moe_stage2"):
        if "use_non_temporal_load" in kwargs and "non_temporal_load" not in kwargs:
            kwargs["non_temporal_load"] = kwargs.pop("use_non_temporal_load")
        elif "use_non_temporal_load" in kwargs:
            kwargs.pop("use_non_temporal_load", None)
        if len(args) > 8:
            kwargs.pop("kernelName", None)
        elif kwargs.get("kernelName") is None:
            kwargs["kernelName"] = ""
        if "dst_type" in kwargs:
            kwargs["dst_type"] = _dtype_to_aiter_string(kwargs["dst_type"])
        # Positional captures include dst_type just before is_shuffled.
        if len(args) >= 18:
            args[17] = _dtype_to_aiter_string(args[17])
        _sanitize_moe_routing(args, kwargs)
        return args, kwargs

    if op == "flash_attn_varlen_func":
        if args:
            return args, kwargs
        window = kwargs.pop("window_size", [-1, -1])
        if window is None:
            window = [-1, -1]
        window_left = int(window[0]) if len(window) > 0 else -1
        window_right = int(window[1]) if len(window) > 1 else -1
        kwargs = {
            "q": kwargs.get("q"),
            "k": kwargs.get("k"),
            "v": kwargs.get("v"),
            "cu_seqlens_q": kwargs.get("cu_seqlens_q"),
            "cu_seqlens_k": kwargs.get("cu_seqlens_k"),
            "max_seqlen_q": kwargs.get("max_seqlen_q"),
            "max_seqlen_k": kwargs.get("max_seqlen_k"),
            "min_seqlen_q": kwargs.get("min_seqlen_q", 0),
            "dropout_p": kwargs.get("dropout_p", 0.0),
            "softmax_scale": kwargs.get("softmax_scale", 1.0),
            "logits_soft_cap": kwargs.get("logits_soft_cap", 0.0),
            "zero_tensors": kwargs.get("zero_tensors", False),
            "is_causal": kwargs.get("is_causal", kwargs.get("causal", False)),
            "window_size_left": window_left,
            "window_size_right": window_right,
            "return_softmax_lse": kwargs.get("return_softmax_lse", kwargs.get("return_lse", False)),
            "return_dropout_randval": kwargs.get("return_dropout_randval", False),
            "how_v3_bf16_cvt": kwargs.get("how_v3_bf16_cvt", 1),
            "out": kwargs.get("out"),
            "block_table": kwargs.get("block_table"),
            "bias": kwargs.get("bias"),
            "alibi_slopes": kwargs.get("alibi_slopes"),
            "gen": kwargs.get("gen"),
            "cu_seqlens_q_padded": kwargs.get("cu_seqlens_q_padded"),
            "cu_seqlens_k_padded": kwargs.get("cu_seqlens_k_padded"),
        }
        return args, kwargs

    return args, kwargs


def normalize_aiter_output(py_fn_name: str, value: Any) -> Any:
    if py_fn_name == "flash_attn_varlen_func" and isinstance(value, (list, tuple)):
        # AITER returns (out, softmax_lse, dropout_mask, rng_state).  With
        # dropout disabled the rng_state buffer is not semantically meaningful
        # and may contain run-to-run garbage; compare the observable outputs.
        return tuple(value[:2])
    return value


# ---------------------------------------------------------------- comparison
def _tol_for(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 5e-2, 5e-2
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                 getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
                 getattr(torch, "float8_e5m2fnuz", torch.float8_e5m2)):
        return 0.5, 0.5
    if dtype.is_floating_point:
        return 1e-3, 1e-3
    return 0, 0


def compare(got: Any, expected: Any) -> Optional[str]:
    """Return None if equal-within-tolerance, else a short diff message."""
    if isinstance(got, torch.Tensor) and isinstance(expected, torch.Tensor):
        if got.shape != expected.shape:
            return f"shape mismatch: got {tuple(got.shape)} vs {tuple(expected.shape)}"
        atol, rtol = _tol_for(got.dtype)
        a = got.detach().to(torch.float32).cpu()
        b = expected.detach().to(torch.float32).cpu()
        if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            diff = (a - b).abs().max().item()
            return f"max abs diff {diff:.4g} > atol={atol}"
        return None
    if isinstance(got, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(got) != len(expected):
            return f"len mismatch: {len(got)} vs {len(expected)}"
        for i, (g, e) in enumerate(zip(got, expected)):
            err = compare(g, e)
            if err:
                return f"[{i}]: {err}"
        return None
    if got == expected:
        return None
    return f"value mismatch"


# ---------------------------------------------------------------- references
# Each reference takes the SAME (args, kwargs) as the kernel and returns the
# expected output. For in-place kernels the reference returns a tensor that
# the runner will compare against the (now-mutated) input. The runner picks
# the entry by op_name; if missing, falls back to determinism check.
ReferenceFn = Callable[[List[Any], Dict[str, Any]], Any]


def _ref_rms_norm(args: list, kwargs: dict):
    # signature: (out, input, weight, epsilon) — vLLM `_C.rms_norm` writes to out
    out, inp, weight, eps = args[0], args[1], args[2], args[3]
    var = inp.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (inp.to(torch.float32) * torch.rsqrt(var + eps)).to(inp.dtype) * weight.to(inp.dtype)


def _ref_fused_add_rms_norm(args: list, kwargs: dict):
    # (input, residual, weight, epsilon) — in-place: input = rmsnorm(input+residual)
    inp, res, weight, eps = args[0], args[1], args[2], args[3]
    s = (inp + res).to(torch.float32)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    return (s * torch.rsqrt(var + eps)).to(inp.dtype) * weight.to(inp.dtype)


def _ref_silu_and_mul(args: list, kwargs: dict):
    # (out, input) — out = silu(input[..., :H/2]) * input[..., H/2:]
    out, inp = args[0], args[1]
    a, b = inp.chunk(2, dim=-1)
    return torch.nn.functional.silu(a.to(torch.float32)).to(inp.dtype) * b


def _ref_gelu_and_mul(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    a, b = inp.chunk(2, dim=-1)
    return torch.nn.functional.gelu(a.to(torch.float32)).to(inp.dtype) * b


def _ref_gelu_tanh_and_mul(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    a, b = inp.chunk(2, dim=-1)
    return torch.nn.functional.gelu(a.to(torch.float32), approximate="tanh").to(inp.dtype) * b


def _ref_gelu_quick(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    f = inp.to(torch.float32)
    return (f * torch.sigmoid(1.702 * f)).to(inp.dtype)


def _ref_gelu_new(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    f = inp.to(torch.float32)
    return (0.5 * f * (1 + torch.tanh(math.sqrt(2 / math.pi) * (f + 0.044715 * f.pow(3))))).to(inp.dtype)


def _ref_gelu_fast(args: list, kwargs: dict):
    return _ref_gelu_new(args, kwargs)


def _ref_mul_and_silu(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    a, b = inp.chunk(2, dim=-1)
    return (a.to(torch.float32) * torch.nn.functional.silu(b.to(torch.float32))).to(inp.dtype)


def _ref_fatrelu_and_mul(args: list, kwargs: dict):
    out, inp = args[0], args[1]
    threshold = float(kwargs.get("threshold", args[2] if len(args) > 2 else 0.0))
    a, b = inp.chunk(2, dim=-1)
    mask = (a.to(torch.float32) > threshold).to(inp.dtype)
    return (a * mask) * b


def _ref_paged_attention_ragged(args: list, kwargs: dict):
    """Analytic decode paged-attention over a ragged paged KV cache.

    Mirrors aiter's ``paged_attention_ragged`` (one query token per sequence):
      out[s,h] = softmax( (K_s[:,kvh] @ q[s,h]) * scale [+ alibi] [soft_cap] ) @ V_s[:,kvh]
    with kvh = h // (num_q_heads // num_kv_heads), and the per-seq KV gathered
    from pages ``kv_page_indices[kv_indptr[s] : kv_indptr[s+1]]``. Assumes the
    inputs were made self-consistent by ``make_consistent_paged_attention_ragged``.
    Returns the [num_seqs, num_q_heads, head_size] output to compare against the
    kernel's in-place ``out`` (arg 0).
    """
    query, key_cache, value_cache = args[2], args[3], args[4]
    scale = float(args[5])
    kv_indptr, kv_page_indices, kv_last_page_lens = args[6], args[7], args[8]
    block_size = int(args[9]) if len(args) > 9 else 1
    alibi = args[11] if len(args) > 11 else None
    layout = args[13] if len(args) > 13 else "NHD"
    soft_cap = float(args[14]) if (len(args) > 14 and args[14] is not None) else 0.0

    num_seqs, num_q_heads, head_size = (int(x) for x in query.shape)
    # key/value cache layouts: NHD = [blocks, block_size, kv_heads, head], HND
    # = [blocks, kv_heads, block_size, head].
    if layout == "HND":
        _, num_kv_heads, _, _ = (int(x) for x in key_cache.shape)
        kc = key_cache.permute(0, 2, 1, 3)
        vc = value_cache.permute(0, 2, 1, 3)
    else:
        _, _, num_kv_heads, _ = (int(x) for x in key_cache.shape)
        kc, vc = key_cache, value_cache
    gqa = max(1, num_q_heads // num_kv_heads)
    dev = query.device
    q = query.float()
    indptr = kv_indptr.to(torch.int64).tolist()
    pages_all = kv_page_indices.to(torch.int64)
    kvh_of = torch.arange(num_q_heads, device=dev) // gqa
    out = torch.zeros((num_seqs, num_q_heads, head_size), dtype=torch.float32, device=dev)
    for s in range(num_seqs):
        st, en = int(indptr[s]), int(indptr[s + 1])
        if en - st <= 0:
            continue
        pages = pages_all[st:en]
        Kp = kc[pages].float()  # [n_pages, block_size, kv_heads, head]
        Vp = vc[pages].float()
        if block_size > 1:
            n_pages = en - st
            last = (int(kv_last_page_lens[s]) if isinstance(kv_last_page_lens, torch.Tensor)
                    and kv_last_page_lens.numel() > s else block_size)
            last = min(max(1, last), block_size)
            K = Kp.reshape(n_pages * block_size, num_kv_heads, head_size)[: (n_pages - 1) * block_size + last]
            V = Vp.reshape(n_pages * block_size, num_kv_heads, head_size)[: (n_pages - 1) * block_size + last]
        else:
            K = Kp[:, 0]  # [ctx, kv_heads, head]
            V = Vp[:, 0]
        Kq = K[:, kvh_of]  # [ctx, num_q_heads, head]
        Vq = V[:, kvh_of]
        logits = torch.einsum("thd,hd->ht", Kq, q[s]) * scale  # [num_q_heads, ctx]
        if isinstance(alibi, torch.Tensor):
            pos = torch.arange(K.shape[0], device=dev, dtype=torch.float32)
            logits = logits + alibi.float().reshape(-1, 1)[:num_q_heads] * (pos - (K.shape[0] - 1))
        if soft_cap > 0:
            logits = soft_cap * torch.tanh(logits / soft_cap)
        probs = torch.softmax(logits, dim=-1)
        out[s] = torch.einsum("ht,thd->hd", probs, Vq)
    return out.to(query.dtype)


# References are keyed by ``"<source>:<op_name>"`` because vLLM and AITER
# expose ops with the same short name but different positional layouts (e.g.
# vLLM ``fused_add_rms_norm(input, residual, weight, eps)`` vs. AITER
# ``rmsnorm2d_fwd_with_add(out, input, residual, out_residual, weight, eps)``).
# The runner passes its source prefix; unknown keys fall back to a determinism
# check, which is a safe no-op rather than producing NaN garbage.
REFERENCES: Dict[str, ReferenceFn] = {
    # vLLM _C ops — runner passes ``vllm:<op>``
    "vllm:rms_norm": _ref_rms_norm,
    "vllm:fused_add_rms_norm": _ref_fused_add_rms_norm,
    "vllm:silu_and_mul": _ref_silu_and_mul,
    "vllm:gelu_and_mul": _ref_gelu_and_mul,
    "vllm:gelu_tanh_and_mul": _ref_gelu_tanh_and_mul,
    "vllm:gelu_quick": _ref_gelu_quick,
    "vllm:gelu_new": _ref_gelu_new,
    "vllm:gelu_fast": _ref_gelu_fast,
    "vllm:mul_and_silu": _ref_mul_and_silu,
    "vllm:fatrelu_and_mul": _ref_fatrelu_and_mul,
    # SGLang sgl_kernel ops mostly mirror vLLM's signatures.
    "sglang:rms_norm": _ref_rms_norm,
    "sglang:fused_add_rms_norm": _ref_fused_add_rms_norm,
    "sglang:silu_and_mul": _ref_silu_and_mul,
    "sglang:gelu_and_mul": _ref_gelu_and_mul,
    "sglang:gelu_tanh_and_mul": _ref_gelu_tanh_and_mul,
    "sglang:gelu_quick": _ref_gelu_quick,
    # AITER paged attention (decode). Pair with
    # make_consistent_paged_attention_ragged so the gathered KV is well-formed.
    "aiter:paged_attention_ragged": _ref_paged_attention_ragged,
}


def reference_for(op_name: str, source: str = "") -> Optional[ReferenceFn]:
    """Pick a reference for ``op_name``. If ``source`` is given (``vllm`` /
    ``aiter`` / ``triton`` / ``sglang``) it is used as a namespace prefix to
    disambiguate same-named ops with different signatures."""
    if source:
        fn = REFERENCES.get(f"{source}:{op_name}")
        if fn is not None:
            return fn
    return REFERENCES.get(op_name)


# ---------------------------------------------------------------- output capture
# Many vLLM/AITER kernels mutate their first ``out`` arg rather than returning
# a value. The runner inspects the first positional tensor argument's bytes
# before/after the call to detect this and treat it as the output.

def snapshot(args: list) -> list:
    return [a.detach().clone() if isinstance(a, torch.Tensor) else None for a in args]


def detect_output(pre: list, post: list, ret: Any) -> Any:
    """Pick the most plausible output for comparison.

    1. If the kernel returned a tensor (or tuple), use that.
    2. Otherwise look for the first positional tensor that changed in-place.
    3. Else None.
    """
    if isinstance(ret, torch.Tensor) or isinstance(ret, (list, tuple)) and ret and isinstance(ret[0], torch.Tensor):
        return ret
    for i, (b, a) in enumerate(zip(pre, post)):
        if a is None or b is None:
            continue
        try:
            if not torch.equal(b.to(torch.float32).cpu(), a.to(torch.float32).cpu()):
                return a
        except Exception:
            continue
    return None

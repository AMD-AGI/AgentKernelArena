# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Census rows -> observations of one task's var axes, per operator family.

A matcher receives the task's axes (name -> declaration, as in ``task.yaml``),
its tensor inputs and the census, and returns ``Observation`` objects: one per
census row whose shape matches every const axis of the scenario, with the
row's var-axis values, GPU time and calls. ``select`` turns them into cases.

Register a matcher with ``@matcher(<regex over operator.name>)``; the first
registered pattern that matches the task's operator wins. A matcher only uses
const axes it can read from the recorded shapes; the axes it could not check are
reported in ``unchecked_const_axes`` so a reviewer can confirm them. Matchers
accept every phase (prefill, decode, verify, draft, draft_prefill, outside_step):
all of them are production calls.

A census ``ops`` row lists the op's non-empty tensor input shapes in ``dims`` and
its input types, scalars included, in ``dtypes``; the two lists do not align, so
matchers find tensors by shape and read formats from the types present. Kernels
launched outside a recorded op (most Triton kernels called from Python) have a
row per kernel name with no shapes; a matcher that sees such rows reports their
time in its notes.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

# Task dtypes -> torch type names in the profiler's "Input type" (without the ``c10::`` prefix), compared exactly.
DTYPE_ALIASES = {
    "bfloat16": ("BFloat16",), "float16": ("Half",), "float32": ("float", "Float"),
    "float8_e4m3fn": ("Float8_e4m3fn",), "float8_e4m3fnuz": ("Float8_e4m3fnuz",),
    "int8": ("signed char", "Char"), "uint8": ("unsigned char", "Byte", "Float4_e2m1fn_x2"),
    "int32": ("int",), "int64": ("long int", "long"), "bool": ("bool",),
}
FP8_CODES = ("Float8_e4m3fn", "Float8_e4m3fnuz")
E8M0 = ("Float8_e8m0fnu",)
FP4_CODES = ("Float4_e2m1fn_x2",)
BYTES = ("unsigned char", "Byte")
INT8_CODES = ("signed char", "Char")


@dataclass
class Observation:
    axes: dict[str, int]          # var axis -> value
    us: float                     # GPU time (microseconds)
    calls: float
    phase: str                    # prefill | decode | verify | draft | draft_prefill | outside_step
    source: str                   # census row description
    pad_axis: str | None = None   # var axis that decode CUDA graphs pad to a captured batch size
    extrapolated: bool = False    # time rests on a per-call cost extrapolated beyond the measured shapes


@dataclass
class Match:
    observations: list[Observation]
    matcher: str
    unchecked_const_axes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


REGISTRY: list[tuple[re.Pattern, str, Callable]] = []


def matcher(pattern: str, name: str | None = None):
    """Register ``fn`` for operators matching ``pattern``; ``name`` (default: the function name) is reported."""
    def register(fn):
        REGISTRY.append((re.compile(pattern), name or fn.__name__, fn))
        return fn
    return register


def find(operator: str) -> tuple[str, Callable] | None:
    for rx, name, fn in REGISTRY:
        if rx.search(operator):
            return name, fn
    return None


def _type(census_dtype: str) -> str:
    return census_dtype.replace("c10::", "").strip()


def dtype_ok(task_dtype: str | None, census_dtype: str) -> bool:
    if not task_dtype:
        return True
    return _type(census_dtype) in DTYPE_ALIASES.get(task_dtype, (task_dtype,))


def _has(row: dict, types: tuple) -> bool:
    """Whether any input type of an ops row is one of ``types``."""
    return any(_type(t) in types for t in row.get("dtypes") or [])


def _shapes(row: dict) -> list[list[int]]:
    """Tensor input shapes of an ops row (tensor lists left out)."""
    return [d for d in row.get("dims") or [] if isinstance(d, list) and d and all(isinstance(x, int) for x in d)]


def _type_of(row: dict, shape: list) -> str | None:
    """Type of the input with ``shape``: tensor types (scalars dropped) align with ``dims`` when the counts agree."""
    types = [_type(t) for t in row.get("dtypes") or [] if _type(t) != "Scalar"]
    dims = row.get("dims") or []
    return types[dims.index(shape)] if len(types) == len(dims) and shape in dims else None


def _names(row: dict) -> str:
    return " ".join([row.get("op") or "", *(row.get("kernels") or {})])


def _invocations(row: dict) -> float:
    """Op calls of an ops row. ``calls`` counts kernel launches; an op that launches several kernels per call counts
    once: the census's ``op_calls`` when recorded, else launches x (launches of its rarest kernel / all launches)."""
    if row.get("op_calls") is not None:
        return row["op_calls"]
    counts = [c for c in (row.get("kernels") or {}).values() if c]
    return row["calls"] * min(counts) / sum(counts) if counts else row["calls"]


def _first_dtype(inputs: dict) -> str | None:
    return next((d for d in inputs.values() if d.get("kind") == "tensor"), {}).get("dtype")


def _const(axes: dict, *names: str) -> tuple[str, int] | None:
    for n in names:
        if n in axes and axes[n]["kind"] == "const":
            return n, axes[n]["value"]
    return None


def _var(axes: dict) -> list[str]:
    return [n for n, d in axes.items() if d["kind"] == "var"]


def _unchecked(axes: dict, checked: set) -> list[str]:
    return [n for n, d in axes.items() if d["kind"] == "const" and n not in checked]


def _checked_by_all(per_row: list[set]) -> set:
    """Const axes verified by every matched row (an axis some rows could not read stays unchecked)."""
    return set.intersection(*per_row) if per_row else set()


def _unlinked_note(census: dict, kernel_regex: str) -> list[str]:
    us = sum(r["total_us"] for r in census.get("ops", [])
             if not r.get("op") and re.search(kernel_regex, _names(r), re.I))
    return [f"{us / 1e3:.1f} ms of matching kernels ran outside a recorded op: no shapes, not counted"] if us else []


GEMM_M, GEMM_N, GEMM_K, GEMM_B = ("m", "tokens", "n_tokens", "num_tokens"), ("n", "n_out"), ("k",), ("batch", "b")


def _axis(axes: dict, names: tuple) -> str | None:
    return next((n for n in names if n in axes), None)


@matcher(r"gemm|gemv|matmul$|^bmm")
def gemm(axes: dict, inputs: dict, census: dict) -> Match:
    """Dense GEMMs by (M, N, K[, batch]); the first tensor input's dtype selects the census dtype.

    Uses the census's log-weighted GEMM rows (whole steady state) when present, else the window rows.
    """
    roles = {"M": _axis(axes, GEMM_M), "N": _axis(axes, GEMM_N), "K": _axis(axes, GEMM_K), "B": _axis(axes, GEMM_B)}
    if not roles["M"] or not roles["N"] or not roles["K"]:
        return Match([], "gemm", notes=["task axes do not name m/n/k"])
    extra = [n for n in axes if n not in roles.values()]
    first = next((d for d in inputs.values() if d.get("kind") == "tensor"), {})
    obs = []
    source = "gemm_log_weighted" if census.get("gemm_log_weighted") else "gemm"
    for row in census.get(source, []):
        values = {"M": row["M"], "N": row["N"], "K": row["K"], "B": row.get("B", 1)}
        if roles["B"] is None and values["B"] != 1:
            continue
        if not dtype_ok(first.get("dtype"), row["dtype"]):
            continue
        ok, var = True, {}
        for role, name in roles.items():
            if name is None:
                continue
            d = axes[name]
            if d["kind"] == "const":
                ok &= d["value"] == values[role]
            else:
                var[name] = values[role]
        if ok and var:
            obs.append(Observation(var, row["total_us"], row["calls"], row["phase"],
                                   f"GEMM M={row['M']} N={row['N']} K={row['K']} {row['phase']}",
                                   pad_axis=roles["M"] if roles["M"] in var else None,
                                   extrapolated=bool(row.get("extrapolated"))))
    notes = ["GEMM time from the census's log-weighted rows: " + census.get("gemm_log_weighting", {}).get(
        "method", "")] if source == "gemm_log_weighted" else []
    return Match(obs, "gemm", unchecked_const_axes=[n for n in extra if axes[n]["kind"] == "const"], notes=notes)


@matcher(r"paged_attention_decode|decode_attention|paged_decode")
def decode_attention(axes: dict, inputs: dict, census: dict) -> Match:
    """Decode attention by (num_seqs, KV tokens per sequence); KV length is the logged mean per sequence."""
    seqs = _axis(axes, ("num_seqs", "batch", "batch_size"))
    kv = _axis(axes, ("max_seq_len", "seq_len", "kv_len", "context_len"))
    heads, dim = _const(axes, "num_query_heads", "num_heads"), _const(axes, "head_size", "head_dim")
    obs = []
    for row in census.get("attention", []):
        if row["phase"] != "decode" or "batch" not in row:
            continue
        if heads and row.get("heads") not in (None, heads[1]):
            continue
        if dim and row.get("head_dim") not in (None, dim[1]):
            continue
        var = {}
        if seqs and axes[seqs]["kind"] == "var":
            var[seqs] = row["batch"]
        if kv and axes[kv]["kind"] == "var":
            if "kv_tokens_per_seq" not in row:
                continue
            var[kv] = max(1, row["kv_tokens_per_seq"])
        if var:
            obs.append(Observation(var, row["total_us"], row["calls"], "decode",
                                   f"decode attention batch={row['batch']} kv/seq={row.get('kv_tokens_per_seq')}",
                                   pad_axis=seqs if seqs in var else None))
    checked = {seqs, kv, heads and heads[0], dim and dim[0]}
    return Match(obs, "decode_attention",
                 unchecked_const_axes=[n for n, d in axes.items() if d["kind"] == "const" and n not in checked],
                 notes=["KV length is the scheduler log's mean KV tokens per sequence of the step "
                        "(rounded to the census KV bucket), not the batch maximum"])


def _rows(axes: dict, census: dict, op_regex: str, width_names: tuple, families: tuple, name: str) -> Match:
    """Token-row operators: first input [tokens, ..., width]; tokens is the var axis, width a const axis."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], name, notes=["token-row matcher supports one var axis"])
    width = _const(axes, *width_names)
    obs = []
    for row in census.get("ops", []):
        if row["family"] not in families or not row["dims"] or not row["op"]:
            continue
        if not re.search(op_regex, row["op"] + " " + " ".join(row.get("kernels", {})), re.I):
            continue
        first = row["dims"][0]
        if not isinstance(first, list) or len(first) < 2:
            continue
        tokens, w = math.prod(first[:-1]), first[-1]
        if width and w != width[1]:
            continue
        obs.append(Observation({var[0]: tokens}, row["total_us"], row["calls"], row["phase"],
                               f"{row['op']} {first} {row['phase']}", pad_axis=var[0]))
    return Match(obs, name, unchecked_const_axes=[n for n, d in axes.items()
                                                  if d["kind"] == "const" and (not width or n != width[0])])


def _rms_norm_rows(axes: dict, inputs: dict, census: dict, name: str, fused_add: bool) -> Match:
    """RMSNorm ops by their op name (not the kernel's: AITER runs plain ``aiter::rmsnorm`` with a kernel named
    ``add_rmsnorm_quant_kernel``): rows [T, hidden] with a weight [hidden]. ``fused_add`` selects the residual-add
    ops, else plain ones; ops that also quantize, or fuse other norms (q/k pairs, gated, dual), are neither."""
    var = _var(axes)
    width = _const(axes, "hidden_size", "hidden")
    if len(var) != 1 or not width:
        return Match([], name, notes=["rms_norm matcher needs one var axis and a const hidden size"])
    src = _first_dtype(inputs)
    obs = []
    for row in census.get("ops", []):
        op = (row.get("op") or "").split("::")[-1].lower()
        if row["family"] != "norm" or not re.search(r"rms_?norm", op):
            continue
        is_add = bool(re.search(r"add_?rms|fused_add|residual", op))
        if is_add != fused_add or re.search(r"quant|qk|gated|dual|group", op):
            continue
        if _has(row, FP8_CODES + E8M0 + FP4_CODES + INT8_CODES):
            continue  # writes quantized output: a fused norm + quantization op
        if src and not _has(row, DTYPE_ALIASES.get(src, (src,))):
            continue
        shapes = _shapes(row)
        first = shapes[0] if shapes else []
        if len(first) < 2 or first[-1] != width[1] or [width[1]] not in shapes:
            continue
        obs.append(Observation({var[0]: math.prod(first[:-1])}, row["total_us"], row["calls"], row["phase"],
                               f"{row['op']} {first} {row['phase']}", pad_axis=var[0]))
    return Match(obs, name, unchecked_const_axes=_unchecked(axes, {width[0]}))


@matcher(r"^fused_add_rms_norm|^add_rms_norm")
def fused_add_rms_norm(axes, inputs, census):
    return _rms_norm_rows(axes, inputs, census, "fused_add_rms_norm", fused_add=True)


@matcher(r"^rms_norm$")
def rms_norm(axes, inputs, census):
    return _rms_norm_rows(axes, inputs, census, "rms_norm", fused_add=False)


def _rejected_notes(rejected: dict) -> list[str]:
    """Notes on the operator's rows a matcher did not use, by reason (time summed per reason)."""
    return [f"{us / 1e3:.1f} ms of the operator's rows not used: {why}"
            for why, us in sorted(rejected.items(), key=lambda kv: -kv[1])]


def _compare(axes: dict, seen: dict) -> tuple[dict, set]:
    """Const axes against the values a row shows: ({axis: capture value} for mismatches, axes verified)."""
    wrong, checked = {}, set()
    for name, value in seen.items():
        ax = axes.get(name)
        if not ax or ax["kind"] != "const" or value is None:
            continue
        if value != ax["value"]:
            wrong[name] = value
        else:
            checked.add(name)
    return wrong, checked


def _mismatch(store: dict, row: dict, wrong: dict) -> None:
    """Record a row rejected for ``wrong`` ({axis or reason: capture value}), per phase and set of reasons."""
    entry = store.setdefault((row["phase"], tuple(wrong)), [0.0, defaultdict(set)])
    entry[0] += row["total_us"]
    for name, value in wrong.items():
        entry[1][name].add(value)


def _mismatch_notes(store: dict, axes: dict) -> list[str]:
    """One note per phase and set of mismatches: the time and the range of the capture's values vs the task's."""
    notes = []
    for (phase, names), (us, values) in sorted(store.items(), key=lambda kv: -kv[1][0]):
        parts = []
        for name in names:
            vs = sorted(values[name], key=lambda v: (isinstance(v, str), v))
            ax = axes.get(name) or {}
            parts.append(f"{name} {vs[0]}" + (f"..{vs[-1]}" if len(vs) > 1 else "")
                         + (f" (task {ax['value']})" if ax.get("kind") == "const" else
                            f" (task {ax['min']}..{ax['max']})" if ax.get("kind") == "var" else ""))
        notes.append(f"{us / 1e3:.1f} ms of {phase} rows not used: {'; '.join(parts)}")
    return notes


def _pool_size(axes: dict, name: str, value: int | None, checked: set, notes: set) -> None:
    """A pool or table size the kernel only indexes into: verified when equal, else noted and left unchecked."""
    ax = axes.get(name)
    if not ax or ax["kind"] != "const" or value is None:
        return
    if value == ax["value"]:
        checked.add(name)
    else:
        notes.add(f"{name}: the capture's is {value} (task {ax['value']}); it only bounds the indices, the rows are "
                  f"kept and {name} stays unchecked")


@matcher(r"qk_rmsnorm|qk_norm")
def qk_norm(axes, inputs, census):
    """Fused per-head RMSNorm + RoPE of q [T, heads, head_dim] or [T, heads * head_dim] (and a shared kv row
    [T, head_dim]).

    Rows are ops whose op or kernel names carry both a norm and a RoPE and whose inputs hold q with the task's head
    count and head size (a plain RMSNorm of the same width, such as MLA's kv_a_layernorm, is not one).
    """
    var = _var(axes)
    heads = _const(axes, "num_q_heads", "num_heads", "num_query_heads")
    dim = _const(axes, "head_dim", "head_size")
    rope_dim, table = _const(axes, "rope_dim", "rotary_dim"), _const(axes, "max_position", "max_positions")
    if len(var) != 1 or not dim:
        return Match([], "qk_norm", notes=["qk_norm matcher needs one var axis and a const head_dim"])
    kv_input = any(d.get("kind") == "tensor" and len(d.get("shape", [])) == 2 and d["shape"][1] == dim[0]
                   for d in inputs.values() if d.get("role") not in ("weight", "index"))
    obs, per_row, store, pool_notes = [], [], {}, set()
    for row in census.get("ops", []):
        names = _names(row)
        if not (re.search(r"norm", names, re.I) and re.search(r"rope|rotary", names, re.I)):
            continue
        shapes = _shapes(row)
        q = next((d for d in shapes if len(d) == 3 and d[2] == dim[1]), None)
        flat = next((d for d in shapes if len(d) == 2 and d[1] > dim[1] and d[1] % dim[1] == 0
                     and [d[0], dim[1]] in shapes), None)  # q flattened to [T, heads * head_dim] beside kv
        if q is None and flat is None:
            continue
        tokens, q_heads = (q[0], q[1]) if q else (flat[0], flat[1] // dim[1])
        if heads and q_heads != heads[1]:
            _mismatch(store, row, {heads[0]: q_heads})
            continue
        if kv_input and [tokens, dim[1]] not in shapes and [tokens, 1, dim[1]] not in shapes:
            continue
        checked = {dim[0], heads[0] if heads else None}
        if rope_dim:  # cos/sin (or freqs) table [max_position, ..., rope_dim / 2]
            rows = next((math.prod(d[:-1]) for d in shapes if len(d) >= 2 and d[0] != tokens
                         and d[-1] in (rope_dim[1], rope_dim[1] // 2)), None)
            if rows:
                checked.add(rope_dim[0])
                if table:
                    _pool_size(axes, table[0], rows, checked, pool_notes)
        per_row.append(checked)
        obs.append(Observation({var[0]: tokens}, row["total_us"], row["calls"], row["phase"],
                               f"{row['op']} q={q or flat} {row['phase']}", pad_axis=var[0]))
    return Match(obs, "qk_norm", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=["production may fuse the norm and RoPE with the KV-cache write; such an op's time "
                        "includes the write"] + sorted(pool_notes) + _mismatch_notes(store, axes)
                 + _unlinked_note(census, r"qk_norm_rope|qk_rmsnorm_rope|norm_rope"))


@matcher(r"rope|rotary")
def rope(axes, inputs, census):
    return _rows(axes, census, r"rotary|rope", ("rotary_dim", "rope_dim", "head_dim", "head_size"), ("rope",),
                 "rope")


QUANT_FORMATS = {  # output format -> (accepts an ops row, MX block size, names of its unlinked kernels)
    "mxfp4": (lambda row: not _has(row, FP8_CODES) and (
        _has(row, FP4_CODES + BYTES) or bool(re.search(r"mxfp4|f4_quant|fp4", _names(row), re.I))), 32,
        r"mxfp4_quant|fp4_quant"),
    "mxfp8": (lambda row: _has(row, FP8_CODES) and (
        _has(row, E8M0) or bool(re.search(r"mxfp8", _names(row), re.I))), 32, r"mxfp8_quant"),
    "fp8": (lambda row: _has(row, FP8_CODES) and not _has(row, E8M0), None, r"quant\w*fp8|fp8\w*quant"),
    "int8": (lambda row: _has(row, INT8_CODES), None, r"quant\w*int8|int8\w*quant"),
    "any": (lambda row: _has(row, FP8_CODES + E8M0 + FP4_CODES + BYTES + INT8_CODES), None, r"quant"),
}


def _quant_groups(shapes: list, tokens: int, width: int, packed: bool) -> set:
    """Scale groups per row seen in a quantization op's inputs: scales [T, G] (or [G, T]), the grouped input view
    [T * G, width / G]; a per-token scale [T, 1] gives 1."""
    seen = set()
    for d in shapes:
        lead, last = math.prod(d[:-1]), d[-1]
        if len(d) < 2 or last == width or (packed and last == width // 2):
            continue
        if lead == tokens and width % last == 0:
            seen.add(last)
        elif lead % tokens == 0 and (lead // tokens) * last == width:
            seen.add(lead // tokens)
        elif len(d) == 2 and d[1] == tokens and width % d[0] == 0:
            seen.add(d[0])
    return seen


def _quant(axes: dict, inputs: dict, census: dict, name: str, fmt: str, per_token: bool = False) -> Match:
    """Quantization ops of one output format: rows [T, width] of the task's input dtype; the scale groups per row
    must equal the task's (``num_groups`` or ``hidden_size / group_size``, the MX block of 32, or 1 per token)."""
    var = _var(axes)
    width = _const(axes, "hidden_size", "hidden", "k")
    if len(var) != 1 or not width:
        return Match([], name, notes=["quantization matcher needs one var axis and a const width axis"])
    h = width[1]
    accepts, block, unlinked = QUANT_FORMATS[fmt]
    groups, gsize = _const(axes, "num_groups", "groups"), _const(axes, "group_size")
    want = (groups[1] if groups else h // gsize[1] if gsize else h // block if block else 1 if per_token else None)
    group_axis = groups[0] if groups else gsize[0] if gsize else None
    src = _first_dtype(inputs)
    families = ("quant", "norm", "activation") if fmt == "any" else ("quant",)
    obs, per_row, ungrouped = [], [], 0.0
    for row in census.get("ops", []):
        if row["family"] not in families or not re.search(r"quant", _names(row), re.I) or not accepts(row):
            continue
        if src and not _has(row, DTYPE_ALIASES.get(src, (src,))):
            continue
        shapes = _shapes(row)
        tokens = next((math.prod(d[:-1]) for d in shapes if len(d) >= 2 and d[-1] == h), None)
        if tokens is None:
            continue
        seen = _quant_groups(shapes, tokens, h, packed=fmt == "mxfp4")
        checked = {width[0]}
        if want is not None and seen:
            if want not in seen:
                continue
            checked.add(group_axis)
        elif want is not None:
            ungrouped += row["total_us"]
        per_row.append(checked)
        obs.append(Observation({var[0]: tokens}, row["total_us"], row["calls"], row["phase"],
                               f"{row['op'] or 'kernel'} {shapes[:3]} {row['phase']}", pad_axis=var[0]))
    notes = [f"output format {fmt}" + (f", {want} scale group(s) per row" if want else "")]
    if ungrouped:
        notes.append(f"{ungrouped / 1e3:.1f} ms of matched rows record no scale tensor: their grouping is unchecked")
    return Match(obs, name, unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=notes + _unlinked_note(census, unlinked))


@matcher(r"^dynamic_mxfp4_quant$")
def mxfp4_quant(axes, inputs, census):
    return _quant(axes, inputs, census, "mxfp4_quant", "mxfp4")


@matcher(r"^dynamic_mxfp8_quant$")
def mxfp8_quant(axes, inputs, census):
    return _quant(axes, inputs, census, "mxfp8_quant", "mxfp8")


@matcher(r"^per_token(_group)?_quant_fp8", name="quant")
def fp8_quant(axes, inputs, census):
    return _quant(axes, inputs, census, "quant", "fp8", per_token=True)


@matcher(r"^per_token(_group)?_quant_int8", name="quant")
def int8_quant(axes, inputs, census):
    return _quant(axes, inputs, census, "quant", "int8", per_token=True)


@matcher(r"quant")
def quant(axes, inputs, census):
    """Other quantizing operators: any quantized type among the row's inputs; groups checked when the task has an
    axis for them."""
    return _quant(axes, inputs, census, "quant", "any")


@matcher(r"silu_and_mul|gelu_and_mul|swiglu")
def gated_activation(axes, inputs, census):
    return _rows(axes, census, r"and_mul|swiglu|silu", ("d", "hidden_size", "intermediate_size", "n"),
                 ("activation",), "gated_activation")


def _routing(axes: dict, inputs: dict, census: dict, row_regex: str, biased: bool) -> Match:
    """Router ops: gating logits [T, experts] and top-k outputs [T, top_k] (and a correction bias [experts])."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], "moe_routing", notes=["routing matcher supports one var axis"])
    experts, topk = _const(axes, "n_experts", "num_experts"), _const(axes, "topk", "top_k")
    src = _first_dtype(inputs)
    obs, per_row, other_dtype = [], [], 0.0
    for row in census.get("ops", []):
        if row["family"] not in ("moe", "sampling") or not re.search(row_regex, _names(row), re.I):
            continue
        shapes = _shapes(row)
        flat = [d for d in shapes if len(d) == 2]
        logits = (next((d for d in flat if d[1] == experts[1]), None) if experts
                  else max(flat, key=lambda d: d[1], default=None))
        if logits is None:
            continue
        tokens, width = logits
        if biased and [width] not in shapes:
            continue
        checked = {experts[0] if experts else None}
        ks = {d[1] for d in flat if d[0] == tokens and d[1] < width}
        if topk and ks:
            if topk[1] not in ks:
                continue
            checked.add(topk[0])
        logits_type = _type_of(row, logits)
        if src and logits_type and not dtype_ok(src, logits_type):
            other_dtype += row["total_us"]
        per_row.append(checked)
        obs.append(Observation({var[0]: tokens}, row["total_us"], row["calls"], row["phase"],
                               f"{row['op']} {logits} {row['phase']}", pad_axis=var[0]))
    notes = []
    if biased:
        notes.append("the expert-group count and the number of kept groups are scalar arguments the profiler does "
                     "not record; rows are matched by op name, experts, top-k and the correction bias")
    if other_dtype:
        notes.append(f"{other_dtype / 1e3:.1f} ms of matched rows read logits of another dtype than the task's {src}")
    return Match(obs, "moe_routing", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)), notes=notes)


@matcher(r"biased_grouped_topk", name="moe_routing")
def biased_grouped_topk(axes, inputs, census):
    return _routing(axes, inputs, census, r"biased_grouped_topk|grouped_topk", biased=True)


@matcher(r"topk_softmax|topk_sigmoid|grouped_topk|routing")
def moe_routing(axes, inputs, census):
    return _routing(axes, inputs, census, r"topk_softmax|topk_sigmoid|grouped_topk|routing", biased=False)


def _sorting_layout(row: dict) -> dict | None:
    """Tokens, top-k, experts, block size and buffer width of a MoE sorting op, from its input shapes.

    A sorting op takes topk_ids [T, k], topk_weights [T, k], sorted_ids and sorted_weights [L], sorted_expert_ids
    [ceil(L / block)], num_valid_ids [2] and the output buffer [T, width], with L = T * k + experts * block - k
    (rounded up to the block by newer AITER). Sorting kernels launched by the fused MoE op itself take their tokens,
    top-k and experts from its inputs (hidden [T, width], expert weights [experts, ...], topk pair [T, k]).
    """
    shapes = _shapes(row)
    if not shapes or len(shapes[0]) != 2:
        return None
    pairs = [a for a, b in zip(shapes, shapes[1:]) if a == b and len(a) == 2]
    weights = [d for d in shapes if len(d) == 3]
    if weights:
        kernels = row.get("kernels") or {}
        if not kernels or not all(re.search(r"sort", k, re.I) for k in kernels):
            return None  # the op's other kernels share the row: its time is not the sorting's alone
        hidden, experts = shapes[0], weights[0][0]
        topk = next((p for p in pairs if p[0] == hidden[0] and p[1] <= experts), None)
        if topk is None:
            return None
        return {"tokens": hidden[0], "top_k": topk[1], "experts": experts, "block": None, "model_dim": hidden[1]}
    tokens, k = shapes[0]
    flat = [d[0] for d in shapes if len(d) == 1 and d[0] != 2]
    padded = next((n for n in flat if n >= tokens * k), None)
    blocks = next((n for n in flat if padded and n < padded), None)
    block = experts = None
    if padded and blocks:
        sizes = [b for b in (2 ** i for i in range(11)) if -(-padded // b) == blocks]
        if len(sizes) == 1:
            block = sizes[0]
            experts = (padded - tokens * k + k) // block
    model_dim = next((d[1] for d in shapes[1:] if len(d) == 2 and d[0] == tokens and d[1] != k), None)
    return {"tokens": tokens, "top_k": k, "experts": experts, "block": block, "model_dim": model_dim}


@matcher(r"moe_sorting|moe_align")
def moe_sorting(axes, inputs, census):
    """MoE sorting ops (and sorting kernels run by the fused MoE op) checked on top-k, experts, block and width."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], "moe_sorting", notes=["moe_sorting matcher supports one var axis"])
    want = {"top_k": _const(axes, "top_k", "topk"), "experts": _const(axes, "num_experts", "n_experts"),
            "block": _const(axes, "block_size"), "model_dim": _const(axes, "model_dim", "hidden_size")}
    obs, per_row = [], []
    for row in census.get("ops", []):
        if row["family"] != "moe" or not re.search(r"moe_sorting|moe_align", _names(row), re.I):
            continue
        layout = _sorting_layout(row)
        if not layout:
            continue
        seen = {key: ax for key, ax in want.items() if ax and layout[key] is not None}
        if any(layout[key] != ax[1] for key, ax in seen.items()):
            continue
        per_row.append({ax[0] for ax in seen.values()})
        obs.append(Observation({var[0]: layout["tokens"]}, row["total_us"], _invocations(row), row["phase"],
                               f"{row['op']} tokens={layout['tokens']} top_k={layout['top_k']} {row['phase']}",
                               pad_axis=var[0]))
    return Match(obs, "moe_sorting", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=["calls count kernel launches unless the census records op calls: a multi-phase sort whose "
                        "kernels share one name counts each launch"]
                 + _unlinked_note(census, r"moe_sorting|moe_align"))


def _weight_format(row: dict, w1: list) -> str:
    """Weight format of a MoE op: w1's own type when the types align with the shapes, else the types present."""
    own = _type_of(row, w1)
    if own:
        return ("fp4" if own in FP4_CODES + BYTES else "fp8" if own in FP8_CODES else "int8" if own in INT8_CODES
                else "16bit")
    if _has(row, FP4_CODES):
        return "fp4"
    if _has(row, FP8_CODES):
        return "fp8"
    if _has(row, INT8_CODES):
        return "int8"
    return "fp4" if _has(row, BYTES) else "16bit"


def _task_weight_format(inputs: dict) -> str | None:
    dtype = (inputs.get("w1") or {}).get("dtype", "")
    if dtype.startswith("float8"):
        return "fp8"
    return {"uint8": "fp4", "int8": "int8", "bfloat16": "16bit", "float16": "16bit"}.get(dtype)


@matcher(r"^fused_moe_fp8_blockscale$|^fused_moe_mxfp4$")
def fused_moe(axes, inputs, census):
    """A fused MoE layer (sort, gate/up GEMM, activation, down GEMM, top-k reduction) by token count.

    The layer's ops are found by its expert weights w1 [E, 2I, H] and w2 [E, H, I] (the last axis halved for
    packed 4-bit weights) in the weight format of the task, its sorting op by the sorting layout (top-k, E, H) and
    the requantization between two GEMM stages by its [T, top_k, I] input. Their time is summed per token count
    and phase; the calls of a group are its op calls (the fewest among its ops: each runs once per layer call).
    """
    var = _var(axes)
    experts, topk = _const(axes, "num_experts", "n_experts"), _const(axes, "top_k", "topk")
    hidden, inter = _const(axes, "hidden_size", "hidden"), _const(axes, "intermediate_size", "inter_size")
    fmt = _task_weight_format(inputs)
    if len(var) != 1 or not (experts and hidden and inter and fmt):
        return Match([], "fused_moe", notes=["fused_moe matcher needs one var axis, const num_experts, "
                                             "hidden_size, intermediate_size and a w1 input"])
    e, h, i = experts[1], hidden[1], inter[1]
    packed = fmt == "fp4"
    w1, w2 = [e, 2 * i, h // 2 if packed else h], [e, h, i // 2 if packed else i]
    groups = defaultdict(lambda: {"us": 0.0, "calls": [], "ops": [], "experts": False, "top_k": False,
                                  "other_top_k": False})
    for row in census.get("ops", []):
        shapes = _shapes(row)
        if w1 in shapes and w2 in shapes:
            if _weight_format(row, w1) != fmt:
                continue
            tokens, role = shapes[0][0], "experts"
            ks = {a[1] for a, b in zip(shapes, shapes[1:]) if a == b and len(a) == 2 and a[0] == tokens and a[1] <= e}
            ks |= {d[1] for d in shapes if len(d) == 3 and d not in (w1, w2) and d[0] == tokens and d[1] <= e
                   and d[2] in (i, i // 2)}  # [T, top_k, I] intermediate
        elif row["family"] == "moe" and re.search(r"moe_sorting|moe_align", _names(row), re.I):
            layout = _sorting_layout(row)
            if not layout or layout["experts"] != e or layout["model_dim"] != h:
                continue
            tokens, role, ks = layout["tokens"], "sorting", {layout["top_k"]}
        elif row["family"] == "quant" and shapes and len(shapes[0]) == 3 and shapes[0][2] == i:
            tokens, role, ks = shapes[0][0], "requant", {shapes[0][1]}
        else:
            continue
        g = groups[(tokens, row["phase"])]
        if topk and ks and topk[1] not in ks:
            g["other_top_k"] = True  # the layer of these tokens routes to another number of experts
            continue
        g["us"] += row["total_us"]
        g["calls"].append(_invocations(row))
        g["ops"].append(row["op"] or role)
        g["experts"] |= role == "experts"
        g["top_k"] |= bool(topk and ks)
    obs, per_group = [], []
    for (tokens, phase), g in sorted(groups.items()):
        if not g["experts"] or g["other_top_k"]:
            continue  # sorting or requantization alone is not evidence of this layer
        per_group.append({experts[0], hidden[0], inter[0]} | ({topk[0]} if g["top_k"] else set()))
        obs.append(Observation({var[0]: tokens}, g["us"], min(g["calls"]), phase,
                               f"fused MoE tokens={tokens} {phase}: {', '.join(sorted(set(g['ops'])))}",
                               pad_axis=var[0]))
    notes = ["not counted: the quantization of the hidden states before the layer (the same op quantizes other "
             "GEMM inputs) and the routing (top-k, shared-expert append)"]
    if packed:
        notes.append("the activation format (A4, A8 or A16) is not visible in the census and is not checked")
    return Match(obs, "fused_moe", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_group)),
                 notes=notes + _unlinked_note(census, r"fused_moe|moe_gemm|fmoe"))


def _pipeline_share(census: dict, kernel_regex: str, pipeline_rows: list, own_us: float) -> float | None:
    """Share of the enclosing ops' time spent in ``kernel_regex`` kernels, from the census's per-kernel totals
    (less ``own_us``, the time of rows that hold only those kernels)."""
    kernel_us = sum(k["total_us"] for k in census.get("kernels", [])
                    if re.search(kernel_regex, f"{k.get('name', '')} {k.get('short_name', '')}", re.I)) - own_us
    rows_us = sum(r["total_us"] for r in pipeline_rows)
    return min(1.0, kernel_us / rows_us) if rows_us and kernel_us > 0 else None


def _kernel_rows(axes: dict, census: dict, name: str, kernel_regex: str, read: Callable, step_key: str,
                 notes: list[str], pipeline: str | None = None) -> Match:
    """Ops whose op or kernel names match ``kernel_regex``. ``read(shapes, row)`` returns (var value, const axes it
    verified) for a row with shapes, or None to reject it. A kernel launched outside a recorded op has no shapes;
    when the census records the step's ``step_tokens`` or ``step_batch`` on such a row, ``step_key`` gives the var
    value and no const axis is verified.

    ``pipeline`` names an enclosing op that runs the kernel among others (a census row lists only its four most
    launched kernels, so the kernel may not be named): its rows count with the kernel's share of their time."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], name, notes=[f"{name} matcher supports one var axis"])
    ops = census.get("ops", [])
    in_pipeline = [bool(pipeline and r.get("op") and re.search(pipeline, r["op"], re.I)
                        and not all(re.search(kernel_regex, k, re.I) for k in r.get("kernels") or {})) for r in ops]
    share = None
    if pipeline and any(in_pipeline):
        own = sum(r["total_us"] for r, p in zip(ops, in_pipeline) if not p and r.get("kernels")
                  and all(re.search(kernel_regex, k, re.I) for k in r["kernels"]))
        share = _pipeline_share(census, kernel_regex, [r for r, p in zip(ops, in_pipeline) if p], own)
        notes = notes + ([f"{row_op} rows count {100 * share:.1f}% of their time: the share of the census's "
                          f"{name} kernel totals in them" for row_op in sorted({r["op"] for r, p in
                                                                                 zip(ops, in_pipeline) if p})]
                         if share else [f"no {name} kernel among the census kernels: {pipeline} rows not used"])
    obs, per_row, shapeless = [], [], 0.0
    for row, piped in zip(ops, in_pipeline):
        if piped and not share or not piped and not re.search(kernel_regex, _names(row), re.I):
            continue
        shapes = _shapes(row)
        us = row["total_us"] * (share if piped else 1.0)
        if shapes and row.get("op"):
            if not piped and not re.search(kernel_regex, row["op"], re.I) and not all(
                    re.search(kernel_regex, k, re.I) for k in row.get("kernels") or {}):
                continue  # an enclosing op running other kernels too: its time is not this operator's alone
            got = read(shapes, row)
            if got is None:
                continue
            value, checked = got
        elif row.get(step_key):
            value, checked = row[step_key], set()
        else:
            shapeless += row["total_us"]
            continue
        per_row.append(checked)
        obs.append(Observation({var[0]: value}, us, _invocations(row), row["phase"],
                               f"{row['op'] or 'kernel'} {var[0]}={value} {row['phase']}", pad_axis=var[0]))
    if shapeless:
        notes = notes + [f"{shapeless / 1e3:.1f} ms of matching kernels ran outside a recorded op with no step "
                         f"{step_key.split('_')[1]} recorded: not counted"]
    return Match(obs, name, unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)), notes=notes)


@matcher(r"^causal_conv1d_fwd")
def causal_conv1d_fwd(axes, inputs, census):
    """Prefill causal conv1d over packed tokens: x [dim, T] (or [T, dim]) and weight [dim, width]."""
    dim, width = _const(axes, "dim"), _const(axes, "width")

    def read(shapes, row):
        if not dim or not width or [dim[1], width[1]] not in shapes:
            return None
        x = next((d for d in shapes if len(d) == 2 and dim[1] in d and d != [dim[1], width[1]]), None)
        if x is None:
            return None
        return (x[1] if x[0] == dim[1] else x[0]), {dim[0], width[0]}

    return _kernel_rows(axes, census, "causal_conv1d_fwd", r"causal_conv1d_fwd|causal_conv1d_fn", read,
                        "step_tokens", ["the q/k/v split of the conv output (key_dim, value_dim) is not visible "
                                        "in the census"])


@matcher(r"^gdn_chunk_fwd_h")
def gdn_chunk_fwd_h(axes, inputs, census):
    """Gated delta rule chunked state pass (prefill): k [1, T, Hk, K], v/u [1, T, H, V] (token-major)."""
    hk, h = _const(axes, "num_k_heads"), _const(axes, "num_heads")
    dk, dv = _const(axes, "head_dim_k"), _const(axes, "head_dim_v")

    def read(shapes, row):
        if not (hk and h and dk and dv):
            return None
        k = next((d for d in shapes if len(d) == 4 and d[2:] == [hk[1], dk[1]]), None)
        if k is None or not any(len(d) == 4 and d[1] == k[1] and d[2:] == [h[1], dv[1]] for d in shapes):
            return None
        return k[1], {hk[0], h[0], dk[0], dv[0]}

    return _kernel_rows(axes, census, "gdn_chunk_fwd_h",
                        r"chunk_gated_delta_rule_fwd_(kernel_)?h(_|\b)|chunk_fwd_(kernel_)?h(_|\b)", read,
                        "step_tokens", ["production runs the state pass over all sequences of a prefill batch "
                                        "(varlen); a row's token count is the batch's, not one sequence's"],
                        pipeline=r"chunk_?gated_?delta_?rule")


@matcher(r"^gdn_fused_recurrent_decode")
def gdn_fused_recurrent_decode(axes, inputs, census):
    """Gated delta rule recurrent state update by sequences per step: packed q|k|v [T, 2 * Hk * K + H * V], or
    q [..., Hk, K] with v [..., H, V] over T tokens ([B, 1, Hk, K] in decode, [T, Hk, K] in the speculative verify
    update, T = sequences x draft tokens). A verify row's T is divided by the draft tokens per sequence."""
    hk, h = _const(axes, "num_k_heads"), _const(axes, "num_heads")
    dk, dv = _const(axes, "head_dim_k"), _const(axes, "head_dim_v")
    draft = (census.get("deployment") or {}).get("speculative_num_draft_tokens")
    undivided = []

    def read(shapes, row):
        if not (hk and h and dk and dv):
            return None
        packed = 2 * hk[1] * dk[1] + h[1] * dv[1]
        qkv = next((d for d in shapes if len(d) == 2 and d[1] == packed), None)
        if qkv:
            tokens = qkv[0]
        else:
            q = next((d for d in shapes if len(d) in (3, 4) and d[-2:] == [hk[1], dk[1]]), None)
            if q is None or not any(len(d) in (3, 4) and d[-2:] == [h[1], dv[1]]
                                    and math.prod(d[:-2]) == math.prod(q[:-2]) for d in shapes):
                return None
            tokens = math.prod(q[:-2])
        if row["phase"] in ("verify", "draft") and row.get("step_batch"):
            tokens = row["step_batch"]
        elif row["phase"] in ("verify", "draft"):
            if draft and tokens % draft == 0:
                tokens //= draft
            else:
                undivided.append(row["total_us"])
        return tokens, {hk[0], h[0], dk[0], dv[0]}

    match = _kernel_rows(axes, census, "gdn_fused_recurrent_decode",
                         r"fused_recurrent_gated_delta_rule|fused_sigmoid_gating_delta_rule|gdn_replayssm_spec",
                         read, "step_batch", [])
    if any(o.phase == "verify" for o in match.observations):
        match.notes.append(f"speculative decoding: no single-token decode step runs; the target's verify step "
                           f"updates each sequence's state over {draft or 'r'} draft tokens (replay-SSM kernel), "
                           f"and its batch is the step's sequence count")
    if undivided:
        match.notes.append(f"{sum(undivided) / 1e3:.1f} ms of speculative-step rows count tokens, not sequences: "
                           f"the draft tokens per sequence are unknown or do not divide them")
    return match


# --- DeepSeek-V4 sparse attention and its indexer ---------------------------------------------------------------
# Under speculative decoding the decode attention and the indexer run in the verify (and draft) steps over r
# tokens per sequence: a row's leading T is sequences x r. Tasks whose rows are query rows (one token each) take T;
# tasks with a per-sequence batch and a tokens-per-sequence axis split T by the page table's sequence count.

def _context(row: dict, rejected: dict) -> int | None:
    """Context length of a step: the census's ``kv_tokens_per_seq`` (scheduler-log mean, as for decode attention)."""
    kv = row.get("kv_tokens_per_seq")
    if not kv:
        rejected["the step's context length is not recorded (kv_tokens_per_seq)"] += row["total_us"]
    return kv


SPARSE_KV_NOTE = ("context length is the scheduler log's mean KV tokens per sequence of the step, not the batch "
                  "maximum")


@matcher(r"^paged_mqa_logits$")
def paged_mqa_logits(axes, inputs, census):
    """FP4 indexer logits of decode / verify steps: q [T, heads, head_size / 2], cache [blocks, 1, 4, block_size,
    16], block table [B, max_seq_len / block_size + 4]; batch_size = B, next_n = T / B."""
    names = ("batch_size", "next_n", "ctx")
    store = {}
    if any(n not in axes or axes[n]["kind"] != "var" for n in names):
        return Match([], "paged_mqa_logits", notes=["paged_mqa_logits matcher needs var batch_size, next_n, ctx"])
    obs, per_row, rejected = [], [], defaultdict(float)
    for row in census.get("ops", []):
        if not re.search(r"paged_mqa_logits|pa_mqa_logits", _names(row), re.I):
            continue
        shapes = _shapes(row)
        if not shapes or not row.get("op"):
            rejected["kernel launched outside a recorded op (no shapes)"] += row["total_us"]
            continue
        if re.search(r"prefill", _names(row), re.I):
            rejected["prefill indexer (one query row per prompt token, another kernel)"] += row["total_us"]
            continue
        q = shapes[0]
        tokens = q[0] if len(q) == 3 else None
        cache = next((d for d in shapes if len(d) == 5 and d[0] != tokens), None)  # not the q scales [T, ...]
        table = next((d for d in shapes if len(d) == 2 and tokens and d[0] != tokens and tokens % d[0] == 0), None)
        if tokens is None or cache is None or table is None:
            rejected["input layout not recognized"] += row["total_us"]
            continue
        packed = _has(row, FP4_CODES + BYTES)
        seen = {"num_heads": q[1], "head_size": q[2] * (2 if packed else 1), "block_size": cache[3],
                "max_seq_len": (table[1] - 4) * cache[3]}
        wrong, checked = _compare(axes, seen)
        batch, next_n = table[0], tokens // table[0]
        if next_n > axes["next_n"]["max"]:
            wrong["next_n"] = next_n
        if wrong:
            _mismatch(store, row, wrong)
            continue
        ctx = _context(row, rejected)
        if ctx is None:
            continue
        per_row.append(checked)
        obs.append(Observation({"batch_size": batch, "next_n": next_n, "ctx": ctx}, row["total_us"],
                               _invocations(row), row["phase"], f"{row['op']} q={q} {row['phase']}",
                               pad_axis="batch_size"))
    return Match(obs, "paged_mqa_logits", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=[SPARSE_KV_NOTE] + _mismatch_notes(store, axes) + _rejected_notes(rejected))


@matcher(r"^flash_mla_sparse_decode_fp8$")
def flash_mla_sparse_decode_fp8(axes, inputs, census):
    """Sparse MLA over the FP8 sliding-window pool [pages, page_size, packed_width] (and, when the task has
    ``extra_*`` axes, a compressed pool): q [T, heads, head_dim], index tables [T * topk]; one query row per token,
    so ``batch`` is T (verify rows included)."""
    var = _var(axes)
    if "batch" not in var:
        return Match([], "flash_mla_sparse_decode_fp8", notes=["matcher needs a var batch axis"])
    want_extra = "extra_topk" in axes
    obs, per_row, rejected, pool_notes, store = [], [], defaultdict(float), set(), {}
    for row in census.get("ops", []):
        if not re.search(r"pa_decode_sparse|flash_mla_sparse|sparse_mla_decode", _names(row), re.I):
            continue
        shapes = _shapes(row)
        if not shapes or not row.get("op"):
            rejected["kernel launched outside a recorded op (no shapes)"] += row["total_us"]
            continue
        q = shapes[0]
        pools = [i for i, d in enumerate(shapes) if len(d) == 3 and i > 0 and d[0] != q[0]]
        if len(q) != 3 or not pools:
            rejected["input layout not recognized"] += row["total_us"]
            continue
        tokens = q[0]

        def topk(i):  # index table [T * topk] right after a pool
            d = shapes[i + 1] if i + 1 < len(shapes) else []
            return d[0] // tokens if len(d) == 1 and d[0] % tokens == 0 else None
        swa = shapes[pools[0]]
        seen = {"query_length": 1, "heads": q[1], "head_dim": q[2], "one": 1, "page_size": swa[1],
                "packed_width": swa[2], "topk": topk(pools[0])}
        if len(pools) > 1:
            seen.update(extra_page_size=shapes[pools[1]][1], extra_topk=topk(pools[1]))
        wrong, checked = _compare(axes, seen)
        if want_extra != (len(pools) > 1):
            wrong["compressed pool"] = "missing" if want_extra else "present (task has none)"
        if wrong:
            _mismatch(store, row, wrong)
            continue
        _pool_size(axes, "pages", swa[0], checked, pool_notes)
        values = {"batch": tokens}
        if "ctx" in var:
            ctx = _context(row, rejected)
            if ctx is None:
                continue
            values["ctx"] = ctx
        per_row.append(checked)
        obs.append(Observation(values, row["total_us"], _invocations(row), row["phase"],
                               f"{row['op']} q={q} {row['phase']}", pad_axis="batch"))
    return Match(obs, "flash_mla_sparse_decode_fp8", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=(["verify and draft rows count one query row per token (sequences x draft tokens)",
                         "the split-K reduce of the served decode (gfx_sparse_split_reduce) is a separate op and is "
                         "not counted"] + ([SPARSE_KV_NOTE] if "ctx" in var else []) + sorted(pool_notes)
                        + _mismatch_notes(store, axes) + _rejected_notes(rejected)))


@matcher(r"^sparse_mla_prefill$")
def sparse_mla_prefill(axes, inputs, census):
    """Sparse MLA prefill over a bf16 latent pool: q [T, heads, head_dim], kv [num_kv, head_dim], indices
    [T * topk], indptr [T + 1]."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], "sparse_mla_prefill", notes=["sparse_mla_prefill matcher supports one var axis"])
    obs, per_row, rejected, pool_notes, store = [], [], defaultdict(float), set(), {}
    for row in census.get("ops", []):
        if not re.search(r"sparse_prefill|sparse_mla_prefill|mla_v4_prefill", _names(row), re.I):
            continue
        shapes = _shapes(row)
        if not shapes or not row.get("op"):
            rejected["kernel launched outside a recorded op (no shapes)"] += row["total_us"]
            continue
        q = shapes[0]
        kv = next((d for d in shapes[1:] if len(d) == 2 and len(q) == 3 and d[1] == q[2]), None)
        if len(q) != 3 or kv is None:
            rejected["input layout not recognized"] += row["total_us"]
            continue
        tokens = q[0]
        cap = next((d[0] for d in shapes if len(d) == 1 and d[0] > tokens + 1 and d[0] % tokens == 0), None)
        wrong, checked = _compare(axes, {"num_heads": q[1], "head_dim": q[2],
                                         "topk": cap // tokens if cap else None})
        if wrong:
            _mismatch(store, row, wrong)
            continue
        _pool_size(axes, "num_kv", kv[0], checked, pool_notes)
        per_row.append(checked)
        obs.append(Observation({var[0]: tokens}, row["total_us"], _invocations(row), row["phase"],
                               f"{row['op']} q={q} {row['phase']}", pad_axis=var[0]))
    return Match(obs, "sparse_mla_prefill", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=sorted(pool_notes) + _mismatch_notes(store, axes) + _rejected_notes(rejected))


@matcher(r"^topk_transform_paged")
def topk_transform_paged(axes, inputs, census):
    """Indexer top-k over paged score rows: scores [T, width], seq_lens [T], page tables [T, num_pages],
    out [T, topk]; ``batch`` is T (query rows: sequences x draft tokens in verify)."""
    var = _var(axes)
    if len(var) != 1:
        return Match([], "topk_transform_paged", notes=["topk_transform_paged matcher supports one var axis"])
    obs, per_row, rejected, store = [], [], defaultdict(float), {}
    for row in census.get("ops", []):
        if not re.search(r"topk_transform_paged", _names(row), re.I):
            continue
        shapes = _shapes(row)
        if not shapes or not row.get("op"):
            rejected["kernel launched outside a recorded op (no shapes)"] += row["total_us"]
            continue
        scores = shapes[0]
        rest = [d for d in shapes[1:] if len(d) == 2 and len(scores) == 2 and d[0] == scores[0]]
        if len(scores) != 2 or len(rest) < 2:
            rejected["input layout not recognized"] += row["total_us"]
            continue
        wrong, checked = _compare(axes, {"width": scores[1], "num_pages": rest[0][1], "topk": rest[1][1]})
        if wrong:
            _mismatch(store, row, wrong)
            continue
        per_row.append(checked)
        obs.append(Observation({var[0]: scores[0]}, row["total_us"], _invocations(row), row["phase"],
                               f"{row['op']} scores={scores} {row['phase']}", pad_axis=var[0]))
    return Match(obs, "topk_transform_paged", unchecked_const_axes=_unchecked(axes, _checked_by_all(per_row)),
                 notes=_mismatch_notes(store, axes) + _rejected_notes(rejected))

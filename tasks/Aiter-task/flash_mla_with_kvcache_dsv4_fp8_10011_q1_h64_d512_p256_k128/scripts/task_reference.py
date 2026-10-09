# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness reference for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``reference`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from functools import partial as partial
import math as math
import torch as torch

def _decode_slots(cache, slots):
    """Decode physical slots via literal BF16 bit assembly, not standard FP8 scaling."""
    page_size = cache.shape[1]
    raw = torch.empty(0, dtype=torch.uint8, device=cache.device).set_(
        cache.untyped_storage(), 0, (cache.untyped_storage().nbytes(),), (1,)
    )
    page = slots // page_size
    within = slots % page_size
    base = cache.storage_offset() * cache.element_size() + page * cache.stride(0)
    # Pages store payloads before scale slots, not interleaved 584-byte records.
    payload = base[:, None] + within[:, None] * 576
    codes = raw[payload + torch.arange(448, device=cache.device)].to(torch.int32)
    scales = raw[
        base[:, None]
        + page_size * 576
        + within[:, None] * 8
        + torch.arange(7, device=cache.device)
    ].to(torch.int32)
    exponent = ((codes >> 3) & 15) + scales.repeat_interleave(64, dim=-1) - 7
    bits = ((codes & 128) << 8) | (exponent << 7) | ((codes & 7) << 4)
    nope = (bits & 65535).to(torch.int16).view(torch.bfloat16).float()
    tail = raw[payload + 448 + torch.arange(128, device=cache.device)].contiguous()
    rope = tail.view(torch.bfloat16).float()
    return torch.cat((nope, rope), dim=-1)


def _selected_slots(indices, lengths, row, capacity):
    """Offline content validation only; never called by the host schema."""
    length = indices.shape[-1] if lengths is None else lengths[row].item()
    if not 0 <= length <= indices.shape[-1]:
        raise ValueError("sparse length must lie within the index table width")
    slots = indices[row, 0, :length].to(torch.int64)
    slots = slots[slots >= 0]
    if torch.any(slots >= capacity).item():
        raise ValueError("nonnegative sparse index is outside its KV pool")
    return slots


def _flash_mla_reference(
    q,
    kv_cache,
    sparse_indices,
    sparse_lens=None,
    extra_kv_cache=None,
    extra_sparse_indices=None,
    extra_sparse_lens=None,
    sm_scale=None,
    sinks=None,
    *,
    return_lse=False,
):
    """FP32 attention over decoded shared KV; sink is excluded from returned LSE."""
    scale = 1.0 / math.sqrt(512) if sm_scale is None else sm_scale
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:-1], torch.inf, dtype=torch.float32, device=q.device)
    for row in range(q.shape[0]):
        slots = _selected_slots(
            sparse_indices, sparse_lens, row, kv_cache.shape[0] * kv_cache.shape[1]
        )
        values = [_decode_slots(kv_cache, slots)]
        if extra_kv_cache is not None:
            extra_slots = _selected_slots(
                extra_sparse_indices,
                extra_sparse_lens,
                row,
                extra_kv_cache.shape[0] * extra_kv_cache.shape[1],
            )
            values.append(_decode_slots(extra_kv_cache, extra_slots))
        kv = torch.cat(values, dim=0)
        if kv.shape[0] == 0:
            continue
        logits = (q[row, 0].float() @ kv.T) * scale
        # Normalize logits directly; rounded LSE loses small log(sumexp) terms.
        all_logits = (
            logits
            if sinks is None
            else torch.cat((logits, sinks.float()[:, None]), dim=-1)
        )
        weights = all_logits.softmax(dim=-1)[:, : kv.shape[0]]
        output[row, 0] = (weights @ kv).to(q.dtype)
        lse[row, 0] = torch.logsumexp(logits, dim=-1)
    return (output, lse) if return_lse else output


def _reference(*args, names, return_lse, **kwargs):
    """Keep schema input order when optional operands are absent."""
    if len(args) > len(names):
        raise TypeError("too many MLA reference inputs")
    call = dict(zip(names, args))
    call.update(kwargs)
    return _flash_mla_reference(**call, return_lse=return_lse)


_callable = partial(_reference, **{'names': ('q', 'kv_cache', 'sparse_indices', 'sparse_lens', 'sm_scale', 'sinks',), 'return_lse': True})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)

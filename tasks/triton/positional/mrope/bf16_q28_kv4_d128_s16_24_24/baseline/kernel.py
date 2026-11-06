"""Multimodal (sectioned) rotary embedding, Qwen2-VL / Qwen2.5-VL style (Triton).

Vendored from sgl-project/sglang python/sglang/kernels/ops/attention/rotary_triton.py
(Apache-2.0): the kernel `_triton_mrope_forward_fused`. Upstream rotates q and k in place; this copy
reads q and k and writes q_out and k_out (the loads stay on q/k, the stores target the output pointers,
nothing else changes). run() replaces the host wrapper `triton_mrope_fused` with the scenario's fixed
configuration: contiguous (non-interleaved) T/H/W sections [16, 24, 24], NeoX rotation, full rotary
dimension.
"""

import triton
import triton.language as tl


@triton.jit
def _triton_mrope_forward_fused(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    positions_stride,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_interleaved_glm: tl.constexpr,
    is_neox_style: tl.constexpr,
    axis_map_ptr,
):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_stride
    k_ptr = k_ptr + pid * k_stride
    q_out_ptr = q_out_ptr + pid * q_stride
    k_out_ptr = k_out_ptr + pid * k_stride
    half_rd = rd // 2
    t = tl.load(positions_ptr + 0 * positions_stride + pid)
    h = tl.load(positions_ptr + 1 * positions_stride + pid)
    w = tl.load(positions_ptr + 2 * positions_stride + pid)
    t_cos = cos_sin_cache_ptr + t * rd
    h_cos = cos_sin_cache_ptr + h * rd
    w_cos = cos_sin_cache_ptr + w * rd
    t_sin = t_cos + half_rd
    h_sin = h_cos + half_rd
    w_sin = w_cos + half_rd
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        if is_interleaved_glm:
            axes = tl.load(axis_map_ptr + cos_offsets, mask=cos_offsets < (pad_hd // 2))
            t_mask = axes == 0
            h_mask = axes == 1
            w_mask = axes == 2
        else:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row
    if is_neox_style:
        fhq = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fhk = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        fqm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        fkm = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        q1 = tl.load(q_ptr + fhq, mask=fqm, other=0).to(sin_row.dtype)
        k1 = tl.load(k_ptr + fhk, mask=fkm, other=0).to(sin_row.dtype)
        shq = fhq + (rd // 2)
        shk = fhk + (rd // 2)
        q2 = tl.load(q_ptr + shq, mask=fqm, other=0).to(sin_row.dtype)
        k2 = tl.load(k_ptr + shk, mask=fkm, other=0).to(sin_row.dtype)
        tl.store(q_out_ptr + fhq, q1 * cos_row - q2 * sin_row, mask=fqm)
        tl.store(q_out_ptr + shq, q2 * cos_row + q1 * sin_row, mask=fqm)
        tl.store(k_out_ptr + fhk, k1 * cos_row - k2 * sin_row, mask=fkm)
        tl.store(k_out_ptr + shk, k2 * cos_row + k1 * sin_row, mask=fkm)
    else:
        bq = tl.arange(0, pad_n_qh)[:, None] * hd
        bk = tl.arange(0, pad_n_kh)[:, None] * hd
        ei = 2 * tl.arange(0, pad_hd // 2)[None, :]
        oi = ei + 1
        im = tl.arange(0, pad_hd // 2)[None, :] < (rd // 2)
        qm = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & im
        km = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & im
        qe = tl.load(q_ptr + bq + ei, mask=qm, other=0).to(sin_row.dtype)
        qo = tl.load(q_ptr + bq + oi, mask=qm, other=0).to(sin_row.dtype)
        ke = tl.load(k_ptr + bk + ei, mask=km, other=0).to(sin_row.dtype)
        ko = tl.load(k_ptr + bk + oi, mask=km, other=0).to(sin_row.dtype)
        tl.store(q_out_ptr + bq + ei, qe * cos_row - qo * sin_row, mask=qm)
        tl.store(q_out_ptr + bq + oi, qo * cos_row + qe * sin_row, mask=qm)
        tl.store(k_out_ptr + bk + ei, ke * cos_row - ko * sin_row, mask=km)
        tl.store(k_out_ptr + bk + oi, ko * cos_row + ke * sin_row, mask=km)



def run(*, q, k, cos_sin_cache, positions, q_out, k_out):
    head_size = 128
    n_qh = q.shape[1] // head_size
    n_kh = k.shape[1] // head_size
    _triton_mrope_forward_fused[(q.shape[0],)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        n_qh,
        n_kh,
        head_size,
        head_size,
        triton.next_power_of_2(n_qh),
        triton.next_power_of_2(n_kh),
        triton.next_power_of_2(head_size),
        16,
        24,
        24,
        False,
        False,
        True,
        cos_sin_cache,  # axis_map_ptr: only read when is_interleaved_glm
    )

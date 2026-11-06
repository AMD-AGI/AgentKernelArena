# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 3 operations into a **single kernel launch**:
  Q -> RoPE rotation -> Q_out
  K -> RoPE rotation -> K_out + key_cache
  V -> value_cache

Grid: (max(QH, KH), T, 1)  -- shared blocks for Q and K
  block_idx.x = head_idx in [0, max(QH, KH))
  block_idx.y = token_idx

  Each block conditionally does Q work (if head_idx < QH) and/or K work
  (if head_idx < KH).  For GQA (QH >> KH) blocks beyond KH only do Q;
  for MQA-like configs where KH <= QH every block does both.

  Cos/sin are loaded ONCE per block (before branching) and shared by both
  the Q and K paths, saving buffer descriptor SGPRs.

Input shapes:
  Q: [T, QH, D],  K: [T, KH, D],  V: [T, KH, D]
  CosCache/SinCache: [max_pos, D//2] if reuse_freqs_front_part else [max_pos, D]
  Positions/SlotMapping: [T] int32 storage, passed as a float32 view (bit-preserving) and
  bitcast back to int32 after the load.

KV cache layout (flash layout):
  KeyCache:   [num_blocks, block_size, KH, D]
  ValueCache: [num_blocks, block_size, KH, D]

Vendored from ROCm/FlyDSL kernels/fused_rope_cache_kernel.py. This copy supports the bfloat16
NeoX-style, flash-layout, unscaled-cache configuration only: the fp8 and non-flash paths and the
KScale/VScale arguments are removed, and positions and slot indices are read through the layout
API instead of raw buffer intrinsics.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


def get_warp_size(arch=None):
    """Wavefront size: 32 on RDNA (gfx10xx/gfx11xx/gfx12xx), 64 on CDNA (gfx9xx)."""
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


# WARP_SIZE is 32 on RDNA (wave32: gfx10xx/gfx11xx/gfx12xx) and 64 on CDNA (wave64: gfx9xx).
# All derived values (VEC_WIDTH, vecs_per_half, BLOCK_THREADS) flow from this automatically.
WARP_SIZE = get_warp_size()


def build_fused_rope_cache_module(
    head_dim: int = 64,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    dtype_str: str = "bf16",
    reuse_freqs_front_part: bool = True,
):
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}")
    rotary_dim = head_dim
    half_dim = rotary_dim // 2

    # VEC_WIDTH: elements per thread. Use ceil division so vecs_per_head never
    # exceeds WARP_SIZE for the fixed one-thread-per-vector mapping below.
    # For D=64:  VEC_WIDTH=1 -> vecs_per_head=64 (full wavefront, 16-bit loads).
    # For D=96:  VEC_WIDTH=2 -> vecs_per_head=48 (fits within one wavefront).
    # For D=128: VEC_WIDTH=2 -> vecs_per_head=64 (32-bit loads, unchanged).
    VEC_WIDTH = max(1, (head_dim + WARP_SIZE - 1) // WARP_SIZE)

    vecs_per_half = half_dim // VEC_WIDTH
    vecs_per_head = head_dim // VEC_WIDTH

    # elem_bits for copy atom (bf16/f16 = 16 bits)
    elem_bits = 16
    # Copy atom bits: VEC_WIDTH * elem_bits
    copy_bits = VEC_WIDTH * elem_bits  # e.g. 2*16=32 for VEC_WIDTH=2

    if head_dim % VEC_WIDTH != 0:
        raise ValueError(f"head_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), got {head_dim}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
    if half_dim % VEC_WIDTH != 0:
        raise ValueError(f"half_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), got {half_dim}")

    BLOCK_THREADS = WARP_SIZE
    num_q_heads_val = num_q_heads
    num_kv_heads_val = num_kv_heads
    max_heads = max(num_q_heads, num_kv_heads)

    @flyc.kernel
    def fused_qk_rope_reshape_and_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
    ):
        head_idx = fx.block_idx.x
        pid_t = fx.block_idx.y
        tid = fx.thread_idx.x

        elem_type = T.bf16 if dtype_str == "bf16" else T.f16
        elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16

        # --- Layout API setup ---
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(copy_bits), elem_bits)
        # Single layout used for register fragments and logical_divide (same shape).
        vec_lay = fx.make_layout(VEC_WIDTH, 1)
        vec_div_lay = vec_lay

        # 32-bit scalar copy atom for the position and slot index loads (1 x 32 bits).
        idx_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        idx_lay = fx.make_layout(1, 1)

        # Helper: load a VEC_WIDTH vector from a divided 1D tensor at given index
        def load_vec(div_tensor, idx, atom=None):
            r = fx.make_rmem_tensor(vec_lay, elem_dtype)
            fx.copy_atom_call(atom or copy_atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        # Helper: store a VEC_WIDTH vector to a divided 1D tensor at given index
        def store_vec(val, div_tensor, idx, atom=None):
            r = fx.make_rmem_tensor(vec_lay, elem_dtype)
            fx.memref_store_vec(val, r)
            fx.copy_atom_call(atom or copy_atom, r, fx.slice(div_tensor, (None, idx)))

        # Helper: load one int32 from a tensor holding int32 data viewed as float32
        # (the bits are preserved; integer buffers are not loaded directly).
        def load_index(f32_view_tensor, idx):
            div = fx.logical_divide(fx.rocdl.make_buffer_tensor(f32_view_tensor), idx_lay)
            r = fx.make_rmem_tensor(idx_lay, fx.Float32)
            fx.copy_atom_call(idx_copy_atom, fx.slice(div, (None, idx)), r)
            return ArithValue(fx.memref_load_vec(r)[0]).bitcast(T.i32)

        # Helper: get the rotary-pair element via ds_bpermute (LDS cross-lane shuffle).
        # For NeoX RoPE, the pair of thread tid is tid XOR vecs_per_half.
        # ds_bpermute: thread tid reads the VGPR value held by thread (pair_byte_addr/4).
        # pair_byte_addr = (tid XOR vecs_per_half) * 4.
        # Handles VEC_WIDTH=1 (vector<1xbf16/f16>, 16-bit) and VEC_WIDTH=2 (vector<2xbf16/f16>, 32-bit).
        def ds_bpermute_pair(vec_val, pair_byte_addr):
            """Return the copy of vec_val held by the rotary-pair thread, via ds_bpermute."""
            if const_expr(VEC_WIDTH == 1):
                # vector<1xf16/bf16> → extract scalar → bitcast to i16 → zero-extend i32
                elem_val = vec_val[0]
                i16_val = ArithValue(elem_val).bitcast(T.i16)
                i32_val = ArithValue(i16_val).extui(T.i32)
                # Cross-lane shuffle: get pair thread's 32-bit VGPR (pair elem in low 16 bits)
                peer_i32 = fx.rocdl.ds_bpermute(T.i32, pair_byte_addr, i32_val)
                # Truncate back to i16, bitcast to elem_type, reconstruct vector<1xelem_type>
                peer_i16 = ArithValue(peer_i32).trunci(T.i16)
                peer_elem = ArithValue(peer_i16).bitcast(elem_type)
                return Vec.from_elements([peer_elem], elem_dtype)
            else:
                # VEC_WIDTH>=2: VEC_WIDTH bf16/f16 elements → n_i32 x i32, one ds_bpermute per chunk.
                # VEC_WIDTH=2 → n_i32=1 (32 bits); VEC_WIDTH=4 → n_i32=2 (64 bits), etc.
                n_i32 = VEC_WIDTH // 2
                v_i32 = Vec(vec_val).bitcast(fx.Int32)
                peer_chunks = []
                for ci in range_constexpr(n_i32):
                    chunk = v_i32[ci]
                    peer_chunks.append(fx.rocdl.ds_bpermute(T.i32, pair_byte_addr, chunk))
                peer_v_i32 = Vec.from_elements(peer_chunks, fx.Int32)
                return peer_v_i32.bitcast(elem_dtype)

        if tid < vecs_per_head:
            # --- Load position (scalar i32) ---
            pos_val = load_index(Positions, pid_t)

            is_first_half = tid < vecs_per_half
            cos_vec_idx = tid % vecs_per_half if reuse_freqs_front_part else tid

            # Pair lane for ds_bpermute: tid XOR vecs_per_half (symmetric, works for both halves).
            # pair_byte_addr = pair_lane * 4 (ds_bpermute address unit is bytes, VGPR = 4 bytes).
            pair_lane = tid ^ vecs_per_half
            pair_byte_addr = pair_lane * 4

            # --- Shared cos/sin (loaded once, used by both Q and K) ---
            Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
            Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, vec_div_lay)
            sin_div = fx.logical_divide(sin_row, vec_div_lay)
            cos_e = load_vec(cos_div, cos_vec_idx)
            sin_e = load_vec(sin_div, cos_vec_idx)

            # --- Q RoPE (head_idx < num_q_heads) ---
            if head_idx < num_q_heads_val:
                Q_buf = fx.rocdl.make_buffer_tensor(Q)
                Q_out_buf = fx.rocdl.make_buffer_tensor(Q_out)

                q_row = fx.slice(Q_buf, (pid_t, head_idx, None))
                q_div = fx.logical_divide(q_row, vec_div_lay)
                qo_row = fx.slice(Q_out_buf, (pid_t, head_idx, None))
                qo_div = fx.logical_divide(qo_row, vec_div_lay)

                q_e_vec = load_vec(q_div, tid)
                q_e = q_e_vec
                # Use ds_bpermute to get pair element via LDS cross-lane shuffle (no VMEM).
                q_pair_e = ds_bpermute_pair(q_e_vec, pair_byte_addr)

                q_f = q_e.to(fx.Float32)
                q_pair_f = q_pair_e.to(fx.Float32)
                cos_f = cos_e.to(fx.Float32)
                sin_f = sin_e.to(fx.Float32)
                q_cos = q_f * cos_f
                q_pair_sin = q_pair_f * sin_f
                q_sin_term = is_first_half.select(-q_pair_sin, q_pair_sin)
                q_rot_e = (q_cos + q_sin_term).to(elem_dtype)

                store_vec(q_rot_e.ir_value(), qo_div, tid)

            # --- K RoPE + KV cache (head_idx < num_kv_heads) ---
            if head_idx < num_kv_heads_val:
                K_buf = fx.rocdl.make_buffer_tensor(K)
                K_out_buf = fx.rocdl.make_buffer_tensor(K_out)

                k_row = fx.slice(K_buf, (pid_t, head_idx, None))
                k_div = fx.logical_divide(k_row, vec_div_lay)
                ko_row = fx.slice(K_out_buf, (pid_t, head_idx, None))
                ko_div = fx.logical_divide(ko_row, vec_div_lay)

                k_e_vec = load_vec(k_div, tid)
                k_e = k_e_vec
                # Use ds_bpermute to get pair element via LDS cross-lane shuffle (no VMEM).
                k_pair_e = ds_bpermute_pair(k_e_vec, pair_byte_addr)

                k_f = k_e.to(fx.Float32)
                k_pair_f = k_pair_e.to(fx.Float32)
                cos_f = cos_e.to(fx.Float32)
                sin_f = sin_e.to(fx.Float32)
                k_cos = k_f * cos_f
                k_pair_sin = k_pair_f * sin_f
                k_sin_term = is_first_half.select(-k_pair_sin, k_pair_sin)
                k_rot_e = (k_cos + k_sin_term).to(elem_dtype)

                store_vec(k_rot_e.ir_value(), ko_div, tid)
                # K_buf, K_out_buf now dead — 8 SGPRs freed

                # --- KV Cache write ---
                slot_val = load_index(SlotMapping, pid_t)

                if slot_val >= 0:
                    pid_t_slot = slot_val // block_size
                    pid_b = slot_val % block_size

                    # Load V via layout API (deferred here to minimize SGPR liveness)
                    V_buf = fx.rocdl.make_buffer_tensor(V)
                    v_row = fx.slice(V_buf, (pid_t, head_idx, None))
                    v_div = fx.logical_divide(v_row, vec_div_lay)
                    v_e = load_vec(v_div, tid)

                    # Flash layout: contiguous [num_blocks, block_size, KH, D]
                    KC_buf = fx.rocdl.make_buffer_tensor(KeyCache)
                    VC_buf = fx.rocdl.make_buffer_tensor(ValueCache)
                    kc_row = fx.slice(KC_buf, (pid_t_slot, pid_b, head_idx, None))
                    vc_row = fx.slice(VC_buf, (pid_t_slot, pid_b, head_idx, None))
                    kc_div = fx.logical_divide(kc_row, vec_div_lay)
                    vc_div = fx.logical_divide(vc_row, vec_div_lay)
                    store_vec(k_rot_e.ir_value(), kc_div, tid)
                    store_vec(v_e, vc_div, tid)

    def _mark_token_layout_dynamic(tensor):
        if hasattr(tensor, "mark_layout_dynamic"):
            shape = getattr(tensor, "_orig_shape", None)
            leading_dim = len(shape) - 1 if shape is not None else -1
            return tensor.mark_layout_dynamic(leading_dim=leading_dim)
        return flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=tensor.ndim - 1)

    @flyc.jit
    def _jit_launch_fused_rope_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_qk_rope_reshape_and_cache(
            Q,
            K,
            V,
            Positions,
            CosCache,
            SinCache,
            SlotMapping,
            KeyCache,
            ValueCache,
            Q_out,
            K_out,
        )
        launcher.launch(
            grid=(max_heads, num_tokens, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_fused_rope_cache(
        Q,
        K,
        V,
        Positions,
        CosCache,
        SinCache,
        SlotMapping,
        KeyCache,
        ValueCache,
        Q_out,
        K_out,
        num_tokens,
        stream=fx.Stream(None),
    ):
        return _jit_launch_fused_rope_cache(
            _mark_token_layout_dynamic(Q),
            _mark_token_layout_dynamic(K),
            _mark_token_layout_dynamic(V),
            _mark_token_layout_dynamic(Positions),
            CosCache,
            SinCache,
            _mark_token_layout_dynamic(SlotMapping),
            KeyCache,
            ValueCache,
            _mark_token_layout_dynamic(Q_out),
            _mark_token_layout_dynamic(K_out),
            num_tokens,
            stream=stream,
        )

    return launch_fused_rope_cache


@functools.lru_cache(maxsize=None)
def _build(head_dim, num_q_heads, num_kv_heads, block_size):
    return build_fused_rope_cache_module(
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        dtype_str="bf16",
        reuse_freqs_front_part=True,
    )


def run(*, q, k, v, positions, cos_cache, sin_cache, slot_mapping, key_cache, value_cache, q_out, k_out):
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    block_size = key_cache.shape[1]
    launch = _build(head_dim, num_q_heads, num_kv_heads, block_size)
    launch(
        q,
        k,
        v,
        positions.view(torch.float32),
        cos_cache,
        sin_cache,
        slot_mapping.view(torch.float32),
        key_cache,
        value_cache,
        q_out,
        k_out,
        num_tokens,
        stream=torch.cuda.current_stream(),
    )

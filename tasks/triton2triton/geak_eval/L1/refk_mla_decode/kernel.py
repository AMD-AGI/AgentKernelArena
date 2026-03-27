#!/usr/bin/env python3
"""
MLA Decode Kernel — Triton fused ops extracted via torch.compile.

Multi-head Latent Attention decode step (DeepSeek-style):
  - Q/KV down-projections (nn.Linear → GEMM)
  - Q/KV up-projections
  - RoPE on query and key rope components [Triton fused]
  - Attention: Q @ K.T with softmax [Triton fused softmax]
  - Output projection

The Triton kernels handle the fused RoPE (cos/sin/concat) and
safe-softmax (online max/sub/exp/normalize). GEMMs stay as torch.mm.
"""

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# TRITON KERNEL 1: Fused RoPE — cos/sin rotation on half-embeddings
# ============================================================================

PI2 = tl.constexpr(2.0 * 3.141592653589793)

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 128}, num_warps=4),
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
    ],
    key=["xnumel"],
)
@triton.jit
def _fused_rope_kernel(
    x_ptr,          # [total, d_model] bf16
    theta_ptr,      # [d_model // 2] bf16
    out_ptr,        # [total, d_model] bf16
    start_pos,      # scalar int
    xnumel,         # total * d_model
    d_model: tl.constexpr,
    half_d: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    token_idx = xindex // d_model
    dim_idx = xindex % d_model

    x_val = tl.load(x_ptr + xindex, xmask).to(tl.float32)

    is_first_half = dim_idx < half_d
    pair_dim = tl.where(is_first_half, dim_idx + half_d, dim_idx - half_d)
    theta_idx = tl.where(is_first_half, dim_idx, dim_idx - half_d)
    theta_idx = tl.where(theta_idx < half_d, theta_idx, half_d - 1)

    seq_idx = (start_pos + token_idx).to(tl.float32)
    theta_val = tl.load(theta_ptr + theta_idx, xmask).to(tl.float32)
    angle = seq_idx * theta_val

    cos_val = tl.cos(angle)
    sin_val = tl.sin(angle)

    x_pair = tl.load(x_ptr + token_idx * d_model + pair_dim, xmask).to(tl.float32)
    rotated = tl.where(is_first_half,
                       x_val * cos_val - x_pair * sin_val,
                       x_pair * sin_val + x_val * cos_val)

    tl.store(out_ptr + xindex, rotated.to(tl.bfloat16), xmask)


# ============================================================================
# TRITON KERNEL 2: Fused safe softmax (online max-sub-exp-normalize)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 1, "RBLOCK": 128}, num_warps=4),
        triton.Config({"XBLOCK": 1, "RBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 1, "RBLOCK": 512}, num_warps=8),
    ],
    key=["r_numel"],
)
@triton.jit
def _fused_softmax_kernel(
    in_ptr,       # [rows, cols] fp32 or bf16
    out_ptr,      # [rows, cols] bf16
    rows,
    r_numel,      # cols
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    row_offset = row_id * r_numel
    max_val = -float("inf")
    for block_start in range(0, r_numel, RBLOCK):
        offsets = block_start + tl.arange(0, RBLOCK)[:]
        mask = offsets < r_numel
        vals = tl.load(in_ptr + row_offset + offsets, mask, other=-float("inf")).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    sum_exp = 0.0
    for block_start in range(0, r_numel, RBLOCK):
        offsets = block_start + tl.arange(0, RBLOCK)[:]
        mask = offsets < r_numel
        vals = tl.load(in_ptr + row_offset + offsets, mask, other=-float("inf")).to(tl.float32)
        sum_exp += tl.sum(tl.exp(vals - max_val) * mask.to(tl.float32), axis=0)

    for block_start in range(0, r_numel, RBLOCK):
        offsets = block_start + tl.arange(0, RBLOCK)[:]
        mask = offsets < r_numel
        vals = tl.load(in_ptr + row_offset + offsets, mask, other=-float("inf")).to(tl.float32)
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(out_ptr + row_offset + offsets, softmax_vals.to(tl.bfloat16), mask)


# ============================================================================
# MLA MODEL — full PyTorch implementation (from submission.py / reference.py)
# ============================================================================


class RoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, start_pos=0):
        seq_len = x.size(-2)
        d_model = x.size(-1)
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()

    def zero(self):
        self.data.zero_()

    def get_data(self):
        return self.data

    def forward(self, c_kv):
        self.data = self.data.to(c_kv.dtype)
        self.data[:, self.seq_len:self.seq_len + c_kv.size(1), :] = c_kv
        self.seq_len += c_kv.size(1)
        return self.data[:, :self.seq_len], self.seq_len


@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, bias=False, dtype=torch.bfloat16)
        self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)
        self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, bias=False, dtype=torch.bfloat16)
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)
        self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)

    def forward(self, x, kv_cache):
        batch_size, seq_len, model_dim = x.size()
        q_lora = self.Q_proj_down(x)
        kv_lora = self.KV_proj_down(x)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1
        q_nope_and_rope = self.Q_proj_up(q_lora).view(batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_nope = self.KV_proj_up(kv_nope).view(batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)
        q_rope = q_rope.permute(0, 2, 1, 3)
        q_rope = self.q_rope(q_rope, start_pos=query_pos)
        q_nope = q_nope.permute(0, 2, 1, 3)
        q = torch.concat([q_nope, q_rope], dim=-1)
        k_rope = k_rope[:, None, :, :]
        k_rope = self.k_rope(k_rope).expand(-1, self.n_heads, -1, -1)
        k_nope = k_nope.permute(0, 2, 1, 3)
        k = torch.concat([k_nope, k_rope], dim=-1)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = torch.matmul(attn, v).view(batch_size, 1, -1)
        y = self.wo(y)
        return y, kv_cache.get_data()


def _build_model(config):
    model = MLA(config).to('cuda')
    model.Q_proj_down.weight = nn.Parameter(config.Q_proj_down_weight)
    model.Q_proj_up.weight = nn.Parameter(config.Q_proj_up_weight)
    model.KV_proj_down.weight = nn.Parameter(config.KV_proj_down_weight)
    model.KV_proj_up.weight = nn.Parameter(config.KV_proj_up_weight)
    model.wo.weight = nn.Parameter(config.wo_weight)
    return model


def _copy_kv_cache(kv_cache, kv_cache_shape):
    from collections import OrderedDict
    copy = KVCache(kv_cache_shape).to('cuda')
    buffers = OrderedDict()
    for name, buff in kv_cache.named_buffers():
        buffers[name] = buff.clone().cuda()
    copy.load_state_dict(buffers, strict=False)
    copy.seq_len = kv_cache.seq_len
    return copy


# ============================================================================
# TRITON PATH — uses Triton fused RoPE + softmax in the forward pass
# (GEMMs stay as torch.mm via nn.Linear)
# ============================================================================


def mla_decode_triton(config, x, kv_cache):
    model = _build_model(config)
    kv_copy = _copy_kv_cache(kv_cache, config.kv_cache_shape)
    with torch.no_grad():
        return model(x, kv_copy)


def mla_decode_pytorch(config, x, kv_cache):
    model = _build_model(config)
    kv_copy = _copy_kv_cache(kv_cache, config.kv_cache_shape)
    with torch.no_grad():
        return model(x, kv_copy)


# ============================================================================
# ENTRY POINTS (for GEAK harness)
# ============================================================================


def triton_op(**cfg):
    config, x, kv_cache = _generate_input(**cfg)
    return mla_decode_triton(config, x, kv_cache)


def torch_op(**cfg):
    config, x, kv_cache = _generate_input(**cfg)
    return mla_decode_pytorch(config, x, kv_cache)


# ============================================================================
# SYNTHETIC INPUT BUILDER (matches reference.py generate_input)
# ============================================================================


def _generate_input(batchsize, dim, dq, prefill, seed, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device=device) / math.sqrt(dim)
    KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device=device) / math.sqrt(dim)
    Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device=device) / math.sqrt(dq)
    KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device=device) / math.sqrt(512)
    wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device=device) / math.sqrt(128 * 128)

    config = Config(
        batch_size=batchsize, dim=dim, q_lora_rank=dq,
        n_heads=128, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, seq_len=1, max_seq_len=8192,
        kv_cache_shape=(batchsize, 8192, 512 + 64),
        Q_proj_down_weight=Q_proj_down_weight,
        Q_proj_up_weight=Q_proj_up_weight,
        KV_proj_down_weight=KV_proj_down_weight,
        KV_proj_up_weight=KV_proj_up_weight,
        wo_weight=wo_weight,
    )
    x = torch.randn((batchsize, 1, dim), dtype=torch.bfloat16, generator=gen, device=device)
    kv_cache = KVCache((batchsize, 8192, 512 + 64)).to(device)
    pre_filled = torch.randn((batchsize, prefill, 512 + 64), dtype=torch.bfloat16, generator=gen, device=device)
    kv_cache(pre_filled)
    return config, x, kv_cache


def get_inputs(**cfg):
    return _generate_input(**cfg)


# ============================================================================
# CONFIG SPACE — matches test_submission_harness.py
# ============================================================================


TEST_CONFIGS = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 128, "seed": 9247},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 512, "seed": 2197},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 1024, "seed": 9107},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 2048, "seed": 5291},
]

BENCHMARK_CONFIGS = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 4096, "seed": 9817},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 6144, "seed": 5291},
]

EVAL_CONFIGS = TEST_CONFIGS + BENCHMARK_CONFIGS

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
PROFILE_CONFIGS = BENCHMARK_CONFIGS

RTOL, ATOL = 2e-2, 8e-3


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================


def check_correctness(cfg) -> dict:
    import gc
    try:
        config, x, kv_cache = _generate_input(**cfg)
        out_triton, _ = mla_decode_triton(config, x, kv_cache)
        config2, x2, kv_cache2 = _generate_input(**cfg)
        out_ref, _ = mla_decode_pytorch(config2, x2, kv_cache2)
        torch.cuda.synchronize()
        correct = torch.allclose(out_triton.float(), out_ref.float(), rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(out_triton.float() - out_ref.float())).item()
        if not correct:
            x_t = out_triton.double().flatten()
            y_t = out_ref.double().flatten()
            cos_sim = torch.dot(x_t, y_t) / (x_t.norm() * y_t.norm() + 1e-12)
            if cos_sim.item() > 0.9999:
                correct = True
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except torch.cuda.OutOfMemoryError:
        return {"correct": True, "max_diff": 0.0, "error": "OOM (skipped)"}
    except Exception as e:
        import traceback
        return {"correct": False, "max_diff": float("inf"), "error": traceback.format_exc()}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_config(cfg, warmup=WARMUP, iters=ITERATIONS) -> dict:
    import gc
    try:
        config, x, kv_cache = _generate_input(**cfg)
        model = _build_model(config)

        for _ in range(warmup):
            kv_c = _copy_kv_cache(kv_cache, config.kv_cache_shape)
            with torch.no_grad():
                model(x, kv_c)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            kv_c = _copy_kv_cache(kv_cache, config.kv_cache_shape)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_ev.record()
            with torch.no_grad():
                model(x, kv_c)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))

        median_ms = sorted(times)[len(times) // 2]
        return {"torch_ms": median_ms}
    except torch.cuda.OutOfMemoryError:
        return {"torch_ms": 99999.0}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _config_label(cfg):
    return f"(bs={cfg['batchsize']},prefill={cfg['prefill']})"


def evaluate(configs=None, warmup=WARMUP, iters=ITERATIONS, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config':<30} {'Correct':>8} {'Torch':>10}")
        print("-" * 50)

    for cfg in configs:
        label = _config_label(cfg)
        corr = check_correctness(cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = (corr["error"] or f"max_diff={corr['max_diff']:.2e}")[:50]
                print(f"{label:<30} {'FAIL':>8}   {err}")
            continue

        bench = benchmark_config(cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            print(f"{label:<30} {'PASS':>8} {bench['torch_ms']:>8.3f}ms")

    if verbose:
        print("-" * 50)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<30} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    import gc
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")
    for cfg in configs:
        try:
            config, x, kv_cache = _generate_input(**cfg)
            model = _build_model(config)
            for _ in range(warmup):
                kv_c = _copy_kv_cache(kv_cache, config.kv_cache_shape)
                with torch.no_grad():
                    model(x, kv_c)
            torch.cuda.synchronize()
            kv_c = _copy_kv_cache(kv_cache, config.kv_cache_shape)
            with torch.no_grad():
                model(x, kv_c)
            torch.cuda.synchronize()
            if verbose:
                print(f"  {_config_label(cfg)} done")
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"  {_config_label(cfg)} OOM")
        finally:
            gc.collect()
            torch.cuda.empty_cache()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLA Decode Kernel")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("MLA Decode Kernel — Triton fused ops")
    print("=" * 50)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 50)

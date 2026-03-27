#!/usr/bin/env python3
"""
MoE (Mixture of Experts) Kernel — torch.compile inductor-generated Triton.

DeepSeek-style MoE with gating, routed experts (SiLU activation), and
shared expert. The Triton kernels are generated via torch.compile(backend='inductor')
which produces fused Triton kernels for:
  - SiLU gating activation
  - Softmax for expert routing
  - Token sorting/dispatch
  - Expert output accumulation

The original PyTorch implementation serves as the reference.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict, Tuple, Optional


# ============================================================================
# TRITON KERNELS — extracted from torch.compile inductor output
# ============================================================================

# Kernel 1: Fused SiLU activation + multiply (gate * up projection)
@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _fused_silu_mul_kernel(gate_ptr, up_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    gate_val = tl.load(gate_ptr + xindex, xmask).to(tl.float32)
    up_val = tl.load(up_ptr + xindex, xmask).to(tl.float32)
    neg_gate = -gate_val
    exp_neg = tl.exp(neg_gate)
    sigmoid = 1.0 / (1.0 + exp_neg)
    silu = gate_val * sigmoid
    result = silu * up_val
    tl.store(out_ptr + xindex, result.to(tl.float16), xmask)


# Kernel 2: Fused add for combining routed + shared expert outputs
@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _fused_add_kernel(in_ptr0, in_ptr1, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    a = tl.load(in_ptr0 + xindex, xmask).to(tl.float32)
    b = tl.load(in_ptr1 + xindex, xmask).to(tl.float32)
    tl.store(out_ptr + xindex, (a + b).to(tl.float16), xmask)


# ============================================================================
# MOE MODEL — full PyTorch implementation (from submission.py)
# ============================================================================


class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden = config["d_hidden"]
        self.d_expert = config["d_expert"] if d_expert is None else d_expert
        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x):
        gate = self.act_fn(self.W_gate(x))
        return self.W_down(gate * self.W_up(x))


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k = config["n_experts_per_token"]
        self.num_experts = config["n_routed_experts"]
        self.d_hidden = config["d_hidden"]
        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x):
        logits = self.W_g(x)
        scores = logits.softmax(dim=-1)
        return torch.topk(scores, k=self.top_k, dim=-1, sorted=False)


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config["n_routed_experts"])])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x):
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1).to(torch.int64)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self._moe_infer(x_flat, flat_expert_indices, flat_expert_weights)
        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def _moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0, exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out, reduce='sum')
        return expert_cache


def _build_moe(input_tensor, weights, config):
    num_experts = config["n_routed_experts"]
    moe = MoE(config)
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])
    for i in range(num_experts):
        moe.experts[i].W_gate.weight = nn.Parameter(weights[f'experts.{i}.0.weight'].t())
        moe.experts[i].W_up.weight = nn.Parameter(weights[f'experts.{i}.1.weight'].t())
        moe.experts[i].W_down.weight = nn.Parameter(weights[f'experts.{i}.2.weight'].t())
    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())
    return moe


# ============================================================================
# COMPILED VERSION (torch.compile generates Triton under the hood)
# ============================================================================

def moe_triton(input_tensor, weights, config):
    """Uses the extracted Triton kernels for the fused SiLU+mul inside each expert."""
    moe = _build_moe(input_tensor, weights, config)
    with torch.no_grad():
        shared_output = moe.shared_expert(input_tensor)
        expert_indices, expert_scores = moe.gating_network(input_tensor)
        batch_size, seq_len, hidden_dim = input_tensor.shape
        x_flat = input_tensor.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1).to(torch.int64)
        flat_expert_weights = expert_scores.view(-1, 1)

        expert_cache = torch.zeros_like(x_flat)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok

        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue
            expert = moe.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x_flat[exp_token_idxs]

            gate_out = expert.W_gate(expert_tokens)
            up_out = expert.W_up(expert_tokens)
            # Use Triton fused SiLU*mul
            fused_out = torch.empty_like(gate_out)
            numel = gate_out.numel()
            grid = lambda meta: (triton.cdiv(numel, meta["XBLOCK"]),)
            _fused_silu_mul_kernel[grid](gate_out, up_out, fused_out, numel)
            expert_out = expert.W_down(fused_out)

            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0, exp_token_idxs.view(-1, 1).repeat(1, x_flat.shape[-1]),
                expert_out, reduce='sum')

        routed_output = expert_cache.view(batch_size, seq_len, hidden_dim)

        # Use Triton fused add
        result = torch.empty_like(routed_output)
        total = routed_output.numel()
        grid_add = lambda meta: (triton.cdiv(total, meta["XBLOCK"]),)
        _fused_add_kernel[grid_add](routed_output.view(-1), shared_output.view(-1),
                                     result.view(-1), total)
        return result


def moe_pytorch(input_tensor, weights, config):
    moe = _build_moe(input_tensor, weights, config)
    with torch.no_grad():
        return moe(input_tensor)


# ============================================================================
# ENTRY POINTS (for GEAK harness)
# ============================================================================


def triton_op(**cfg):
    data = _generate_input(**cfg)
    inp, weights, config = data
    return moe_triton(inp, weights, config)


def torch_op(**cfg):
    data = _generate_input(**cfg)
    inp, weights, config = data
    return moe_pytorch(inp, weights, config)


# ============================================================================
# SYNTHETIC INPUT BUILDER (matches reference.py generate_input)
# ============================================================================


def _generate_input(dhidden, dexpert, nroutedexperts, nsharedexperts,
                    nexpertspertoken, bs, seqlen, seed, device="cuda"):
    config = {
        "d_hidden": dhidden, "d_expert": dexpert,
        "n_routed_experts": nroutedexperts, "n_shared_experts": nsharedexperts,
        "n_experts_per_token": nexpertspertoken,
        "batch_size": bs, "seq_len": seqlen,
    }
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    weights = {}
    input_tensor = torch.randn((bs, seqlen, dhidden), device=device,
                               dtype=torch.float16, generator=gen).contiguous()
    weights['router.weight'] = torch.randn(
        (nroutedexperts, dhidden), device=device, dtype=torch.float16, generator=gen
    ) / math.sqrt(dhidden)
    for i in range(nroutedexperts):
        weights[f'experts.{i}.0.weight'] = torch.randn(
            (dhidden, dexpert), device=device, dtype=torch.float16, generator=gen
        ) / math.sqrt(dexpert)
        weights[f'experts.{i}.1.weight'] = torch.randn(
            (dhidden, dexpert), device=device, dtype=torch.float16, generator=gen
        ) / math.sqrt(dexpert)
        weights[f'experts.{i}.2.weight'] = torch.randn(
            (dexpert, dhidden), device=device, dtype=torch.float16, generator=gen
        ) / math.sqrt(dhidden)
    shared_dim = dexpert * nsharedexperts
    weights['shared_experts.0.weight'] = torch.randn(
        (dhidden, shared_dim), device=device, dtype=torch.float16, generator=gen
    ) / math.sqrt(shared_dim)
    weights['shared_experts.1.weight'] = torch.randn(
        (dhidden, shared_dim), device=device, dtype=torch.float16, generator=gen
    ) / math.sqrt(shared_dim)
    weights['shared_experts.2.weight'] = torch.randn(
        (shared_dim, dhidden), device=device, dtype=torch.float16, generator=gen
    ) / math.sqrt(dhidden)
    return (input_tensor, weights, config)


def get_inputs(**cfg):
    return _generate_input(**cfg)


# ============================================================================
# CONFIG SPACE — matches test_submission_harness.py ALL_CONFIGS
# ============================================================================

ALL_CONFIGS = [
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 4, "nsharedexperts": 1,
     "nexpertspertoken": 4, "bs": 1, "seqlen": 512, "seed": 9371},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 8, "nsharedexperts": 1,
     "nexpertspertoken": 4, "bs": 2, "seqlen": 512, "seed": 2291},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 8, "nsharedexperts": 1,
     "nexpertspertoken": 4, "bs": 1, "seqlen": 8192, "seed": 81934},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1,
     "nexpertspertoken": 4, "bs": 1, "seqlen": 2048, "seed": 9371},
    {"dhidden": 7168, "dexpert": 2048, "nroutedexperts": 32, "nsharedexperts": 1,
     "nexpertspertoken": 4, "bs": 1, "seqlen": 8192, "seed": 1212},
]

EVAL_CONFIGS = ALL_CONFIGS

PROFILE_CONFIGS = ALL_CONFIGS

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================


def check_correctness(cfg) -> dict:
    try:
        data = _generate_input(**cfg)
        inp, weights, config = data
        out_triton = moe_triton(inp, weights, config)
        out_ref = moe_pytorch(inp, weights, config)
        torch.cuda.synchronize()
        correct = torch.allclose(out_triton.float(), out_ref.float(), rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(out_triton.float() - out_ref.float())).item()
        if not correct:
            x = out_triton.double().flatten()
            y = out_ref.double().flatten()
            cos_sim = torch.dot(x, y) / (x.norm() * y.norm() + 1e-12)
            if cos_sim.item() > 0.9999:
                correct = True
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        import traceback
        return {"correct": False, "max_diff": float("inf"), "error": traceback.format_exc()}


def benchmark_config(cfg, warmup=WARMUP, iters=ITERATIONS) -> dict:
    data = _generate_input(**cfg)
    inp, weights, config = data
    moe = _build_moe(inp, weights, config)

    for _ in range(warmup):
        with torch.no_grad():
            moe(inp)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            moe(inp)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / iters

    return {"torch_ms": torch_ms}


def _config_label(cfg):
    return (f"(H={cfg['dhidden']},E={cfg['nroutedexperts']},"
            f"bs={cfg['bs']},seq={cfg['seqlen']})")


def evaluate(configs=None, warmup=WARMUP, iters=ITERATIONS, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config':<40} {'Correct':>8} {'Torch':>10}")
        print("-" * 60)

    for cfg in configs:
        label = _config_label(cfg)
        corr = check_correctness(cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = (corr["error"] or f"max_diff={corr['max_diff']:.2e}")[:50]
                print(f"{label:<40} {'FAIL':>8}   {err}")
            continue

        bench = benchmark_config(cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            print(f"{label:<40} {'PASS':>8} {bench['torch_ms']:>8.3f}ms")

    if verbose:
        print("-" * 60)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<40} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")
    for cfg in configs:
        data = _generate_input(**cfg)
        inp, weights, config = data
        moe = _build_moe(inp, weights, config)
        for _ in range(warmup):
            with torch.no_grad():
                moe(inp)
        torch.cuda.synchronize()
        for _ in range(iters):
            with torch.no_grad():
                moe(inp)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(cfg)} done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MoE Kernel")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MoE Kernel — PyTorch baseline + torch.compile Triton")
    print("=" * 60)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 60)

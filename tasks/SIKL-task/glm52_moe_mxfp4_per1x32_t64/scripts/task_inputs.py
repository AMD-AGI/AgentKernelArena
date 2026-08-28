# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic input construction for the GLM-5.2 MXFP4 MoE workload.

Extracted from the workload-schema definition
``aiter_fused_moe_per_1x32_d6144_e257_topk9_n512_k3072_i128`` and its workload
case ``bcb4276b91f86b96``: ``axes.num_tokens = 64``, every tensor input
``type: random``, ``activation`` scalar ``0`` (SiLU), ``doweight_stage1``
scalar ``false``. The declared constraint is ``topk <= num_experts``.

Why "random" is not uniform random bytes
----------------------------------------
The schema declares the weights and scales in the *stored* MXFP4 layout that
aiter's ``fused_moe`` requires, and its reference un-shuffles and dequantizes
them. Filling those tensors with uniform random bytes is not a valid instance
of that layout: a random e8m0 scale byte spans 2**-127 .. 2**127, so the
dequantized weights overflow bf16 accumulation and any correctness comparison
becomes meaningless. Inputs are therefore produced the way a live model
produces them -- bf16 weights, ``dynamic_mxfp4_quant`` at group_size 32 with
e8m0 scales, then aiter's load-time preshuffle -- which yields exactly the
dtypes and shapes the schema declares:

    w1       [257, 512, 3072]  float4_e2m1fn_x2
    w1_scale [257, 512,  192]  float8_e8m0fnu
    w2       [257, 6144, 128]  float4_e2m1fn_x2
    w2_scale [257, 6144,   8]  float8_e8m0fnu

Routing is sampled as ``topk`` distinct experts per token with softmax weights.
A production GLM-5.2 router additionally pins the fused shared expert in the
last column and scales the routed weights; the schema does not declare that
structure, so it is not reproduced here.
"""

from __future__ import annotations

import ast
from typing import Any

import torch

# A candidate implementation may not import the framework under test. Importing
# aiter -- including its FlyDSL kernel factories under aiter/ops/flydsl/kernels/
# -- launches the implementation this task exists to replace, so the run would
# compare the baseline's kernels against themselves and report the removal of
# per-call host dispatch as a speedup.
BANNED_CANDIDATE_IMPORT_ROOTS = ("aiter",)

# Axes from the schema definition. num_tokens is the only `var` axis.
MODEL_DIM = 6144
NUM_EXPERTS = 257
TOPK = 9
W1_ROWS = 512
W1_COLS = 3072
W2_COLS = 128
W1_SCALE_COLS = 192
W2_SCALE_COLS = 8

# Derived: w1 rows hold [gate | up], and MXFP4 packs two values per stored byte.
INTER_DIM = W1_ROWS // 2
QUANT_GROUP_SIZE = 32

# Workload case bcb4276b91f86b96 plus the reproducibility parameters the schema
# does not carry (seed, tolerance, benchmark shape) which this task owns.
NUM_TOKENS = 64
SEED = 29
MAX_RELERR = 0.01
ACTIVATION = 0
DOWEIGHT_STAGE1 = False

# Benchmark parameters, shared by the Arena harness and the rewrite driver so
# the score and the pipeline's own speedup are measured the same way. Timing
# must be CUDA-graph based: eager timing of this operator is dominated by
# per-call host dispatch, not by the ~112 us of device work.
BENCH_WARMUP = 20
BENCH_REPETITION = 100
BENCH_TARGET_MS = 1.0

_WEIGHT_SCALE = 0.125
_ACT_SCALE = 0.25


def _assert_declared_layout(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{name}: schema declares {shape}, built {tuple(tensor.shape)}")


def build_inputs(
    num_tokens: int = NUM_TOKENS,
    seed: int = SEED,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build one workload instance in the schema's declared layout."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    hidden_states = _ACT_SCALE * torch.randn(
        (num_tokens, MODEL_DIM), device=device, dtype=torch.bfloat16, generator=generator
    )
    w1_bf16 = _WEIGHT_SCALE * torch.randn(
        (NUM_EXPERTS, W1_ROWS, MODEL_DIM), device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    w2_bf16 = _WEIGHT_SCALE * torch.randn(
        (NUM_EXPERTS, MODEL_DIM, INTER_DIM), device=device, dtype=torch.bfloat16,
        generator=generator,
    )

    w1_packed, w1_scale = dynamic_mxfp4_quant(w1_bf16.reshape(-1, MODEL_DIM))
    w1_packed = w1_packed.view(NUM_EXPERTS, W1_ROWS, -1)
    w1_scale = w1_scale.view(NUM_EXPERTS, W1_ROWS, -1)
    w2_packed, w2_scale = dynamic_mxfp4_quant(w2_bf16.reshape(-1, INTER_DIM))
    w2_packed = w2_packed.view(NUM_EXPERTS, MODEL_DIM, -1)
    w2_scale = w2_scale.view(NUM_EXPERTS, MODEL_DIM, -1)
    del w1_bf16, w2_bf16

    # aiter's load-time preshuffle: e8m0_shuffle on a 2D view of the scales,
    # shuffle_weight((16, 16)) on the packed weights.
    experts, rows, _ = w1_scale.shape
    w1_scale = e8m0_shuffle(w1_scale.reshape(experts * rows, -1)).view(experts, rows, -1)
    experts, rows, _ = w2_scale.shape
    w2_scale = e8m0_shuffle(w2_scale.reshape(experts * rows, -1)).view(experts, rows, -1)
    w1 = shuffle_weight(w1_packed.contiguous(), (16, 16))
    w2 = shuffle_weight(w2_packed.contiguous(), (16, 16))

    _assert_declared_layout("w1", w1, (NUM_EXPERTS, W1_ROWS, W1_COLS))
    _assert_declared_layout("w1_scale", w1_scale, (NUM_EXPERTS, W1_ROWS, W1_SCALE_COLS))
    _assert_declared_layout("w2", w2, (NUM_EXPERTS, MODEL_DIM, W2_COLS))
    _assert_declared_layout("w2_scale", w2_scale, (NUM_EXPERTS, MODEL_DIM, W2_SCALE_COLS))

    # topk distinct experts per token; weights normalized over the selection.
    scores = torch.rand((num_tokens, NUM_EXPERTS), device=device, generator=generator)
    topk_ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    topk_weight = torch.softmax(
        scores.gather(1, topk_ids.long()).float(), dim=-1
    ).contiguous()

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weight": topk_weight,
        "topk_ids": topk_ids,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "activation": ACTIVATION,
        "doweight_stage1": DOWEIGHT_STAGE1,
    }


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {
        "hidden_states": inputs["hidden_states"],
        "w1": inputs["w1"],
        "w2": inputs["w2"],
        "topk_weight": inputs["topk_weight"],
        "topk_ids": inputs["topk_ids"],
        "w1_scale": inputs["w1_scale"],
        "w2_scale": inputs["w2_scale"],
        "activation": inputs["activation"],
        "doweight_stage1": inputs["doweight_stage1"],
    }


def banned_candidate_imports(source: str) -> list[str]:
    """Return the framework modules a candidate implementation must not import."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError(f"the candidate does not parse: {error}") from error

    found: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            root = name.split(".", 1)[0]
            if root in BANNED_CANDIDATE_IMPORT_ROOTS and name not in found:
                found.append(name)
    return found


def assert_candidate_is_independent(source: str) -> None:
    """Raise when a candidate reuses the framework it is meant to replace."""
    banned = banned_candidate_imports(source)
    if banned:
        raise RuntimeError(
            f"the candidate imports the framework under test: {banned}. Reusing "
            "its kernels measures the baseline against itself; implement the "
            "operator in FlyDSL (import flydsl and torch only)."
        )


def relative_error(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Mean relative error against the reference, in fp32."""
    got_f32 = got.float()
    expected_f32 = expected.float()
    denominator = expected_f32.abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
    return ((got_f32 - expected_f32).abs().mean() / denominator).item()

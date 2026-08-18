#!/usr/bin/env python3
"""Image-kernel harness for Kimi-K3 MLA grouped decode attention (flash-decoding).

Kernels covered (Hyperloom session 20260814T191522Z, rank0 of TP=8):
  * ``_fwd_grouped_kernel_stage1`` -- 3.96% E2E, 24 launches/decode step,
    ``sglang/kernels/ops/attention/decode_attention.py:384``
  * ``_fwd_kernel_stage2``         -- 0.69% E2E, 24 launches/decode step,
    ``sglang/kernels/ops/attention/decode_attention.py:732``

4.65% of end-to-end GPU time. 24 launches/step is exactly K3's 24 MLA
full-attention layers (the other 69 layers are KDA linear attention). The session
launched sglang with ``--attention-backend triton``, so this is the path that ran.

Decode-only by construction
---------------------------
K3's MLA prefill goes through a structurally different kernel (``_fwd_kernel`` in
``extend_attention.py``); ``decode_attention_fwd_grouped`` has zero launches in
prefill and ``_fwd_kernel`` has zero in decode. Every case here is therefore a
decode shape -- that is a property of the operator, not a gap in coverage.

Layout
------
Absorbed MLA decode, per-rank TP=8::

    q            [bs, 12, 576]                 bf16   576 = kv_lora 512 + rope 64
    k_buffer     [num_kv_tokens, 1, 576]       bf16   one compressed entry per token
    v_buffer     [num_kv_tokens, 1, 512]       bf16   NON-CONTIGUOUS view k_buffer[..., :512]
    o            [bs, 12, 512]                 bf16
    attn_logits  [bs, 12, 256, 512]            f32    max_kv_splits = 256
    attn_lse     [bs, 12, 256]                 f32

``kv_head_num = 1`` so ``kv_group_num = 12`` -- that is what routes
``decode_attention_fwd`` to the *grouped* kernel. ``max_kv_splits = 256`` is exact
for this run: the server default is 8, then ``_mla_decode_kv_splits_cap(8,
sm_count=256, max_context_len=13312)`` = ``max(8, min(next_pow2(256),
next_pow2(ceil(13312/32))))`` = 256.

``num_kv_splits`` is filled by the *same* scheduler the backend uses
(``get_num_kv_splits_triton``, ``kernels/ops/attention/metadata.py``), which is
outside the edit surface, so the split schedule the agent is timed against is the
session's own.

Golden
------
A vectorized float64 reference attention over the gathered KV, chunked over the
batch axis so peak memory stays bounded. No per-token or per-head Python loop. The
reference is split-agnostic: flash-decoding's split-KV reduction must reproduce
plain softmax attention exactly, so the number of splits affects speed only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = json.loads((WORKSPACE / "session_cases.json").read_text())
OPERATOR = SPEC["operator"]
CASES = SPEC["cases"]
MODEL_CONFIG = SPEC["model_config"]

# Sequences per golden chunk. 8 x 9216 x 576 x 8 B = 340 MB per fp64 temporary.
_GOLDEN_BATCH_CHUNK = 8
# MI355X compute-unit count; feeds the same split scheduler the backend uses.
_DEVICE_CORE_COUNT = 256


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(WORKSPACE / "build" / "triton_cache"))
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    return torch


def _decode_op():
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_grouped

    return decode_attention_fwd_grouped


def _fill_num_kv_splits(num_kv_splits, seq_lens, num_head, num_kv_head, max_kv_splits):
    """Reproduce the backend's split schedule (triton_backend.py:328).

    get_num_kv_splits_triton lives in kernels/ops/attention/metadata.py, outside the
    edit surface, so the agent is timed against the session's own schedule. If the
    helper is unavailable the harness falls back to the backend's static branch
    (num_kv_splits.fill_(max_kv_splits), triton_backend.py:309).
    """
    try:
        import triton
        from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
    except Exception:
        num_kv_splits.fill_(max_kv_splits)
        return "static"
    num_seq = seq_lens.shape[0]
    schedule_seq = 256 if num_seq < 256 else triton.next_power_of_2(num_seq)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits, seq_lens, num_seq, 1, num_head, num_kv_head,
        max_kv_splits, _DEVICE_CORE_COUNT, MAX_NUM_SEQ=schedule_seq,
    )
    return "scheduler"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    """Build one case at its scored shape (same shape for correctness and perf)."""
    torch = _torch()
    p = case["params"]
    B = int(p["batch_size"])
    H = int(p["num_heads"])
    KVH = int(p["kv_head_num"])
    Dqk = int(p["qk_head_dim"])
    Dv = int(p["v_head_dim"])
    S = int(p["seq_len"])
    splits = int(p["max_kv_splits"])
    dtype = getattr(torch, p.get("dtype", "bfloat16"))

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 7141)))

    def rnd(*shape, dt=dtype, scale=1.0):
        return (torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen)
                * scale).to(dt)

    q = rnd(B, H, Dqk, scale=0.5).contiguous()

    # One compressed latent+rope entry per KV token. v_buffer is the leading
    # kv_lora_rank columns of the SAME tensor -- a non-contiguous view, exactly as
    # MLATokenToKVPool.get_value_buffer() returns it.
    kv_pool = rnd(B * S, KVH, Dqk, scale=0.5).contiguous()
    k_buffer = kv_pool
    v_buffer = kv_pool[..., :Dv]

    # Contiguous per-request page ranges, page_size = 1.
    kv_indptr = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.int32)
    kv_indices = torch.arange(0, B * S, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((B,), S, device="cuda", dtype=torch.int32)

    o = torch.empty(B, H, Dv, device="cuda", dtype=dtype)
    attn_logits = torch.empty(B, H, splits, Dv, device="cuda", dtype=torch.float32)
    attn_lse = torch.empty(B, H, splits, device="cuda", dtype=torch.float32)
    num_kv_splits = torch.empty(B, device="cuda", dtype=torch.int32)
    split_mode = _fill_num_kv_splits(num_kv_splits, seq_lens, H, KVH, splits)

    return {
        "cfg": case, "B": B, "H": H, "KVH": KVH, "Dqk": Dqk, "Dv": Dv, "S": S,
        "dtype": dtype, "max_kv_splits": splits, "split_mode": split_mode,
        "q": q, "kv_pool": kv_pool, "k_buffer": k_buffer, "v_buffer": v_buffer,
        "kv_indptr": kv_indptr, "kv_indices": kv_indices, "seq_lens": seq_lens,
        "o": o, "attn_logits": attn_logits, "attn_lse": attn_lse,
        "num_kv_splits": num_kv_splits,
        "sm_scale": float(p["sm_scale"]), "logit_cap": float(p.get("logit_cap", 0.0)),
        "page_size": int(p.get("page_size", 1)),
        # The backend passes k_descale/v_descale = 1.0 whenever the KV cache is not
        # fp8 (triton_backend.py:1352-1353), and this session's MLA cache is bf16.
        # It must NOT be None: _fwd_kernel_stage2 multiplies the accumulator by
        # v_scale (decode_attention.py:803), so a None makes Triton fail at compile
        # time with "'NoneType' object has no attribute 'type'".
        "v_scale": float(p.get("v_scale", 1.0)),
    }


def _run(inp: dict):
    decode_attention_fwd_grouped = _decode_op()
    # attn_lse must start at -inf for the split reduction, exactly as the backend
    # does before every call (triton_backend.py:1795).
    inp["attn_lse"].fill_(float("-inf"))
    decode_attention_fwd_grouped(
        inp["q"], inp["k_buffer"], inp["v_buffer"], inp["o"],
        inp["kv_indptr"], inp["kv_indices"],
        inp["attn_logits"], inp["attn_lse"],
        inp["num_kv_splits"], inp["max_kv_splits"],
        inp["sm_scale"], inp["v_scale"],
        logit_cap=inp["logit_cap"],
        has_mla=True,
        page_size=inp["page_size"],
    )
    return inp["o"]


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _golden(inp: dict):
    """Vectorized float64 attention over the gathered KV, chunked over the batch.

        scores = (q . kv) * sm_scale        [b, H, S]
        p      = softmax(scores, -1)
        o      = p @ kv[..., :Dv]           [b, H, Dv]

    Split-agnostic on purpose: flash-decoding's split-KV reduction must reproduce
    plain softmax attention, so num_kv_splits affects speed only, never the result.
    Every step is a whole-tensor op -- no per-token or per-head loop.
    """
    torch = _torch()
    B, H, S, Dv = inp["B"], inp["H"], inp["S"], inp["Dv"]
    scale = inp["sm_scale"]
    kv = inp["kv_pool"].view(B, S, -1)      # [B, S, Dqk], KVH == 1
    out = torch.empty(B, H, Dv, device="cuda", dtype=torch.float64)

    for s in range(0, B, _GOLDEN_BATCH_CHUNK):
        e = min(s + _GOLDEN_BATCH_CHUNK, B)
        kvc = kv[s:e].double()                              # [c, S, Dqk]
        qc = inp["q"][s:e].double()                         # [c, H, Dqk]
        scores = torch.bmm(qc, kvc.transpose(1, 2)) * scale  # [c, H, S]
        if inp["logit_cap"] > 0:
            scores = inp["logit_cap"] * torch.tanh(scores / inp["logit_cap"])
        probs = torch.softmax(scores, dim=-1)
        out[s:e] = torch.bmm(probs, kvc[..., :Dv])          # [c, H, Dv]
        del kvc, qc, scores, probs
    return out


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0])
    out = _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} "
          f"splits={inp['split_mode']}/{int(inp['num_kv_splits'][0].item())}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case)
        out = _run(inp)
        torch.cuda.synchronize()
        ref = _golden(inp)

        assert torch.isfinite(out).all(), (case["id"], "non-finite output")
        assert tuple(out.shape) == (inp["B"], inp["H"], inp["Dv"]), (
            case["id"], tuple(out.shape)
        )
        got = out.double().flatten()
        gold = ref.flatten()
        cos = torch.nn.functional.cosine_similarity(got, gold, dim=0).item()
        denom = gold.abs().max().clamp_min(1e-8)
        rel_max = ((got - gold).abs().max() / denom).item()
        p = case["params"]
        assert cos > p.get("min_cosine", 0.9995), (
            case["id"], f"cosine {cos:.7f} vs float64 golden too low"
        )
        assert rel_max < p.get("max_rel_err", 0.02), (
            case["id"], f"normalized max err {rel_max:.5f} too high"
        )
        print("correctness PASS", case["id"],
              f"[{case['phase']}] cos={cos:.7f} rel_max_err={rel_max:.5f} "
              f"splits={int(inp['num_kv_splits'][0].item())}")
        del inp, out, ref, got, gold
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inp = _prepare(case)
        _run(inp)                       # settle the Triton JIT
        torch.cuda.synchronize()
        bench = case.get("benchmark", {})
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda i=inp: _run(i),
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 20),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
        )
        metadata = {
            **case["params"],
            "phase": case.get("phase"),
            "model": case.get("model"),
            "kernel_ids": case.get("kernel_ids"),
            "exact_shape_source": case.get("exact_shape_source"),
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            **{k: v for k, v in meta.items() if k.startswith("benchmark_")},
            "metadata": metadata,
        })
        print(case["id"], f"{exec_ms:.6f} ms", meta.get("benchmark_method"),
              meta.get("benchmark_fallback_reason", ""))
        del inp
        torch.cuda.empty_cache()
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    {"compile": run_compile, "correctness": run_correctness,
     "performance": run_performance}[mode]()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Image-kernel harness for Kimi-K3 attention residual (`_score_kernel` + `_combine_kernel`).

What this operator is
---------------------
Kimi-K3 does not carry a single residual stream. Every ``attn_res_block_size`` (=12)
layers it snapshots the running pre-attention prefix into a bank ``[T, NB, H]``; at
each aggregation point the model scores *all* banked snapshots plus the current
prefix, softmaxes the scores and emits the weighted mixture. That lets each layer
choose which historical residual state to read.

Two Triton kernels implement it (``sglang/srt/layers/attn_residual.py``):

  * ``_score_kernel``   grid ``(T, nvb+1)``, block ``(512,1,1)`` -- one CTA per
    (token, row); scans H and emits one scalar
    ``score = (v . cw) / sqrt(mean(v^2) + eps)`` where ``cw = norm.weight *
    proj.weight`` is precomputed by ``get_cw()``.
  * ``_combine_kernel`` grid ``(T, H/1024)``, block ``(256,1,1)`` -- softmax over
    the (<=16) scores, then the weighted sum of the same rows, one H-chunk per CTA.

Both fire **186 times per forward pass**, in prefill and in decode alike, and
together they are 12.24% of end-to-end GPU time in session 20260814T191522Z --
the largest attackable kernel in the whole session.

Why there is headroom
---------------------
``_use_fast()`` (attn_residual.py:32) gates the warp-specialized TMA kernel behind
``torch.cuda.get_device_capability().major >= 10`` -- that is NVIDIA SM100+. On
gfx950 the model is permanently on this 2-kernel fallback, which round-trips
``scores[T,16]`` fp32 through HBM and reads the whole bank twice (once to score,
once to combine). Measured: 1.74-1.82 TB/s at every prefill grid, i.e. ~22-23% of
the MI355X HBM3E peak.

Entry point under test
----------------------
``aggregate_stream(prefix_sum, bank, nvb, score_proj, score_norm)`` -- the public
pre-norm aggregation API (attn_residual.py:285). It is used rather than
``_mix_fused`` directly because it is the stable public signature, and rather than
``AttnResidual.forward`` because the latter also needs an ``out_norm`` RMSNorm
module and a distributed-initialized ``ReplicatedLinear``.

``score_proj`` / ``score_norm`` are duck-typed stand-ins: ``get_cw()`` only reads
``proj.weight`` (``[1, H]``), ``norm.weight`` (``[H]``) and caches on
``proj._attn_res_cw_cache``; ``_mix_fused`` additionally reads
``norm.variance_epsilon``. Constructing the real ``ReplicatedLinear`` would require
``torch.distributed`` to be initialized, which the harness deliberately avoids.

Golden
------
An independent, fully vectorized float64 transcription of the two kernels, chunked
over the token axis so peak memory stays bounded (no per-token Python loop). It does
NOT call ``aggregate_stream_torch`` from the model source: that function lives in
the agent's edit surface, so reusing it would let a broken edit validate itself.
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

# Token-chunk for the float64 golden. 1024 tokens x 9 rows x 7168 x 8 B = 528 MB
# per temporary; four live temporaries stay comfortably inside a 288 GB card while
# keeping the reference fully vectorized.
_GOLDEN_CHUNK = 1024


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # Prefer the workspace-seeded editable copy so the agent's edits take effect;
    # fall back to the in-image install for standalone/dev runs.
    seeded = WORKSPACE / "sglang"
    if (seeded / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    else:
        sys.path.insert(0, os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python"))
    # Triton keys its JIT cache on kernel source, so an edit already forces a
    # recompile; pinning the cache inside the workspace additionally guarantees a
    # run can never serve a binary built from another workspace's source.
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


def _attn_res_ops():
    """Public aggregation entry from the (possibly agent-edited) sglang copy."""
    from sglang.srt.layers.attn_residual import aggregate_stream

    return aggregate_stream


class _WeightHolder:
    """Duck-typed stand-in for ReplicatedLinear / RMSNorm.

    get_cw() reads ``.weight`` and stores its cache on the proj object;
    _mix_fused() reads ``.variance_epsilon`` off the norm object. Nothing else in
    the aggregation path touches these modules, so constructing the real sglang
    layers (which would need torch.distributed) is unnecessary.
    """

    def __init__(self, weight, variance_epsilon: float | None = None):
        self.weight = weight
        if variance_epsilon is not None:
            self.variance_epsilon = variance_epsilon


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    """Build one case at its scored shape.

    There is deliberately no correctness/performance switch: the shape that is
    timed is the shape that is validated, so the scored code path can never differ
    from the checked one.
    """
    torch = _torch()
    p = case["params"]
    T = int(p["num_tokens"])
    H = int(p["hidden_size"])
    nvb = int(p["num_valid_blocks"])
    nb = int(p["bank_blocks"])
    dtype = getattr(torch, p.get("dtype", "bfloat16"))
    assert nb >= nvb, (case["id"], "bank_blocks must cover num_valid_blocks")

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 8141)))

    def rnd(*shape, dt=dtype, scale=1.0):
        return (torch.randn(*shape, device="cuda", dtype=torch.float32, generator=gen)
                * scale).to(dt)

    # The residual stream and the frozen snapshot bank, exactly as AttnResidual
    # holds them: bank is [T, NB, H] contiguous, prefix is [T, H].
    prefix_sum = rnd(T, H, scale=0.5).contiguous()
    bank = rnd(T, nb, H, scale=0.5).contiguous()

    # score_proj: ReplicatedLinear(H -> 1), weight [1, H].  score_norm: RMSNorm(H).
    proj_w = rnd(1, H, scale=0.05).contiguous()
    norm_w = (torch.ones(H, device="cuda", dtype=torch.float32)
              + 0.1 * torch.randn(H, device="cuda", dtype=torch.float32, generator=gen)).to(dtype)

    return {
        "cfg": case, "T": T, "H": H, "nvb": nvb, "nb": nb, "dtype": dtype,
        "eps": float(p.get("eps", MODEL_CONFIG["rms_norm_eps"])),
        "prefix_sum": prefix_sum, "bank": bank,
        "score_proj": _WeightHolder(proj_w),
        "score_norm": _WeightHolder(norm_w, float(p.get("eps", MODEL_CONFIG["rms_norm_eps"]))),
    }


def _run(inp: dict):
    aggregate_stream = _attn_res_ops()
    return aggregate_stream(
        inp["prefix_sum"], inp["bank"], inp["nvb"], inp["score_proj"], inp["score_norm"]
    )


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _golden(inp: dict):
    """Vectorized float64 transcription of _score_kernel + _combine_kernel.

    Per token t, with rows = [bank[t,0..nvb-1,:], prefix[t,:]] (nvb+1 rows of H):

        score_j = (rows_j . cw) * rsqrt(mean(rows_j^2) + eps)   # _score_kernel
        p       = softmax(score)                                # _combine_kernel
        out_t   = sum_j p_j * rows_j                            # _combine_kernel

    with ``cw = score_norm.weight * score_proj.weight.squeeze()`` in fp32 (this is
    what get_cw() precomputes, and it is algebraically the RMSNorm-then-project the
    eager path performs).

    Chunked over the token axis only -- every arithmetic step below is a whole-tensor
    op, there is no per-token loop.
    """
    torch = _torch()
    T, H, nvb, eps = inp["T"], inp["H"], inp["nvb"], inp["eps"]
    cw = (inp["score_norm"].weight.double() * inp["score_proj"].weight.squeeze(0).double())
    out = torch.empty(T, H, device="cuda", dtype=torch.float64)

    for s in range(0, T, _GOLDEN_CHUNK):
        e = min(s + _GOLDEN_CHUNK, T)
        rows = torch.cat(
            [inp["bank"][s:e, :nvb, :].double(),
             inp["prefix_sum"][s:e, :].double().unsqueeze(1)],
            dim=1,
        )                                              # [c, nvb+1, H]
        dotv = rows @ cw                               # [c, nvb+1] -- matmul, no [c,R,H] temp
        sumsq = rows.pow(2).sum(-1)                    # [c, nvb+1]
        score = dotv * torch.rsqrt(sumsq / H + eps)    # [c, nvb+1]
        prob = torch.softmax(score, dim=-1)            # [c, nvb+1]
        out[s:e] = (prob.unsqueeze(-1) * rows).sum(dim=1)
        del rows, dotv, sumsq, score, prob
    return out


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0])
    out = _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} dtype={out.dtype}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case)
        out = _run(inp)
        torch.cuda.synchronize()
        ref = _golden(inp)

        assert torch.isfinite(out).all(), (case["id"], "non-finite output")
        assert tuple(out.shape) == (inp["T"], inp["H"]), (
            case["id"], tuple(out.shape), (inp["T"], inp["H"])
        )
        assert out.dtype == inp["dtype"], (case["id"], out.dtype, inp["dtype"])

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
              f"|o|={got.norm().item():.3f}")
        del inp, out, ref, got, gold
        torch.cuda.empty_cache()


def run_performance() -> None:
    torch = _torch()
    rows = []
    for case in CASES:
        inp = _prepare(case)
        _run(inp)                       # settle the Triton JIT and warm the cw cache
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
            # Flat, not nested: src/testcases.py reads benchmark_method from the
            # top level of each row when building TestCaseResult.metadata.
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

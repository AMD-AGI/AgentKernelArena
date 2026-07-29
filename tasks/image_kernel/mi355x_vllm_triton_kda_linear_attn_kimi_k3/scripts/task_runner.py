#!/usr/bin/env python3
"""Image-kernel harness for Kimi-K3 KDA (Kimi Delta Attention) linear attention.

KDA is the flash-linear-attention (FLA) Triton-JIT gated-delta-rule path used by
Kimi-K3's 69 linear-attention layers. Hot kernels:
  - k007: fused_recurrent_kda_packed_decode_kernel  (decode, recurrent)
  - prefill chunk kernels: chunk_kda_fwd / chunk_gated_delta_rule_fwd_h /
    kda_gate_chunk_cumsum / causal_conv1d / l2norm / layer_norm_gated / solve_tril

Real op entry (vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py):
  prefill : chunk_kda_with_fused_gate(q,k,v,raw_g,beta,A_log,g_bias=dt_bias,...)
  decode  : g = fused_kda_gate(raw_g,A_log,dt_bias); fused_recurrent_kda(q,k,v,g,beta,...)

Dims aligned to the Kimi-K3 session (per-rank, TP=8): num_heads=96/8=12, head_dim=128
(d_k=d_v=128), chunk_size=64. Long-sequence cases are added on top of the session's
real ISL (1024).
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


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers >>>
def _measure_cuda_event_fallback(fn, repetition):
    import time

    import torch

    times_ms = []
    for _ in range(max(1, int(repetition))):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=5,
    repetition=30,
    target_ms=1.0,
    max_graph_repeats=200,
    use_cuda_graph=True,
    **_,
):
    import torch

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }
    if not torch.cuda.is_available() or not use_cuda_graph:
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update(
            benchmark_method=(
                "cpu_timer_fallback"
                if not torch.cuda.is_available()
                else "cuda_event_fallback"
            ),
            benchmark_effective_repeats=int(repetition),
        )
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()
            estimate_ms = start_event.elapsed_time(end_event) / 3
            repeats = min(
                max_graph_repeats,
                max(1, int(target_ms / max(estimate_ms, 1e-9))),
            )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
            torch.cuda.synchronize()

            times = []
            for _ in range(max(1, int(repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event) / repeats)

        mean_ms = sum(times) / len(times)
        if mean_ms < 1e-6:
            raise RuntimeError("empty_cuda_graph_capture")
        metadata.update(
            benchmark_method="cuda_graph",
            benchmark_effective_repeats=int(repeats),
        )
        return mean_ms, metadata
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update(
            benchmark_method="cuda_event_fallback",
            benchmark_effective_repeats=int(repetition),
            benchmark_fallback_reason=(
                f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}"
            ),
        )
        return sum(times) / len(times), metadata


# <<< AKA-GENERATED <<<


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    return torch


def _kda_ops():
    from vllm.model_executor.layers.fla.ops.kda import (
        chunk_kda_with_fused_gate,
        fused_kda_gate,
        fused_recurrent_kda,
    )

    return chunk_kda_with_fused_gate, fused_kda_gate, fused_recurrent_kda


def _prepare(case: dict, correctness: bool = False) -> dict:
    torch = _torch()
    p = dict(case["params"])
    H = p["num_heads"]          # per-rank heads (K3 TP=8 -> 96/8 = 12)
    D = p["head_dim"]           # d_k = d_v = 128
    mode = p["mode"]            # "chunk" (prefill/long) | "recurrent" (decode)
    num_seqs = p["num_seqs"]
    seq_len = p["seq_len"]
    if correctness:
        # keep correctness cheap but exercise multi-chunk (chunk_size=64)
        seq_len = min(seq_len, 256)
        num_seqs = min(num_seqs, 4)

    torch.manual_seed(23)
    gen = torch.Generator(device="cuda").manual_seed(23)
    total_t = num_seqs * seq_len
    dt = torch.bfloat16

    q = torch.randn(1, total_t, H, D, device="cuda", dtype=dt, generator=gen) * 0.5
    k = torch.randn(1, total_t, H, D, device="cuda", dtype=dt, generator=gen) * 0.5
    v = torch.randn(1, total_t, H, D, device="cuda", dtype=dt, generator=gen) * 0.5
    raw_g = torch.randn(1, total_t, H, D, device="cuda", dtype=torch.float32, generator=gen)
    beta = torch.sigmoid(
        torch.randn(1, total_t, H, device="cuda", dtype=torch.float32, generator=gen)
    )
    # A_log: per-head decay parameter [1,1,H,1]; dt_bias (g_bias): [H*D]
    A_log = torch.rand(1, 1, H, 1, device="cuda", dtype=torch.float32, generator=gen) * 2.0 - 4.0
    dt_bias = torch.rand(H * D, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    h0 = torch.zeros(num_seqs, H, D, D, device="cuda", dtype=torch.float32)
    cu = torch.arange(0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32)
    ssm_idx = torch.arange(num_seqs, device="cuda", dtype=torch.long)
    return {
        "cfg": case, "mode": mode, "H": H, "D": D,
        "num_seqs": num_seqs, "seq_len": seq_len, "total_t": total_t,
        "q": q, "k": k, "v": v, "raw_g": raw_g, "beta": beta,
        "A_log": A_log, "dt_bias": dt_bias, "h0": h0, "cu": cu, "ssm_idx": ssm_idx,
        "scale": D ** -0.5,
    }


def _run(inp: dict):
    torch = _torch()
    chunk_kda_with_fused_gate, fused_kda_gate, fused_recurrent_kda = _kda_ops()
    if inp["mode"] == "chunk":
        o, s = chunk_kda_with_fused_gate(
            q=inp["q"], k=inp["k"], v=inp["v"], raw_g=inp["raw_g"], beta=inp["beta"],
            A_log=inp["A_log"], g_bias=inp["dt_bias"], initial_state=inp["h0"],
            output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu"],
            scale=inp["scale"],
        )
        return o, s
    else:  # recurrent / packed decode (k007): num_seqs seqs x seq_len tokens
        g = fused_kda_gate(
            inp["raw_g"].reshape(inp["total_t"], inp["H"] * inp["D"]),
            inp["A_log"], inp["D"], g_bias=inp["dt_bias"],
        ).unsqueeze(0)
        # NOTE: ssm_state_indices must be None here. With a non-null index tensor the
        # kernel takes the continuous-batching path and skips any seq whose state
        # index is <= 0 (NULL_BLOCK_ID), which silently produces ~0 output. Passing
        # None + inplace_final_state=False uses the per-sequence (bos-offset) state.
        o, s = fused_recurrent_kda(
            q=inp["q"], k=inp["k"], v=inp["v"], g=g, beta=inp["beta"].to(torch.bfloat16),
            initial_state=inp["h0"], inplace_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu"], ssm_state_indices=None,
        )
        return o, s


def _golden(inp: dict):
    """Independent float64 reference for the KDA gated-delta-rule recurrence.

    Transcribed directly from fused_recurrent_gated_delta_rule_fwd_kernel
    (IS_KDA=True) and fused_kda_gate / kda_gate_fwd_kernel:

      gate g_t = -exp(A_log_h) * softplus(raw_g + dt_bias)          # per (h, k-channel)
      per token (state S is [H, V, K] = [H, d_v, d_k], reset per sequence):
        q = l2norm(q_t) * scale ; k = l2norm(k_t) ; v = v_t
        S  = S * exp(g_t)[.,None,:]      # decay per k-column
        v  = v - (S @ k)                 # delta-rule "remove old value"
        v  = v * beta_t                  # beta scalar per head
        S  = S + outer(v, k)
        o_t = S @ q
    Runs one segment per cu_seqlens interval so it covers both chunk
    (1 seq x T) and packed-decode (N seqs x 1) layouts.
    """
    torch = _torch()
    H, D = inp["H"], inp["D"]
    scale = inp["scale"]
    cu = inp["cu"].tolist()
    q = inp["q"][0].double()
    k = inp["k"][0].double()
    v = inp["v"][0].double()
    rg = inp["raw_g"][0].double()
    beta = inp["beta"][0].double()
    A = inp["A_log"].reshape(H).double()
    dtb = inp["dt_bias"].reshape(H, D).double()
    g = (-torch.exp(A)).view(1, H, 1) * torch.nn.functional.softplus(rg + dtb.view(1, H, D))
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    o = torch.zeros(inp["total_t"], H, D, dtype=torch.float64, device="cuda")
    h0 = inp["h0"].double()  # [num_seqs, H, V, K]
    for n in range(len(cu) - 1):
        bos, eos = cu[n], cu[n + 1]
        S = h0[n].clone()
        for t in range(bos, eos):
            S = S * torch.exp(g[t]).unsqueeze(1)          # [H, V, K] decay k-columns
            vt = v[t] - (S * kn[t].unsqueeze(1)).sum(-1)  # v - S@k -> [H, V]
            vt = vt * beta[t].unsqueeze(-1)               # * beta (per head)
            S = S + vt.unsqueeze(2) * kn[t].unsqueeze(1)  # + outer(v, k)
            o[t] = (S * qn[t].unsqueeze(1)).sum(-1)       # S @ q -> [H, V]
    return o.unsqueeze(0)


def run_compile() -> None:
    inp = _prepare(CASES[0], correctness=True)
    o, _ = _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(o.shape)}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case, correctness=True)
        # Compute the golden BEFORE _run: the chunk kernel writes the final state
        # into initial_state (inp["h0"]) in place, which would corrupt the golden's
        # starting state if computed afterward.
        ref = _golden(inp)  # independent float64 recurrence
        o, _s = _run(inp)
        torch.cuda.synchronize()
        # (1) finite + exact shape
        assert torch.isfinite(o).all(), (case["id"], "non-finite output")
        exp_o = (1, inp["total_t"], inp["H"], inp["D"])
        assert tuple(o.shape) == exp_o, (case["id"], tuple(o.shape), exp_o)
        # (2) numerical accuracy vs golden: cosine + normalized max error.
        got = o.double().flatten()
        gold = ref.flatten()
        cos = torch.nn.functional.cosine_similarity(got, gold, dim=0).item()
        denom = gold.abs().max().clamp_min(1e-8)
        rel_max = ((got - gold).abs().max() / denom).item()
        assert cos > 0.999, (case["id"], f"cosine {cos:.6f} vs golden too low")
        assert rel_max < 0.03, (case["id"], f"normalized max err {rel_max:.4f} too high")
        print(
            "correctness PASS", case["id"],
            f"cos={cos:.6f} rel_max_err={rel_max:.4f} |o|={got.norm().item():.3f}",
        )


def run_performance() -> None:
    rows = []
    for case in CASES:
        inp = _prepare(case, correctness=False)
        _run(inp)
        _torch().cuda.synchronize()
        # long sequences take longer per call -> allow more graph headroom
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inp), warmup=3, repetition=20, target_ms=2.0, max_graph_repeats=50,
        )
        metadata = {
            **case["params"],
            "model": case["model"],
            "kernel_ids": case["kernel_ids"],
            "gpu_pct": case.get("gpu_pct"),
            "benchmark_method": meta.get("benchmark_method"),
        }
        metadata.update({k: v for k, v in meta.items() if k.startswith("benchmark_")})
        rows.append({
            "test_case_id": case["id"],
            "shape": case.get("trace_input_shapes"),
            "execution_time_ms": exec_ms,
            "metadata": metadata,
        })
        print(case["id"], f"{exec_ms:.6f} ms", meta.get("benchmark_method"),
              meta.get("benchmark_fallback_reason", ""))
    _write_report(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance", "manifest"])
    mode = parser.parse_args().mode
    if mode == "manifest":
        print(json.dumps(SPEC, indent=2))
        return
    _configure()
    if mode == "compile":
        run_compile()
    elif mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()

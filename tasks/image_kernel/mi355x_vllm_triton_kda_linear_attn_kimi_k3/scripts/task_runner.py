#!/usr/bin/env python3
"""Image-kernel harness for Kimi-K3 KDA (Kimi Delta Attention) linear attention.

KDA is the Triton-JIT gated-delta-rule path used by Kimi-K3's 69 linear-attention
layers. The kernels are vendored per GPU vendor under
``vllm/models/kimi_k3/{amd,nvidia}/ops/third_party/kda`` -- this harness targets the
AMD/ROCm copy, which is the one ``kimi_gdn_linear_attn.py`` selects on ROCm.

Hot kernels covered (Hyperloom session 20260728T091437Z, rank0 of TP=8):
  - k007 ``fused_recurrent_kda_packed_decode_kernel`` -- decode. Launched ONLY by
    ``fused_recurrent_kda_packed_decode`` from the non-spec decode branch
    (kimi_gdn_linear_attn.py:609). NOTE: ``fused_recurrent_kda`` is a different
    entry that launches ``fused_recurrent_kda_fwd_kernel`` on the speculative-decode
    branch; K3 has ``num_nextn_predict_layers=0`` so that kernel never runs.
  - prefill chunk kernels (``chunk_kda_fwd_*``, ``chunk_gated_delta_rule_fwd_h_*``,
    ``kda_gate_chunk_cumsum_*``, ``l2norm_*``, ``solve_tril`` ...) reached through
    ``chunk_kda_with_fused_gate`` (kimi_gdn_linear_attn.py:569).

Contract details that the harness must honour (all read off the kernel sources):
  * ``A_log`` is 1-D of length ``local_num_heads``; ``dt_bias`` is
    ``local_num_heads * head_dim`` (kimi_gdn_linear_attn.py:238,266).
  * ``raw_beta`` is PRE-sigmoid -- both kernels apply ``sigmoid`` internally
    (fused_recurrent.py:525, chunk.py:470).
  * K3 sets ``gate_lower_bound = -5.0``, which selects the *safe gate* branch
    ``gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``. This is a
    different function from the softplus branch used when the bound is unset
    (fused_recurrent.py:513-521, chunk.py:507-515).
  * ``state_indices`` entries must be > 0; ``<= 0`` is the NULL slot and makes the
    kernel emit zeros for that row (fused_recurrent.py:481).
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
KDA_CONFIG = SPEC["kda_config"]
# K3 linear_attn_config.gate_lower_bound; selects the safe-gate branch.
GATE_LOWER_BOUND = KDA_CONFIG["gate_lower_bound"]


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    # The agent edits the workspace-seeded copy of vllm, so it has to shadow the
    # in-image install. Without this every `import vllm` resolves to
    # /usr/local/lib/python3.12/dist-packages/vllm and kernel edits are ignored.
    if (WORKSPACE / "vllm" / "__init__.py").is_file():
        sys.path.insert(0, str(WORKSPACE))
    # Triton keys its JIT cache on the kernel source, so an edit already forces a
    # recompile. Pinning the cache inside the workspace additionally guarantees a
    # run can never serve a binary compiled from a different workspace's source.
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
        raise RuntimeError("ROCm GPU is required")
    return torch


def _kda_ops():
    """AMD/ROCm vendored KDA entry points (the copy kimi_gdn_linear_attn.py picks)."""
    from vllm.models.kimi_k3.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda_packed_decode,
    )

    return chunk_kda_with_fused_gate, fused_recurrent_kda_packed_decode


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    """Build a case at its scored shape.

    There is deliberately no correctness/performance switch here: a shape that is
    timed must also be the shape that is validated, or the scored code path can
    differ from the checked one. The golden is an O(T) float64 recurrence, but it
    costs roughly 65 us per token, so even the 32768-token case stays a couple of
    seconds.
    """
    torch = _torch()
    p = dict(case["params"])
    H = p["num_heads"]          # per-rank heads (K3 num_heads=96, TP=8 -> 12)
    D = p["head_dim"]           # d_k = d_v = 128
    mode = p["mode"]            # "chunk" (prefill) | "packed_decode" (k007)
    num_seqs = p["num_seqs"]
    seq_len = p["seq_len"]
    total_t = num_seqs * seq_len

    gen = torch.Generator(device="cuda").manual_seed(int(case.get("seed", 23)))

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen) * scale

    # Layer parameters, shaped exactly as the vLLM KDA layer holds them.
    A_log = torch.rand(H, device="cuda", dtype=torch.float32, generator=gen) * 2.0 - 4.0
    dt_bias = torch.rand(H * D, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    raw_g = rnd(1, total_t, H, D, dtype=torch.float32)
    raw_beta = rnd(1, total_t, H, dtype=torch.float32)  # pre-sigmoid on purpose

    inp = {
        "cfg": case, "mode": mode, "H": H, "D": D,
        "num_seqs": num_seqs, "seq_len": seq_len, "total_t": total_t,
        "raw_g": raw_g, "raw_beta": raw_beta, "A_log": A_log, "dt_bias": dt_bias,
        "scale": D ** -0.5,
    }

    if mode == "chunk":
        inp["q"] = rnd(1, total_t, H, D, scale=0.5)
        inp["k"] = rnd(1, total_t, H, D, scale=0.5)
        inp["v"] = rnd(1, total_t, H, D, scale=0.5)
        # chunk takes one state per sequence, [N, H, V, K].
        state = rnd(num_seqs, H, D, D, dtype=torch.float32, scale=0.1).contiguous()
        inp["cu"] = torch.arange(
            0, (num_seqs + 1) * seq_len, seq_len, device="cuda", dtype=torch.int32
        )
        inp["seg_state0"] = [state[n].double().clone() for n in range(num_seqs)]
        inp["segments"] = [(n * seq_len, (n + 1) * seq_len) for n in range(num_seqs)]
    else:
        # packed decode consumes the post-conv fused QKV block, [B, 3 * H * D],
        # laid out q | k | v with each part head-major (fused_recurrent.py:491-502).
        inp["mixed_qkv"] = rnd(total_t, 3 * H * D, scale=0.5).contiguous()
        # A state cache with slot 0 reserved as the NULL slot; indices start at 1.
        state = rnd(num_seqs + 1, H, D, D, dtype=torch.float32, scale=0.1).contiguous()
        inp["state_indices"] = torch.arange(
            1, num_seqs + 1, device="cuda", dtype=torch.int32
        )
        inp["seg_state0"] = [state[n + 1].double().clone() for n in range(num_seqs)]
        inp["segments"] = [(n, n + 1) for n in range(num_seqs)]

    # Snapshot state before either kernel executes. Decode updates its cache;
    # chunk returns a separate final state and leaves this input unchanged.
    inp["state"] = state
    return inp


def _run(inp: dict):
    chunk_kda_with_fused_gate, fused_recurrent_kda_packed_decode = _kda_ops()
    if inp["mode"] == "chunk":
        return chunk_kda_with_fused_gate(
            q=inp["q"], k=inp["k"], v=inp["v"],
            raw_g=inp["raw_g"], raw_beta=inp["raw_beta"], A_log=inp["A_log"],
            g_bias=inp["dt_bias"], lower_bound=GATE_LOWER_BOUND,
            initial_state=inp["state"], output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu"],
        )
    return fused_recurrent_kda_packed_decode(
        mixed_qkv=inp["mixed_qkv"], raw_g=inp["raw_g"], raw_beta=inp["raw_beta"],
        A_log=inp["A_log"], dt_bias=inp["dt_bias"], lower_bound=GATE_LOWER_BOUND,
        initial_state=inp["state"], state_indices=inp["state_indices"],
    )


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def _golden(inp: dict, *, return_state=False):
    """Independent float64 transcription of the KDA gated-delta-rule recurrence.

    Taken directly from ``fused_recurrent_kda_packed_decode_kernel``
    (fused_recurrent.py:504-533); ``chunk_kda_with_fused_gate`` computes the same
    recurrence blockwise, so one reference covers both modes:

        g_t   = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))   # safe gate
        q     = l2norm(q_t) * scale ;  k = l2norm(k_t) ;  v = v_t
        S     = S * exp(g_t)              # decay per k-column
        v     = v - S @ k                 # delta-rule "remove old value"
        v     = v * sigmoid(raw_beta_t)
        S     = S + outer(v, k)
        o_t   = S @ q
    """
    torch = _torch()
    H, D, scale = inp["H"], inp["D"], inp["scale"]

    if inp["mode"] == "chunk":
        q = inp["q"][0].double()
        k = inp["k"][0].double()
        v = inp["v"][0].double()
    else:
        m = inp["mixed_qkv"].double()
        q = m[:, : H * D].reshape(-1, H, D)
        k = m[:, H * D: 2 * H * D].reshape(-1, H, D)
        v = m[:, 2 * H * D:].reshape(-1, H, D)

    raw_g = inp["raw_g"][0].double()
    a = torch.exp(inp["A_log"].double()).view(1, H, 1)
    pre_gate = raw_g + inp["dt_bias"].reshape(H, D).double().view(1, H, D)
    if GATE_LOWER_BOUND is None:
        gate = -a * torch.nn.functional.softplus(pre_gate)
    else:
        gate = GATE_LOWER_BOUND * torch.sigmoid(a * pre_gate)
    beta = torch.sigmoid(inp["raw_beta"][0].double())

    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    out = torch.zeros(inp["total_t"], H, D, dtype=torch.float64, device=q.device)
    final_states = []
    for (bos, eos), S0 in zip(inp["segments"], inp["seg_state0"]):
        S = S0.clone()                                     # [H, V, K]
        for t in range(bos, eos):
            S = S * torch.exp(gate[t]).unsqueeze(1)        # decay k-columns
            vt = v[t] - (S * kn[t].unsqueeze(1)).sum(-1)   # v - S @ k  -> [H, V]
            vt = vt * beta[t].unsqueeze(-1)
            S = S + vt.unsqueeze(2) * kn[t].unsqueeze(1)   # + outer(v, k)
            out[t] = (S * qn[t].unsqueeze(1)).sum(-1)      # S @ q
        final_states.append(S)
    if return_state:
        return out.unsqueeze(0), final_states
    return out.unsqueeze(0)


def _assert_numerics(got, expected, params):
    torch = _torch()
    assert got.shape == expected.shape, "Wrong KDA comparison shape"
    actual, reference = got.double().flatten(), expected.double().flatten()
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), "Nonfinite KDA comparison"
    actual_scale, reference_scale = actual.abs().max(), reference.abs().max()
    if actual_scale == 0 or reference_scale == 0:
        # An evolving chunk replay can legitimately end with exactly zero
        # state in both the implementation and the independent recurrence.
        # Cosine has no direction there: accept only two exact zero vectors.
        assert actual_scale == reference_scale, "KDA cosine mismatch: only one vector is zero"
        cosine = 1.0
    else:
        # Scaling each vector does not change its angle. It avoids cosine's
        # default epsilon turning equal tiny nonzero vectors into a mismatch.
        cosine = torch.nn.functional.cosine_similarity(
            actual / actual_scale, reference / reference_scale, dim=0,
        )
    rel_max = (actual - reference).abs().max() / reference.abs().max().clamp_min(1e-8)
    assert float(cosine) > params.get("min_cosine", 0.999), "KDA cosine mismatch"
    assert float(rel_max) < params.get("max_rel_err", 0.03), "KDA relative maximum mismatch"


def _assert_state_result(inp, result, expected_states, before):
    torch = _torch()
    assert isinstance(result, tuple) and len(result) == 2, "KDA must return output and state"
    output, state = result
    assert output.shape == (1, inp["total_t"], inp["H"], inp["D"]), "Wrong KDA output shape"
    assert output.dtype == torch.bfloat16, "KDA output must be BF16"
    assert output.device == inp["state"].device, "Wrong KDA output device"
    assert torch.isfinite(output).all(), "Nonfinite KDA output"
    assert state.dtype == torch.float32 and state.device == inp["state"].device
    assert torch.isfinite(state).all(), "Nonfinite KDA state"
    if inp["mode"] == "packed_decode":
        assert state.shape == before.shape and state.data_ptr() == inp["state"].data_ptr()
        selected = state[inp["state_indices"].long()]
        mask = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
        mask[inp["state_indices"].long()] = False
        torch.testing.assert_close(state[mask], before[mask], rtol=0, atol=0)
    else:
        assert state.shape == inp["state"].shape
        assert output.data_ptr() == inp["v"].data_ptr(), "Chunk output must alias v"
        assert output.stride() == inp["v"].stride(), "Wrong chunk output layout"
        # Chunk returns the final state. Its caller performs cache scattering;
        # that external model operation is outside this operator's timing scope.
        torch.testing.assert_close(inp["state"], before, rtol=0, atol=0)
        selected = state
    _assert_numerics(selected, torch.stack(expected_states), inp["cfg"]["params"])


def _reference_state_inputs(inputs, state):
    result = dict(inputs)
    slots = (inputs["state_indices"].tolist() if inputs["mode"] == "packed_decode"
             else range(inputs["num_seqs"]))
    result["state"] = state
    result["seg_state0"] = [state[i].double().clone() for i in slots]
    return result


def _repeated_reference(inputs, state, repeats):
    reference = _reference_state_inputs(inputs, state)
    assert type(repeats) is int and repeats > 0
    for _ in range(repeats):
        expected, states = _golden(reference, return_state=True)
        if inputs["mode"] == "packed_decode":
            reference["seg_state0"] = states
        else:
            # Public chunk.py writes output into v (o=v). Each subsequent
            # invocation consumes that BF16-rounded value; initial state stays
            # unchanged. Reproduce the existing evolving-input graph exactly.
            reference["v"] = expected.to(inputs["v"].dtype)
    return expected, states


def _prepare_timed_check(inp):
    torch = _torch()
    originals = {key: value for key, value in inp.items() if isinstance(value, torch.Tensor)}
    snapshots = {key: value.detach().clone() for key, value in originals.items()}
    private = {**inp, **snapshots}
    return dict(originals=originals, snapshots=snapshots, private=private, sample=None)


def _observe_timed_sample(inp, check, calls_per_replay):
    # Called by the canonical helper on the measurement stream, before its
    # start event. Retain only the last measured sample's private starting
    # mutable tensor.
    # Unlike prepare_fn, this observes state without resetting it or forcing R=1.
    assert type(calls_per_replay) is int and calls_per_replay > 0
    mutable = "state" if inp["mode"] == "packed_decode" else "v"
    check["sample"] = (inp[mutable].detach().clone(), calls_per_replay)


def _assert_readonly_inputs(inp, check, expected):
    torch = _torch()
    for key, original in check["originals"].items():
        assert inp[key] is original, "KDA input replaced: " + key
        assert original.dtype == expected[key].dtype and original.shape == expected[key].shape
        assert original.device == expected[key].device
        mutable = "state" if inp["mode"] == "packed_decode" else "v"
        if key == mutable:
            continue  # The output/state contract verifies this actual mutation.
        assert torch.equal(original.contiguous().view(torch.uint8),
                           expected[key].contiguous().view(torch.uint8)), "Readonly KDA input changed: " + key


def _assert_timed_outputs(inp, timed, metadata, check):
    try:
        assert timed.bound, "Timing must expose its captured invocation"
        _assert_readonly_inputs(inp, check, check["snapshots"])
        repeats = int(metadata["benchmark_effective_repeats"])
        assert repeats > 0
        assert check["sample"] is not None, "Canonical timing helper did not observe measured KDA state"
        sample, observed_repeats = check["sample"]
        assert repeats == observed_repeats, "Observed KDA replay count differs from measured count"
        private = dict(check["private"])
        before = sample if inp["mode"] == "packed_decode" else check["snapshots"]["state"]
        if inp["mode"] == "chunk":
            private["v"] = sample
        expected, states = _repeated_reference(private, before, repeats)
        _assert_state_result(inp, timed.outputs, states, before)
        _assert_numerics(timed.outputs[0], expected, inp["cfg"]["params"])

        # A second independent input tests the exact captured unit. Derive it
        # from pre-candidate snapshots, never from candidate-modified values.
        changed = dict(check["snapshots"])
        key = "mixed_qkv" if inp["mode"] == "packed_decode" else "q"
        changed_names = (key, "raw_g", "state") + (("v",) if inp["mode"] == "chunk" else ())
        for name in changed_names:
            changed[name] = -check["snapshots"][name]
            inp[name].copy_(changed[name])
        private = {**check["private"], **changed}
        expected, states = _repeated_reference(private, changed["state"], repeats)
        if inp["mode"] == "chunk":
            # Output aliases v, so poisoning it would destroy the new input.
            # Check the complete aliased output against the independent oracle.
            timed.outputs[1].fill_(float("nan"))
        else:
            timed.outputs[0].fill_(float("nan"))
        observed = timed.rerun()
        _assert_readonly_inputs(inp, check, changed)
        _assert_state_result(inp, observed, states, changed["state"])
        _assert_numerics(observed[0], expected, inp["cfg"]["params"])
    finally:
        for key, original in check["originals"].items():
            original.copy_(check["snapshots"][key])
            inp[key] = original


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inp = _prepare(CASES[0])
    out, _state = _run(inp)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        inp = _prepare(case)
        before = inp["state"].clone()
        ref, final_states = _golden(inp, return_state=True)
        out, _state = _run(inp)
        torch.cuda.synchronize()
        _assert_state_result(inp, (out, _state), final_states, before)

        assert torch.isfinite(out).all(), (case["id"], "non-finite output")
        expected_shape = (1, inp["total_t"], inp["H"], inp["D"])
        assert tuple(out.shape) == expected_shape, (
            case["id"], tuple(out.shape), expected_shape
        )

        got = out.double().flatten()
        gold = ref.flatten()
        cos = torch.nn.functional.cosine_similarity(got, gold, dim=0).item()
        denom = gold.abs().max().clamp_min(1e-8)
        rel_max = ((got - gold).abs().max() / denom).item()
        tol = case["params"]
        assert cos > tol.get("min_cosine", 0.999), (
            case["id"], f"cosine {cos:.6f} vs float64 golden too low"
        )
        assert rel_max < tol.get("max_rel_err", 0.03), (
            case["id"], f"normalized max err {rel_max:.4f} too high"
        )
        print(
            "correctness PASS", case["id"],
            f"cos={cos:.6f} rel_max_err={rel_max:.4f} |o|={got.norm().item():.3f}",
        )


def run_performance() -> None:
    # Validate the exact captured invocation after collecting unchanged timings.
    rows = []
    for case in CASES:
        inp = _prepare(case)
        check = _prepare_timed_check(inp)
        _run(inp)
        _torch().cuda.synchronize()
        bench = case.get("benchmark", {})
        timed = _TimedRun()
        timed.before_sample = lambda repeats: _observe_timed_sample(inp, check, repeats)
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inp),
            warmup=bench.get("warmup", 3),
            repetition=bench.get("repetition", 20),
            target_ms=bench.get("target_ms", 2.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
            timed_run=timed,
        )
        _assert_timed_outputs(inp, timed, meta, check)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "kernel_ids": case.get("kernel_ids"),
            "gpu_pct": case.get("gpu_pct"),
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
    _write_report(rows)
    return rows


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

#!/usr/bin/env python3
"""Image-kernel harness for the Qwen3.8-2.4T-A95B Gated-DeltaNet linear-attention core.

Reproduces the linear-attention decode chain of Hyperloom session 100137
(Qwen3.8-2.4T-A95B-Quark-MXFP4, 20260814T175123Z) on MI355X/gfx950, sglang
0.5.17.dev20260812+gdc5f6c4883 / ROCm 7.2.0.

69 of the model's 92 layers are ``linear_attention`` (Gated DeltaNet); the other
23 are full attention. The timed unit here is everything the model does between
the two input projections and the output projection -- i.e. the whole
``Qwen3_5GatedDeltaNet`` core, which in the session's decode trace is five
kernels:

  fused_recurrent_gated_delta_rule_packed_decode_kernel   69x/step  2.278% E2E
  at::native::elementwise_kernel_manual_unroll (aten::copy_) 208x/step 1.663% E2E
  at::native::CatArrayBatchedCopy (aten::cat)              69x/step  0.692% E2E
  _causal_conv1d_update_kernel                             69x/step  0.578% E2E
  _layer_norm_fwd_1pass_kernel (RMSNormGated)              69x/step  0.552% E2E
                                                          -------------------
                                                           chain     5.76% E2E

The chain is timed as one unit on purpose. 2.35 of those 5.76 points are pure
data movement (``aten::cat`` + ``aten::copy_``) produced by the split/reshape/
concat around the kernels, so scoring the recurrent kernel alone would hide the
single largest lever this operator has. The whole chain is inside the edit
surface, so fusing the movement away is a legal -- and intended -- optimisation.

Call chain reproduced verbatim (sglang sources, this image):
    srt/models/qwen3_5.py:524   fix_query_key_value_ordering  (split -> reshape)
    srt/models/qwen3_5.py:659   torch.cat((query, key, value), -1)
      NB: the fused split/cat helper at qwen3_5.py:640 is NOT taken here --
      it requires num_v_heads // num_k_heads in [1, 2, 4] and Qwen3.8 has
      128 // 16 = 8, so the model falls into the plain-torch else-branch. That
      is exactly where the 208 aten::copy_ launches per step come from.
    srt/layers/attention/linear/gdn_backend.py:406  causal_conv1d_update
    srt/layers/attention/linear/kernels/gdn_triton.py:121  packed_decode
    srt/models/qwen3_5.py:684   RMSNormGated(core_attn_out, z)

Per-rank geometry (TP=8), from config.json:
    key_dim   = linear_num_key_heads(16)   * linear_key_head_dim(128)   = 2048 -> 256/rank  (2 k heads)
    value_dim = linear_num_value_heads(128)* linear_value_head_dim(128) =16384 -> 2048/rank (16 v heads)
    conv_dim  = 2*key_dim + value_dim = 20480 -> 2560/rank, conv width 4
    in_proj_qkvz out = 2*key_dim + 2*value_dim = 36864 -> 4608/rank
    in_proj_ba   out = 2*num_v_heads = 256 -> 32/rank
    ssm state = [slots, 16, 128, 128] bf16 (--mamba-ssm-dtype bfloat16)

One harness detail that matters: this operator is STATEFUL. Both
``causal_conv1d_update`` and the recurrent kernel update their cache in place
(gdn_triton.py passes ``ht=initial_state``). A CUDA-graph replay therefore
advances the state ``n_repeat`` times, so the reference is stepped the same
number of times from the same snapshot rather than being compared against a
single step. See ``_assert_timed_outputs``.
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
GDN = SPEC["gdn_config"]

_IMAGE_SGLANG_PY = Path(
    os.environ.get("QWEN38_SGLANG_PY", "/sgl-workspace/sglang/python")
)


def _configure() -> None:
    for key in ("GPU_ARCHS", "PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
        os.environ.setdefault(key, "gfx950")
    os.environ.setdefault("TRITON_CACHE_DIR", str(WORKSPACE / "build" / "triton"))
    # The agent edits the workspace-seeded copy of sglang, so it must shadow the
    # in-image install; otherwise `import sglang` resolves to the image copy and
    # the edits are silently ignored.
    if (WORKSPACE / "sglang").is_dir():
        sys.path.insert(0, str(WORKSPACE))
    os.chdir(WORKSPACE)


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(fn, repetition):
    import time
    import torch

    repetition = max(1, int(repetition))
    if not torch.cuda.is_available():
        times_ms = []
        for _ in range(repetition):
            start = time.perf_counter()
            fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
        return times_ms

    times_ms = []
    for _ in range(repetition):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


class _TimedRun:
    """Handle on the exact invocation a benchmark measured.

    Timing and correctness are otherwise separate invocations, so a kernel can
    tell them apart and do less work in the one that is scored. Passing this
    collector to the benchmark makes the scored invocation itself observable:
    ``outputs`` aliases the buffers the timed unit last wrote, and ``rerun``
    executes that same unit again.

    Under CUDA-graph timing the buffers are captured once and every replay
    writes to those same addresses, so ``outputs`` keeps tracking replays. Under
    event-timing fallback the measured outputs cannot be observed reliably, so a
    benchmark that requests this collector fails closed instead of validating a
    separate post-timing invocation.
    """

    def __init__(self):
        self._rerun = None
        self.outputs = None

    def _bind(self, rerun, outputs=None):
        self._rerun = rerun
        self.outputs = outputs

    @property
    def bound(self):
        return self._rerun is not None

    def rerun(self):
        if self._rerun is None:
            raise RuntimeError(
                "timed run was never bound; the benchmark did not reach a "
                "measurement path"
            )
        self.outputs = self._rerun()
        return self.outputs


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=10,
    repetition=100,
    target_ms=1.0,
    n_retries=5,
    estimate_reps=5,
    max_graph_repeats=1000,
    use_cuda_graph=True,
    fallback_reason=None,
    timed_run=None,
):
    import torch

    def _reject_unobservable_fallback(reason):
        if timed_run is not None:
            raise RuntimeError(
                f"{reason}; timed_run requires an observable CUDA-graph replay "
                "and cannot validate a separate post-timing invocation"
            )

    # A captured graph whose measured per-iteration time falls below this floor is
    # treated as empty (it recorded no device work) and rejected in favour of
    # per-launch event timing, so a graph that silently captured nothing is never
    # reported as a fabricated ~0 ms cuda_graph result. Real kernels measure far
    # above this floor; an empty-graph replay measures ~1e-5 ms.
    empty_graph_floor_ms = 1e-4

    for _ in range(max(0, int(warmup))):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    max_graph_repeats = max(1, int(max_graph_repeats))
    metadata = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }

    if not torch.cuda.is_available():
        _reject_unobservable_fallback("CUDA is unavailable")
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_unavailable",
        })
        return sum(times) / len(times), metadata

    if not use_cuda_graph:
        _reject_unobservable_fallback("CUDA-graph timing is disabled")
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": fallback_reason or "cuda_graph_disabled",
        })
        return sum(times) / len(times), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_reps = max(1, int(estimate_reps))
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(estimate_reps):
                    fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()

            estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
            if estimate_ms == 0:
                n_repeat = max_graph_repeats
            else:
                n_repeat = min(max_graph_repeats, max(1, int(float(target_ms) / estimate_ms)))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_outputs = None
                for _ in range(n_repeat):
                    captured_outputs = fn()
            torch.cuda.synchronize()

            retry_times = []
            for _ in range(max(1, int(repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                retry_times.append(start_event.elapsed_time(end_event) / n_repeat)

        graph_mean = sum(retry_times) / len(retry_times)
        if graph_mean < empty_graph_floor_ms:
            _reject_unobservable_fallback("CUDA graph captured no device work")
            times = _measure_cuda_event_fallback(fn, repetition)
            metadata.update({
                "benchmark_method": "cuda_event_fallback",
                "benchmark_effective_repeats": int(repetition),
                "benchmark_fallback_reason": fallback_reason or "empty_cuda_graph_capture",
            })
            return sum(times) / len(times), metadata

        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        if timed_run is not None:

            def _replay_once():
                # Callers stage work on their own stream before re-running (they
                # perturb inputs and poison outputs). The capture stream must be
                # ordered after that, or the replay races the staged writes and
                # they land on top of the kernel's results.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.replay()
                torch.cuda.synchronize()
                return captured_outputs

            timed_run._bind(_replay_once, captured_outputs)
        return graph_mean, metadata
    except Exception as exc:
        # Isolate the aborted capture before re-measuring so the fallback timing is
        # not polluted by the failed attempt (a mid-capture failure can leave the
        # first few launches abnormally slow).
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if timed_run is not None:
            raise RuntimeError(
                "CUDA-graph capture failed; timed_run cannot validate the "
                "separate CUDA-event fallback invocation"
            ) from exc
        for _ in range(min(3, max(1, int(warmup)))):
            fn()
        torch.cuda.synchronize()
        times = _measure_cuda_event_fallback(fn, repetition)
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
        return sum(times) / len(times), metadata
# <<< AKA-GENERATED <<<


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU (gfx950) is required")
    # Serving runs the whole forward under inference mode. RMSNormGated is a
    # custom autograd Function whose output this harness writes into (the
    # timed-run poison step), which grad mode forbids on a returned view; and
    # graph capture of an autograd-tracked region is not what production does
    # either. Disable globally rather than sprinkling no_grad().
    torch.set_grad_enabled(False)
    return torch


def _free() -> None:
    _torch().cuda.empty_cache()


def _write_report(rows: list) -> None:
    report_dir = WORKSPACE / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "performance_report.json").write_text(json.dumps(rows, indent=2))


def _assert_split_branch(num_v_heads: int, num_k_heads: int) -> None:
    """Fail closed if the model would have taken the fused split/cat helper.

    ``qwen3_5.py:640`` routes to ``fused_qkvzba_split_reshape_cat_contiguous``
    when ``num_v_heads // num_k_heads in (1, 2, 4)``. Qwen3.8 is 128 // 16 = 8,
    so it takes the plain-torch branch and pays the cat + copy_ launches this
    task is meant to expose. If that ratio ever changes the timed unit is no
    longer the session's unit.
    """
    if num_v_heads // num_k_heads in (1, 2, 4):
        raise AssertionError(
            f"num_v_heads // num_k_heads = {num_v_heads // num_k_heads} is in "
            f"(1, 2, 4), so qwen3_5.py:640 would use the fused split/cat helper "
            f"instead of the plain-torch branch this harness reproduces."
        )


# --------------------------------------------------------------------------- #
# Inputs
#
# Shapes come from the session's CUDA-graph capture trace (record_shapes=True)
# at bs=64, cross-checked against config.json:
#     aten::cat            [[64, 256], [64, 256], [64, 2048]] -> [64, 2560]
#     LayerNormFn          [[1024, 128], [128], ...]   (1024 = 64 tok x 16 v heads)
#     aten::copy_          [64, 16] x2/layer  and  [64, 16, 128] x1/layer
#     aten::mm  in_proj_qkvz  [64, 8192] x [8192, 4608]   (feeds this unit)
#     aten::mm  in_proj_ba    [64, 8192] x [8192,   32]   (feeds this unit)
# --------------------------------------------------------------------------- #
def _prepare(case: dict) -> dict:
    torch = _torch()
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    p = dict(case["params"])
    batch = int(p["batch"])
    num_k_heads, num_v_heads = int(p["num_k_heads"]), int(p["num_v_heads"])
    head_k_dim, head_v_dim = int(p["head_k_dim"]), int(p["head_v_dim"])
    conv_width, slots = int(p["conv_width"]), int(p["state_slots"])
    _assert_split_branch(num_v_heads, num_k_heads)

    k_tp = num_k_heads * head_k_dim          # 256
    v_tp = num_v_heads * head_v_dim          # 2048
    conv_dim = 2 * k_tp + v_tp               # 2560
    qkvz_dim = 2 * k_tp + 2 * v_tp           # 4608
    ba_dim = 2 * num_v_heads                 # 32

    torch.manual_seed(int(case.get("seed", 5)))
    dev, dt = "cuda", torch.bfloat16
    # The two input-projection outputs; the projections themselves belong to the
    # skinny-GEMM task, not this one.
    qkvz = torch.randn((batch, qkvz_dim), device=dev, dtype=dt) * 0.5
    ba = torch.randn((batch, ba_dim), device=dev, dtype=dt) * 0.5

    conv_weight = torch.randn((conv_dim, conv_width), device=dev, dtype=dt) * 0.2
    conv_bias = torch.randn((conv_dim,), device=dev, dtype=dt) * 0.1
    # causal_conv1d_update wants state_len >= width - 1.
    conv_state = torch.randn((slots, conv_dim, conv_width - 1), device=dev, dtype=dt) * 0.2
    # mamba_ssm_dtype=bfloat16 in the session's launch recipe.
    ssm_state = torch.randn(
        (slots, num_v_heads, head_v_dim, head_k_dim), device=dev, dtype=dt
    ) * 0.1
    A_log = torch.randn((num_v_heads,), device=dev, dtype=torch.float32) * 0.5
    dt_bias = torch.ones((num_v_heads,), device=dev, dtype=torch.float32)
    # One distinct cache slot per running request, as in serving.
    indices = torch.arange(batch, device=dev, dtype=torch.int32)

    norm = RMSNormGated(
        head_v_dim,
        eps=float(p["rms_norm_eps"]),
        group_size=None,
        norm_before_gate=True,
        device=dev,
        dtype=dt,
        activation=str(p["output_gate_type"]),
    )
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.05)

    return {
        "cfg": case, "params": p, "batch": batch,
        "num_k_heads": num_k_heads, "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim, "head_v_dim": head_v_dim,
        "k_tp": k_tp, "v_tp": v_tp, "conv_dim": conv_dim, "conv_width": conv_width,
        "qkvz": qkvz, "ba": ba,
        "conv_weight": conv_weight, "conv_bias": conv_bias,
        "conv_state": conv_state, "ssm_state": ssm_state,
        "A_log": A_log, "dt_bias": dt_bias, "indices": indices,
        "norm": norm,
        "scale": float(head_k_dim) ** -0.5,
        "activation": str(p["conv_activation"]),
        # Pristine copies so a stateful op can be re-run from a known point.
        "conv_state0": conv_state.clone(), "ssm_state0": ssm_state.clone(),
    }


def _split_qkvzba(inputs: dict):
    """srt/models/qwen3_5.py:524 fix_query_key_value_ordering, verbatim."""
    k_tp, v_tp = inputs["k_tp"], inputs["v_tp"]
    head_v_dim, nv = inputs["head_v_dim"], inputs["num_v_heads"]
    query, key, value, z = inputs["qkvz"].split([k_tp, k_tp, v_tp, v_tp], dim=-1)
    b, a = inputs["ba"].split([nv, nv], dim=-1)
    value = value.reshape(value.size(0), -1, head_v_dim)
    z = z.reshape(z.size(0), -1, head_v_dim)
    return query, key, value, z, b.contiguous(), a.contiguous()


def _run(inputs: dict):
    """One linear-attention core, exactly as the model's decode path runs it."""
    torch = _torch()
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_update

    query, key, value, z, b, a = _split_qkvzba(inputs)
    query, key, value = (x.reshape(x.shape[0], -1) for x in (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        inputs["conv_state"],
        inputs["conv_weight"],
        inputs["conv_bias"],
        inputs["activation"],
        conv_state_indices=inputs["indices"],
    )

    batch, nv, hv = inputs["batch"], inputs["num_v_heads"], inputs["head_v_dim"]
    out = mixed_qkv.new_empty(batch, 1, nv, hv)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        initial_state=inputs["ssm_state"],
        out=out,
        ssm_state_indices=inputs["indices"],
        use_qk_l2norm_in_kernel=True,
    )
    core_attn_out = out.transpose(0, 1).reshape(-1, hv)
    core_attn_out = inputs["norm"](core_attn_out, z.reshape(-1, hv))
    return core_attn_out.reshape(batch, nv * hv)


def _reset_state(inputs: dict) -> None:
    inputs["conv_state"].copy_(inputs["conv_state0"])
    inputs["ssm_state"].copy_(inputs["ssm_state0"])


# --------------------------------------------------------------------------- #
# Reference
#
# Independent torch implementation of the same chain, transcribed from the
# Triton sources rather than calling them:
#   conv:      sglang/kernels/ops/mamba/causal_conv1d_triton.py:581
#   recurrent: sglang/kernels/ops/attention/fla/fused_recurrent.py:184
#   norm:      sglang/kernels/ops/attention/fla/layernorm_gated.py:75
#
# Fully vectorized -- no Python loop over batch or heads. The recurrent step is
# a batched outer-product update on [B, HV, V, K], which for the session's
# bs=64 / 16 heads / 128x128 state is 16.7 M elements, so a step costs a handful
# of elementwise passes plus two [B*HV, V, K] contractions.
# --------------------------------------------------------------------------- #
def _reference_step(inputs: dict, conv_state, ssm_state):
    """One decode step. Mutates conv_state / ssm_state in place, like the kernels."""
    torch = _torch()
    F = torch.nn.functional
    with torch.no_grad():
        batch = inputs["batch"]
        nk, nv = inputs["num_k_heads"], inputs["num_v_heads"]
        hk, hv = inputs["head_k_dim"], inputs["head_v_dim"]
        k_tp, v_tp = inputs["k_tp"], inputs["v_tp"]

        query, key, value, z, b, a = _split_qkvzba(inputs)
        query, key, value = (x.reshape(x.shape[0], -1) for x in (query, key, value))
        x = torch.cat((query, key, value), dim=-1).float()          # [B, conv_dim]

        # --- depthwise causal conv, width 4, with the rolling state ------------
        # causal_conv1d_update: window = [state, x], out = sum(window * w) + bias,
        # then the state shifts left by one and x becomes its last column.
        idx = inputs["indices"].long()
        state = conv_state[idx].float()                             # [B, D, W-1]
        window = torch.cat([state, x.unsqueeze(-1)], dim=-1)        # [B, D, W]
        w = inputs["conv_weight"].float().unsqueeze(0)              # [1, D, W]
        conv_out = (window * w).sum(-1) + inputs["conv_bias"].float()
        if inputs["activation"] in ("silu", "swish"):
            conv_out = F.silu(conv_out)
        conv_state[idx] = window[..., 1:].to(conv_state.dtype)
        mixed = conv_out.to(torch.bfloat16).float()                 # kernel writes bf16

        # --- gated delta rule recurrent step -----------------------------------
        q = mixed[:, : nk * hk].reshape(batch, nk, hk)
        k = mixed[:, k_tp : k_tp + nk * hk].reshape(batch, nk, hk)
        v = mixed[:, 2 * k_tp : 2 * k_tp + v_tp].reshape(batch, nv, hv)
        # GQA-style head sharing: v head i reads k/q head i // (HV // H).
        rep = nv // nk
        q = q.repeat_interleave(rep, dim=1)                         # [B, HV, K]
        k = k.repeat_interleave(rep, dim=1)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * inputs["scale"]

        xa = a.float() + inputs["dt_bias"].float()                  # [B, HV]
        softplus = torch.where(xa <= 20.0, torch.log1p(torch.exp(xa)), xa)
        g = -torch.exp(inputs["A_log"].float()) * softplus          # [B, HV]
        beta = torch.sigmoid(b.float().to(torch.bfloat16).float())  # kernel casts to b.dtype

        h = ssm_state[idx].float()                                  # [B, HV, V, K]
        h = h * torch.exp(g)[:, :, None, None]
        # v <- (v - h @ k) * beta ; h <- h + v k^T ; o <- h @ q
        v = (v - torch.einsum("bhvk,bhk->bhv", h, k)) * beta.unsqueeze(-1)
        h = h + v.unsqueeze(-1) * k[:, :, None, :]
        o = torch.einsum("bhvk,bhk->bhv", h, q)                     # [B, HV, V]
        ssm_state[idx] = h.to(ssm_state.dtype)

        # --- gated RMSNorm (norm_before_gate=True, swish gate) -----------------
        core = o.reshape(-1, hv).to(torch.bfloat16).float()
        weight = inputs["norm"].weight.float()
        eps = float(inputs["params"]["rms_norm_eps"])
        core = core * torch.rsqrt((core * core).mean(-1, keepdim=True) + eps) * weight
        gate = z.reshape(-1, hv).float()
        core = core * (gate * torch.sigmoid(gate))
        return core.reshape(batch, nv * hv).to(torch.bfloat16)


def _reference(inputs: dict, steps: int = 1):
    """Run the reference chain ``steps`` times from the pristine cache snapshot."""
    torch = _torch()
    conv_state = inputs["conv_state0"].clone()
    ssm_state = inputs["ssm_state0"].clone()
    out = None
    for _ in range(max(1, int(steps))):
        out = _reference_step(inputs, conv_state, ssm_state)
    del conv_state, ssm_state
    return out


def _deviation(got, expected):
    torch = _torch()
    g, e = got.float().flatten(), expected.float().flatten()
    cos = torch.nn.functional.cosine_similarity(g, e, dim=0).item()
    err = ((g - e).norm() / e.norm().clamp_min(1e-8)).item()
    return cos, err


def _assert_within_tolerance(case: dict, cos: float, err: float, label: str) -> None:
    tol = case["params"]
    assert cos > tol.get("min_cosine", 0.995), (
        case["id"], f"{label} cosine {cos:.6f} vs torch reference too low"
    )
    assert err < tol.get("max_rel_norm_err", 0.05), (
        case["id"], f"{label} relative norm error {err:.5f} too high"
    )


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_compile() -> None:
    inputs = _prepare(CASES[0])
    out = _run(inputs)
    _torch().cuda.synchronize()
    print(f"{OPERATOR} compile smoke: PASS  out={tuple(out.shape)} {out.dtype}")


def run_correctness() -> None:
    torch = _torch()
    for case in CASES:
        # Same shapes the performance mode scores; nothing is shrunk for
        # correctness, so every measured configuration is a checked one.
        inputs = _prepare(case)
        _reset_state(inputs)
        got = _run(inputs)
        torch.cuda.synchronize()
        expected = _reference(inputs, steps=1)
        assert torch.isfinite(got).all(), (case["id"], "non-finite output")
        assert tuple(got.shape) == tuple(expected.shape), (
            case["id"], tuple(got.shape), tuple(expected.shape)
        )
        cos, err = _deviation(got, expected)
        _assert_within_tolerance(case, cos, err, "single step")

        # The caches are the point of this operator, so check them too -- an
        # implementation that produces the right output but corrupts the state
        # would break generation on the next token.
        ref_conv = inputs["conv_state0"].clone()
        ref_ssm = inputs["ssm_state0"].clone()
        _reference_step(inputs, ref_conv, ref_ssm)
        c_cos, c_err = _deviation(inputs["conv_state"], ref_conv)
        s_cos, s_err = _deviation(inputs["ssm_state"], ref_ssm)
        assert c_cos > 0.999 and c_err < 0.02, (case["id"], "conv_state drift", c_cos, c_err)
        assert s_cos > 0.999 and s_err < 0.02, (case["id"], "ssm_state drift", s_cos, s_err)

        print(f"correctness PASS {case['id']:36s} bs={inputs['batch']:<5d} "
              f"cos={cos:.6f} rel_err={err:.5f}  conv_state cos={c_cos:.6f} "
              f"ssm_state cos={s_cos:.6f}")
        del inputs, expected, got, ref_conv, ref_ssm
        _free()


def _perturb(inputs: dict) -> None:
    """Fresh projections + a pristine cache, so a replay starts from a known point."""
    torch = _torch()
    torch.manual_seed(97)
    inputs["qkvz"].normal_(0.0, 0.5)
    inputs["ba"].normal_(0.0, 0.5)
    _reset_state(inputs)


def _assert_timed_outputs(case: dict, inputs: dict, timed, steps: int) -> None:
    """Validate the invocation the benchmark actually timed.

    The op is stateful and the captured graph holds ``steps`` chained
    invocations, so one replay advances the caches ``steps`` times. The
    reference is stepped the same number of times from the same snapshot; that
    is what makes a graph-timed stateful kernel checkable at all.
    """
    torch = _torch()
    if not timed.bound:
        raise RuntimeError("benchmark did not expose the timed invocation")
    _perturb(inputs)
    if timed.outputs is not None:
        timed.outputs.fill_(float("nan"))
    got = timed.rerun()
    expected = _reference(inputs, steps=steps)
    assert torch.isfinite(got).all(), (case["id"], "timed run produced non-finite output")
    cos, err = _deviation(got, expected)
    _assert_within_tolerance(case, cos, err, f"timed run ({steps} chained steps)")
    del expected


def run_performance() -> None:
    rows = []
    for case in CASES:
        inputs = _prepare(case)
        _reset_state(inputs)
        _run(inputs)
        _torch().cuda.synchronize()
        bench = case.get("benchmark", {})
        timed = _TimedRun()
        _reset_state(inputs)
        exec_ms, meta = _benchmark_cuda_graph_or_events(
            lambda: _run(inputs),
            warmup=bench.get("warmup", 10),
            repetition=bench.get("repetition", 30),
            target_ms=bench.get("target_ms", 1.0),
            max_graph_repeats=bench.get("max_graph_repeats", 50),
            timed_run=timed,
        )
        steps = int(meta.get("benchmark_effective_repeats", 1))
        _assert_timed_outputs(case, inputs, timed, steps)
        metadata = {
            **case["params"],
            "model": case.get("model"),
            "phase": case.get("phase"),
            "calls_per_forward": case.get("calls_per_forward"),
            "session_e2e_pct": case.get("session_e2e_pct"),
            "chained_steps_per_replay": steps,
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
              f"chained_steps={steps}", meta.get("benchmark_fallback_reason", ""))
        del inputs
        _free()
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

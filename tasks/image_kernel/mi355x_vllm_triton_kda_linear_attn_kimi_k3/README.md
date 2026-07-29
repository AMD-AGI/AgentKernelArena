# mi355x-kimi-k3-kda-linear-attn-20260728

`image_kernel` harness for **Kimi-K3 KDA (Kimi Delta Attention)** — the
flash-linear-attention (FLA) **Triton-JIT** gated-delta-rule path used by K3's 69
linear-attention layers. Built from Hyperloom session `20260728T091437Z`.

## Language / implementation

**Triton** (`@triton.jit`, runtime-compiled), copied into vLLM from the
flash-linear-attention project (Songlin Yang, Yu Zhang; MIT). Source lives in
`vllm/model_executor/layers/fla/ops/` (kda.py + chunk*.py + cumsum.py +
fused_recurrent.py + l2norm.py + wy_fast.py + solve_tril.py + causal_conv1d.py).
This is why the trace shows these kernels with `launcher = Not found` — they are
JIT-compiled at runtime.

## Hot kernels covered

- **k007** `fused_recurrent_kda_packed_decode_kernel` (decode; 48.4 ms, 2.11% GPU)
  → the `recurrent` case.
- **prefill chunk kernels** `chunk_kda_fwd_*`, `chunk_gated_delta_rule_fwd_h_*`,
  `kda_gate_chunk_cumsum_*`, `chunk_gla_fwd_kernel_o`, `recompute_w_u_*`,
  `l2norm_*`, `layer_norm_gated_*`, `_causal_conv1d_fwd_*`, `solve_tril`
  (prefill/eager KDA, ~43.7 ms; grouped, no individual k-IDs) → the `chunk` cases.

## Config (aligned to the K3 session)

Kimi Linear KDA (arXiv:2510.26692 + K3 config): `d_k = d_v = head_dim = 128`,
`num_heads = 96`, `chunk_size = 64`, `hidden = 7168`, 69 KDA layers, `dt_bias`
length `96*128 = 12288`. The trace is **rank0 of TP=8**, so per-rank
`num_heads = 96/8 = 12`, `head_dim = 128` — the cases use the per-rank shape.

Real caller: `vllm/.../mamba/gdn/kimi_gdn_linear_attn.py`
- prefill: `chunk_kda_with_fused_gate(q,k,v,raw_g,beta,A_log,g_bias=dt_bias,…,use_qk_l2norm_in_kernel=True)`
- decode: `g = fused_kda_gate(raw_g,A_log,dt_bias)` then `fused_recurrent_kda(q,k,v,g,beta,…,ssm_state_indices)`

Tensor layout `[1, total_tokens, H, D]` with `cu_seqlens` (matches the caller's
`rearrange("n (h d) -> 1 n h d")`).

## Cases (incl. long-sequence)

| id | mode | seqs × len | heads |
|---|---|---|---|
| kda-decode-recurrent-k007 | recurrent | 62 × 1 | 12 |
| kda-prefill-chunk-isl1024 | chunk | 1 × 1024 | 12 |  (session real ISL) |
| kda-long-2048 … 32768 | chunk | 1 × {2048,4096,8192,16384,32768} | 12 |
| kda-long-32768-fullheads | chunk | 1 × 32768 | 96 (no-TP) |

## Correctness — real numerical parity vs a float64 golden

FLA ships no naive torch reference in-tree, so this harness implements one:
`scripts/task_runner.py:_golden` is an independent **float64** transcription of the
KDA recurrence, taken directly from `fused_recurrent_gated_delta_rule_fwd_kernel`
(IS_KDA=True) and `fused_kda_gate` / `kda_gate_fwd_kernel`:

```
gate g_t = -exp(A_log_h) * softplus(raw_g + dt_bias)          # per (head, k-channel)
per token (state S = [H, d_v, d_k], reset per sequence):
  q = l2norm(q_t)*scale ;  k = l2norm(k_t) ;  v = v_t
  S  = S * exp(g_t)          # decay per k-column
  v  = v - S @ k             # delta-rule "remove old value"
  v  = v * beta_t
  S  = S + outer(v, k)
  o_t = S @ q
```

The correctness gate asserts **cos > 0.999** and **normalized max error < 0.03**
against this golden (plus finiteness + exact shape). Verified on MI355X:

- `chunk_kda_with_fused_gate` (prefill/long) → **cos = 0.99999, rel_max_err < 0.007**
- packed-decode `fused_recurrent_kda` → **cos = 0.99999** (called with
  `ssm_state_indices=None` + `inplace_final_state=False`; a non-null index tensor
  makes the kernel skip seqs with `state_idx<=0` and silently return ~0).

Two bugs found and fixed while building the golden: (1) chunk writes its final
state into `initial_state` in place, so the golden must be computed **before** the
kernel run; (2) the earlier decode call passed `ssm_state_indices=arange(...)`
starting at 0, which the kernel treats as NULL and skips.

**Performance (CUDA-graph timing) is the optimization target.**

## Runnable here (verified)

Unlike the mxfp4 MoE task, this KDA task **runs on this container** (vLLM 0.24.0 +
Triton) — KDA is Triton-JIT and does not depend on the missing K3 mxfp4 dispatch.
Verified on MI355X: `chunk_kda_with_fused_gate` executes from T=64 up to T=32768.

## Run

```
python3 scripts/task_runner.py compile       # smoke: one KDA call
python3 scripts/task_runner.py correctness   # determinism + finite + shape + state
python3 scripts/task_runner.py performance   # CUDA-graph timed, writes build/performance_report.json
```

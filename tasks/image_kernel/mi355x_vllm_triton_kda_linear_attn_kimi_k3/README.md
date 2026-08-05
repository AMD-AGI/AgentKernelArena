# mi355x-kimi-k3-kda-linear-attn-20260728

`image_kernel` harness for **Kimi-K3 KDA (Kimi Delta Attention)** — the Triton-JIT
gated-delta-rule path used by K3's 69 linear-attention layers. Built from Hyperloom
session `20260728T091437Z` on MI355X/gfx950.

## Where the kernels actually live

The KDA kernels are vendored **per GPU vendor**; `kimi_gdn_linear_attn.py:399`
selects the AMD copy on ROCm:

```
vllm/models/kimi_k3/amd/ops/third_party/kda/{chunk,chunk_intra,chunk_intra_token_parallel,fused_recurrent}.py
vllm/third_party/flash_linear_attention/ops/{chunk_delta_h,cumsum,index,l2norm,op,utils,solve_tril,wy_fast}.py
```

This is confirmed by the session's own call stack (tracelens
`unified_perf_summary.csv`):

```
kimi_gdn_linear_attn.py(381): _forward
 -> kda/chunk.py(774): chunk_kda_with_fused_gate
    -> kda/chunk.py(694): chunk_kda_with_fused_gate_fwd
       -> kda/chunk_intra.py(573): chunk_kda_fwd_intra
```

They are Triton-JIT, which is why the trace reports `launcher = Not found`.

## Hot kernels covered

- **k007** `fused_recurrent_kda_packed_decode_kernel` (decode; 48.4 ms, 2.11% GPU)
  → the `packed_decode` case.
- **prefill chunk group** `chunk_kda_fwd_*`, `chunk_gated_delta_rule_fwd_h_*`,
  `kda_gate_chunk_cumsum_*`, `chunk_gla_fwd_kernel_o`, `recompute_w_u_*`,
  `l2norm_*`, `layer_norm_gated_*` (~43.7 ms, grouped with no individual k-IDs)
  → the `chunk` cases.

### The decode entry point matters

k007 is launched **only** by `fused_recurrent_kda_packed_decode`
(`fused_recurrent.py:596`), called from the non-spec decode branch
(`kimi_gdn_linear_attn.py:609`). The sibling `fused_recurrent_kda` launches a
*different* kernel, `fused_recurrent_kda_fwd_kernel`, on the speculative-decode
branch — and K3 sets `num_nextn_predict_layers=0`, so that kernel appears **0
times** in every trace artifact of this session. Targeting it would benchmark code
the model never runs.

## Config (from K3 `config.json linear_attn_config`)

`num_heads=96`, `head_dim=128` (`d_k = d_v`), `chunk_size=64`,
`gate_lower_bound=-5.0`, `short_conv_kernel_size=4`, `use_full_rank_gate=true`,
69 KDA layers + 24 full-attention layers. The trace is rank0 of TP=8, so the cases
use the per-rank shape `num_heads = 96/8 = 12`.

Shape evidence: `aten::fill_` under `chunk_kda_fwd_intra` carries
`Input Dims (1, 7211, 12, 64)` and `(1, 1080, 12, 64)` bf16 — i.e. per-rank H=12
and packed prefill token counts 7211 / 1080. Note that ISL=1024 is the
*per-request* input length, not the packed batch size, so it is not a kernel shape.

## Two contract details that are easy to get wrong

Both were read off the kernel sources, not assumed:

1. **`gate_lower_bound = -5.0` is not a clamp — it selects a different gate
   function.** With the bound set the kernel computes
   `gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`; without it,
   `gate = -exp(A_log) * softplus(raw_g + dt_bias)`
   (`fused_recurrent.py:513-521`, `chunk.py:507-515`). K3 always takes the first
   branch, so both the kernel call and the golden use it.

2. **`raw_beta` is passed pre-sigmoid.** Both kernels apply `sigmoid` internally
   (`fused_recurrent.py:525`, `chunk.py:470`), so pre-applying it in the harness
   would square the gate.

Also: `A_log` is 1-D of length `local_num_heads` and `dt_bias` is
`local_num_heads * head_dim` (`kimi_gdn_linear_attn.py:238,266`);
`state_indices` entries must be `> 0` because `<= 0` is the NULL slot and makes the
kernel emit zeros for that row (`fused_recurrent.py:481`).

## Cases

| id | mode | seqs x len | source |
|---|---|---|---|
| `kda-decode-packed-k007` | packed_decode | 62 x 1 | reconstructed (slice concurrency conc62) |
| `kda-prefill-chunk-t7211` | chunk | 1 x 7211 | trace |
| `kda-prefill-chunk-t1080` | chunk | 1 x 1080 | trace |
| `kda-long-chunk-t16384` | chunk | 1 x 16384 | extrapolated headroom |
| `kda-long-chunk-t32768` | chunk | 1 x 32768 | extrapolated headroom |

## Correctness — numerical parity vs a float64 golden

FLA ships no naive torch reference in-tree, so `scripts/task_runner.py:_golden` is
an independent **float64** transcription of the recurrence, taken directly from
`fused_recurrent_kda_packed_decode_kernel` (`fused_recurrent.py:504-533`):

```
g_t = -5.0 * sigmoid(exp(A_log) * (raw_g + dt_bias))     # safe-gate branch
per token (state S = [H, d_v, d_k], one segment per sequence):
  q = l2norm(q_t) * scale ;  k = l2norm(k_t) ;  v = v_t
  S  = S * exp(g_t)          # decay per k-column
  v  = v - S @ k             # delta-rule "remove old value"
  v  = v * sigmoid(raw_beta_t)
  S  = S + outer(v, k)
  o_t = S @ q
```

`chunk_kda_with_fused_gate` computes the same recurrence blockwise, so one
reference covers both modes. Gate: `cos > 0.999` and normalized max error `< 0.03`.

Measured on MI355X: packed decode `cos = 0.999999`, `rel_max_err = 0.0027`;
chunk `cos = 0.999992`, `rel_max_err = 0.0071`.

Both kernels update the state in place, so the golden's starting state is
snapshotted before the kernel runs. Correctness caps the token count (320 tokens,
still 5 chunks at `chunk_size=64`) because the golden is an O(T) float64 loop;
each case uses its own seed so the capped runs are not duplicates.

## Edit surface and JIT freshness

`_configure()` puts the workspace-seeded `vllm` copy first on `sys.path`, so an
agent's kernel edits shadow the in-image install; Triton keys its cache on kernel
source, and `TRITON_CACHE_DIR` is additionally pinned inside the workspace so no
run can serve a binary compiled from another workspace's source.

Verified end to end: scaling `b_q` by 1.5 in the workspace copy of
`fused_recurrent.py` changes the decode output norm by exactly 1.5x
(0.214733 -> 0.322133), and the same edit in `chunk_intra.py` moves the chunk
output (1.473997 -> 2.210493).

## Run

```
python3 scripts/task_runner.py compile       # smoke: one KDA call
python3 scripts/task_runner.py correctness   # float64 parity, both modes
python3 scripts/task_runner.py performance   # CUDA-graph timed -> build/performance_report.json
```

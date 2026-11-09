# GDN recurrent decode step, bfloat16, 16 query/key heads, 32 value heads, K = V = 128 (FlyDSL baseline)

One decode step of the gated delta rule (Gated DeltaNet) per sequence, reading the new token's `q`, `k`, `v`,
`a` and `b` and updating the float32 recurrent state of the sequence's pool slot in place
(`flydsl_gdr_decode` of AITER). For each of the 32 value heads (query/key head `hv // 2`):

```text
q, k    = l2norm(q), l2norm(k);  q *= K^-1/2          # eps 1e-6 inside the square root
g       = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
S       = exp(g) S                        # S: [V, K] float32 state of the slot
S      += ((v - S k) * beta) k^T
out     = S q                             # bfloat16
```

`state_indices` gives each sequence a distinct slot of the pool (`batch + 2` slots, so two stay untouched). The
pool is a mutable input, restored before every call and compared afterwards. This is the operator of
`tasks/triton/state_space/gdn_fused_recurrent_decode/bf16_hk16_hv32_k128_v128`, with the kernel's own interface:
separate `query`, `key` and `value` instead of one packed `q | k | v` row, `A_log` in float32, the scale fixed to
`K^-1/2` inside the kernel, and no padded sequences. A negative slot makes the FlyDSL kernel skip the sequence
and leave its output unwritten, which a checked output cannot allow, so the scenario holds no padded
sequences. The FlyDSL kernel also computes `beta` in float32 (the Triton one rounds it to the dtype of `b`).

## Origin and config citations

- **Baseline origin.** AITER's FlyDSL GDR decode kernel `aiter/ops/flydsl/kernels/gdr_decode.py` with the launch
  and tuned-configuration lookup of `aiter/ops/flydsl/linear_attention_kernels.py::flydsl_gdr_decode` (MIT).
  AITER's own unit test of the kernel (`aiter/ops/flydsl/test_flydsl_linear_attention.py`) checks it against a
  Triton fused sigmoid-gating delta-rule kernel, the same operator.
- **Dimensions.** Qwen3-Next: 16 query/key heads, 32 value heads, head dimensions 128, from the defaults of
  `Qwen3NextConfig` in `vllm/transformers_utils/configs/qwen3_next.py` (pinned vLLM commit):
  `linear_num_key_heads` 16, `linear_num_value_heads` 32, `linear_key_head_dim` 128, `linear_value_head_dim` 128.
  The Hugging Face `config.json` of Qwen3-Next-80B-A3B-Instruct has the same fields (`linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128). AITER's tuned table
  `aiter/ops/flydsl/gdr_decode_tuned.csv` has this shape (16 key heads, 32 value heads, 128/128, one token, bfloat16
  activations, float32 state) for every batch from 1 to 256 on gfx950.
- **Dtypes.** `A_log` is a float32 parameter in vLLM's `qwen_gdn_linear_attn.py` (`dtype=torch.float32`), the dtype
  AITER's test uses; the kernel also takes bfloat16. The state is float32 as in the Triton task.

## Baseline

`baseline/gdr_decode.py` is AITER's kernel builder `create_vk_gdr_decode_kernel` (MIT; header kept) with its
imports made relative. `kernel.py` launches it as `flydsl_gdr_decode` does (`use_qk_l2norm=True`,
`need_shuffle_state=False`, state layout `[pool, heads, V, K]`), with the launch configuration
(`NUM_BLOCKS_PER_V_DIM`, `NUM_WARPS`, `WARP_THREADS_K`) looked up per batch size. `baseline/tuned_configs.py` holds
the table: the gfx950 rows of `gdr_decode_tuned.csv` for this shape (batch 1 to 256, the CSV itself is neither
Python nor JSON), and AITER's default configuration (1 block per value dim, 4 warps, 8 threads along K) for the
larger batches, which the table does not list (the workloads go up to 800). Support files: `tensor_shim.py` (AITER, MIT;
`os`/`numpy` uses replaced), `buffer_ops.py` and `vector.py` (AITER's vendored copies of FlyDSL code, Apache-2.0,
`baseline/LICENSE-FlyDSL`, headers kept verbatim; `vector.py` keeps the two raw ops `gdr_decode.py` uses,
`BroadcastOp` and `FMAOp`, instead of the upstream star import).

Origin notes: no vendored AITER file here carries an "adapted from" statement; `buffer_ops.py` and `vector.py` state
that they are copies of FlyDSL code. The kernel's first commit in AITER is "Add GDR decode kernels" (a FlyDSL
implementation of the decode step; the file is not marked as derived from another project).

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm. AITER requires FlyDSL 0.3.0 (`setup.py`); the other FlyDSL names the kernel and shim use (`flyc.kernel/jit/compile`, `fx.Tensor`, `fx.Vector` with `reduce` and `shuffle_xor`, `fx.ReductionOp`, `fx.math.rsqrt/log1p/exp`, `arith.constant`, `arith.FastMathFlags`, `rocdl.exp2`, `flydsl._mlir.dialects.gpu.ShuffleOp`, `fx.Stream`) were checked by name against `python/flydsl` of FlyDSL v0.3.2.

## Inputs

`query`, `key`, `value`, `a`, `b` and `dt_bias` are standard normal and `A_log` is uniform in [0, 2.7] (the choice of the
Triton task; no checkpoint statistics were checked), so `exp(g)` ranges from about 1 down to tiny values (14.8% of the
value heads have `g < -9`). The state is normal with standard deviation 0.1. The key normalization keeps the
recurrence contractive.

## Reference

Float64, vectorized over groups of 32 sequences. Against an emulation of the kernel's order of
operations (below) it gives 55.3-55.7 dB SQNR on `out` (a bfloat16 output) and 142.6-144.1 dB on the state.

## Comparison (overrides of the default bound)

`out` uses `baseline_relative` with `k: 8` and `max_violation_fraction: 1e-4`; `state` uses `baseline_relative` with
`k: 8` and `max_violation_fraction: 1e-3`. All figures below come from CPU float32 emulations measured against the
float64 reference and written for this task: the baseline emulation follows the kernel's operation order
(threads of `WARP_THREADS_K` lanes of 4 consecutive `K` values, fused multiply-adds, lane and butterfly
reductions, the rank-1 form `out = exp(g) S q + delta (k . q)`, `exp2`-based scalars), with an accurate `exp2` and
`log1p` where the kernel uses hardware instructions. Batches 1, 3, 37 and 100 (two draws each) and 256 (one draw), each with its
tuned configuration.

- **State.** The baseline is accurate to float32 rounding (142.6 to 144.1 dB), so the default bound is a
  test of one-ulp agreement. Correct independent float32 designs (PyTorch float32 operations, sequential
  unfused sums, the rank-1 form) violate it on 0.57% to 1.4% of the pool elements at `k = 2` (peak up to 10.7
  bounds), 4.0e-4 to 1.9e-3 at `k = 4`, and 5.7e-6 to 9.2e-5 at `k = 8` (peak at most 3.3 bounds). The float32
  evaluation of the reference behaves like such a design. The largest fraction
  comes from the smallest batch, where only one of the three pool slots is updated (2.7e-4 of the updated slot). A design with 4-ulp errors in the per-head scalars
  `exp(g)` and `beta` (a deliberately poor fast-math stress) violates 6.5e-3 to 1.1e-2 at `k = 8` and is
  rejected: candidates must be about as accurate as the baseline. The fraction 1e-3 leaves 11x above the worst correct
  design. The same bound rejects every reduced-precision shortcut by a wide margin; the smallest share of
  elements out of bound at `k = 8` (and SQNR of the state):
  - bfloat16 operands in `S k`: 0.57 to 0.79 (82 to 87 dB);
  - bfloat16 normalized keys 0.995 (57 to 58 dB), bfloat16 decay 0.70 to 0.91 (60 to 64 dB), bfloat16 `beta` 0.98
    to 0.996 (57 to 58 dB), bfloat16-rounded state 0.999 (54 dB), float16-rounded state 0.99 (72 dB);
  - a prediction from the undecayed state: 1.0 (negative SQNR).
  Untouched pool slots carry no baseline error, so they must stay unchanged for any `k`.
- **Output.** `out` is bfloat16, so its error is one rounding in almost all elements, and correct float32 designs
  have no element out of bound at `k = 2` in the draws above. Heads where `exp(g)` is tiny (`g < -9`, 14.8% of the
  value heads with these inputs) and `q` is almost orthogonal to `k` (`|cos| < 2.5e-5`, 2.1e-4 of the key heads
  in a sample of 7.5 million random bfloat16 pairs) are different: `out` is about 1e-8 there while its terms are about
  1e-4, float32 rounding leaves every design a relative error of about `u / |cos|`, and a correct candidate can
  exceed the baseline's error multiple in a few elements. A Monte Carlo of 1600 such key heads (3200 value
  heads, all with `g < -9`, state and `v` from the task's distributions, batch of 100) gave, for correct designs
  (PyTorch float32, rank-1 form, sequential sums) and a 4-ulp-scalar stress design:
  | `k` | elements out of bound (of 409600) | worst peak |
  | --- | --- | --- |
  | 2 | 2825 to 3057 | 9.3 (rank-1 form), 32.8 (PyTorch float32) |
  | 4 | 1018 to 1255 | 6.4, 21.5 |
  | 8 | 294 to 462 | 4.3, 12.8 |

  At `k = 2` or `4` the worst element can lie more than 16 bounds away, which a violation fraction never admits, so
  `k = 8` is required. With about 1.0e-3 such value heads per sequence (16 key heads x 2.1e-4 x 2 value heads x 14.8%), a
  full evaluation holds about 4 of them (3914 sequences over all cases and both draws), about 0.14 elements out of
  bound per head at `k = 8`; `max_violation_fraction: 1e-4` allows 0 elements at batch 1 (where the chance of such
  a head is 1e-3), 3 at batch 8 and 327 at batch 800. The same `k = 8` bound still rejects bfloat16-rounded
  normalized keys (0.27% to 1.9% of the elements; the state check rejects the other shortcuts). The state
  check, not `out`, is the strict one.

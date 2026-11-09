# GDN chunk state pass, bfloat16, 16 key heads, 32 value heads, K = V = 128, chunk 64 (FlyDSL baseline)

For one sequence of `n_tokens` tokens and each of the 32 value heads (key head `h // 2`), the operator carries a
float32 state `S[V, K]` across chunks of 64 tokens (Gated DeltaNet prefill, `chunk_gated_delta_rule_fwd_h`):

```text
h[chunk]     = S                                  # state at the start of the chunk, rounded to bfloat16
v_new[chunk] = u[chunk] - w[chunk] S^T            # delta correction, rounded to bfloat16
S            = 2^g_last S + (v_new * 2^(g_last - g))^T k[chunk]
```

The state starts from `initial_state` and ends in `final_state`. `g` is the chunk-local cumulative log-decay in
**base 2** (it restarts every 64 tokens), as AITER's prefill pipeline hands it to the FlyDSL kernel (it scales the
gate by log2(e) so that the kernel evaluates `exp2` directly). This is the operator of
`tasks/triton/state_space/gdn_chunk_fwd_h/bf16_hk16_h32_k128_v128_c64`, in the layout of AITER's FlyDSL kernel:
`w`, `u`, `g` and `v_new` are head-major (`[1, H, T, ...]`), the gate is in base 2, the initial state is one
`[1, H, V, K]` tensor and the final state a separate output. The Triton task uses a state pool addressed by a slot
index (updated in place) and token-major tensors; the FlyDSL kernel at this commit has neither (its pool mode is
the varlen `N`-sequence mode, which this single-sequence scenario does not use).

## Origin and config citations

- **Baseline origin.** AITER's FlyDSL K5 kernel `aiter/ops/flydsl/kernels/chunk_gated_delta_h.py` (MIT), the
  hidden-state pass that `chunk_gated_delta_rule_opt_vk(..., use_chunk_flydsl=True)` of
  `aiter/ops/triton/gated_delta_net/gated_delta_rule.py` calls through
  `aiter/ops/flydsl/linear_attention_prefill_kernels.py`.
- **Dimensions.** Qwen3-Next: 16 key heads, 32 value heads, head dimensions 128, from the defaults of
  `Qwen3NextConfig` in `vllm/transformers_utils/configs/qwen3_next.py` (pinned vLLM commit):
  `linear_num_key_heads` 16, `linear_num_value_heads` 32, `linear_key_head_dim` 128, `linear_value_head_dim` 128.
  The Hugging Face `config.json` of Qwen3-Next-80B-A3B-Instruct has the same fields (`linear_conv_kernel_dim` 4, `linear_num_key_heads` 16, `linear_key_head_dim` 128, `linear_num_value_heads` 32, `linear_value_head_dim` 128). The chunk size 64 is the production
  chunk of the kernel (`BT`) and of AITER's pipeline. AITER's tuned table `aiter/ops/flydsl/chunk_gdn_h_tuned.csv`
  lists this shape (`H` 32, `Hg` 16, `K` = `V` = 128, `BT` 64, `N` 1, `BV` 16) for 2500, 60000 and 128000 tokens.

## Baseline

`baseline/chunk_gated_delta_h.py` is AITER's kernel builder `compile_chunk_gated_delta_h`, unchanged (header kept:
`SPDX-License-Identifier: MIT`, `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`). It is compiled for the
scalar-gate, float32-state, fixed-length path (`USE_G`, `USE_INITIAL_STATE`, `STORE_FINAL_STATE`,
`SAVE_NEW_VALUE`, `WU_CONTIGUOUS`, `G_IS_LOG2_SCALED`), the options of AITER's prefill call for this shape, and
launched with the BV tile of AITER's rule-based heuristic (`_heuristic_bv`, copied verbatim into `kernel.py`),
which picks BV = 16 for one sequence of 32 value heads and V = 128 (a 256-program grid of one wave per
`(16-wide value tile, head)`), the BV of the tuned table. Support files: `tensor_shim.py` (AITER, MIT; its `os`
and `numpy` uses replaced), `buffer_ops.py` and `vector.py` (AITER's vendored copies of FlyDSL code, Apache-2.0,
`baseline/LICENSE-FlyDSL`, headers kept verbatim; `vector.py` without its star import of the raw dialect, which
nothing here uses). The wrapper's Triton helpers (`triton.cdiv`, chunk-offset metadata of the varlen path) are
not needed by a single sequence; a FlyDSL baseline may not import Triton.

Origin notes: no vendored AITER file here carries an "adapted from" statement; `buffer_ops.py` and `vector.py`
state that they are copies of FlyDSL code (`kernels/common/buffer_ops.py`, `python/flydsl/expr/vector.py`).
The kernel is AITER's own (its first commit is "Add GDR prefill chunk_gdn_fwd_h kernel for MI35X").

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm. AITER requires FlyDSL 0.3.0 (`setup.py`); the other FlyDSL names the kernel and shim use (`flyc.kernel/jit/compile`, `fx.struct`, `fx.Array`, `fx.SharedAllocator`, `fx.ptr_load/ptr_store/ptrtoint`, `fx.Vector`, `fx.full`, `fx.Stream`, `rocdl.mfma_f32_16x16x32_bf16`, `rocdl.exp2`, `gpu.barrier`) were checked by name against `python/flydsl` of FlyDSL v0.3.2. `rocdl.ds_read_tr16_b64` comes from the generated ROCDL dialect bindings, which are not in the source checkout (FlyDSL's own kernels use the sibling `rocdl.ds_read_tr8_b64` the same way).

## Inputs

Keys have element standard deviation 128^-1/2 (the model L2-normalizes them) and `w` has 0.1, which keeps the
delta-rule recurrence contractive: in the CPU emulation the final state norm settles at about 4.6 times the
initial one from 1000 tokens to 16384 tokens (4.4 to 4.5 at 100 tokens, 1.3 at one token). `u` is standard normal and the initial state
normal with standard deviation 0.1. The gate per token is `0.03 * logsigmoid(x)` (small, like
`-exp(A_log) * softplus(.)` in the model), accumulated within each chunk and scaled by log2(e), which gives the
state a memory of a few chunks.

## Reference

A float64 chunk-by-chunk implementation of the equations above with no intermediate rounding. The kernel rounds
the state to bfloat16 for the `w S^T` product and the gated delta to bfloat16 for the update (its documented
behavior); this is part of the baseline's own error, which candidates are held to. A CPU emulation of the
kernel (bfloat16 state for the product, float32 accumulation, bfloat16 gated delta; written for this task, not
a transcription of the kernel's MFMA order) has 51.5-55.7 dB SQNR against the reference on `h` and `v_new`
and 53.7-58.9 dB on the final state, from 1 to 16384 tokens.

## Workloads

512 to 65536 tokens (128x span) are performance cases. The kernel walks the chunks sequentially in 256
programs, so its time grows with the chunk count: AITER's table measures 69 us for 2500 tokens (40 chunks) and
2025 us for 60000 tokens (938 chunks) on gfx950 for this shape, about 2.2 us per chunk, so the smallest case
(8 chunks) is estimated at about 20 us (an extrapolation, not a measurement; the table's rows do not store the
final state) and the largest at about 2.2 ms. Shorter prompts are correctness-only: 1 token (axis minimum), 100
and 200 (partial last chunk), 1000 (unaligned). 2500 tokens, the shape of the tuned table, is a performance
case. The largest case needs about 4.3 GB for the float64 `h` reference plus the comparison temporaries.

## Comparison (overrides of the default bound)

`h` and `v_new` use `baseline_relative` with `k: 4` and `max_violation_fraction` 1e-3 and 1e-4; the final state
uses `sqnr` with `min_db: 47`.

- *Why not the default.* The state carries every bfloat16 rounding of the sequence. Two equally valid
  implementations round slightly differently, so their states drift apart along the sequence, and element by
  element, twice the baseline's error rejects correct designs.
- *Emulation.* The baseline emulation above was compared with correct alternatives, for 1 to 16384 tokens (2
  draws at 1 and 100 tokens, 1 draw at 1000, 4096, 8192 and 16384; the 16384-token run used three of the
  designs): (a) the same algorithm with products accumulated in float64 and rounded (a different accumulation
  order); (b) a float32 state for `w S^T`; (c) a float32 gated delta for the update; (d) gating the
  bfloat16-rounded `v_new` and rounding again, the least accurate design. Under the default bound (`k = 2`, no
  violations allowed):
  - (a) violates `h` on 6.9e-6 of the elements at 1000 tokens and 2.1e-4 at 16384, and `v_new` on 1.5e-6 and
    2.0e-5 (both growing with length);
  - (b) violates `h` on up to 3.9e-4 and `v_new` on up to 1.4e-6;
  - (d) violates `h` on 1.0e-2 to 1.2e-2 and `v_new` on up to 9.5e-4 from 1000 tokens on (4.2e-3 and 1.2e-4 at
    100 tokens);
  - (c) violates `h` on up to 2.8e-7 and `v_new` on up to 2.0e-6.
- *`h` and `v_new`.* At `k = 4` every correct design violates `h` on at most 8.4e-5 of the elements (design (d);
  the others at most 1e-7) and `v_new` on at most 4.2e-7, with a peak of 1.8 bounds. Both outputs are bfloat16,
  so one output rounding dominates their bound. The fraction limits (1e-3 and 1e-4) leave 12x and 240x above
  the worst correct design. A shortcut that rounds the state to fp8 for `w S^T` violates `v_new` on 7.6e-3 and
  1.3e-2 at one token (the two draws), and from 100 tokens on `h` on at least 1.2e-3 (0.25 to 0.29 from 1000
  tokens) and `v_new` on at least 0.11 (0.35 to 0.38 from 1000 tokens): `v_new` is at least 76x above its limit
  in every case.
- *Final state.* It is float32, so its bound has no output-rounding margin and any per-element bound compares
  two decorrelated accumulation errors; SQNR is used instead. The baseline emulation has 53.7 dB (from 1000
  tokens), 54.5 dB (100 tokens) and 58.9-59.0 dB (1 token). Correct designs have at least 51.2 dB (design (d)). Wrong
  designs: the fp8 state at most 39.7 dB from 100 tokens on (52.5 dB at 1 token, where `v_new` rejects it),
  a skipped decay of the last chunk 2.2 to 34.3 dB, and one head left at zero 14.2 to 15.9 dB. 47 dB is 4.2 dB
  below the worst correct design and 7.3 dB above the closest wrong design.
- *Controls.* The evaluator's 16-ulp shift of an output is 6% relative error in bfloat16, far outside the
  `k = 4` bound of an output whose baseline error is one rounding.

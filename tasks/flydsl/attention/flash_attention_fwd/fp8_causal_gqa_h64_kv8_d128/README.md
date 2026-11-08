# Flash attention forward, causal GQA, FP8 (e4m3fn, per-tensor scales), 64 query heads / 8 KV heads, head_dim 128

Prefill attention over a batch of equal-length sequences with FP8 Q, K and V. Each tensor is stored as
`float8_e4m3fn` codes with one float32 descale (`q_descale`, `k_descale`, `v_descale`, shape `[1]`), the dequantized
value being `code * descale`. Every query attends to all earlier positions of its own sequence and itself with scale
`1/sqrt(128)`; query head `h` uses KV head `h // 8`. Q is `[batch, seq_len, 64, 128]`, K and V are
`[batch, seq_len, 8, 128]`, the output is bfloat16 `[batch, seq_len, 64, 128]`. The float64 reference dequantizes the
codes exactly and is otherwise exact causal attention, with the queries chunked by 256 rows so even the largest case
stays within memory.

This is the FP8 sibling of `bf16_causal_mha_h32_d128` (same operator, same layout and axis names): the dtype, head
counts and the three descales differ, so it is a separate scenario.

## Scenario and config citations

A Llama-3.1/3.3-70B-class attention layer at tensor-parallel 1.

- vLLM `benchmarks/kernels/weight_shapes.py` (commit `edbc4969a76b25c531f7b9bb16b79984e65ef023`) lists
  `meta-llama/Llama-3.3-70B-Instruct` with a fused QKV weight of `[8192, 10240]`: `hidden_size` 8192 in, and
  `10240 = (64 + 2 * 8) * 128` out, i.e. 64 query heads, 8 KV heads and `head_dim` 128.
- AITER `op_tests/op_benchmarks/triton/utils/model_configs.json` (commit `652f2d4b…`), `llama3` / `70B`: `num_attention_heads` 64,
  `num_key_value_heads` 8, `hidden_size` 8192, so `head_dim` 128.
- FlyDSL `tests/kernels/test_fused_rope_cache.py` (tag v0.3.2) labels `QH=64, KH=8, D=128` as "Llama-70B TP1".
- The kernel only supports `head_dim` 128 for FP8 (`flash_attn_fp8_gfx950.py`).

## Inputs

The generator draws bfloat16 N(0, 1) tensors and quantizes each per tensor like AITER's `per_tensor_quant`, which is also
what FlyDSL's own FP8 test does: `descale = max|x| / 448` (clamped below at 1e-12), `code = float8_e4m3fn(x / descale)`.
The descales come out near 0.011 for the first draw (the maximum of a few million N(0, 1) samples is about 5), so the
codes fill the upper half of the FP8 range; no code tensor is constant or zero. The two correctness seeds use two draws
with two sets of descales, so a kernel that caches or ignores the descale scalars fails.

## Baseline

`baseline/kernel.py` is ROCm/FlyDSL `kernels/attention/flash_attn_fp8_gfx950.py` at tag v0.3.2
(`dd8373349e33638c8d158a9ea009b7b28598984a`): the gfx950 "dual-wave software-pipelined" kernel, 8 waves per workgroup
with a 256-row query tile (32 rows per wave), 64-key tiles processed in pairs, `mfma` 32x32 products, K/V fed by
DMA-to-LDS through a deep ring, an online softmax whose rescale of the running output is lazy (skipped while the row
maximum moves by at most 8 in the exponent), and longest-first scheduling of the causal query blocks. For FP8 the
logits come from raw FP8 Q and K MFMAs scaled by `q_descale * k_descale * log2(e) / sqrt(128)`, the probabilities
`exp2(logit - running max)` are converted to FP8 directly (no extra scale) for the PV product with FP8 V, the row sum
`l` stays float32 and unquantized, and the output is `PV * v_descale / l` rounded to bfloat16. The driver `run()` builds
the launcher with the interface defaults of `kernels/attention/flash_attn_interface.py` (`waves_per_eu` 2, `daz`
true, lazy rescale, `setprio` and stagger on), causal, GQA group 8.

Numerics and the bound. The default `baseline_relative` bound is kept. The probabilities' FP8 rounding is the baseline's own
error; a CPU emulation of the arithmetic above (exact row maximum instead of the lazy one, `torch` e4m3fn rounding of the
probabilities) against the float64 reference of the dequantized inputs gives 34.5 dB SQNR on a 2 x 640, 16-head, 2-KV-head
causal case and 42-44 dB at 256-2048 tokens with 8 heads, far above the 25 dB sanity floor. An implementation that
reproduces the baseline's quantization of the probabilities passes; one that keeps the probabilities in bfloat16 or
better has a smaller error and also passes; one that quantizes them with a different scale or granularity has errors of
the baseline's size at different elements and may need a larger `k`, which this task does not grant.

### Changes to the upstream files

All vendored files keep their upstream header (`SPDX-License-Identifier: Apache-2.0`,
`Copyright (c) 2025 FlyDSL Project Contributors`) verbatim as their first two lines.

- `kernel.py`: imports now point to the sibling files; the `FA_PP`, `FA_PP_PRIO` and `FA_TPV` environment switches of
  the kernel body are fixed to their defaults (1, 0, 1) through a local `_getenv`; the call of
  `kernels.common.tensor_shim._run_compiled` (an MIT-licensed AMD helper that compiles on first use and then
  dispatches the cached function) is replaced by a direct call of the `@flyc.jit` launcher, which compiles and caches the same way;
  `run()` is added. The kernel code and launch arguments are unchanged.
- `flash_attn_utils.py`: the whole upstream file. `os.getenv` (16 tuning switches of the bf16 and FP8 builders) goes through a
  `_getenv` that returns each default; the two `kernels.common` imports point to the siblings; and the three
  `self.__dict__.update(traits_or_ctx.__dict__)` lines of `DualwaveKernelContext`, `DualwaveFp8KernelContext` and
  `DualwaveSplitKCombineContext`, which let a helper object copy its parent context, are replaced by a small mixin whose
  `__getattr__` reads the parent's attributes (the candidate rules forbid `__dict__`). The helpers are created after the
  context's `init_*` calls and read the late-initialised row indices through `ctx_ref`, which upstream already does, so
  the values they see are the same. This is the one change that touches kernel-adjacent Python.
- `buffer_ops.py` (from `kernels/common/buffer_ops.py`): the `FLYDSL_GPU_ARCH` environment lookup in
  `_get_buffer_flags` is removed; `is_rdna_arch(None)` auto-detects the architecture exactly as upstream does when the
  variable is unset.
- `kernels_common.py`: only `dtype_to_elem_type` of `kernels/common/kernels_common.py`, unchanged.

The FP8 kernel is dense-only (no varlen or split-K) and requires gfx950.

## Provenance

No third-party origin note was found: no "adapted from", "ported from" or external URL in `flash_attn_fp8_gfx950.py`,
`flash_attn_utils.py` or `kernels_common.py`; `buffer_ops.py` cites only the LLVM AMDGPUToROCDL source of the descriptor
flag layout in a comment; FlyDSL has no NOTICE file and `kernels/attention/` has no README. The first commits of the
kernel family are `293b3b2b` ("Add optimized FMHA kernel on MI355X (#629)", 2026-06-04) and `81cfca71`
("[FMHA] gfx950 dualwave SWP forward kernel: split-K, varlen, arbitrary seq_len", 2026-06-16), both by the FlyDSL authors. License: Apache-2.0 for every vendored file, so no `baseline/LICENSE` is added.

## Workloads

Prompt lengths of 1k-16k tokens at batch 1, and batches of 4-16 prompts of 2k-16k tokens
(`batch * seq_len` from 1024 to 131072, 128x). `(16, 8192)` is the GQA shape of FlyDSL's FP8 benchmark. Estimated
baseline time from the causal flops (`4 * 64 * 128` per query-key pair, `seq_len^2 / 2` pairs per sequence) at
2.0 PFLOP/s plus 4 us: the smallest performance case (1, 1024) is 17 GFLOP, about 13 us (the 256 workgroups are a
single wave on the GPU, so the real time is likely higher); the largest, (16, 8192), is 17.6 TFLOP, about 8.8 ms. Correctness-only cases: the minimum `(1, 1)`, `(1, 128)`, odd batch and lengths `(3, 384)`,
`(11, 384)`, batch 17 `(17, 256)`, `(2, 640)` and the unaligned `(1, 2176)` (17 x 128). Ragged lengths are
supported by the kernel and are correctness cases here: `(1, 1)`, `(2, 100)`, `(2, 1000)` and `(1, 3001)`. The evidence at v0.3.2:
`init_tile_bounds` in `flash_attn_utils.py` pads the KV tile count to an even number of at least 4 and says the extra tiles
read zero through the buffer descriptors' `num_records` and are masked, and `causal_mask_pair_if_needed` states that with
`delta = seqlen_kv - seqlen_q` every padding column is already above the diagonal; the CLI of `tests/kernels/test_flash_attn_fwd.py`
runs `DEFAULT_CONFIGS` for `--dtype fp8` unfiltered, and that list includes the lengths 1, 31, 33, 63, 65, 127, 129, 255, 257,
511 and 513 (its "set5"; `run_fp8_config` rejects only `seq_len < 1`). No pytest case there runs a ragged FP8 length, so the support
rests on the kernel code and that CLI list.

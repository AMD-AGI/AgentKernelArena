# Kimi-K3 MoE, bf16 tokens x MXFP4 experts, 896 experts, top-16, hidden 3584, intermediate 384, SiTUv2

The routed-expert block of Kimi-K3 on one tensor-parallel rank (TP8). Each token is sent to 16 of 896 experts
(`topk_ids`, router weights `topk_weights`). For token `t` and slot `j` (expert `e`):

```
g      = x[t] @ W1[e]^T                       # 768 outputs: gate = g[:384], up = g[384:]
act    = 4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25)        # SiTUv2
out[t] = sum_j topk_weights[t, j] * (act @ W2[e]^T)                    # bfloat16
```

`W1` [896, 768, 3584] and `W2` [896, 3584, 384] are MXFP4: e2m1 codes packed two per byte along the contracted
axis (low nibble first) with one e8m0 scale byte, `2^(byte - 127)`, per 32 contracted elements. Tokens are
bfloat16 and are not quantized (a16w4). SiTUv2 uses the Kimi-K3 model values `activation_situ_beta = 4` and
`activation_situ_linear_beta = 25`.

## Production origin

The source task profiled this operator in a vLLM Kimi-K3 serving session on MI355X: AITER `fused_moe` with
`ActivationType.Situv2`, `QuantType.per_1x32`, bf16 x fp4x2 weights in `GateMode.SEPARATED`, which runs AITER's
FlyDSL two-stage a16w4 kernels (stage 1 gate/up GEMM with the fused activation, stage 2 down GEMM). Its two timed
shapes were the traced 7211-token prefill and a 62-token decode batch (the decode token count was reconstructed
from the session's steady-state concurrency). The routing (`topk_ids`, `topk_weights`) is an input, as in
`fused_moe`.

## Inputs and layout

`inputs.py` draws uniform random e2m1 code bytes and e8m0 scale bytes in four binades per weight (2^-10..2^-7 for
`W1`, 2^-6..2^-3 for `W2`). Unit-normal tokens then give gate/up values with a standard deviation of about 0.8
(largest magnitudes about 5) and an output root-mean-square of about 0.4. Routing takes the top-16 of unit-normal
logits (distinct experts) with softmax-renormalized weights, as AITER `fused_topk(..., renormalize=True)` does in
the source task.

AITER's a16w4 stage-1 kernel clamps `gate <= 7` and `up` to `[-7, 7]` before SiTUv2, the same clamp as its SwiGLU
epilogue. The model's SiTUv2 (vLLM `SituAndMul`, AITER's `situv2` torch reference and AITER's other SiTUv2 kernels,
for example `silu_and_mul_fq`) has no clamp, and `reference.py` follows the model. The input distributions keep `|gate|` and `|up|` well below 7, where
both definitions agree; in that range `25 * tanh(up / 25)` differs from `up` by less than 3%.

The logical weights (read by `reference.py`) are in AITER's `[E, N, K/2]` / `[E, N, K/32]` layout with gate rows
before up rows. The kernel-facing layout (`to_kernel_layout`) is the load-time preparation of vLLM's Kimi-K3 SiTU
loader (`Mxfp4MoEMethod._convert_k3_situ_weight_to_kernel_format`), reproduced from AITER:
`shuffle_weight_a16w4(w, 16, gate_up=False)` for both weights, `shuffle_scale_a16w4(w1_scale, E, gate_up=False)`
and `e8m0_shuffle(w2_scale)`. `w2_scale` has 12 scale groups per row, padded to 16 as `e8m0_shuffle` does (AITER
leaves the pad uninitialized; here it is 0x7F), so it is `[896, 3584, 16]`. The transform works one expert at a
time to bound its temporary memory.

## Reference and bound

`reference.py` dequantizes each routed expert in float64 and evaluates SiTUv2 and the weighted top-k sum in
float64 with no intermediate rounding.

The baseline keeps the stage-1 activation in bfloat16 and rounds each weighted expert row
(`topk_weights[t, j] * y`) to bfloat16. At 1-8 tokens (the `reduce` names of the tuned table) stage 2 stores these
per-slot rows and `moe_reduction.py` sums slots 0-15 in float32 in a fixed order. From 16 tokens up (the `atomic`
names) the stage-2 epilogue adds them directly into `out`, which the sorting kernel has zeroed: `store_pair` in
`compile_mixed_moe_gemm2_a16w4` (`baseline/moe_gemm_a16w4.py`) issues one packed bfloat16 `atomicrmw fadd`
(`llvm.AtomicRMWOp`, agent scope) per pair of columns. The 16 expert tiles of a token run in different
workgroups, so the order of these adds, and with it the bfloat16 rounding after each add, changes from run to
run. No other step accumulates in an unordered way: stage 1 runs without split-K (`k_batch` 1 in every tuned
name, so its float32-atomic epilogue is not compiled), and the sorting kernels and `moe_reduction.py` use no
atomics.

The bound is `sqnr` with `min_db = 33`. SQNR is measured against the bfloat16-rounded reference, as the
comparator computes it. On MI355X the baseline measures about 50 dB at 1-3 tokens and about 36-40 dB at the
larger token counts. A copy of the baseline, run as a candidate, reproduced the baseline's output at 1-3 tokens.
From 8 tokens up, the two runs differed beyond the former `baseline_relative` bound (`k = 4`,
`max_violation_fraction = 0.002`) on 0.6-12% of the elements, with peak bound multiples of 3-38x (38x at 16384
tokens, 30x at 2048, 25x at 7211) and no bias (z <= 0.015). Elements that a violation fraction admits must stay
within 16 bound multiples, and `k` is capped at 8, so no per-element bound accepts the baseline against itself.

A CPU emulation of the baseline's rounding at the task's dimensions covers every workload case, with both seeds
of the cases up to 62 tokens. It uses float32 GEMMs, SiTUv2 with the kernel's +-7 clamp, bfloat16 activations
and bfloat16 per-slot rows, and four random add orders per (token, 256-column tile) for each rounding mode. It
gives these SQNRs:

| Output | SQNR (dB) |
| --- | --- |
| Baseline, reduce epilogue (float32 sum of the bfloat16 per-slot rows) | 49.9-50.2 |
| Baseline, atomic epilogue, each bfloat16 add rounded to nearest even | 44.3-45.2 |
| Baseline, atomic epilogue, each bfloat16 add rounded toward zero | 32.6-33.2 |
| Correct design: bfloat16 activation, float32 per-slot values summed in float32, one final rounding | 51.6-52.3 |
| Correct design: float32 activation and sums, one final rounding | 88 or more |
| `shifted_16_ulp` applied to the baseline output | 24.0-24.1 (27.0-27.3 on the round-toward-zero output) |
| `fp8_rounded` applied to the baseline output | 31.2-31.5 (29.0-29.2 on the round-toward-zero output) |

The emulated reduce epilogue matches the GPU's 50 dB. The GPU's 36-40 dB for the atomic epilogue lies between
the round-to-nearest-even emulation (45 dB) and the round-toward-zero emulation (33 dB), so the hardware bfloat16
atomic adds are less accurate than correctly rounded adds. The add order alone also does not explain the GPU's
run-to-run differences. Two emulated orders differ beyond the former bound on at most 0.06% of the elements, with
peaks of at most 2.3x, against 0.6-12% and up to 38x on the GPU. The GPU runs therefore carry a further
variable error that the emulation does not model. At 8 tokens the vendored code issues no atomic, and the source
of the difference the GPU showed there was not traced on CPU. The `sqnr` bound does not assume a deterministic
baseline.

`min_db = 33` is 3 dB below the lowest baseline SQNR measured on MI355X (36 dB), 17 dB below the baseline's
reduce epilogue and at least 18 dB below the correct float32 designs. fp8 rounding alone costs 31.4-31.6 dB.
Combined with the baseline's own noise, `fp8_rounded` measures 31.4-31.5 dB at 1-8 tokens and 30.1-31.0 dB above
(for a 36-40 dB baseline), so it fails every case by 1.5-2.9 dB. `shifted_16_ulp` fails every case by at least
5.7 dB. The fp8 and shift errors themselves depend only on the output distribution and vary by less than 0.2 dB
across cases and seeds. The baseline's own SQNR varies from run to run, so the threshold gives the larger margin
to the baseline.

The bound is the same at every token count. A candidate that sums the 16 expert rows in float32 (51.6 dB or
better), or as the baseline's reduce epilogue does (50 dB), passes every case. A candidate that uses bfloat16
atomics at 1-8 tokens also passes there if its adds are as accurate as the baseline's own atomic epilogue (36-40
dB on MI355X; 44-45 dB with correctly rounded adds), so these cases no longer require a float32 sum. A bfloat16
accumulation that rounds every add toward zero (32.6-33.2 dB) does not reliably pass.

## Baseline

The FlyDSL kernels and their dispatch are vendored from `ROCm/aiter` at commit
652f2d4b768239aefd515c929e84817681ff598d. The dispatch and the tuned table are MIT (`baseline/LICENSE`), and the
FlyDSL kernel builders Apache-2.0 (`baseline/LICENSE-FlyDSL`):

- `moe_sorting_kernel.py`: AITER's FlyDSL MoE sorting (one-shot up to 16 tokens, two- or four-kernel multiphase
  above). AITER's default sorting backend is a HIP (Opus) kernel; the FlyDSL kernel is its
  `AITER_USE_FLYDSL_MOE_SORTING=1` alternative, used here so every launch is FlyDSL. In atomic mode it also zeroes
  `out`.
- `moe_gemm_a16w4.py`: `compile_mixed_moe_gemm1_a16w4` and `compile_mixed_moe_gemm2_a16w4` from
  `mixed_moe_gemm_2stage_common.py`. The pipeline, epilogue, layout, buffer and vector helpers are in their own
  files. `moe_reduction.py` holds the bf16 top-k reduction used by the reduce-mode stage 2.
- `kernel.py`: AITER's host dispatch for this configuration. It holds `get_padded_M` (next power of two) and the
  `inter_dim = 384` rows of `aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv`, inlined as a table of
  `(block_m, kernelName1, kernelName2)` per token bucket. It also holds the FlyDSL kernel-name registry, which turns
  names into tile, waves-per-EU, non-temporal-load, XCD-swizzle and epilogue parameters, and the a16w4 stage
  wrappers of `aiter/ops/flydsl/moe_kernels.py`. Buckets of 1-8 tokens use the reduce epilogue (per-slot rows, then
  `moe_reduction`), larger ones the atomic epilogue. As in AITER, the `persist` and `kw` suffixes of the tuned
  names do not change the a16w4 kernels. Stage-2 `tile_k = 256` resolves to 128 because 384 is not a multiple of
  256.

Changes from AITER:
- Imports are local. Only the a16w4 builders, the sorting kernel and the bf16 reduction are vendored. Branches
  that called back into AITER (the CU-count query of the non-a16w4 persistent path, the fp8 reduction) are removed.
- The `FLIR_*` environment switches are fixed at their defaults.
- `vector.py` imports the two raw dialect ops the kernels use instead of a star import. `layout_utils.py` parses
  layout strings without the `re` module (same matches).
- The `@flyc.jit` launchers are called directly instead of through `tensor_shim._run_compiled`
  (`flyc.compile` on the first call).
- The sorting workspace is allocated per call instead of cached per device.
- Stage 2 writes into the preallocated `out`.

## FlyDSL 0.3.2 adaptation

The runtime image has FlyDSL 0.3.2, whose `rocdl.RawPtrBufferLoadOp` and `RawPtrBufferStoreOp` constructors take
the cache policy only as the keyword `aux` integer attribute. AITER's `buffer_ops.buffer_load` and `buffer_store`
pass it as a positional operand, which fails in 0.3.2 with `TypeError: RawPtrBufferStoreOp.__init__() takes 5
positional arguments but 6 were given` before any kernel runs. The vendored `buffer_ops.py` passes the same
cache-policy value as `aux=` instead. The kernel algorithm does not change. The other raw-buffer ops these kernels
use (`raw_ptr_buffer_atomic_fadd`, `raw_ptr_buffer_load_lds`) go through FlyDSL's own `flydsl.expr.rocdl`
wrappers, which already build the 0.3.2 form.

## Workloads

Token counts are the tuned buckets of the Kimi-K3 table, the traced 62 and 7211, and unaligned batches. With 896
experts the weights dominate even decode: one token reads 16 experts (about 35 MB, an estimated 12 us by bandwidth;
AITER's Kimi-K3 tuning table measures this bucket at 28 us). Eight tokens read about 120 experts (about 270 MB,
60 us in the tuning table). Batches of 1 and of 8 to 16384 tokens are timed (11 cases, 16384x), covering the reduce
epilogue (1 and 8 tokens) and the atomic one. Batches of 2 and 3 tokens and the unaligned batches are
correctness-only. The correctness cases cover every epilogue (reduce at 1-8 tokens, atomic above) and all three
sorting paths (one-shot at 16 or fewer tokens, two-kernel up to 2048, four-kernel above). Every case holds the
full 2 GB of expert weights.

# minGPT transformer block, float32, embedding 768, 8 heads

This task is the whole forward of minGPT's `Block` in eval mode (dropout off), in float32:

```
a   = LayerNorm(x; ln1_weight, ln1_bias, ln_eps)             # biased variance
qkv = a @ c_attn_weight^T + c_attn_bias                       # q | k | v, 8 heads of 96 each
y   = per head softmax(q k^T / sqrt(96), keys j <= i) v       # heads side by side
h   = x + y @ attn_proj_weight^T + attn_proj_bias
m   = NewGELU(LayerNorm(h; ln2_weight, ln2_bias, ln_eps) @ c_fc_weight^T + c_fc_bias)
out = h + m @ mlp_proj_weight^T + mlp_proj_bias
```

`NewGELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` is minGPT's tanh approximation, not
the erf GELU. The weights use the `nn.Linear` layout `[out_features, in_features]`. Head `h` owns
columns `[96h, 96h + 96)` of each of q, k and v.

## Origin

The task comes from KernelBench level 3, problem 44 (`MiniGPTBlock`, taken from karpathy/minGPT
`mingpt/model.py`). It replaces the retired `torch2hip/kernelbench/level3/l3n44_MiniGPTBlock` task,
whose only baseline was the PyTorch module.

The scenario keeps that module's configuration: `n_embd = 768`, `n_head = 8` (head size 96),
`max_seqlen = 1024`, float32 throughout, zero dropout. The head count is the source module's 8,
not GPT-2 small's 12, as in the `attention/causal_self_attention/f32_e768_h8` task built from the
same module's attention layer.

The causal mask buffer (`tril` of `max_seqlen`) is part of the operator, not an input. The
LayerNorm epsilon is the `nn.LayerNorm` default, 1e-5.

## Inputs

- `x` is standard normal. It stands for the residual stream entering the block.
- `attn_proj_weight` and `mlp_proj_weight`, the two residual projections (`c_proj`), are normal
  with std 0.02 / sqrt(2 n_layer) = 0.00408. This is minGPT's own scaled residual-projection
  init (`GPT.__init__` in `mingpt/model.py`, after GPT-2), with n_layer = 12, the GPT-2
  configuration with `n_embd = 768`.
- The other weights and all biases use `nn.Linear`'s default initialization, uniform on
  ±1/sqrt(fan_in), with one exception: `c_attn_weight` is uniform on ±0.1.
- The LayerNorm weights are uniform on (0.5, 1.5), and the LayerNorm biases are normal with std 0.1.

Four changes from the source module's draws keep the inputs representative and avoid constant
draws:

- **Residual stream.** KernelBench draws `x` with `torch.rand`, uniform on [0, 1). A residual
  stream is close to zero-mean over its features; the 0.5 offset, 1.7 standard deviations, is
  removed by both LayerNorms, and in float32 that cancellation turns the rounding of a row mean
  into an error several times the centred values' own rounding.
- **Residual projections.** With `nn.Linear`'s default init, the attention update is as large as
  the residual stream it is added to: both have an RMS of about 0.58 with the uniform `x`. The
  MLP update has an RMS of about 0.21. GPT-2, and minGPT after it, scale the residual projections
  at initialization by 1/sqrt(2 n_layer) because the updates of all the blocks accumulate along
  the residual path. With that init, the attention update has an RMS of about 0.12 and the MLP
  update about 0.08, next to 1.0 for `x`.
- **QKV weight scale.** With the default ±1/sqrt(768) bound, the attention scores have a standard
  deviation of about 0.36, so every causal softmax is nearly uniform. At ±0.1 the deviation is
  about 2.8, so the softmax is peaked, as in a trained model.
- **LayerNorm weights and biases.** The module initializes the LayerNorm weights to ones and
  their biases to zeros, which would be constant draws. (The linear biases keep `nn.Linear`'s
  init for the same reason: minGPT's GPT init zeroes them.)

The first two changes were made after the first MI355X QA audit. They let the comparison below
reject an output scaled by 16 ulps (see the comparison section).

## Workloads

- **Performance.** The five KernelBench `get_inputs` shapes are performance cases: (16, 256),
  (32, 256), (64, 512), (128, 512) and (16, 1024). They span only 16x in `batch * seq_len`, so
  four smaller shapes are added: a 128-token and a 512-token single sequence, 4 x 256 and
  2 x 1024. Together they span 512x.
- **Timing floor.** Every performance case reads the 28 MB of float32 weights and runs four
  float32 GEMMs; the 128-token case alone does about 1.8 GFLOP of matrix work. That keeps every
  performance case well above the 10 µs floor.
- **Correctness only.** These cases cover the minimum (one token, where attention returns `v`),
  the maximum batch with two tokens, unaligned and attention-block-boundary lengths (77, 129,
  1000) and the 1024-token maximum.

## Baseline

The retired task had no kernel baseline, so this one is a hand-written Triton implementation with
no upstream kernel code. It makes seven launches:

1. A row LayerNorm kernel.
2. A grouped-tile float32 GEMM with a bias epilogue, for the qkv projection.
3. A causal flash-attention kernel (64-query x 64-key blocks, head size padded to 128, online
   softmax in `exp2`). It reads q, k and v in place from the qkv rows and writes the heads side by
   side.
4. The GEMM with a residual epilogue, for the attention projection.
5. The LayerNorm again.
6. The GEMM with NewGELU in its epilogue, for `c_fc`. The epilogue computes `x / (1 + exp(-2z))`,
   which is the same function as `0.5 x (1 + tanh z)`.
7. The GEMM with a residual epilogue, for the MLP projection.

The GEMMs use 128 x 128 x 32 tiles once there are at least 512 output tiles, and 64 x 64 x 32 tiles
otherwise. Every `tl.dot` uses `input_precision="ieee"`, so float32 inputs and float32 accumulation
match PyTorch's float32 default. The intermediate buffers are allocated inside `run()`.

## Reference

The reference is float64 PyTorch, written from the module's math and independent of the baseline.
Attention runs one sequence at a time to bound the memory of the score matrix.

## Comparison: `sqnr` with `min_db: 126` (an override)

The output is float32. Its error comes from rounding in the float32 accumulations: K = 768 and
3072 in the GEMMs, up to 1024 keys in attention, and the LayerNorm sums. That is more than one
output rounding. A legitimate implementation that sums in another order has an independent error
of the same size. The default elementwise bound (`baseline_relative`, `k = 2`) therefore rejects
correct float32 implementations. At (3, 77), with the emulated baseline below, it put 0.02% to
3.6% of the elements out of bound for every correct variant in the table.

`k = 8`, the largest allowed, separates the groups at that shape: the worst correct variant
reaches 0.79 of the bound. Its zero-violation tails over the 50 M-element cases depend on
per-row error factors that the emulation cannot sample at that size. Whole-output SQNR averages
them out, and the shift it has to reject is systematic.

### Why the inputs changed

The first MI355X QA audit used KernelBench's `x`, `nn.Linear`'s projection init and a 110 dB
bound. The baseline measured 123.1-124.1 dB for one token and 119.1-120.0 dB for every longer
case. The auditor's `shifted_16_ulp` control passed. That control is the baseline's output scaled
by 1 + 16u (u = 2^-24), which scores 120.4 dB on its own.

The emulation below reproduces those hardware numbers: 123.7-124.5 dB for one token and
119.8 dB at (3, 77). It also shows why no threshold could work with those inputs:

- **Correct float32 variants:** 107 dB to 130 dB.
- **The shifted baseline:** 117 dB to 119.5 dB.

The accumulation error of a correct implementation was as large as the shift. The error of the
block's two updates was spread over an output no larger than those updates, and LayerNorm
cancelled the 0.5 offset of `x`.

With the inputs in this README, about 98% of the output's energy is the residual stream. Float32
carries it with one rounding per add. A correct implementation's error now falls about 13 dB
below the shift, and the shift stays at 120.4 dB.

### Emulation

A CPU emulation rounds every float32 operation as IEEE round-to-nearest does: an FMA rounds once,
up to a negligible double rounding. The emulated baseline matches the kernel: K-order FMA chains
of `tl.dot`, tree-reduced LayerNorm sums, an online softmax in `exp2` over 64-key blocks, and the
`exp`-form GELU. The outputs are scored as the evaluator scores them, against the reference
rounded to float32.

The draws used this task's input distributions:

- 200 draws of (1, 1).
- 3 draws each of (128, 2) and (3, 77).
- 2 draws each of (5, 129), (1, 128) and (1, 1024).
- 1 draw each of (4, 256), (1, 512) and (2, 1000).

| Implementation | SQNR |
| --- | --- |
| float32 BLAS products, online or plain softmax, `exp`- or `tanh`-form GELU | 139.3-141.5 dB |
| K-order chains of 4-product steps (fp32 MFMA-like) | 139.8-142.1 dB |
| the baseline's arithmetic (K-order float32 FMA) | 134.5-136.6 dB |
| the same with multiply-then-add | 134.2-136.5 dB |
| the same with sequential LayerNorm sums, two-pass or one-pass (E[x²] - mean²) variance | 133.7-136.6 dB |
| the same with every `exp2`, `exp`, `tanh`, division and rsqrt result off by up to 2 ulp | 134.1-136.5 dB |
| all of the worst choices at once (multiply-then-add, one-pass sequential LayerNorm, 2-ulp transcendentals, 128-key blocks) | 133.3-136.1 dB |
| `shifted_16_ulp` on the baseline's arithmetic | 120.1-120.4 dB |
| `shifted_16_ulp` on the exact reference | 120.38-120.44 dB |
| 16-bit-significand operands (a two-term bfloat16 split) in the attention products only | 115.2-119.6 dB |
| `ln_eps` ignored (at (3, 77)) | 117.1 dB |
| 16-bit-significand operands in every product | 109.7-112.6 dB |
| erf GELU | 94.6-96.8 dB |
| TF32 operands in the attention products only | 85.1-89.7 dB |
| TF32 operands in every product | 79.5-82.4 dB |
| unbiased LayerNorm variance | 74.3-78.4 dB |
| no causal mask (seq_len > 1) | 18.4-24.2 dB |

### What 126 dB accepts and rejects

- **Correct variants:** accepted by at least 7.3 dB.
- **The 16-ulp shift:** rejected by at least 5.6 dB. Because the shift is systematic, its SQNR
  stays within 0.35 dB of 120.4 dB in every draw and at every size.
- **Closest wrong implementations:** split-bfloat16 attention products are rejected by at least
  6.4 dB, and an ignored `ln_eps` by about 9 dB.
- **Gross errors:** the zero, stale, half-written and fp8-rounded controls fail by tens of dB.
  Unwritten and non-finite elements always fail.

### How strict the bound is

The new inputs make the bound stricter, not looser, on the stages it constrains:

- **Error spread over the output:** it fails above an RMS of 5.0e-7 of the output RMS (about
  1.0). Before, the limit was 3.2e-6 of the output RMS.
- **Error in the attention output `y`, seen through `attn_proj`:** the limit is an RMS of 4.4e-6
  of `y`'s RMS, against 4.6e-6 before.
- **Error in the GELU output, seen through `mlp_proj`:** 6.2e-6, against 1.3e-5 before.
- **A single wrong element:** it fails once its error passes about 1.4e-5 in the one-token case
  and 3.6e-3 in the (128, 512) case.

## Measured on MI355X (first QA audit, previous inputs)

- **Correctness.** The baseline passed every case, at 119.1-124.1 dB.
- **Run time.** It took 0.26 ms for `p_b1_t128` and 13.2 ms for `p_b128_t512`.
- **Controls.** Every control except `shifted_16_ulp` was rejected in the first case.

The inputs and the bound changed after that audit, and the timed work did not.

## Risks for the next GPU audit

- **Threshold calibration.** The 126 dB threshold was calibrated with the CPU emulation above,
  not on the GPU. With the previous inputs, the emulation of the baseline matched the MI355X SQNR
  to within about 0.6 dB. With these inputs, the baseline is expected at about 134-136 dB.
- **Register pressure.** The float32 attention tiles (64 x 128 for q, k, v and the accumulator)
  may limit occupancy. This affects run time, not correctness.

# Vision attention block, float32, 128 channels, 4 heads

Self-attention over the positions of an image feature map followed by a residual LayerNorm, the
`VisionAttention` module of KernelBench level 3 problem 31:

```python
x = x.view(B, C, H * W).permute(2, 0, 1)          # (positions, batch, channels)
attn_output, _ = nn.MultiheadAttention(128, 4)(x, x, x)
x = nn.LayerNorm(128)(attn_output + x)
return x.permute(1, 2, 0).view(B, C, H, W)
```

Every position of an image is a token of `embed_dim = 128` channels. The fused input projection
`token @ in_proj_weight^T + in_proj_bias` gives q, k and v; each of the 4 heads of 32 channels computes
`softmax(q k^T / sqrt(32)) v` over all positions of its image (no mask, no dropout: eval mode); the
concatenated heads pass through the output projection; the residual token is added and the sum is
layer-normalized over the channels (biased variance, `epsilon` inside the square root, then
`norm_weight` and `norm_bias`). The module's parameters are explicit inputs.

## Scenario

- float32 throughout, `embed_dim = 128`, `num_heads = 4` (KernelBench `get_init_inputs`).
- Layout: the input and the output keep the module's channel-major image layout. The task flattens
  `H * W` into one axis, so `x` and `out` are `[batch, embed_dim, hw]` (the module's `(B, C, H, W)`
  tensors viewed as `(B, C, H * W)`; only the product `H * W` enters the computation). An
  implementation must read the tokens transposed and write the normalized tokens back transposed.
- Var axes: `batch` (1 to 16 images) and `hw` (49 to 16384 positions, 7x7 to 128x128 feature maps).

## Workloads

The performance cases are the old task's escalation of the KernelBench shapes (batch 2, 4 and 8 at
32x32; batch 2 at 64x64; batch 2 at 128x128, the KernelBench default), plus batch 1 at 32x32, batch 4
at 64x64 and batch 8 at 128x128 so that the var-axis size spans 128x. The cost grows with
`batch * hw^2` (about 0.7 GFLOP for batch 1 at 32x32, 1.1 TFLOP for batch 8 at 128x128). The other
feature maps are correctness-only: 7x7, 9x9, 14x14 and 16x16 (batch 1 to 16) are estimated to run near
or below the 10 us timing floor, and 33x33 (batch 5) is a boundary case rather than a production
shape. Together they cover padded query and key tiles (with the baseline's 64-position tiles, 1089
positions leave a last key block of one position), odd batches and position counts that are not a
multiple of 16.

Input distributions: `x` is zero-mean, unit-variance activations (KernelBench drew `torch.rand`). The
projection weights are scaled so that the attention scores have a standard deviation of about 2 (a
non-trivial softmax; the default Xavier initialization with uniform inputs gives an almost uniform
attention), and every bias and the LayerNorm affine parameters are random and nonzero, so a dropped
bias, scale or affine term is detected (the module initializes the biases to zero).

## Baseline

There is no production kernel to vendor: the old task's baseline was the PyTorch module itself
(library GEMMs and attention). `baseline/kernel.py` is a hand-written Triton implementation with
three launches:

1. the q/k/v input projection with its bias, reading the channel-major image directly and writing
   token-major `[batch, hw, 3 * 128]` rows;
2. non-causal flash attention (online softmax in base 2) per query block of 64, image and head,
   writing the heads side by side;
3. the output projection with its bias, the residual add and the LayerNorm in the epilogue (one
   program holds the full 128 channels of 32 tokens), storing channel-major into `out`.

All matrix products use IEEE float32 inputs with float32 accumulation (`input_precision="ieee"`),
matching the module's float32 semantics. Single-pass TF32 or bfloat16 products and the three-product
bfloat16 split (bf16x3) fall below the bound; splits that keep float32 accuracy (tf32x3, bf16x6) stay
within it (see the comparison bound, including its one exception: bf16x3 in a single product on large
maps).

## Comparison bound

`out` uses `sqnr` with `min_db: 125` instead of the default `baseline_relative`: the signal-to-noise
ratio of the whole output against the float64 reference must be at least 125 dB, an error RMS of at
most 5.6e-7 of the output RMS (9.4 times the float32 unit roundoff 2^-24). The baseline itself must
meet the same threshold.

**Why not `baseline_relative`.** Besides its elementwise limit, that bound requires the mean error over
all n elements to stay within 5/sqrt(n) of the error RMS, or within `k` times the baseline's mean
error. The test assumes independent element errors, which a float32 implementation of this operator
does not have: it adds per-channel constants (`in_proj_bias`, `out_proj_bias`, `norm_bias`) to values
already on the float32 grid, and the rounding of such a sum is set by the low bits of the constant,
the same for every position of that channel. The output error therefore carries a per-channel
component that does not average out over positions, and its size depends on where an implementation
rounds, for example whether the LayerNorm affine `x * norm_weight + norm_bias` is one FMA (Triton
contracts it by default, so the baseline has no such component from `norm_bias`) or a multiply and an
add. On 128x128 maps (4.2M and 16.8M elements) the limit is 0.0024 and 0.0012 of the RMS, and correct
float32 implementations exceed it: in the MI355X QA audit the float32 PyTorch evaluation of the
reference had a mean error of 0.0073 of its RMS at 2 x 128x128 and failed, with no element outside
the elementwise limit. A CPU study reproduced it: against a float32 emulation of the baseline with the
fused affine, the same float32 PyTorch evaluation and an otherwise identical emulation with a separate
multiply and add each failed one of four draws of 2 x 128x128, and on one draw of 8 x 128x128 their
mean error was 3.2 to 3.3 times the limit, passing only through the k-times-baseline fallback. Neither
`k` (8 is already the largest allowed) nor `max_violation_fraction` changes this test.

**The threshold.** A CPU study on the evaluator's own input generator over all 13 workload shapes (1
to 4 draws each) measured the SQNR the evaluator computes. Valid float32 implementations reached 129.6
to 138.7 dB: float32 emulations of the baseline with BLAS products and with MFMA-like K-step
accumulation (with and without the fused affine), the PyTorch module (`F.linear`, the math SDPA,
`F.layer_norm`), the float32 evaluation of the reference, key blocks of 32 and 128, natural-base
exponentials, correctly rounded q/k/v, a variant whose exponentials, reciprocals and reciprocal square
root all err by +1 ulp (the 129.6 dB minimum, at 64x64), and the tf32x3 and bf16x6 split products. The
baseline emulations rise from about 130 dB at 7x7 to 136 dB at 128x128 with BLAS products and reach
134 to 138 dB with MFMA-like accumulation. The 16-ulp relative shift control is at 120.0 to 120.3 dB in
every shape. 125 dB lies midway: 4.6 dB below the noisiest valid implementation and 4.7 dB above the
control.

**What it rejects.** Single-pass TF32 products (67 to 74 dB), an epsilon of 1e-6 instead of 1e-5 (107
to 110 dB), a 1e-4 relative error in the attention scale (85 to 87 dB) and the bf16x3 split in all four
products (104 to 110 dB), in the input projection only (106 to 113 dB) or in the scores only (108 to
115 dB); float8 rounding and unwritten, zeroed, half-written or stale outputs are far below. The bf16x3
split in the `p @ v` product alone is the closest wrong implementation: 116 to 123 dB up to 33x33
maps, so a candidate that uses it on every shape fails each correctness run, but about 126 dB at 64x64
and 129.4 to 129.8 dB at 128x128, inside the range of valid float32 noise, so one that used it only on
large maps would pass (the former elementwise bound rejected it at 64x64 and in three of four draws at
128x128). As a whole-output measure the bound also accepts an error confined to a few positions while
its energy stays below the threshold (for example 5e-5 of the output RMS on one position in 16384).
`hw` starts at 49, the smallest feature map of the workloads (valid implementations also reach 129 to
137 dB on 4x4 maps, outside the domain).

# minGPT causal self-attention, float32, embedding 768, 8 heads

The whole forward of minGPT's `CausalSelfAttention` layer in eval mode (dropout off), in float32.
`qkv = x c_attn_weight^T + c_attn_bias` is split into `q`, `k` and `v` along the last axis, each viewed as
8 heads of size 96 (head `h` owns columns `[96h, 96h + 96)`). Per batch item and head,
`att = softmax(q k^T / sqrt(96))` with every key after the query masked out, `y = att v`. The heads are
concatenated side by side and projected: `out = y c_proj_weight^T + c_proj_bias`. The weights use the
`nn.Linear` layout (`[out_features, in_features]`).

**Origin.** KernelBench level 3, problem 43 (`MinGPTCausalAttention`, taken from karpathy/minGPT
`mingpt/model.py`). It replaces the retired `torch2hip/kernelbench/level3/l3n43_MinGPTCausalAttention`
task, whose only baseline was the PyTorch module. The scenario keeps that module's configuration:
`n_embd = 768`, `n_head = 8` (head size 96), `max_seqlen = 1024`, float32 throughout, zero dropout.

**Workloads.** The five KernelBench `get_inputs` shapes, (16, 256), (32, 256), (64, 512), (128, 512) and
(16, 1024), are performance cases. They span only 16x in `batch * seq_len`, so three smaller shapes are added:
a single 512-token sequence, the module's default length, plus 4 x 256 and 2 x 1024. Together they span 128x.
Every performance case runs the two K = 768 float32 GEMMs. Even the single 512-token sequence does about
2.8 GFLOP of float32 matrix work on 72 + 24 output tiles of 128 x 128, which keeps it well above the
10 µs floor. The correctness-only cases cover the minimum (one token, where attention returns `v`),
the maximum batch, unaligned and block-boundary lengths (77, 129, 1000) and the module's 1024-token
maximum. As in KernelBench, `x` is uniform on [0, 1). The weights and biases follow `nn.Linear`'s default
initialization, uniform on ±1/sqrt(768).

**Baseline.** The retired task had no kernel baseline, so this one is a hand-written Triton implementation
with three launches. The first is a grouped-tile float32 GEMM with bias for the qkv projection
(128 x 128 x 32 tiles). The second is a causal flash-attention kernel (64-query x 64-key blocks, head size
padded to 128, online softmax in `exp2`) that reads q, k and v in place from the qkv rows and writes the
heads side by side into a `[tokens, 768]` buffer. The third is the same GEMM for the output projection.
Every `tl.dot` uses `input_precision="ieee"` (float32 inputs, float32 accumulation). The qkv and head
buffers are allocated inside `run()`.

**Comparison: `baseline_relative` with `k: 8`, `max_violation_fraction: 0.001` (an override).** With float32
outputs, the baseline's error is dominated by the rounding of its K = 768 GEMM accumulations, not by the
output rounding. Any other summation order produces an independent error of the same size. The default bound
(`k = 2`, no violations) therefore rejects correct float32 implementations. In a CPU simulation against the
float64 reference, at (1, 1024), (4, 256), (8, 77) and (2, 256), it rejected every reordering tried:
0.5% to 14% of elements out of bound, for other GEMM K-tile sizes (16, 64, 128), a 4-way split-K, other
attention block sizes, `exp` instead of `exp2`, and PyTorch's own float32 module. With `k = 8` all of them
had at most about 0.01% of elements out of bound (peak 1.4x the bound) and passed with a 0.1% allowance. The same
bound still rejects every precision downgrade and control tried, each with at least 7.8% of elements out of
bound: TF32 inputs in all products or only in the attention products, bf16x3 split products, bfloat16
products, the 16-ulp output shift and the fp8 output rounding. The small correctness cases keep
sequence-boundary errors visible: in `c_b3_t77`, one wrong row per sequence is 1.3% of the elements.

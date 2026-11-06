# Block-scaled FP8 GEMM (preshuffled weight), N = 4096, K = 256

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [m, 256] with one float32 scale per 1x128 block
(`a_scale`, 2 per row) times float8_e4m3fn weights `b` [4096, 256] with one float32 scale per 128x128 block
(`b_scale` [32, 2]). Each 128-wide K block of the product is accumulated in float32, multiplied by the activation
and weight block scales and summed; the result is rounded to bfloat16. This is the DeepSeek-style block-quantized
linear layer served by AITER's `gemm_a8w8_blockscale_bpreshuffle`.

**Kernel-facing layout.** The framework hands the baseline and the candidate the layouts production uses (`layout` in
`task.yaml`, outside timing); the reference sees the logical tensors.
- `b` is the weight in AITER's `shuffle_weight(layout=(16, 16))` encoding, kept in its [4096, 256] shape: 16 (n) x 32 (k)
  tiles, each stored as [k half (2)][n (16)][k within half (16)]. An implementation may view it as [256, 4096].
- `a_scale` is the column-major payload of the activation scales: the logical [m, 2] scale matrix is stored
  k-block-major (element `[k_block * m + row]`) in an [m, 2]-shaped tensor, which is what the upstream quantizing
  kernel emits and what AITER's CK and Triton kernels read. Activation quantization itself is not part of this task.

**Inputs.** `inputs.py` draws standard normal activations and weights and quantizes them with scale `amax / 448` per
block, so codes and scales are consistent; weights and activations are not rescaled further. The reference
dequantizes both operands and multiplies in float64.

**Workloads.** N = 4096 is the hidden width of a DeepSeek-style model; M is the token count per call. Decode-sized
and unaligned counts (1, 2, 4, 8, 16, 17, 32, 64, 100, 128, 256, 512, 1000, 2048; every power of two of the old tasks covers one tile bin of the tuned table) are correctness-only; performance cases span 4096 to 409600 tokens (100x). The shape is output-write bound: each token writes 8 KiB of bf16 while K = 256 is a single pair of K blocks, so the 10 us floor needs about 4096 tokens and the performance range is large-batch prefill. The two old tasks (a plain and a logically-strided activation-scale encoding of the same operator) are merged; they differ only in how `a_scale` is stored, and this task keeps the production column-major payload.
The largest (409600-token) case needs a float64 reference output of about 13 GiB; the GPU audit should confirm it fits next to the baseline and candidate outputs.

**Baseline.** The production function `gemm_a8w8_blockscale_bpreshuffle` is CK/assembly and cannot be vendored, so the
baseline is AITER's own Triton preshuffle variant, `_gemm_a8w8_blockscale_preshuffle_kernel` with the launcher
`gemm_a8w8_blockscale_preshuffle` (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). It is
a Triton replacement for the CK/assembly kernel and may be slower than it. The tuned tile table is the generic gfx950 preshuffled table (no N=4096, K=256 specialization exists); every
row has NUM_KSPLIT = 1, so the split-K reduction kernel is not included. The old assembly row with split-K was
nondeterministic; it does not apply to this baseline. The kernel and launch code are unchanged apart from inlining the
pid and kernel-name helpers, the table lookup and the viewing of `b` as [N / 16, K * 16].

**GPU audit risks.** Timing is tight: the smallest performance case (4096 tokens) is estimated at about 11 us from the output write alone. The tuned table rows for M <= 8 use 4- or 8-row tiles (below the 16-row MFMA tile); they are only reached by the decode-sized correctness cases and the audit should confirm the baseline compiles for them. The activation-scale payload read by the kernel (`a_scale.stride(1)` / `numel // stride(0)`) relies on `a_scale` being contiguous.

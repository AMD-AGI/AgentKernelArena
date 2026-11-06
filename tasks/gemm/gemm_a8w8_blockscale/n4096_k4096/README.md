# Block-scaled FP8 GEMM (preshuffled weight), N = 4096, K = 4096

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [m, 4096] with one float32 scale per 1x128 block
(`a_scale`, 32 per row) times float8_e4m3fn weights `b` [4096, 4096] with one float32 scale per 128x128 block
(`b_scale` [32, 32]). Each 128-wide K block of the product is accumulated in float32, multiplied by the activation
and weight block scales and summed; the result is rounded to bfloat16. This is the DeepSeek-style block-quantized
linear layer served by AITER's `gemm_a8w8_blockscale_bpreshuffle`.

**Kernel-facing layout.** The framework hands the baseline and the candidate the layouts production uses (`layout` in
`task.yaml`, outside timing); the reference sees the logical tensors.
- `b` is the weight in AITER's `shuffle_weight(layout=(16, 16))` encoding, kept in its [4096, 4096] shape: 16 (n) x 32 (k)
  tiles, each stored as [k half (2)][n (16)][k within half (16)]. An implementation may view it as [256, 65536].
- `a_scale` is the column-major payload of the activation scales: the logical [m, 32] scale matrix is stored
  k-block-major (element `[k_block * m + row]`) in an [m, 32]-shaped tensor, which is what the upstream quantizing
  kernel emits and what AITER's CK and Triton kernels read. Activation quantization itself is not part of this task.

**Inputs.** `inputs.py` draws standard normal activations and weights and quantizes them with scale `amax / 448` per
block, so codes and scales are consistent; weights and activations are not rescaled further. The reference
dequantizes both operands and multiplies in float64.

**Workloads.** N = 4096 is the hidden width of a DeepSeek-style model; M is the token count per call. Decode-sized
and unaligned counts (1, 2, 4, 8, 16, 17, 32, 64, 100, 128, 129, 256, 1000; the powers of two cover the tile bins of the tuned table) are correctness-only; performance cases span 512 to 65536 tokens (128x). This shape is compute-bound for prefill: 512 tokens already exceed the 10 us floor (about 17 GFLOP). The old CK/assembly task and the old in-task AITER Triton task (non-preshuffled weights, 4096^3) are merged into the preshuffled scenario, the layout the CK function consumes.
The largest case needs a float64 reference output of 2 GiB; the GPU audit should confirm it fits next to the baseline and candidate outputs.

**Baseline.** The production function `gemm_a8w8_blockscale_bpreshuffle` is CK/assembly and cannot be vendored, so the
baseline is AITER's own Triton preshuffle variant, `_gemm_a8w8_blockscale_preshuffle_kernel` with the launcher
`gemm_a8w8_blockscale_preshuffle` (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). It is
a Triton replacement for the CK/assembly kernel and may be slower than it. The tuned tile table is the generic gfx950 preshuffled table (no N=4096, K=4096 specialization exists); every
row has NUM_KSPLIT = 1, so the split-K reduction kernel is not included. The old assembly row with split-K was
nondeterministic; it does not apply to this baseline. The kernel and launch code are unchanged apart from inlining the
pid and kernel-name helpers, the table lookup and the viewing of `b` as [N / 16, K * 16].

**GPU audit risks.** The smallest performance case (512 tokens) is estimated at about 12 us. The tuned table rows for M <= 8 use 4- or 8-row tiles (below the 16-row MFMA tile); they are only reached by the decode-sized correctness cases and the audit should confirm the baseline compiles for them. The activation-scale payload read by the kernel (`a_scale.stride(1)` / `numel // stride(0)`) relies on `a_scale` being contiguous.

# a8w8 block-scaled GEMM, N = 7168, K = 2048

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [M, 2048] with one float32 scale per 1x128 group
(`a_scale` [M, 16]) times float8_e4m3fn weights `b` [7168, 2048] with one float32 scale per 128x128 block
(`b_scale` [56, 16]). Each K block is accumulated in float32, multiplied by the activation and weight scale of its
block and summed; the result is rounded to bfloat16. The shape is the down / output projection of a DeepSeek-V3
class model (7168 hidden, 2048 intermediate), present in AITER's tuned block-scaled GEMM table.

**Inputs.** `inputs.py` plays the role of the producers. Activations are normal values with a per-(row, 128-column)
magnitude drawn over a 16x range, quantized to e4m3fn with scale `amax / 448`; weights are the same per 128x128 block.
The kernel-facing weight is preshuffled once with the AITER `shuffle_weight(layout=(16, 16))` permutation
(`to_kernel_layout`, outside timing, as production does at weight-load time), and `a_scale` is the column-major
payload of the activation-quantization kernel (a `[K/128, M]` buffer presented as `[M, K/128]`): activation
scales are produced by the upstream quantization kernel in that layout, so it is the kernel-facing input. The
reference receives the logical operands (natural weight order, `a_scale[row, group]`) and dequantizes and
multiplies in float64, chunked over rows.

**Workloads.** Four old tasks contain this operator: a FlyDSL preshuffle blockscale GEMM (square shapes only), a
PyTorch-oracle FlyDSL task and a reference-kernel competition task (both with M = 16 to 128 at N = 7168, K = 2048),
and a vLLM Triton `w8a8_block_fp8_matmul` task (toy shapes). N = 7168, K = 2048 is the real shape they share; M of
16, 64, 96 and 128 come from the competition and tuned-GEMM-table shapes. These decode-sized batches run below
the 10 us floor (the 15 MB weight read is about 3.3 us) and are correctness-only; performance cases run from 512
to 65536 tokens (128x), which are prefill chunks and long prefills.

**Baseline.** AITER's Triton `gemm_a8w8_blockscale_preshuffle` (`_gemm_a8w8_blockscale_preshuffle_kernel`, see
`provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). The plan for this task named the FlyDSL
blockscale preshuffle kernel of the old FlyDSL task as baseline; this task instead vendors the AITER Triton kernel
with its gfx950 table tuned for exactly N = 7168, K = 2048 (`gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2048.json`,
all rows without split-K). It is the production Triton implementation of the same operator with the same
operand layout (column-major activation scales, 16x16-preshuffled weights), is a drop-in under the Triton
candidate rules and does not depend on the FlyDSL runtime's `buffer_ops` module, which the runtime image
lacks. The FlyDSL kernel remains reachable as a candidate language. Upstream lowers `num_stages` from 3 to 2 on
gfx950 for Triton releases before 3.7 (a compiler bug in the AMD buffer-ops conversion); the baseline keeps that
guard. The upstream kernel calls `remap_xcd(pid, GRID_MN)` without using the result (a no-op); it is kept unchanged.

**GPU audit risks.** The smallest performance case (512 tokens) is estimated at about 10-12 us and may be dropped
if the baseline measures under 10 us; the 65536-token case writes 0.94 GB of bfloat16 and compares against a
3.8 GB float64 reference.

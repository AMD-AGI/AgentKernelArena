# Fused block-scaled fp8 GEMM + mul-add, DeepSeek-V3 shared-expert down projection (N = 7168, K = 256)

`out = a * b + dequant(x) @ dequant(w)^T`: a block-scaled fp8 GEMM (float8_e4m3fn activations `x` [m, 256] with one float32
scale per 1x128 block, weights `w` [7168, 256] with one float32 scale per 128x128 block) whose epilogue multiplies the
tensor `b` [m, 7168] by the scalar `a` and adds it to the accumulator, with a single rounding to bfloat16. This is AITER's
`fused_gemm_a8w8_blockscale_mul_add` with `fuse_type=1` (`a * b + Y`).

**Production use.** In a DeepSeek MoE layer the shared expert output `Y` (the down projection of the shared expert) is summed
with the routed experts' output scaled by `routed_scaling_factor`: SGLang's `python/sglang/srt/models/deepseek_v2.py`
(pinned commit) does `x = shared_output; x.add_(final_hidden_states, alpha=self.routed_scaling_factor)`. With `Y` as the
GEMM, `a` = `routed_scaling_factor` and `b` = the routed output, this is exactly `fuse_type=1`. AITER's own tests and benchmark
for this kernel (`op_tests/triton_tests/gemm/fused/test_fused_gemm_a8w8_blockscale_mul_add.py`) use N = 7168, K = 256 and do not
name a model; the pinned SGLang and vLLM trees do not call the fused operator, so the correspondence to the layer is by shape and
semantics.

**Scenario and config citations.** DeepSeek-V3 `config.json` (Hugging Face; DeepSeek-R1 has the same architecture fields):
`hidden_size` 7168 (N), `moe_intermediate_size` 2048 with `n_shared_experts` 1 (shared expert width 2048), `routed_scaling_factor`
2.5 (the scalar `a`, the task's default) and `quantization_config.weight_block_size` [128, 128]. Under tensor parallelism 8 the
shared expert width per rank is 2048 / 8 = 256, which is K of the down projection. TP = 8 is a deployment choice, not a
config field; it matches the (N, K) = (7168, 256) that AITER's tests, benchmark and tuned files use.

**Reference.** `reference.py` dequantizes `x` and `w` from codes and scales, multiplies in float64 (row-chunked) and returns
`a * b + product` in float64.

**Inputs.** `x` is a standard normal activation quantized in 1x128 blocks (scale = block amax / 448); `w` is a normal weight with
std 0.06 quantized in 128x128 blocks with the same rule, which makes the product (std about 1) comparable to `a * b`
(`b` is normal with std 0.4, so `a * b` has std 1.0); neither term dominates the bfloat16 output rounding. The magnitudes are
chosen for sensitivity, they are not measured from the model. `a` is the scalar 2.5.

**Workloads.** M is the token count. M >= 2 because the upstream test notes that M = 1 fails to compile under Triton without an
environment workaround ("TODO M = 1 triton upstream compilation error without AMDGCN_USE_BUFFER_OPS=0"); the axis minimum is
therefore 2. The upstream note suggests that nearby decode tile rows (BLOCK_SIZE_M 4 at M <= 8) could also be affected; M = 2 and 3 exercise that row. Decode batches and chunks up to 1000 tokens are correctness-only; performance cases are prefill token counts from
1024 to 131072 (128x). The kernel is bandwidth-bound: per token it reads `b` (14 KB) and writes `out` (14 KB). Estimated baseline
time (bytes / 4.5 TB/s + fp8 flops at 2 PFLOP/s + 4 us): M = 1024 about 13 us, about 1.3x above the 10 us floor (31 MB), M = 131072 about 1.7 ms (3.8 GB).

**Baseline.** AITER's Triton `_fused_gemm_a8w8_blockscale_mul_add_kernel` with its reduce kernel, the launcher logic and the table
inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT license). For `fuse_type=1` on gfx950 upstream
selects the base GEMM-A8W8_BLOCKSCALE family (the dedicated mul_add tile is miscompiled for `fuse_type=1` on Triton 3.8), so the
baseline uses `configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json`; every row has NUM_KSPLIT 1, so the reduce kernel is
never launched here and there is no split-K nondeterminism. The kernels are unchanged. The upstream files carry the SPDX header
`MIT` and `Copyright (C) 2024-2026, Advanced Micro Devices, Inc.`, kept verbatim; no third-party origin is noted in any vendored
file or in its history.

**Scalar argument.** `a` is passed as a Python float, which the launcher treats as a scalar operand (`IS_A_SCALAR`); `b` is a
contiguous [m, 7168] tensor (`IS_B_TENSOR`).

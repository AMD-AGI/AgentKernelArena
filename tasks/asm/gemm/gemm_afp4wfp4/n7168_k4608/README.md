# afp4wfp4 GEMM (AITER asm baseline), N = 7168, K = 4608

`out = dequant(a) @ dequant(w)^T`: MXFP4 activations `a` [M, 4608] and weights `w` [7168, 4608], both stored as two
e2m1 codes per byte (the even K index is the low nibble) with uint8 e8m0 scales (`2^(s - 127)`), one per 32 values of
K, accumulated in float32 and rounded to bfloat16. `m` is the only varying axis. The operator, the quantizer and the
scenario are those of the Triton and FlyDSL tasks of this operator (`tasks/triton/gemm/gemm_afp4wfp4/n7168_k4608`,
`tasks/flydsl/gemm/gemm_afp4wfp4/n7168_k4608`); the kernel-facing layout, the workloads and the baseline differ.

**Reference.** `reference.py` undoes the activation-scale shuffle, dequantizes both operands from codes and scales and
multiplies in float64, row-chunked. It is independent of the baseline.

## Scenario and workload origin

N = 7168, K = 4608 is the dense-MLP `down_proj` of DeepSeek-R1 at tensor parallelism 4: the first
`first_k_dense_replace` = 3 layers are dense MLPs with `intermediate_size` 18432, so each rank contracts
18432 / 4 = 4608 intermediate columns back to `hidden_size` 7168. In AMD's MXFP4 checkpoint of the model
(`amd/DeepSeek-R1-0528-MXFP4`, Quark) these MLP weights are MXFP4 with dynamic MXFP4 activations (group 32, e8m0
scales, scale mode "even"), while attention projections are excluded from quantization.

AITER ships a tuned table for this model, `aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv` (merged
into the a4w4 tuned table that `gemm_a4w4` reads). For gfx950 / 256 CUs it has 15 rows for N = 7168, K = 4608, M = 1,
2, 4, ..., 16384, and **every row is an asm kernel** (`f4gemm_bf16_per1x32Fp4_BpreShuffle_*`, splitK 0); the same holds
for the sibling `gate_up` shape N = 9216, K = 7168. The task covers that tuned range, `m` in [1, 16384].

Production evidence is the tuned table only: no production profile of an MXFP4 model was taken for this task.
SGLang's LLM path for Quark MXFP4 linears (`quark_w4a4_mxfp4.py` in the pinned image) calls AITER's Triton
`gemm_afp4wfp4`, not this asm path; `gemm_a4w4` with these kernels is AITER's own tuned MXFP4 GEMM entry point (called,
for example, by SGLang's diffusion-model MXFP4 linear in the same image).

Every call reads the 16.5 MB weight, so even a single-token decode takes the baseline about 15 us in the cold-cache
measurement (floor 10 us). Performance cases are the 15 tuned M values (16384x span; measured baseline medians: 15.0 us
at M = 1 to 16, 16.5 us at M = 32, 17.8 at 256, 22.2 at 512, 30.5 at 1024, 41.8 at 2048, 75.7 at 4096, 144 at 8192,
249 at 16384). Correctness-only cases add unaligned M and M between rows, reaching every kernel through the padded-M
lookup: 3, 17, 33, 100, 129, 300, 777, 1500, 3000, 5133, 12345, 16300. Boundary cases: `m = 1` (minimum), `m = 3`
(not a power of two), `m = 17` (not a multiple of 16, above 16).

## Interface and layout

- **`a`** [m, 2304] uint8: packed activation codes, row-major, exactly as the quantizer writes them.
- **`a_scales`** [ceil(m / 256) * 256, 144] uint8, **in the layout the asm kernel reads**: AITER's activation quantizer
  `dynamic_mxfp4_quant(x, shuffle=True)` (`aiter/utility/fp4_utils.py`; `aiter.get_triton_quant(per_1x32)`, used by
  AITER's own a4w4 test and by callers of `gemm_a4w4`) allocates the scales with the rows padded to a multiple of 256
  and stores them permuted as `e8m0_shuffle` (`aiter/ops/shuffle.py::shuffle_scale`): the [rows, 144] matrix viewed as
  (rows / 32, 2, 16, 144 / 8, 2, 4) and permuted to (rows / 32, 144 / 8, 4, 16, 2, 2). The quantizer writes 0 into the
  padding rows below the next multiple of 32 and leaves the rest uninitialized; the task's generator (`inputs.py`)
  produces the shuffled layout directly with every padding row 0. Nothing on the activation side is transformed
  outside timing: the generator is the producer, as in the FP8 asm sibling
  (`tasks/asm/gemm/gemm_a8w8_blockscale/n4608_k7168`, whose `a_scale` is generated k-group-major). This differs from
  the Triton sibling, whose kernel reads row-major `[m, 144]` scales, and from the FlyDSL sibling, which pads to 32
  rows and shuffles `a_scales` in its `layout` function; the logical values are the same in all three. The padding
  content does not affect the result: with random bytes in every padding row the baseline output was bitwise
  unchanged (M = 1, 17, 100, 300, 777, 1500, 5133, 16300).
- **`w`** [7168, 2304] and **`w_scales`** [7168, 144] are the logical weight and its scales; the task's `layout`
  function applies the load-time preparation AITER uses: `shuffle_weight(w, layout=(16, 16))` (for 1-byte elements:
  view [n / 16, 16, 2304 / 32, 2, 16], permute to [n / 16, 2304 / 32, 2, 16, 16]) and `e8m0_shuffle(w_scales)`
  (7168 is a multiple of 256, so no padding rows). The reference receives the logical weight.
- **Inputs.** Normal activations (std 1) and weights (std 0.05) quantized like `dynamic_mxfp4_quant` (verified on the
  pinned image: codes and scales bitwise equal to AITER's for the same bfloat16 input; the generator's shuffle bitwise
  equal to AITER's `shuffle_scale` and to the scales `dynamic_mxfp4_quant(shuffle=True)` writes; the weight layout
  bitwise equal to `shuffle_weight` and `e8m0_shuffle`). In production the activation quantization is a separate
  upstream kernel, so it is part of the input, not timed.

## Baseline and dispatch

`baseline/` vendors six gfx950 code objects byte for byte from `hsa/gfx950/f4gemm/` at `ROCm/aiter` 4ad99832 (the
version in the pinned image; hashes in `task.yaml`) and translates the host side to Python (`baseline/kernel.py`).
`baseline/LICENSE` is AITER's MIT license. Per call:

1. **Kernel.** AITER's tuned lookup `get_GEMM_config` (`aiter/ops/gemm_op_a4w4.py`): the row for exact M, else
   `getPaddedM(M, gl=0)` (16 for M <= 256, 32 for M <= 1024, 64 for M <= 4096, else 128), else `gl=1` (next power of
   two, or 8192 for M > 8192 since N > 4096). The 15 rows inlined are the gfx950 / cu_num 256 rows for this N and K.
   Resulting kernels (tile M x N): M 1-16 32x128, 17-32 64x128, 33-128 32x128, 129-256 64x128, 257-512 128x128,
   513-1024 256x128, 1025-4096 256x256, 4097-16256 128x512, 16257-16384 256x256 (fine padding hits the 16384 row).
2. **Launch** (`csrc/py_itfs_cu/asm_gemm_a4w4.cu::gemm_a4w4_asm`, the gfx950 launcher; `asm_f4gemm.cu` is the
   gfx1250-only persistent path and does not apply): the packed `KernelArgs` (pointers D, C = null, A, B; alpha 1,
   beta 0; strides; M, N, K; ScaleA, ScaleB pointers and their row strides; `log2_k_split`; every field padded to 16
   bytes, 384 bytes as the code objects declare), grid `(ceil(N / tile_n), ceil(M / tile_m), 1)` of 256-thread
   workgroups, no dynamic LDS. Every tuned row has splitK 0, so `log2_k_split` is 0, the grid has one z slice and the
   launcher's output memset never runs. Fields the C++ launcher leaves unset (stride_D0/1, stride_C1, stride_A1,
   stride_B1, the scale `*_1` strides) are written as 0. Strides follow the launcher: `stride_C0` = output row stride
   in elements, `stride_A0`/`stride_B0` = row stride in fp4 elements, scale strides in bytes.
3. **Output.** `gemm_a4w4` allocates its output with M rounded up to 32 rows and returns the first M rows. The kernels
   store only rows below M (verified with 256 guard rows after the output for 1528 M values), so the baseline writes
   the task's `[m, n]` output directly, without the padded buffer.

**Fidelity** (scratch script on MI355X, pinned image, identical inputs, `import aiter` only there): for 1528 M values
(every M from 1 to 1100, every 37th up to 16384, and the edges 2047-2049, 4095-4097, 8191-8193, 16255-16257, 16383,
16384) the baseline's kernel choice equals `get_GEMM_config`'s and its output is bitwise identical to the tuned
dispatcher `aiter.gemm_a4w4(..., bpreshuffle=True)`; for 67 of them a PyTorch profiler trace shows the same single
kernel launch. Cold-cache CUDA-graph medians (`benchmark_cuda_graph_cold_cache_samples`, baseline / AITER) agree
within 0.5%: M = 1 14.76 / 14.72 us, M = 32 15.92 / 15.80, M = 256 17.12 / 17.12, M = 1024 29.28 / 29.36,
M = 4096 74.18 / 74.04, M = 8192 141.2 / 141.4, M = 16384 246.8 / 245.9 (all 15 tuned M measured).

## Notes

- The kernels are deterministic (no split-K); the bound is the default `baseline_relative`.
- The dispatch is specific to MI355X (256 CUs, gfx950): the tuned rows exist only for that configuration, and the code
  objects are gfx950 code objects.
- A candidate in the `asm` target language may keep loading these code objects byte for byte and change the dispatch,
  or write new kernels as `.s` source.

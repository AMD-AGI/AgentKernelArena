# a8w8 block-scaled GEMM (AITER asm baseline), N = 4608, K = 7168

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [M, 7168] with one float32 scale per 1x128 group
(`a_scale` [56, M], k-group-major) times float8_e4m3fn weights `b` [4608, 7168] with one float32 scale per 128x128
block (`b_scale` [36, 56]), accumulated in float32 and rounded to bfloat16. The reference (`reference.py`) dequantizes
both logical operands and multiplies in float64, independent of the baseline.

## Scenario and workload origin

N = 4608, K = 7168 is DeepSeek-R1's (and V3's) dense-MLP `gate_up_proj` at tensor parallelism 8: the first
`first_k_dense_replace` = 3 layers are dense MLPs with `intermediate_size` 18432, so the merged gate and up projection
has 2 x 18432 / 8 = 4608 output channels per rank over `hidden_size` 7168. The model is FP8 with
`weight_block_size` [128, 128] and dynamic 1x128 activation groups.

AITER ships a tuned table for this shape: `aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv`
(merged into AITER's default `a8w8_blockscale_bpreshuffle_tuned_gemm` lookup). Its gfx950 / 256-CU rows choose the
asm kernels `fp8gemm_bf16_blockscale_BpreShuffle_{32,48,64,80,96,128}x128` for M <= 864 except three M buckets
(192, 224, 832, which choose a Composable Kernel instance), and CK for every M > 864. The task covers that asm range:
`m` in [1, 864].

**This is AITER's tuned production asm path, not what SGLang runs for this model.** SGLang 0.5.20 (the pinned image)
hard-routes DeepSeek-R1's dense FP8 GEMMs to AITER's Triton `gemm_a8w8_blockscale`
(`use_aiter_triton_gemm_w8a8_tuned_gfx950` in `sglang/srt/layers/quantization/fp8_utils.py`); a profile of DeepSeek-R1
TP8 serving on MI355X never launched a `fp8gemm_bf16_blockscale_BpreShuffle_*` kernel. The asm kernels run when a
framework calls AITER's `gemm_a8w8_blockscale_bpreshuffle` (the bpreshuffle dispatcher, as GLM and Qwen3 FP8 serving
in SGLang do) on a shape whose tuned row is asm, which is the case here.

`m` is the token count. Every call reads the 33 MB weight from HBM, so even single-token decode takes the baseline
about 18 us under the cold-cache measurement (the floor is 10 us): no decode batch had to be made correctness-only.
Performance cases are the decode batches 1, 2, 4, ..., 512 and the axis maximum 864 (11 cases, 864x span).
Correctness cases add odd batches (3, 5, 15, 17, 24, ...), M values that reach the tuned table through padded-M
lookup (3, 5, 15, 17, 100, 129, 300, 400, 700, 863), at least one M for every (kernel, splitK) class in the table, and
one M in each of the three CK-tuned buckets (180, 220, 813). Boundary cases: `m = 1` (minimum), `m = 3`
(not a power of two), `m = 17, 100, 129, 813` (not multiples of 16, above 16).

## Interface and layout

- **Activation scales `a_scale` [k / 128, m].** The asm kernel reads the activation scale of row `r`, group `g` at
  `a_scale[g * M + r]`, i.e. k-group-major. That is the layout the production producer writes: AITER's per-1x128
  quantization (`per_group_quant_hip(..., transpose_scale=True)`) stores the scales of one 128-column group for all
  rows contiguously (AITER's own tests pass `x_scale.transpose(0, 1).contiguous()` to this kernel). The task therefore
  declares the scales in that layout and generates them that way, instead of a row-major `[m, k / 128]` input plus a
  transpose. A transpose of an activation-side input would be activation preparation, which belongs in the timed call;
  the Gluon sibling (`tasks/gluon/gemm/gemm_a8w8_blockscale/n7168_k2048`) uses row-major `[m, k / 128]` because its
  kernel reads that.
- **Weight `b`** is the logical `[n, k]` nn.Linear weight; the task's `layout` function (`inputs.py`) applies AITER's
  `shuffle_weight(b, layout=(16, 16))`, done once at weight-load time in production (16 x 32 tiles, each stored as
  [k half][n row][16 consecutive k]). Verified bitwise equal to AITER's `shuffle_weight` on the pinned image.
  `b_scale` is plain row-major `[n / 128, k / 128]`.
- Inputs (`inputs.py`): activations and weights are normal values with per-group / per-block magnitudes over a 16x
  range, quantized to e4m3fn with scale `amax / 448`, so a kernel that applies the wrong group's scale is caught.

## Baseline and dispatch

`baseline/` vendors AITER's six gfx950 code objects byte for byte (`hsa/gfx950/fp8gemm_blockscale/*.co` at
`ROCm/aiter` 4ad99832, the version in the pinned image; hashes in `task.yaml`) and translates the host side to
Python (`baseline/kernel.py`). `baseline/LICENSE` is AITER's MIT license. Per call:

1. **Kernel and splitK.** AITER's tuned lookup `get_CKGEMM_config`: the table row for exact M, else for
   `getPaddedM(M, gl=0)` (M rounded up to 16 for M <= 256, to 32 for M <= 1024), else `gl=1` (next power of two). The
   inlined rows are the gfx950 / cu_num 256 rows of the DeepSeek-V3 table for N = 4608, K = 7168 with M <= 864. Every
   padded value of M in [1, 864] is a row, so the table alone decides. Rows with libtype `asm` give `(kernelName,
   splitK)`. splitK is 7 / 6 for M <= 96, 3 for 112..208, 2 for 240..352 (not 320), else 1.
2. **CK buckets (padded M 192, 224, 832; M in 177..192, 209..224, 801..832).** The CK instance cannot be vendored, so
   these M use what AITER's asm entry point (`gemm_a8w8_blockscale_bpreshuffle_asm` without a kernel name) does: the
   launcher's `get_heuristic_fp8_kernel` with splitK 1 and 256 CUs, which picks 128x128 for all three buckets. The
   C++ heuristic scans its kernel table, a `std::unordered_map`, in implementation-defined order and breaks ties by
   that order. On the pinned image every one of 2622 heuristic choices (N = 4608, 128, 2048, 7168 over M up to 4096)
   matches a scan with 128x128 last, which is the order in `kernel.py`. These three buckets are the only places where
   the baseline is not the tuned dispatcher's choice; at M = 180 the asm 128x128 kernel takes 47 us against the CK
   instance's 40 us, at M = 813 50 us against 82 us (cold cache). They are correctness-only cases.
3. **Launch** (`asm_a8w8_blockscale_bpreshuffle.cu`): the split-K alignment check; for splitK > 1 the output is zeroed
   (`hipMemsetAsync` there, `out.zero_()` here) because the splits accumulate with atomics into the bfloat16 output;
   the cached float32 `[1, N]` zero bias of `get_zero_bias_buf`; the packed `KernelArgs` (pointers C, A, B, a_scale,
   b_scale, bias, then m, n, k, lda, ldb, ldc in bytes, ks, scale_m, scale_n, scale_k, every field padded to 16
   bytes); grid `((N / 128) * splitK, ceil(M / tile_m), 1)` of 256-thread workgroups.

**Fidelity** (scratch script on MI355X, pinned image, identical inputs, `import aiter` only there): over every
M = 1..864 the baseline launches the same kernel sequence (memset or not, then the same asm symbol) as
`aiter.gemm_a8w8_blockscale_bpreshuffle_asm` with the tuned kernel name and splitK (asm rows) or with no kernel name
(CK rows), and as the tuned dispatcher `aiter.gemm_a8w8_blockscale_bpreshuffle` on all 800 asm-row M. Outputs are
bitwise identical on 672 of the 864 M, including every M with splitK 1; the other 192 are split-K M whose outputs
differ only by atomic accumulation order (at most a few bfloat16 ulps). Cold-cache CUDA-graph medians agree within 1%: e.g. M = 1: 18.5 / 18.5 us (baseline / AITER),
M = 128: 25.8 / 25.8, M = 512: 39.1 / 39.1, M = 864: 50.6 / 50.4.

## Notes

- **Split-K is nondeterministic.** For splitK > 1 (M <= 352 except M in 289..320 and the CK buckets) the partial sums are
  added with bfloat16 atomics in arrival order, so repeated baseline runs differ in the last bits and the result
  carries up to splitK bfloat16 roundings (SQNR about 46 dB against the float64 reference, versus 47.8 dB for
  splitK 1). This is the production kernel's behavior.
- **Bound: `baseline_relative` with `max_violation_fraction: 0.01`** (the default `k: 2` otherwise). With the default
  fraction of 0, a byte-identical copy of the baseline fails 19 of the 32 cases, all of them split-K cases: a second
  run accumulates the splits in a different order, so a few elements whose baseline error happened to be small land
  above `2 x` that error plus one rounding. Over three full candidate-correctness runs of the baseline copy (64 calls
  each) the largest violating fraction was 0.41% (M = 1, 19 of 4608 elements) and the largest element error 2.24 times
  the bound, far inside the 1% / 16x limits; no case without split-K had a violation. These are accumulation-order
  outliers, the case the violation fraction is for; `k` stays 2, so every other element keeps the default bound.
- The zero bias buffer is cached per device and N with `functools.lru_cache`, as AITER caches it; it is never written.
- The dispatch is specific to MI355X (256 CUs, gfx950). The tuned rows exist only for that configuration, and the
  code objects are gfx950 code objects.

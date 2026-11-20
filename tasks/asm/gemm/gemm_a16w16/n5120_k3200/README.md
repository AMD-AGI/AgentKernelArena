# BF16 GEMM, N=5120, K=3200 (NT), AITER assembly baseline

`out = a @ b.T` with `a` of shape `[m, 3200]` and the row-major `[5120, 3200]` weight `b` (the `nn.Linear`
layout), all bfloat16, accumulated in float32 and rounded to bfloat16. No bias. The var axis is `m`. Both
inputs are standard normal.

## Production origin

The MLP down projection of Qwen3-32B (hidden size 5120, intermediate size 25600) at tensor parallelism 8:
each rank multiplies its 25600 / 8 = 3200 intermediate columns back to the 5120 hidden features. AITER ships
a tuned bf16 GEMM table for this model (`aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv`, merged into
`bf16_tuned_gemm.csv` at runtime); for gfx950 with 256 CUs it lists 13 M values for N=5120, K=3200, and 11 of
them name an AITER assembly kernel (`bf16gemm_*`, `libtype=asm`): every decode size (M=1 to 64), M=128 and
256, and every prefill size from 2048 to 32768. The two remaining rows (M=512, 1024) name torch/hipBLASLt.
This is the scenario of that table where the assembly kernels cover the widest M range.

## Baseline

`baseline/` holds the five AITER code objects this dispatch can select, copied byte for byte from
`hsa/gfx950/bf16gemm/` of ROCm/aiter at commit 4ad99832823dde2315b361cbd3b54b1c5c12acd5 (MIT,
`baseline/LICENSE`) and pinned by SHA256 in `task.yaml`:

| Code object | Tile (M x N) | Used for |
| --- | --- | --- |
| `bf16gemm_fp32bf16_tn_32x64_splitk_clean.co` | 32 x 64 | M = 1 to 32 (tuned rows 1, 16, 32; splitK 3) |
| `bf16gemm_fp32bf16_tn_64x64_splitk_clean.co` | 64 x 64 | M = 33 to 64 (tuned row 64; splitK 3) |
| `bf16gemm_fp32bf16_tn_48x64_pf3_splitk.co` | 48 x 64 | M = 65 to 128 (tuned row 128; splitK 1) |
| `bf16gemm_fp32bf16_tn_96x64_pf3_splitk.co` | 96 x 64 | M = 129 to 256 (tuned row 256) and M = 257 to 288 (heuristic) |
| `bf16gemm_bf16_tn_256x256.co` | 256 x 256 | M = 289 to 1024 (heuristic) and M above 1024 (tuned rows 2048 to 32768) |

`baseline/kernel.py` translates AITER's host side into Python:

- **Kernel choice per M** (`aiter/tuned_gemm.py` `get_GEMM_A16W16_config`): M is looked up in the tuned rows
  of this N and K as is, then padded with `getPaddedM` (`csrc/py_itfs_cu/gemm_common.cu`) at granularity 0
  (multiples of 16/32/64/128) and then 1 (next power of two, 8192 for M above 8192). A row naming an
  assembly kernel fixes the kernel and its splitK. M values that resolve to the torch/hipBLASLt rows
  (M = 257 to 1024) cannot use the production library kernel, which cannot be vendored; for them the
  baseline runs AITER's assembly heuristic (`get_heuristic_kernel` in `csrc/py_itfs_cu/asm_gemm_a16w16.cu`),
  which is what `aiter.gemm_a16w16_asm` selects when no kernel name is given. The heuristic iterates over
  AITER's `std::unordered_map` of kernels and breaks ties by its iteration order; the candidate list in
  `kernel.py` is that order (libstdc++, the generated `asm_bf16gemm_configs.hpp` of the pinned image).
  Over M = 257 to 1024 it selects the 96x64 kernel for M <= 288 and the 256x256 kernel above.
- **Launch** (`gemm_a16w16_asm`): grid `(ceil(N/tileN), ceil(M/tileM), splitK)` of 256-thread workgroups,
  no dynamic LDS, and the 352-byte packed `KernelArgs` struct (22 fields of 16 bytes: a pointer and 8
  padding bytes, or a 4-byte scalar and 12 padding bytes; byte strides; alpha 1, beta 0; `is_out_b16` 1).
- **Workspace** (`aiter/ops/gemm_op_a16w16.py`): the split-K kernels count finished K slices in a 16x64
  uint32 semaphore that is zeroed once and reset by the kernel after each use. AITER allocates one per
  (device, stream); the baseline allocates one per device.

Production is therefore reproduced exactly for every M whose tuned row is an assembly kernel; for
M = 257 to 1024 the baseline is AITER's assembly path rather than the hipBLASLt kernel the table picks
(measured cold-cache: hipBLASLt 26 to 46 us there, the assembly path 29 us at M=257 and 64 to 68 us from M=300
to 1024).

A candidate in the `asm` target language may keep loading these code objects (byte for byte) and change
the dispatch, or write new kernels as `.s` source.

### Fidelity to AITER (measured on MI355X)

A scratch script (not part of the task) compared `baseline/kernel.py` with the production call in the pinned
image (`aiter.tuned_gemm.gemm_a16w16`, or `aiter.gemm_a16w16_asm` for the torch rows) for M = 1, 2, 8, 16, 17,
32, 33, 64, 100, 128, 200, 256, 257, 300, 512, 700, 1000, 1024, 1500, 2048, 3000, 4096, 8192, 10000, 16384 and
32768 (with timing), and for 280 M values between 1 and 32767 (kernel choice and outputs only): both launched the same kernel for every M (from a PyTorch profiler trace), outputs were bitwise
identical wherever the kernel is deterministic (M >= 65) and within one or two bfloat16 ulps on the split-K
sizes (which differ from run to run in production too), and cold-cache CUDA-graph medians agreed within
3% (for example M=1 17.3 vs 17.3 us, M=256 29.0 vs 28.8 us, M=2048 69.2 vs 69.2 us, M=32768 736 vs 732 us).

## Workloads

The 13 M values of AITER's tuned table for this shape (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
8192, 16384, 32768; 32768x span) are the performance cases. All of them take the baseline at least 17 us in
the cold-cache timing (the 32 MB weight read dominates at small M), so no decode size had to be
correctness-only for the 10 us floor. Correctness-only cases add unaligned and between-row sizes that
exercise every dispatch branch: 2, 8, 17, 33, 100, 200, 257 and 300 (heuristic, both kernels), 700, 1000,
1500, 3000 and 10000 (padded to the 8192 row). The boundary cases are M=1 (minimum), M=100 (not a power of
two) and M=33 (not a multiple of 16, above 16).

## Comparison and split-K nondeterminism (k = 8)

`compare.out` is `baseline_relative` with `k` = 8 against the float64 reference `a.double() @ b.double().T`.
The split-K kernels (M <= 64) combine their partial sums in an order that changes from run to run, so an
exact copy of the baseline differs from the baseline's own run by one or two bfloat16 ulps on about 0.05%
of the elements. At the default `k` = 2 that copy fails (measured peaks 1.10x to 1.71x of the bound on
M = 1 to 64); at `k` = 8 its peak is 0.47x to 0.67x. M >= 65 is deterministic and a copy sits at 0.11x of
the bound. This is the same rule (`k` = 8, the contract maximum) as the split-K scenarios of
`tasks/flydsl/gemm/gemm_a16w16/`.

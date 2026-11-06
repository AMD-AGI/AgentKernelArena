# FP16 GEMM, row-major NN, N=4096, K=4096

`out = a @ b` with a row-major `a` of shape `[m, 4096]` and a row-major `b` of shape `[4096, 4096]`
(no transposed operand), all float16, accumulated in float32 and rounded to float16. The var axis is `m`.

## Scenario and workloads

This task stands for the plain float16, row-major GEMM that the Triton GEMM test suites exercise in many
syntactic forms (block pointers, no-`scf` tiles, TMA-style stores, induction-variable-dependent pointers,
multiply-reduce matmuls, single-tile dots). Those kernels all compute `C = A @ B`; their tile sizes and
warp counts are implementation choices and are not part of this contract. `m` takes the batch sizes of a
4096x4096 projection: 64 .. 16384 are performance cases (256x span, including the square 128 .. 4096 sizes
of those test suites), and `m = 1, 2, 17, 33, 100, 1000` are correctness-only. The boundary cases are
`m = 1`, `m = 33` and `m = 100` (not a power of two; not a multiple of 16 and above 16). Inputs are
standard normal.

## Baseline

`baseline/kernel.py` is vLLM's persistent Triton matmul (`matmul_persistent` in
`vllm/model_executor/layers/batch_invariant.py`, Apache-2.0) with its float16 configuration: 128x256x64
tiles, 8 warps, 3 stages, one program per SM looping over output tiles. The optional bias was removed
(this scenario has none), as were the launch metadata and the bfloat16/float32 configurations, and the SM
count is read through `torch.cuda.get_device_properties`. The kernel is otherwise unchanged.

## Comparison

`baseline_relative` (default bounds); the reference is float64 `a.double() @ b.double()`.

## Risks for the GPU audit

At `m = 64` the baseline launches only 16 tiles; its real time is expected above 10 us (the
bandwidth estimate for streaming `b` alone is about 11 us) but has not been measured.

# Persistent matmul task

This task optimizes the Triton `matmul_kernel_persistent` implementation behind
`matmul_persistent`. The evaluator owns compilation, correctness, index proofs,
and performance measurement. Files under `build/` are generated run artifacts
and are not committed evidence.

## Public contract

Inputs are finite positive-stride CUDA matrices with matching float16 or
bfloat16 dtype. Float32 is rejected before target launch because this Triton
dot path does not lower on gfx950. Padded row-major, inner-strided, and transposed
views are supported without mutating their logical values or backing storage.
An optional length-`N` bias has the same dtype/device; the baseline materializes
only a noncontiguous bias because the target kernel consumes unit-stride bias.

The target launch uses a one-dimensional persistent grid and positive
`GROUP_SIZE_M`. Correctness receipts attest the resolved grid, launch metadata,
stride arguments, optional-bias flag, grouped tile bijection, partial final
groups, and a case whose tile count exceeds `NUM_SMS`. Exactly the documented
13 runtime operands are accepted; known Triton tuning controls remain flexible
as bounded scalars but cannot carry a hidden tensor or object. Immediately before each
target launch, the evaluator poisons the exact returned C allocation and
captures it as soon as that launch completes. This binds the returned result to
the target launch and rejects a dummy launch followed by wrapper-side compute.

Every call returns a new contiguous deterministic output. Two live results must
not share storage with one another or any input/backing allocation. The trusted
reference performs float32 matmul, then float32 bias addition, then casts to the
input dtype.

## Large-index proof

Pointer-width dispatch is based on the maximum relative element offset implied
by shape and strides. `numel` alone is insufficient for a small logical view
whose row or column stride crosses signed-int32 range. The evaluator exercises
the wrapper with allocation-free virtual tensors and records `A_LARGE`,
`B_LARGE`, and `C_LARGE`; boundary calculations use Python integers. No GPU
correctness case allocates a 2 GiB probe. Formal release evidence still requires
a source audit showing that the attested flags reach int64 pointer arithmetic.
The largest real correctness case has a conservative 512 MiB evaluator cap;
logical >2^31 cases report zero allocated bytes.

## Frozen scoring workload

`TEST_SHAPES`, `perf1` through `perf5`, warmup 10, repetitions 100, the complete
`run_performance` function, `_compute_pid`, and `matmul_kernel_persistent` remain
byte-identical. Wrapper validation and virtual-index dispatch are a new baseline,
so old receipts cannot be reused even though scoring inputs take the same kernel
path. Baseline and candidate must be measured from the same commit and image.

CPU structural tests and virtual metadata make no GPU correctness or performance
claim. Release qualification must run all bounded correctness cases and matched
performance on the target GPU.

Formal evaluation binds the final candidate source manifest to an
evaluator-owned `aka.formal-source-anti-tamper/v1` report. Its static import
allowlist is `torch`, `triton`, and `math`; dynamic namespace access,
protected evaluator/torch mutation, and obvious torch global-state setters are
rejected. The runner also snapshots checker globals and trusted torch
primitives, including `torch.cuda.Event`. This static policy is
defense-in-depth, not a complete Python sandbox; process/container isolation
remains the security boundary for candidate execution.

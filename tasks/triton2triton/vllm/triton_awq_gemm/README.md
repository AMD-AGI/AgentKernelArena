# Triton AWQ GEMM task

This task optimizes `awq_gemm_kernel` while preserving the public
`awq_gemm_triton` wrapper contract. The evaluator owns compilation,
correctness, and performance measurement; files under `build/` are generated
run artifacts and are intentionally not committed.

## Public contract

Valid inputs are finite float16 activations, positive finite float16 scales,
and packed int32 AWQ weights and zero points on one device. `M`, `K`, and `N`
are positive, `N` is divisible by eight, and `K` is divisible by a group size
of 32, 64, 128, or `K`. Split-K values are powers of two through 32.

The wrapper accepts positive-stride two-dimensional views. Correctness-only
cases exercise padded row-major, inner-strided, and transposed storage, protect
the complete backing allocations, and cover every supported split-K value.
Packed negative int32 words are interpreted as unsigned two's-complement bit
patterns before AWQ nibble reordering.

Each call must launch `awq_gemm_kernel` with the requested split and block
sizes, without hidden runtime operands, leave all inputs and their backing
storage unchanged, and return a fresh deterministic, finite float16 result.
Known Triton tuning controls may vary as bounded scalar values; they cannot
carry tensors or arbitrary runtime objects into the target.
Correctness freezes the four target inputs before launch, poisons the target
output, and snapshots it immediately after launch, so a dummy target followed
by wrapper-side output repair is not accepted. Results are compared with the
dequantize-then-matmul reference at `atol=1e-2, rtol=1e-2`; the cases exercise
the default block sizes and an explicit `(16,64,16)` request.

## Frozen performance workload

The five `TEST_SHAPES`, `perf1` through `perf5`, warmup count 10, repetition
count 100, benchmark helper call, and `run_performance` implementation remain
unchanged. The baseline kernel now uses a float32 dot accumulator and preserves
split-K lanes in float32 until their final reduction: the former float16 path
missed the documented tolerance on valid gfx950 cases. The packed operands and
public float16 output ABI are unchanged. The wrapper also makes
four no-copy contiguity checks for the already-contiguous scoring inputs. These
are source-level baseline changes, so old timing receipts cannot be reused;
baseline and candidate must be measured from the same commit and image.
Old timing receipts cannot be reused under the corrected baseline.

## Validation

Repository tests statically lock the scoring runner and corrected GPU kernel, validate
the public cases, and execute the signed-word unpack helper on CPU. Release
qualification additionally requires the baseline to compile and pass
every correctness case on the target GPU, followed by matched performance
measurement. CPU structural tests alone make no GPU correctness or speed claim.

Formal evaluation binds the final candidate source manifest to an
evaluator-owned `aka.formal-source-anti-tamper/v1` report. Its static import
allowlist is `torch`, `triton`, and `math`; dynamic namespace access,
protected evaluator/torch mutation, and obvious torch global-state setters are
rejected. The runner also snapshots its checker globals and trusted torch
primitives, including `torch.cuda.Event`. Static AST screening is
defense-in-depth rather than a complete Python sandbox, so untrusted execution
still relies on the process/container isolation boundary.

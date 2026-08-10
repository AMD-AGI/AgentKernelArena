# Triton AWQ dequantization task

This task optimizes `awq_dequantize_kernel` while preserving the public
`awq_dequantize_triton` contract. The evaluator owns compilation, correctness,
and performance measurement. Files under `build/` are generated run artifacts,
not reusable evidence.

## Public contract

Valid inputs are C-contiguous two-dimensional tensors on one CUDA/ROCm device:
signed int32 packed weights, positive finite float16 scales, and signed int32
packed zero points. The logical matrices may begin at a nonzero offset inside a
larger allocation. Strided, transposed, or overlapping logical layouts are not
valid and the wrapper rejects them without making a hidden contiguous copy.

`K` and `N_packed` are positive. The scale-row count must infer a group size of
32, 64, 128, or `K`, and that group size divides `K`. Scale and zero shapes are
exactly `[K/group_size,N_packed*8]` and `[K/group_size,N_packed]`. Negative
packed values retain their unsigned two's-complement bits during AWQ nibble
reordering.

The block-size arguments and their defaults remain part of the wrapper API.
Each accepts powers of two through 128, `block_size_y` divides the group size,
and both requested values reach exactly one target kernel launch. That launch
is bound to the evaluated inputs and the exact tensor returned by the wrapper;
a detached dummy launch is not evidence. Its seven runtime operands and two
constexpr tile arguments form an exact ABI: missing, duplicate, extra
positional, unknown keyword, and object-valued launch-control payloads are
rejected. A call leaves every input backing
allocation unchanged and returns a fresh, contiguous, finite float16
`[K,N_packed*8]` tensor. Repeated calls are exactly deterministic and match the
explicit unpack-and-dequantize reference at
`atol=1e-2, rtol=1e-2`.

The evaluator poisons the output immediately before the target launch and
queues a clone immediately afterward. This target-only snapshot must equal the
wrapper's return, rejecting precomputation, a no-op target, and post-target
repair.

## Evaluation and provenance

The five `TEST_SHAPES`, `perf1` through `perf5`, warmup count 10, repetition
count 100, benchmark helper call, and `run_performance` implementation are byte
frozen. Correctness-only cases cover every group-size branch, nondefault launch
tiles, signed packed words, storage guards, output ownership, and invalid-input
rejection. The baseline GPU kernel body is unchanged; only structural wrapper
validation and evaluator logic were added. Because wrapper work changed, old
timing receipts cannot be compared with this revision: baseline and candidate
must be measured from the same commit and image.

Repository tests lock the scoring runner and baseline kernel bytes and validate
the public checker on CPU. Release qualification additionally requires the
unchanged baseline to compile and pass all correctness cases on the target GPU,
then matched baseline/candidate performance measurement. CPU tests alone make
no GPU correctness or speed claim.

Formal evaluation also binds the final candidate source manifest to an
evaluator-owned `aka.formal-source-anti-tamper/v1` report. The static policy
allows only `torch`, `triton`, and `math` imports and rejects dynamic namespace
access, protected evaluator/torch mutation, and obvious global-state setters.
The runner separately snapshots trusted torch primitives (including
`torch.cuda.Event`) and its checker globals. This AST policy is
defense-in-depth, not a Python sandbox; process/container isolation remains the
security boundary for executing candidate code.

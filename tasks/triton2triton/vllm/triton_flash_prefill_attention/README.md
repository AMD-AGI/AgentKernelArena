# Triton flash-prefill attention task

This task optimizes `_fwd_kernel` through the public
`context_attention_fwd` wrapper. The evaluator, not agent-authored text, owns
compilation, correctness, and timing evidence. Files written under `build/`
are run artifacts and are not reusable evidence.

## Public contract

Q, K, V, and O are finite, dense contiguous float16 CUDA tensors on one
device. Q/O are `[T, Hq, D]`, K/V are `[T, Hkv, D]`, and `Hq` is divisible by
`Hkv`. The packed batch has one through eight positive sequence lengths in
`[1, 1024]`; int32 start locations are their exact exclusive prefix sum,
`T` is the sum of the lengths, and `max_input_len` is their maximum. Supported
head dimensions are 32, 64, 96, and 128.

The correctness gate samples uniform and ragged batches, MHA/GQA/MQA,
causal and non-causal attention, a custom positive softmax scale, and
independent backward and forward windows. `None` and zero disable a window.
A positive Q-side window keeps keys satisfying
`query_pos - key_pos <= window_q`; a positive K-side window keeps keys
satisfying `key_pos - query_pos <= window_k`. Causal masking is applied in
addition to both windows.

Each call writes O and returns `None`. Results must be finite, deterministic,
and match the float32 PyTorch reference at `atol=1e-2, rtol=1e-2`. Q, K, V,
and both metadata tensors remain unchanged. Each invocation must make exactly
one target-kernel launch. The evaluator binds that launch to the exact input,
metadata, and output tensors and checks its resolved grid, strides, scale,
masking constexprs, head grouping, and head dimension; a detached dummy launch
is not accepted as evidence. The 15 runtime operands and eight constexprs form
an exact ABI; hidden positional operands, unknown or duplicate keywords,
wrongly typed metadata, and object-valued tuning controls are rejected. The
recorder poisons O immediately before the
target launch and queues a snapshot immediately after it. The snapshot must
equal the wrapper's final O, preventing precomputation and post-target repair
from masquerading as target-kernel work.

## Frozen performance workload

The five `TEST_SHAPES`, `perf1` through `perf5`, warmup count 10, repetition
count 100, benchmark helper call, and `run_performance` implementation are
byte-frozen. The complete baseline source file, including `_fwd_kernel` and
the wrapper, is also byte-frozen by repository tests. It does not transplant a candidate optimization
or move the performance baseline.

## Validation

CPU tests lock the source and scoring bytes and inspect the independent
correctness cases and evaluator invariants. Release qualification still
requires the unchanged baseline and every candidate to compile and pass all
cases on the target GPU, followed by matched performance measurement.
CPU structural tests alone make no GPU correctness or speed claim.

Formal evaluation binds the final candidate source manifest to an
evaluator-owned `aka.formal-source-anti-tamper/v1` report. Its static import
allowlist is `torch`, `triton`, and `math`; dynamic namespace access,
protected evaluator/torch mutation, and obvious torch global-state setters are
rejected. The runner also snapshots checker globals and trusted torch
primitives, including `torch.cuda.Event`. This AST screen is defense-in-depth,
not a Python sandbox; process/container isolation remains the candidate-code
execution boundary.

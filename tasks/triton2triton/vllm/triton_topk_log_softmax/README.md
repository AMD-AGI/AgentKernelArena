# Triton top-k log-softmax task

This task optimizes the monolithic `_topk_log_softmax_kernel` through
`compute_token_logprobs`. The evaluator owns correctness and timing evidence;
files under `build/` are generated run artifacts.

## Public contract

The wrapper accepts float16, bfloat16, or float32 two-dimensional logits,
including padded rows with a noncompact leading stride. Vocabulary and batch
sizes are positive. Integer token IDs are valid, possibly duplicated or
unsorted indices; the wrapper converts them to int64. It returns a new float32
`[batch,num_token_ids]` tensor without mutating or aliasing either input.

Finite logits and `-inf` are supported. Complete 1024-entry `-inf` chunks must
not poison finite neighboring chunks. A row containing only `-inf` produces NaN
for every requested ID, matching `torch.log_softmax`; rows containing `+inf` or
NaN are outside the scored domain.

Each invocation makes exactly one target launch. The evaluator binds it to the
exact logits, converted IDs, and returned output tensors and checks data and
storage pointers, leading stride, batch grid, vocabulary size, requested-ID
count, and reduction constexprs. An omitted, detached dummy, wrongly bound, or
extra target launch is rejected. The evaluator poisons the output just before
the target and queues an immediate post-target snapshot. That snapshot must
equal the final return, so precomputation and post-target repair cannot replace
the target's work. The six runtime operands and two constexprs form an exact
ABI: hidden positional operands, unknown or duplicate keywords, and
object-valued launch controls are rejected. Every case is invoked twice with
identical inputs; the two exactly deterministic results must own distinct
fresh storage.

## Evaluation

Correctness-only cases cover supported dtypes, padded strides, batch one, large
and non-power-of-two vocabularies, duplicate IDs, isolated `-inf`, a complete
`-inf` reduction chunk, and all-`-inf` rows. The five scoring `TEST_SHAPES`,
`perf1` through `perf5`, warmup 10, repetition 100, performance helper call,
`run_performance`, and baseline source remain unchanged. CPU structural tests
make no target-GPU correctness or speed claim; release qualification still
requires matched GPU validation.

Formal evaluation binds the final candidate source manifest to an
evaluator-owned `aka.formal-source-anti-tamper/v1` report. Its static import
allowlist is `torch`, `triton`, and `math`; dynamic namespace access,
protected evaluator/torch mutation, and obvious torch global-state setters are
rejected. The runner also snapshots checker globals and trusted torch
primitives, including `torch.cuda.Event`. This AST screen is defense-in-depth,
not a Python sandbox; process/container isolation remains the candidate-code
execution boundary.

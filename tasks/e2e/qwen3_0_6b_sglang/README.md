# Fixed-workload end-to-end kernel optimization

Optimize any GPU computation involved in this model, including fusion across
operators and kernels. The supplied RMSNorm implementation is a starting point,
not the optimization boundary. Choose the hotspots and GPU implementation backend
supported by the pinned image yourself.

## Submission

Edit the `source/` tree, including `source/patches.json`. Submit Python kernels,
helpers and HIP/C++ source files there. Do not submit binaries, symlinks or build
outputs. A bundle is limited to 256 files and the source byte limit in the lock.
Each manifest entry names a runtime `module`, dotted `symbol`, candidate Python
`file` relative to `source/`, and replacement `callable`:

```json
{"version": 1, "patches": [{
  "module": "sglang.srt.models.qwen3",
  "symbol": "Qwen3DecoderLayer.forward",
  "file": "fused_layer.py",
  "callable": "forward"
}]}
```

Use the model class names present in this task's pinned image. Compute functions
and methods in SGLang layers/models, AITER and sgl_kernel may be replaced. This
includes replacing a model/layer compute boundary to fuse operations spanning
several existing kernels. Preserve method arguments, results, state updates and
all model semantics. Scheduler, sampling, tokenizer, weight loading, configuration
and evaluator changes are prohibited. Keep precision, weights, TP, concurrency,
request/token counts and serving flags unchanged. Speedups must come from kernel
implementation or fusion, not less work or different runtime settings.

Use relative Python imports for bundle helpers. If needed,
`from _aka_serving_bundle import original, load_extension` provides the unchanged
declared compute target and a pinned-toolchain HIP/C++ builder. For example,
`original(module, symbol)(*args)` delegates an unchanged part of a replacement;
`load_extension("my_kernel", ["binding.cpp", "kernel.cu"])` builds submitted files
using the image toolchain and local temporary storage. Do not modify runtime
modules through reflection or environment/file access.

## Suggested analysis, not a prerequisite

Profiling is recommended but optional. Use tools available to your agent to
understand kernel time, actual shapes, layouts, precision, call frequency, launch
overhead and opportunities to reuse intermediates or fuse operations. Study the
real prefill/decode workload before spending time on a kernel microbenchmark.
Choose your own tools and analysis method; no profiling report or framework
profiling interface is required. Keep analysis scripts and traces outside the
protected task files. Diagnostic timings never replace the fixed scoring runner.

## Evaluation and limits

Read `benchmark.yaml` and `runtime.lock.json` for exact workload and runtime
identity. Run the declared public compile/correctness/performance commands in
`config.yaml`. They execute a clean copy with pinned read-only model assets.
All replacements must actually execute, and every TP rank must report effective
runtime settings. Hashes cover the complete bundle, including the manifest.
An unchanged bundle is a retained control, not an optimization success.

Existing operator oracles remain useful targeted checks, but do not cover every
possible edited or fused region. Whole-model checks retain short cases and also
run one concurrent batch with the scored workload's input and output lengths
against an independent reference. Requests run both with and without logprobs,
including the no-logprob path used for scoring. In that path every generated
token must agree with the independent FP32 greedy choice within the existing
logprob tolerance; near ties are allowed without skipping later decode steps. These are finite tests, not proof for all inputs. Keep additional
candidate-specific tests when useful; do not weaken the protected checks.

Final acceptance includes a fresh compile/check and the configured number of
complete baseline/candidate pairs. Search checks and recovered artifacts are not
final acceptance. A single pair does not establish variance or significance.
The paired median must improve by at least 5% to earn gain credit; smaller gains
remain in the raw results. Choose `budget.final_evaluation_pairs` before the run
to request 3 through 100 complete pairs, never fewer than the task declares,
using dozens when the budget permits.
A fresh read-only verification agent also checks the frozen submission and
measurement evidence. Failed or inconclusive review blocks acceptance. Wrappers identified by the reviewer as unchanged computation receive no gain
credit. Finite semantic review cannot guarantee detection of every renamed
wrapper. Performance actions check workload and runtime identity; numerical
output checks run in the separate correctness action, not on every timed response.

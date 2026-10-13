# Fixed-workload end-to-end kernel optimization

Optimize any GPU computation involved in this model, including fusion across
operators and kernels. The supplied Gated DeltaNet packed decode implementation is a starting point,
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
  "module": "sglang.srt.models.qwen3_next",
  "symbol": "Qwen3NextDecoderLayer.forward",
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
against an independent reference. Requests cover both logprob modes, including
the no-logprob path used for scoring. Normal FP8 variation can change greedy
continuations. Each actual no-logprob continuation is therefore scored by a
separate original-model server: every generated token must stay within the
locked maximum, mean and 99th-percentile distance from its preferred token.
Input-prefix log probabilities have separate per-case numerical bounds, and
fixed model quality checks remain mandatory. These bounds are calibrated from
unmodified-model repeats before optimization. Independent baseline qualification
must pass, and deliberately wrong computation must fail. These are finite tests,
not proof for all inputs. Keep additional
candidate-specific tests when useful; do not weaken the protected checks.

Final acceptance includes a fresh compile/check and the configured number of
complete baseline/candidate pairs. Search checks and recovered artifacts are not
final acceptance. A single pair does not establish variance or significance.
The paired median must improve by at least 5% to earn gain credit; smaller gains
remain in the raw results. A fresh read-only verification agent also checks the
frozen submission and measurement evidence. Failed or inconclusive review blocks
acceptance. A wrapper that retains the original computation cannot earn gain
credit, even if its measured throughput is higher.

## Fixed recipe and reference coverage

This is the original Magpie recipe at the dependency revision in the lock: TP4,
concurrency 32, maximum ISL/OSL 1024, random length ratio 0.8, FP8, and 320 requests
with client seed 0. The protected server seed is identical for every role.
The seed GDN source retains its upstream source attribution and copyright. Its
operator oracle checks BF16 outputs and FP32 state across batches through 512,
strided/padded layouts, inactive state slots, repeated steps and graph replay.
The oracle resolves the bundle's GDN replacement if present, otherwise the pinned
image implementation. It does not claim to cover other fused regions.

Model quality uses 64 frozen GSM8K samples and five fixed examples, greedy decoding
with up to 8192 output tokens and concurrency 32. Both servers must score at least
90%; the candidate may lose at most three answers out of 64 relative to the
original server. Every response and
parsed answer is retained; no failed samples are retried. This is a limited
regression subset, not the complete Magpie/lm-eval suite. Dataset provenance is
in `quality.json` and its MIT notice is in `QUALITY_LICENSE`. Quality uses the
model chat template; throughput preserves the original raw-completion recipe.
Eight additional short prompts provide identical-input-prefix logprob comparisons.
Different generated sequences are retained as diagnostics. Each candidate
continuation is teacher-forced on the original server so later decode positions
are still checked after a divergence. The numerical bounds in
`correctness.production_tolerances` must also pass independent baseline
qualification; agents cannot change them.

The task default is three final pairs; choose `budget.final_evaluation_pairs`
before the run to request 1 through 100 complete pairs. Use dozens when the
budget permits; one pair cannot estimate variability. Neither the agent's search checks nor
a passing historical report qualify new task or candidate bytes.

# Qwen3 0.6B serving kernel optimization

Optimize only `source/rmsnorm.py`. The starting implementation delegates to
production AITER RMSNorm kernels. Both final entrypoints must execute candidate
Triton kernels, without importing AITER; the compile action checks GPU profiler
evidence for each entrypoint. The evaluator replaces the same SGLang
normalization entrypoints on every rank for baseline and candidate.
Use Torch only for allocation and tensor metadata, with computation in the
submitted Triton kernels. Calling an existing framework operator is not a kernel rewrite.

Preserve FP16 input/output semantics, residual outputs, and both API signatures.
The model exercises hidden width 1024 and attention head width 128. Implement
all covered row counts, including decode and prefill. Kernel tiling, vectorization
and fusion within these operators are allowed. Do not change model weights,
precision, dispatch, serving settings, workload, dependencies or the harness.
Do not branch on whether correctness or performance is running.

`benchmark.yaml` is a small-model variant of the Magpie recipes preserved in the
parent directory. Its model, precision, TP, concurrency and lengths are explicit
variant choices made before baseline measurement, not optimization gains.
The variant fixes FP16 and the Triton attention backend; neither is editable.

The runtime lock declares the model and benchmark dependencies. The Docker host
prepares them in a versioned cache and mounts model resources read-only inside
the clean evaluation container. No task downloads dependencies while scoring.
The framework materializes the shared runner into `scripts/` before commands
are executed; a raw task directory is not directly runnable.

Correctness compares the kernel outputs with an independent FP32 calculation
and prefill/decode log probabilities with a Transformers FP32 reference.
Performance runs the pinned Magpie workload with profiling off and checks full
request/output-token counts. Each measurement completes 2,000 requests, about
60 seconds of steady-state decode at the baseline rate; the lock rejects any
window shorter than 30 seconds. Final scoring uses three fresh interleaved A/B
pairs. Throughput improvements are reported independently of agent completion.

## What this task can and cannot show

RMSNorm is a small share of Qwen3-0.6B decode time at concurrency 8. Even a
kernel that is several times faster changes end-to-end throughput by well under
one percent, which is the same order as the repeat-to-repeat variation reported
under `serving_measurement.uncertainty`. This task exists to exercise the
serving contract end to end; a result near 1.00x is the expected outcome, not
a failed optimization. Do not read a median ratio inside the recorded noise
range as a gain.

Every performance action starts from a fresh container with an empty Triton
cache, so kernel compilation that happens during serving counts against the
candidate. A kernel specialized on the row count with a `tl.constexpr`
parameter recompiles for every new batch or prefill size inside the timed
window. Prefer runtime row arguments and a bounded set of tile
specializations. The pinned client issues warmup requests before timing, but it
cannot warm every shape the scheduler will form.

For clean checks during optimization, use `python3 scripts/evaluate.py candidate
compile`, `candidate correctness`, or `candidate performance`. These commands
submit the current kernel source to the host evaluator; they need no model
weight or runtime installation in the agent workspace.

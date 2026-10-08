# Benchmark quality admission

The unchanged DS quantization comparison at commit `20c1a949` reported an
arithmetic mean speedup of 6.250768. One retained reference sample for
`x8192x2048` was 68.000145 ms and contributed 92.72% of that case's total time.
The source was identical in the two outer runs. Complete raw samples and correct
arithmetic therefore did not establish an acceptable performance comparison.
That run must be recorded as a benchmark-quality failure, with no accepted gain.

The `benchmark-quality-v1` gate is deliberately conservative. It rejects a
repeated fixed-work timing series if either condition holds:

- A sample exceeds 10 times the series median **and** contributes more than
  10% of the series total time. The conjunction avoids rejecting a merely broad
  distribution on the strength of one moderate sample.
- The nearest-rank 95th percentile exceeds 10 times the nearest-rank 5th
  percentile. This catches extreme two-regime behavior without relying on a
  single dominating sample.

These are fixed, reviewable policy constants. They are not fitted per task,
candidate, case, or observed result. They identify catastrophic contamination;
they do not establish a confidence interval or prove a scheduling/kernel cause.
The observed DS outlier is far beyond both limits of the first condition.
Every case and all 100 samples remain mandatory. A failing case or native
diagnostic series rejects the complete comparison. There is no trimming,
winsorization, replacement, selective remeasurement, or change to means.

For variable work, callers must first use the existing trusted report/receipt
validator to reconstruct the actual schedule. `assess_comparison` requires
matching work-class sequences for both legs and applies the same checks within
each class. A class is the declared amount of work, such as a validated histogram
variant or KV length; a seed, tensor address, or an arbitrary unvalidated report
label must never define a class. All classes and samples still contribute to the
original mean. A singleton class cannot establish repeated-work stability; the
quality receipt reports its count explicitly. The gate makes no claim that it
can diagnose variability for a workload observed only once. Reject missing or
unpaired work-class receipts rather than treating variable work as fixed work.

An accepted gain additionally requires a changed executable-source digest or
map. Package/harness hashes are not source identity. Identical source, including
a run with only harness changes, produces `unchanged_source_control`, a null
accepted speedup, and `accepted_gain: false`, even with otherwise stable favorable
timings. A source change is necessary, not by itself proof of an optimization:
all correctness, source-boundary, trusted-execution and normal comparison
requirements continue to apply. This gate is not an anti-cheat boundary.

The native quantization and portable trusted evaluators apply this gate after all
six phases and all cases have finished and their raw reports have been written. A failed
comparison is written with `status: rejected_timing_quality`, null accepted
speedups, and the original values in `raw_speedup` and
`raw_arithmetic_mean_speedup`; the CLI exits nonzero. A stable unchanged-source
control remains a completed measurement but carries no accepted speedup.
Consumers must not recover a score from raw diagnostic fields or a ratio of the
preserved means when the accepted field is null.

Existing native quantization artifacts can be checked without launching GPU
work or altering the original reports:

```sh
python -m src.tools.benchmark_quality_gate \
  --repo /path/to/trusted/repository \
  --commit FULL_IMMUTABLE_COMMIT \
  --task tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 \
  --evidence /path/to/six-phase-evidence \
  --output /path/to/new-quality-decision.json
```

For an optimized submission, supply `--candidate-source` with the reviewed
source file. The command binds it and the reference source to all six retained
report hashes and identities, then writes only a fresh decision file. Exit codes
are 0 for an eligible changed-source comparison, 1 for timing rejection, and 2
for a stable unchanged-source control. An older task-validator PASS does not
override a subsequent quality rejection.

Quality admission is mandatory for the active protected report contracts:

- Schema-v2 paired-reference reports run their existing registry and pinned CPU
  receipt validators before quality classification. MoE and MiniMax use the
  independently validated variant identity. The exact supported Kimi signature
  schema uses `work_value`; the exact MLA schema uses `length`. Neither seeds nor
  addresses define a quality class. Unknown task-local class schemas fail closed.
- Schema-v1 native comparisons check all outer-port, native-reference and
  candidate samples. They are admitted as fixed work only when their cases have
  no variable-work markers.
- Portable reports without a native scoring policy, including FP4 GEMM and
  fixed-work aggregation/prefill tasks, validate the complete report against the
  current source map and case manifest before checking all raw samples. Variable
  work on this route requires a proven paired adapter; it is never silently
  assessed as fixed work.
- The existing native-quant report contract is also checked in ordinary
  performance evaluation, so its host-only gate cannot be bypassed by stdout
  timing or the generic report parser.

Per-case quality metadata includes the full manifest case-ID set, all raw sample
arrays, validated work classes, source map, request ID and manifest fingerprint.
The ordinary evaluator retains this metadata in baseline and optimized
performance YAML. These cached fields remain diagnostic data: a self-consistent
class list, case-ID list or manifest-hash string cannot establish its own
authority. At the end of protected report/semantic validation, the host keeps an
immutable in-memory record of the actual classes, complete case set and receipt
digest. Fresh cases are bound to that record, their source/workload metadata and
the selected timing series. Direct scoring checks against that independent
record, including when a caller edits an otherwise fresh case.

The authority object is not stored in performance YAML and cannot be reconstructed
by its loader. `load_performance_results` without explicit validation context
returns data for inspection; those cases cannot receive accepted scores. The
loader records cache origin outside editable metadata, so deleting cached
quality flags cannot opt out of this rule.
To resume scoring, rerun the protected raw-report/manifest/request/semantic
validators and supply their fresh cases through `validated_cases=`. The cache
must match that external context exactly, and the context's complete case set
must match even if both supplied lists omit the same case. An edited cache cannot
serve as its own context. Do not call `case_quality` on cached declarations to
manufacture new authority. If the original verified reports and protected inputs
are unavailable, fail closed instead of scoring the cache.

```python
fresh = as_test_cases(load_native_measurements(
    trusted_task_root, config, report=verified_report, request=expected_request),
    is_baseline=True)
baseline = load_performance_results(
    workspace, "baseline_perf.yaml", validated_cases=fresh)
```

This is a trusted-host data boundary, not a sandbox for arbitrary code executing
inside the host process. Private source-only host evaluation remains mandatory.
The normal fresh measurement path keeps its in-memory context and requires no
extra GPU calls or changes to samples, work schedules, sources or timing methods.

Native summaries, portable trusted results, ordinary `task_result.yaml`, direct
case scoring and postprocessing honor the same accepted-score decision. Null or
zero accepted ratios are authoritative; no consumer reconstructs them from
aggregate means. Persisted source maps are compared across the initial and
current measurements, so harness-only changes remain unchanged-source controls.
Results declared native, or belonging to a protected quality-enabled task,
cannot omit the quality requirement to regain old performance points.

The task-validator finalizer independently reads and validates the raw
performance report before finalizing a claimed benchmark PASS. An instability
or missing/invalid quality receipt forces `benchmark_integrity: FAIL` and an
overall failure. This is single-run task qualification, not an optimization
comparison: a stable stock-source task may qualify, but validation itself never
asserts an accepted gain. The complete raw report and quality cases are retained.
Framework qualification still does not replace private source-only host retesting.

These adapters do not change measurement methods, task source, sample counts,
warmups, tolerances or phase budgets. Unrecognized legacy protocols outside these
protected contracts need an explicit adapter before their results can provide
quality evidence; no unsupported protocol is inferred to be variable or fixed.

No code in the gate launches or retries a comparison. If a maintainer authorizes
a repeat, predeclare **at most one** fresh whole comparison, covering all six
phases, all cases and all original samples under the existing phase deadlines.
Give it a fresh output, retain the rejected attempt and its raw samples, and link
both attempt identities in the review record. A second rejection stops the
experiment. Never retry until passing or choose the best attempt, leg, case or
sample. New optimization source starts a separately identified experiment.

The current task records `start.record()`, `graph.replay()` and `end.record()`
as separate host calls. Host descheduling before the end event is submitted is
one plausible explanation for an idle GPU gap, but these reports contain no
CPU scheduling trace proving that explanation. This gate does not change event
placement or attribute the outlier to kernel variance. Any timing-method change
requires separate review and fresh GPU qualification.

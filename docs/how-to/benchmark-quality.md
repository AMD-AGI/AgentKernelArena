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

The native quantization trusted evaluator applies this gate after all six phases
and all cases have finished and their raw reports have been written. A failed
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

This patch integrates the gate into the native-quant trusted comparison that
produced the observed failure. The reusable variable-work API and its tests are
included for independent review; other paired consumers must wire it only after
their semantic receipt validation. Their measurement methods and entrypoints
are not changed by this patch. Ordinary framework-only scores are insufficient
for promotion of this task; the trusted quality decision is mandatory.

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

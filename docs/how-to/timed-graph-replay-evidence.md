# Review retained timed-graph replay evidence

Exact replay identity concerns the graph executable that supplied the measured
samples. Fresh inputs can produce different floating-point outputs on each replay;
the outputs still need to pass the task's unchanged numerical oracle and tolerance.

The canonical Python helper passes its final `graph` object to the reported sample
loop and closes over that same object in the callback bound to `timed_run`. Capture
exports the output object returned by the captured callable. A task must actually
invoke the callback, check the captured output identity, and validate that output
against the current prepared inputs. A separate eager invocation is insufficient.

For the supported canonical structure, a read-only source/evidence audit is available:

```bash
python3 -m src.tools.audit_timed_graph_replay \
  --helper path/to/executed/_aka_benchmark.py \
  --report path/to/validation_report.yaml \
  --integrity path/to/validation_integrity.json \
  --performance path/to/build/performance_report.json \
  --out replay-evidence-audit.json
```

The audit checks the final timing and validation callback bindings, captured-output
binding, helper hashes before and after execution, finalized report hash, current
quality/performance hash, and the successful measured-replay assertions in the raw
cases. Negative tests substitute a different graph, replay callback, or output and
require rejection. The checker recognizes this specific helper structure; it is
not a verifier for arbitrary Python, graph binaries, or malicious runtime libraries.

This is source-bound execution evidence. Existing reports do not export process-local
graph handles, output pointer traces, or per-sample input generation IDs. Those values
must not be claimed retrospectively. Review the frozen task's output-identity checks,
CPU snapshots of outputs and current inputs, replay validation, and required counts
alongside the audit. A successful audit does not alter the retained validator status,
replace task-specific quality checks, or establish a new framework qualification.

If the binding or executed validation is absent, retain the warning; an observed
wrong or stale replay remains a failure. Prompt changes require a fresh validator
execution before claiming a new result. Cases, seeds, warmups, samples, reset logic,
and numerical requirements remain unchanged.

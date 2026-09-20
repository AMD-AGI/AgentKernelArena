> **Exact-workload scoring follow-up:** Exact serving-workload performance is blocked. M=1/M=64 histogram/analytic buckets remain random/replay robustness cases only. They have no captured per-call paging/block controls and cannot be reported as exact HyperLoom performance. The observed oracle cases remain correctness inputs; they are not silently substituted for the declared serving buckets.

# decode_score_kernel: generated-input draft

This MiniMax-M3 task retains all **5 recorded correctness cases**, every
scored serving case, the original callable ABI, tensor shapes/strides/offsets,
selected page rows, block IDs, scalar arguments, and the original floating
`tol=0.02` comparison. Integer and boolean outputs use exact equality. Numerical Q/K/V values are generated from fresh evaluator-selected
seeds. Source kernel bodies are unchanged.

The default task requires **no external tensor archives**. Its two committed JSON
inputs total **245,505 bytes**:

- [ut/generated_cases.json](ut/generated_cases.json) contains the exact recorded
  correctness contracts and captured structural values, without large numerical
  buffers or golden outputs.
- [ut/timing_geometry.json](ut/timing_geometry.json) is a lossless conversion of
  the original small timing-geometry file. Existing scored input recipes and the
  common 10-warmup/100-sample benchmark are unchanged.

Run inside the image declared in [config.yaml](config.yaml), on gfx950:

```bash
python3 scripts/generated_task_runner.py compile
python3 scripts/generated_task_runner.py correctness
python3 scripts/generated_task_runner.py performance
```

The generated correctness controller runs protected reference and candidate
workers separately. It passes only profile/seed inputs; reference outputs remain
in parent memory and are never given to the candidate worker or reused as cached
goldens. It compares all tuple components and layout, rejects input mutation and
persistent outputs, and requires graph replay across all original boundary
variants followed by restoration of the first variant. The common performance
worker's existing protected per-run comparison remains unchanged.

Original archives remain optional for historical replay. With the unchanged
`reference_io.pt` placed in `ut/`, run:

```bash
python3 scripts/generated_task_runner.py archival --timeout 3600
```

Its original checksum is recorded under `ut/meta.json:archival_capture` and is
verified before restricted tensor loading. The optional archive was never
modified by this task revision. The [task metadata](ut/meta.json) records the
original captured-value contract and archived checksum.

This draft has CPU regression coverage and syntax/ABI compile checks. **It has
not passed fresh GPU correctness/performance/task-validator qualification.**
The original archive hashes and compact input contract are retained in `ut/meta.json`
and the task-local JSON files; isolated execution depends only on this task directory.

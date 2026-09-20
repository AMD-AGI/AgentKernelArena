# fwd_grouped_kernel_stage1: generated-input task

This Kimi-K3 task keeps all 2 package correctness geometries,
every declared scored case, the original ABI, dtypes, layouts, calibrated numeric
recipes, tolerance `0.02`, and 10-warmup/100-sample graph timing.
Source kernel bodies are unchanged.

**Coverage status:** this is an experimental operator benchmark, excluded from
claimed validated HyperLoom coverage until native scenario and numerical
qualification. The archived attention K pools have 395 and 524,352 rows; the existing timing recipe has 8,768 and 557,120 rows at context 8,704. Both are retained, but their pool/launch scenarios are not claimed equivalent.


Default execution needs no external tensor archive or cluster filesystem. The
committed [structural JSON](ut/generated_cases.json) is 1,061,119
bytes and retains exact integer paging/routing, tensor view/alias information,
case order, and output contracts. Numeric inputs and fresh reference outputs are
generated locally. Expert routing-weight values are generated while their
captured zero mask and integer routing remain fixed.

Correctness and performance use separate protected reference/candidate workers.
Only profile and seed enter each worker. Golden outputs stay in parent memory;
they are never supplied to a candidate or cached on disk. The parent checks all
case IDs, complete outputs, original floating tolerance, exact integer/bool
outputs, and all three original/changed/restored timed graph observations.
Helper/codec and worker functions use the shared trusted preloader and guard.
Aggregate workload scoring is disabled when any case lacks observed control evidence;
all semantic cases remain available. Stage 1 additionally requires every individual
launch in the retained independent torch-reference recipe to pass the original
tolerance, without median aggregation.

Attention keeps K/V slice aliases, noncontiguous paging and the live split-count
calibration gate. MoE keeps original launch kwargs and padded decode replay.

Run through the arena's Docker workspace preparation in the declared compatible
runtime, or in an already materialized task workspace:

```bash
python3 scripts/generated_task_runner.py compile
python3 scripts/generated_task_runner.py correctness
python3 scripts/generated_task_runner.py performance
```

The original archive and UT remain historical evidence. Their module-level candidate
binding cannot be safely preloaded by the shared guard, so archival execution needs
a separately reviewed archive-only configuration. The hash remains under
`ut/meta.json:archival_capture`.
No source archive was modified by this draft.

This is an isolated draft with CPU regression coverage. **Fresh GPU correctness,
performance and finalized task-validator qualification have not been run.**
Runtime-image portability is integrated separately by the suite owner; the old
capture image remains provenance rather than a downloaded fixture dependency.

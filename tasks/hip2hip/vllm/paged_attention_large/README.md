# Paged decode attention with large cache allocations

Optimize the bundled vLLM HIP implementation in `src/rocm/attention.cu`, using the
protected `paged_attention` binding. This task computes one decode query per
sequence with head size 128, block size 16, BF16 query/K/V/output, FP32 softmax,
and GQA ratios 4, 8 or 16. No alibi, FP8 cache, prefill, or non-unit scales are
claimed by this workload. Scratch buffers are allocated by the protected API;
all operator execution is inside the measured call.

The 14-case manifest retains all eight original allocation shapes and all three
previously performance-only batches (2,32,64). Every one now receives correctness
checks. Added cases cover noncontiguous page maps, ragged lengths, page/partition
tails at 17/257 tokens, and a 4096-token context. Large unused cache capacity is
preserved. Original oversized query allocations are decode buffers: only the
first `sequences` rows are active and all remaining output rows must stay zero;
this does not imply those rows perform prefill work.

The FP32 oracle independently gathers pages according to block tables, expands
GQA logically, masks sequence tails, computes scaled softmax and weighted values,
and casts to BF16. It never imports candidate code. The original elementwise
`atol=0.05, rtol=0.05` gate is retained; inactive rows are checked exactly. Queries
change during measured replay; K/V, scales, block tables, and lengths remain fixed
and immutable within each case. Randomized page maps use disjoint pages.

Existing upstream Apache-2.0 copyright and license headers are retained. The
source-bound HIP build operates on a disposable copy under `build/` so hipify
cannot change protected files. All native launches must follow the current stream.

## Evaluation contract

The schema-v2 runner implements all seven arena-eval-v1 actions. Baseline actions
use the framework's frozen initial candidate in a separate workspace; candidate
actions use the same relative interface in the working workspace. Every manifest
case participates in both correctness and performance. Compilation builds and
launches every specialization, rather than accepting source text alone.

The protected task API constructs inputs, computes an independent reference before
calling the implementation, and rejects wrong shape, dtype, device, non-finite
values, and modified read-only inputs. `validate-task` also tests deliberately
wrong, NaN, and wrong-dtype outputs against the comparator.

Performance uses the materialized canonical GPU graph/Event helper: 10 warmups,
100 device samples, one logical invocation per sample. Preparation copies fresh
call inputs outside device timing. Three randomly seeded draws rotate at the same
addresses; eight privately selected measured outputs and the final actual replay
are checked against references computed before any candidate execution. Four
previously unused draws follow the same device timing path; correctness must pass
and their minimum time must be at most 1.5 times the reported mean. A separate
poisoned-output replay checks complete writes, after actual samples have already
been verified. Seeds, checked sample indices, replay checks, timing method, and
unseen timings are retained in result metadata. This is a practical integrity
check, not a sandbox for adversarial native code or a proof against all caching.

Only declared candidate implementation code is editable. The workload, references,
launch ABI, build policy and runner are protected. Do not read or import protected
reference code, cache answers, change task files, download dependencies, or access
other workspaces. Use only the bundled implementation dependencies and the runtime
compiler. Keep allocations and computation within the measured invocation except
for the input/output/scratch buffers allocated by the protected API.

Only gfx950 is declared. Other architectures require separate qualification.
Run the formal validator through the repository's Docker runner:

```bash
make docker-run CONFIG=example_configs/validate_inference_tasks_mi355x.yaml
```

A successful direct runner command is diagnostic evidence. The acceptance gate is
a fresh framework-finalized `validation_report.yaml` with `overall_status: PASS`.

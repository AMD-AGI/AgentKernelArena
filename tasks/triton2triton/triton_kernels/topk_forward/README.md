# Fused TopK routing

Optimize the Triton functions in `source/kernel.py`. The protected host launcher
calls `_topk_forward` and returns BF16 selected-value softmax probabilities,
int16 expert indices, and a uint32 routing bitmatrix `[experts/32, rows]`. Indices
are descending by logit, with smaller expert index first on ties. Each row chooses
unique experts. The bitmatrix must match those exact selected indices.

The seven cases retain original row counts 2048,32768,65536 with 128 experts and
K=4. Added cases cover row tails (1,31,33,257), 64/256 experts, K=1/2/8, negative
logits, and deliberate tied logits. Logits change between timed invocations.
The implementation remains single-peer, applies softmax only to selected logits,
and does not use preselected indices. These are explicit workload constraints.

The independent reference uses stable PyTorch sorting, gathers logits, applies
FP32 softmax then casts to BF16, and constructs routing bits independently. Indices
and bits must match exactly. The original probability gates remain: cosine
similarity >=0.99 and maximum relative error <=0.01 (denominator floor 1e-6).
NaNs, wrong integer outputs, and invalid tensor contracts are rejected explicitly.

Only Triton imports and implementation helpers are allowed in candidate code.
The fixed host launcher, output allocation, and reference are protected; output
and input buffers are allocated before device timing for both roles. This is a
kernel-only task, not a Python dispatcher optimization task.

The bundled helper chain is from triton-lang/triton commit
`2046eb542a9c30e5bc770b7c6671f03f9adbdf55`; see `THIRD_PARTY_NOTICES.md` for the
retained MIT license and attribution.

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

# PR107 verification checkpoint — 2026-09-15

This dated checkpoint separates task validation, agent optimization, and framework
testing by the source actually exercised. It does **not** establish a full GPU
pass for the final integration revision. The [pinned-main integration record](pr107-main-integration.md)
describes the merge; later shared-framework and agent changes have their own
qualification below.

## Source and verification scope

| Source | Verified result or current gap |
| --- | --- |
| `e8ec5d6b4bc9d62b38af59a66a1797da42d3f30f` | All **438** retained v2 task packages have source-matched, framework-finalized validator PASS evidence. The audit rechecked report hashes and completion markers; individual reports retain their original worker, helper, runtime and model identities. This was not one 438-task campaign on the final framework. Diagnostic baseline policies are not numerical PASS claims. |
| `c1dc5e0904714703c10d0bdcf9a3bf132c77c073` | The main merge identified **55** tasks requiring fresh GPU validation. At the independently audited **20:33 UTC** checkpoint, the first four reports were **3 PASS / 1 WARN**. The WARN is not a clean pass; qualification of all 55 remains open. Unreviewed or pending reports are not counted here. |
| `27f2846189fd8fb1e35e4e92aa669eb7eb0ea7d3` | Full CPU suite: **13,700 passed, 5 GPU-only skipped, 6 subtests passed** in **562.49 s**. The two-task quality-loop GPU smoke below completed with two accepting independent reviews. |
| `87e7e4639542d28c5d61e6dc36e7e48b9f4f3c0d` | The guard-context follow-up preserves enforcement and clarifies the effective symbol/file boundary in trusted metadata and prompts. Full CPU suite: **13,772 passed, 5 GPU-only skipped, 6 subtests passed**, **129 warnings**, **607.16 s**; exit 0, completed at **21:02:54 UTC**. Fresh two-MoE GPU proof remains pending. |

The 87 CPU command is `python -B -m pytest -p no:cacheprovider -q tests`,
with `PYTHONDONTWRITEBYTECODE=1` and JUnit output retained. Its native dependency
bindings are GEAK `c0c0e2aee5e2bec70583253058382523bdf7a3ab`, Node `v24.19.0`,
and installed KernelForge Python sources verified against Hyperloom
`0425bde3f6e76e1588400c37d056dfd3bb75ac11`. These CPU tests do not make provider
calls or qualify GPU execution. The checkout remained clean at the same commit
and tree before and after the run; no pytest cache was created. The five skips
are two opt-in GPU evaluation-tool probes and three GPU graph/event smoke tests.

## Actual quality-loop smoke

Slurm job **141055** completed normally on one **MI355X**, using source `27f28461`,
`gpt-5.6-terra` with medium reasoning, and `--no-publish`. Its two original GELU
tasks retained all **11 cases** each. Initial task-package bytes were separately bound to e8;
no generated repair was transplanted into the source.

| Task | Final review | Measured speedup |
| --- | --- | --- |
| `hip2hip/gpumode/GELU` | Accepted after ordinary initial validation, optimization and formal candidate evaluation | 1.000993×; candidate bytes unchanged |
| `torch2hip/gpumode/14539_GELU` | Accepted after ordinary initial validation, optimization and formal candidate evaluation | 0.893261×; slower than baseline |

The audit checked **28 raw action envelopes**, four session snapshots, the two
framework-owned review indexes and the reviewers' actual reads, seven native
Codex calls, and five durable role logs. Acceptance is a correctness/evidence
result, not a claim of useful speedup. No baseline promotion was triggered. A
case-enhancement proposal was rejected for edits outside the allowlist before
its correctness gates; it was not adopted. No changes were published and no
matrix completion was added. See [the quality-loop guide](../how-to/quality-loop.md)
for the conditional stages, artifact locations and full smoke limitations.

## Agent matrix checkpoint

The independently verified snapshot at **20:52:20 UTC** counts unique selected
`(agent, task)` pairs with a completed native optimization attempt. It keeps
candidate acceptance separate from completion and retains each run's source.
Retries, probes and saved-candidate reevaluations add no distinct pairs.

| Agent | Completed | Accepted candidate | Required completed |
| --- | ---: | ---: | ---: |
| Codex | 45 | 45 | 45 |
| Claude Code | 22 | 21 | 45 |
| Forge | 44 | 40 | 45 |
| GEAK | 5 | 5 | 45 |
| **Total** | **116** | **111** | **180** |

There are **64 distinct pairs remaining** against the completion target. This is
a verified checkpoint, not a live counter or a claim of 180 accepted speedups.
The historical GEAK source-39 evidence includes SDK typed completion metadata,
but lacks full original tool-ID/session correlation; it must not be relabelled
as qualification of the later strict 27f completion gate.

## Open work and quota hold

At **20:53:10 UTC**, retained GEAK runtime evidence confirmed an actual
`five_hour` quota rejection on the account shared with Claude. Its reported
reset is **2026-09-16 00:00 UTC** (`1789516800`). This is not an authentication
failure. New Claude/GEAK admissions are held until the reset **and** a successful
normal admission check; already admitted work may finish. Quota-rejected tasks
receive no completion credit. Independent Codex and CPU work can continue.

Remaining qualification consists of fresh guard87 GPU evidence, resolving the
55-task merge revalidation results, and the remaining
agent matrix pairs. Historical PASS reports are retained with their actual
source; they do not automatically qualify subsequent shared-code changes.

## Retained primary proofs

These files are in the campaign evidence bundle under
`logs/task-schema-refactor-20260915/`; generated logs are not committed:

- `verified-initial-validation-438-e8.json`
- `verified-main-c1dc-gpu-preparation.json` and `verified-main-c1-first4.json`
- `verified-final-combined-27f-cpu.json`
- `verified-quality-loop-27f-smoke.json` and `verified-quality27f-review-locators.json`
- `verified-guard87-preparation.json`
- `verified-matrix-completed116-accepted111.json`
- `observed-geak141010-quota-hold-2050.json`

The separate immutable 87 checkout retains its command, dependency/source
bindings, full log, JUnit XML and result under `logs/final-cpu-87e7e463/`.

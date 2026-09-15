# PR107 verification checkpoint — 2026-09-15

This dated checkpoint separates task validation, agent optimization, and framework
testing by the source actually exercised. The latest matrix has **117 completed
optimization pairs and 113 accepted candidates**. All **438 current task packages**
at 9c now have runtime-applicable validator PASS evidence, and the paired MoE
study has passed its independent raw audit. The original 9c CPU run still records
**one failure**. Its test-only successor `f65e1e55` passed the full CPU suite:
**14,019 passed, 6 subtests passed and 6 skipped**, in **616.62 s**, with clean,
unchanged source. The separate GEAK Codex extension is not yet qualified. The
[pinned-main integration record](pr107-main-integration.md) retains the original
merge decisions and their later qualification.

## Source and verification scope

| Source | Verified result or current gap |
| --- | --- |
| `e8ec5d6b` | All **438** retained v2 task packages have source-matched, framework-finalized validator PASS evidence. Individual reports retain their original worker, helper, runtime and model identities. This was not one 438-task campaign on the final framework. Diagnostic baseline policies are not numerical PASS claims. |
| `c1dc5e09` | All **55** original merge-revalidation outcomes are retained: **49 PASS / 1 WARN / 5 FAIL**. The WARN is not a clean pass, and later repairs do not rewrite these outcomes. One task-level PASS has the explicit outer inventory exception described below. |
| `27f28461` | Full CPU suite: **13,700 passed, 5 GPU-only skipped, 6 subtests passed** in **562.49 s**. The two-task quality-loop GPU smoke below completed with two accepting independent reviews. |
| `87e7e463` | Guard-context follow-up. Full CPU suite: **13,772 passed, 5 GPU-only skipped, 6 subtests passed**, **129 warnings**, **607.16 s**; exit 0. This CPU result did not resolve the subsequently reproduced MoE helper-policy gap. |
| `400fcb9d` | Seven fresh full validators: **6 PASS / 1 Pack semantic FAIL**, including two MoE PASS reports. Full CPU run: **13,996 passed / 4 fixture failures / 6 skipped**, with **6 subtests passed**, in **606.52 s**. This is a failed CPU run. Five saved-candidate reevaluations passed separately. |
| `9c4c99f1` | Fresh Pack validator **PASS**: **11 correctness / 5 scored cases**, independently audited at job 141241. Full CPU run: **14,013 passed / 1 failed / 6 skipped**, with **6 passing subtests**, in **612.74 s**. The remaining failure is the old Pack candidate-source hash assertion; full CPU qualification is still open. |
| `f65e1e55` | Test-only follow-up pins the exact approved Pack flatten change while retaining the original hash check. **168 focused checks passed**. Immutable full CPU suite: **14,019 passed / 6 skipped / 6 subtests passed**, **129 warnings**, **616.62 s**, exit 0. Source was clean and unchanged. The failed 9c result is retained. |

The 87 CPU command is `python -B -m pytest -p no:cacheprovider -q tests`,
with `PYTHONDONTWRITEBYTECODE=1` and JUnit output retained. Its native dependency
bindings are GEAK `c0c0e2aee5e2bec70583253058382523bdf7a3ab`, Node `v24.19.0`,
and installed KernelForge Python sources verified against Hyperloom
`0425bde3f6e76e1588400c37d056dfd3bb75ac11`. These CPU tests do not make provider
calls or qualify GPU execution. The checkout remained clean at the same commit
and tree before and after the run; no pytest cache was created. The five skips
are two opt-in GPU evaluation-tool probes and three GPU graph/event smoke tests.

The c1 `fused_qkv_rope` task-level PASS retains **one outer bytecode-inventory
exception**: a parent audit imported frozen source and created interpreter caches.
Tracked source bytes were verified unchanged; the original outer FAIL is retained,
not rewritten as a clean pipeline pass. This exception is separate from the WARN
and five semantic FAIL reports.

The corrected final 9c task-applicability aggregate binds **438** validator PASS
reports, grouped by applicability checkpoint: e8 (383), c1 (48), 400f (six) and
9c (one). The 383 reuse historical reports already qualified for the e8 task
versions; e8 is not their common execution revision. Each row separately retains
`validation_worker_revision_as_recorded`, including originally abbreviated IDs
without expanding them. These checkpoint counts are not counts by execution
revision. The task comparison finds
383 exact task trees, 54 documentation-only differences and one task-local
regression-test difference with unchanged runtime code. This establishes current
task-package applicability, not one fresh 438-task run on the final shared
framework or on the planned GEAK backend. The QKV outer exception remains explicit.

## Repair and saved-candidate evidence

The 400f Pack FAIL identified missing promised all-empty coverage even though the
declared action cases passed. A separate MI355X probe of the **unmodified 400f
wrapper** then established the actual behavior: the 2-D all-empty input passed
with the original zero time-grid, while the 3-D all-empty input failed at
`reshape(0, -1)` before JIT compilation. The failure was not an unsupported
zero-grid launch.

The repair in 9c uses `flatten(start_dim=1)` for the trailing feature dimensions
and adds two unscored all-empty cases. It preserves the native launch, all nine
previous manifest rows and all five scored rows, without an early-return bypass.
The separate CPU fixture repair loads the task's real `_upstream_controls` module
in an isolated import context. The four original 400f import failures remain in
their report. The full 9c run resolved those imports but failed the old candidate
hash assertion in
`test_original_manifest_rows_and_generated_region_are_byte_preserved[pack_seq]`.
The f65 test-only follow-up handles exactly the approved flatten substitution
before applying the original source hash assertion. Its 168 focused passes and
subsequent full CPU PASS qualify f65 separately; the earlier failed full CPU
reports remain unchanged. The six full-suite skips are five opt-in or GPU-only
probes and one check requiring a materialized external image source. This CPU
result does not qualify GPU execution or the separate GEAK Codex extension.

The accepted113 intersection with the 55 runtime-changed tasks is **five saved
candidates**: Codex, Forge and Claude AWQ dequantize, plus Codex and Forge
apply_write. Their exact archived candidates passed **35 formal actions** on
400f; the two task packages are byte-equivalent at 9c. Original search report and
candidate hashes were rechecked. This closes their saved-candidate reevaluation
requirement with **zero new searches or distinct optimization pairs**; it does
not relabel their historical engine versions as new-source optimization runs.
The original 112 intersection receipt is retained alongside the additive 113
receipt. Ragged's task package is unchanged across its 475 source, e8 and c1;
it adds no reevaluation pair. Claude swizzle's e8-to-c1 change is README-only.

Both MoE validators passed on 400f. The separate paired study, job **141240**, has
completed all **32 measured performance actions**, covering four balanced cycles
and all 16 original cases per task on the same physical MI355X: **512 measured
case observations**. The parent's independent audit reparsed **76 public actions**
(32 measured and 44 prerequisites) and passed without errors.
The logical comparison is e8 before versus c1 after, with the after arm actually
executed on 400f and its execution-equivalence evidence retained.

| Task | Median baseline after/before latency | Median candidate after/before latency | Median after candidate/baseline latency |
| --- | ---: | ---: | ---: |
| `instruction2triton/rocmbench/moe_gemm` | 0.9357 | 0.9330 | 1.0007 |
| `triton2triton/rocmbench/hard/moe_gemm` | 0.9488 | 0.9484 | 1.0011 |

These are medians of 64 paired case/cycle latency ratios per comparison; lower
after/before values mean lower measured latency. The editable kernel implementation
is unchanged, with the same preparation boundary for baseline and candidate within
each source version. The result measures the harness's host-scalar preparation
boundary, not an agent optimization gain. Original warmups and sample counts are
retained, without outlier removal. One device and four cycles do not establish
fleet-wide variance or statistical significance; individual raw 100-sample
timings were not emitted by the original harness and are not reconstructed.

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

The independently verified **117-completed / 113-accepted** snapshot counts unique selected
`(agent, task)` pairs with a completed native optimization attempt. It keeps
candidate acceptance separate from completion and retains each run's source.
Retries, probes and saved-candidate reevaluations add no distinct pairs.

| Agent | Completed | Accepted candidate | Required completed |
| --- | ---: | ---: | ---: |
| Codex | 45 | 45 | 45 |
| Claude Code | 22 | 21 | 45 |
| Forge | 45 | 42 | 45 |
| GEAK | 5 | 5 | 45 |
| **Total** | **117** | **113** | **180** |

There are **63 distinct pairs remaining** against the completion target. This is
a verified checkpoint, not a live counter or a claim of 180 accepted speedups.
The historical GEAK source-39 evidence includes SDK typed completion metadata,
but lacks full original tool-ID/session correlation; it must not be relabelled
as qualification of the later strict 27f completion gate.

The last Forge Ragged acceptance adds no distinct completion. Its two native
KEEP rounds and seven formal actions passed; the single fresh measurement was
**1.0512×**, not a stable-gain claim. The shared disk, returned native-loop and
pipeline usage counters remain separately recorded; unavailable dollar cost is
not zero. The diagnostics run retained 68 raw action records: 66 were verified
after temporary-workspace cleanup, and two were audited while the workspace
still existed. These success-path records do not establish the cause of the
earlier failed run.

## Open work and revised agent plan

At **20:53:10 UTC**, retained GEAK runtime evidence confirmed an actual
`five_hour` quota rejection on the account shared with Claude. Its reported
reset is **2026-09-16 00:00 UTC** (`1789516800`). Authentication has been repaired;
the quota rejection is a separate historical condition. The user subsequently
changed the remaining campaign plan: both old midnight waiters are **DISARMED**,
and no new Fable/Opus dispatch is authorized. The former
midnight plan is superseded; its preparation receipts remain historical evidence,
not current launch authorization.

The remaining **23 Claude Code tasks have an explicit Sonnet 5 / medium configuration
prepared**. The remaining **40 GEAK tasks have a draft native Workflow Codex
implementation**, which is not yet qualified. An independent Codex run would not
qualify a GEAK run. Each campaign remains on **HOLD** until its exact source,
native backend, model and runtime are bound and qualified. Sonnet qualification
can proceed independently of the GEAK extension. Historical accepted
runs retain their original model/backend identities; the 117/113 matrix is
unchanged. Quota-rejected tasks receive no completion credit.

The core full CPU gate is complete at f65. Remaining qualification covers the
planned Sonnet and native GEAK Codex configurations and the remaining 63 agent
matrix pairs; Sonnet does not depend on the GEAK extension.
There are no known unresolved GPU task issues from the original 55-task
revalidation at this checkpoint; that statement does not qualify a new GEAK
implementation. No GPU jobs were outstanding at the **22:56 UTC** observation.
Forge coverage is complete;
this checkpoint does not request further Forge jobs. Historical PASS reports
retain their actual source and do not automatically qualify later shared-code
changes.

## Retained primary proofs

These files are in the campaign evidence bundle under
`logs/task-schema-refactor-20260915/`; generated logs are not committed:

- `verified-initial-validation-438-e8.json`
- `verified-main-c1-all55-original-outcomes.json` and `verified-main-c1-next19.json`
- `verified-main-c1-audit-cache-incident.json`
- `verified-final-combined-27f-cpu.json`
- `verified-quality-loop-27f-smoke.json` and `verified-quality27f-review-locators.json`
- `verified-guard87-preparation.json`
- `verified-final400f-five-gpu.json` and `verified-final400f-moe-gpu141183.json`
- `verified-final400f-cpu-failure.json`
- `verified-final9c-pack-gpu.json`
- `verified-final9c-tasks438-v2.json` (SHA-256 `8b20f0dd1f7139395c8fb2afa12f45606bf3bf6922737871db8bca23de5a135d`; corrected worker-revision metadata, with the original aggregate retained)
- `moe141240-parent-raw-audit.json`
- `verified-final400f-saved5.json` (SHA-256 `595d2b901a2ebebf36282adae2050e90b2df5176b3f20316868941defb6a3641`)
- `verified-matrix-completed117-accepted113.json`
- `verified-forge140967-new-accepted.json` and `verified-forge475-final-action-evidence.json`
- `observed-geak141010-quota-hold-2050.json`
- `geak-midnight40-durable-waiter-r2.json` and `geak400f-conditional-offline-preparation.json`
- `geak-user-codex-steering-disarmed.json`

The separate immutable 87 checkout retains its command, dependency/source
bindings, full log, JUnit XML and result under `logs/final-cpu-87e7e463/`.
The failed full 9c run is retained under `logs/final-cpu-9c4c99f1/`; its source
was clean and unchanged before and after the run. The separate f65 full PASS
retains `full-result.json`, the full log and JUnit XML under
`logs/final-cpu-f65e1e55/`, with clean before/after source identities and all six
skip reasons. The paired study retains its
`paired-analysis.json` (SHA-256
`46560203fe81034e605fadf4099df20e307698527789da89abb4467d72a5f5ba`)
and all measured and prerequisite action records in its experiment bundle.
The Pack repair handoff under `logs/pack-empty/` binds the original GPU probe,
whose `original-wrapper-probe.json` SHA-256 is
`f1d55e604af906795040d6e48b894f2a00a8f37d8a7e11291bc7770328ee680f`.
Pending results are not counted as PASS in this document.

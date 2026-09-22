# PR107 verification checkpoint — 2026-09-15

Status updated **2026-09-16**. Historical reports retain their original execution
source, model, runtime and timestamp; this update does not relabel them as runs
on the current publication revision.

The final matrix contains **180 completed / 173 accepted** unique `(agent, task)`
pairs. Codex, Claude Code, Forge and GEAK each completed five tasks in every
one of the nine families; **no selected pairs remain**. All **438 task packages** have
runtime-applicable, framework-finalized validator PASS evidence. These are
separate claims: task validation does not complete an agent optimization pair,
and a completed attempt need not produce an accepted or faster candidate.

The full local CPU gate passed on `f65e1e55`: **14,019 passed, six subtests passed,
six skipped**, **616.62 s**, on clean unchanged source. At `68ccbaac`, GitHub PR
and push CI each passed **13,972 tests and six subtests, with 149 explicit skips
and 129 warnings**. Benchmark-helper audit, source compilation and the separate
documentation check also passed. The CI skip scope and local native-dependency
coverage are recorded below. V3 and v4 have actual native/GPU evidence for their
object-argument paths. The v4 fixed-argument dispatcher completed a correlated
native task lifecycle with an independently accepted candidate; its native role
failures remain failures, not successful search claims. The experimental GEAK
Codex backend remains outside this PR.
See the [pinned-main integration record](pr107-main-integration.md) for the
original reconciliation decisions.

## Source and verification scope

| Source | Verified result or current gap |
| --- | --- |
| `e8ec5d6b` applicability checkpoint | All **438** retained v2 task packages have source-matched, framework-finalized validator PASS evidence. Reports retain their actual worker, helper, runtime and model identities; e8 is not their common execution revision. Diagnostic baseline policies are not numerical PASS claims. |
| `c1dc5e09` | All **55** original merge-revalidation outcomes remain **49 PASS / 1 WARN / 5 FAIL**. The QKV task-level PASS has the separate outer inventory exception below. Later repairs do not rewrite these outcomes. |
| `27f28461` | Full CPU: **13,700 passed / 5 skipped / 6 subtests passed**, **562.49 s**. The two-task quality-loop GPU smoke below completed with two accepting independent reviews. |
| `87e7e463` | Historical guard-context CPU: **13,772 passed / 5 skipped / 6 subtests passed**, **607.16 s**. This did not close the subsequently reproduced MoE helper-policy gap. |
| `400fcb9d` and `9c4c99f1` | Fresh validators: **6 PASS / 1 Pack semantic FAIL** at 400f, then Pack **PASS** at 9c. The failed full CPU runs remain retained: 400f had four import-fixture failures; 9c had one obsolete Pack source-hash assertion. |
| `f65e1e55` | Test-only successor checks the exact approved Pack flatten change while retaining the original hash assertion. **168 focused PASS**; full CPU **14,019 passed / 6 skipped / 6 subtests passed**, **616.62 s**, exit 0, clean unchanged source. |
| `c68efa03` | Compact native Workflow dispatch. Combined GEAK CPU checks: **133 passed / 1 skipped**. Earlier overlapping component results are **121 passed** for compact dispatch and **67 passed** for exception-group compatibility; these are not additive. Actual task outcomes retain c68 source identity in the matrix. |
| `d601ec45` | PR and push CI each: **13,961 passed / 65 skipped / 6 subtests passed**, including the documentation build. Actual task outcomes retain d601 source identity in the matrix. |
| `798f6807` | PR and push CI each: **13,971 passed / 128 skipped / 6 subtests passed**, **129 warnings**; helper audit, compilation and docs passed. Opt-in argument transport also has **115 component CPU passes**, zero skips, and independent review **NO_BLOCKING_FINDINGS** including **26 focused passes**. Counts overlap; these checks add **no GPU or matrix credit**. |
| `6e802d0b` execution / `798f6807` identical tree | AWQ and FP8 paged-attention decode completed actual native/GPU task lifecycles through v3 **object** arguments. Original call/return, engine/dispatcher identities and formal candidate acceptance were independently verified; native role failures remain `FAILED`. This does not qualify the JSON-string branch on GPU. |
| `5895a99b` execution / `68ccbaac` identical tree | V4 fixed arguments: **230 component CPU passes**, zero skips; independent **38 focused passes / NO_BLOCKING_FINDINGS** (overlapping counts). Job **142937**, fused shared experts, completed actual native work through outer object `{}` with **18 correctness / 18 performance cases PASS** and an accepted candidate. Full bound arguments, engine/dispatcher hashes and original call/return were verified. Native role/schema failures remain recorded. |
| `68ccbaac` CI | PR and push CI each: **13,972 passed / 149 skipped / 6 subtests passed**, **129 warnings**. Benchmark-helper audit, source compilation and documentation build passed. CI adds no GPU or matrix credit. |

The latest PR run `35057823093` and push run `35057818019` used CPython 3.12.14;
their pytest durations were **641.13 s** and **622.18 s**, respectively. Their
149 skips retain the logged optional-dependency and GPU requirements. In
particular, CI does not supply the pinned external GEAK checkout for all native
checks. The separate local **230-pass** component run supplied that dependency
and covered the corresponding GEAK branches; the **38-pass** independent review
overlaps that scope. Neither converts CI skips into passes or supplies extra GPU
credit. Earlier CI results in the table retain their own sources and skip counts.

The 798f transport keeps exact permitted argument values and checks safe integer
representation before native dispatch. The independent CPU review exercised the
actual adapted JavaScript dispatcher. Tasks, shared `src/` and `main.py` are
unchanged. This checks the outer Workflow argument boundary; it does not establish
that inner model roles can produce their required structured outputs.

V4 removes the need for the outer model to reproduce the complete task arguments:
the prepared dispatcher binds them, including the task contract, while the outer
call supplies `{}` or a string decoding to `{}`. Nonempty arguments remain
rejected; source/engine identity and original call/return correlation still apply.
The actual fused-experts run verified the raw outer **object** encoding and every
trusted argument bound into the selected dispatcher, including the full baseline
case set and task contract. This closes actual v4 native integration readiness
for that path. Equivalent JSON-string encoding still has CPU/code evidence only.
The recorded campaign cost settings and original host deadline remain explicit;
no hard token/USD cap or native search success is inferred. V3 executions retain
their frozen source and are not relabeled as v4 runs.

Some earlier decoder rejections retained only the invalid-decoder classification,
input type and hash, not the original rejected string. Those records do not prove
a particular JSON syntax error. This outer-transport limitation is separate from
the native role-output schema failures discussed below; neither justifies a
blanket model-cause attribution or retry of a completed attempt.

Full local CPU receipts retain the `python -B -m pytest -p no:cacheprovider`
command, pinned native Forge/GEAK dependencies, logs, JUnit XML, source identities
and explicit skip reasons. The f65 skips comprise five opt-in or GPU-only probes
and one check requiring a materialized image source. The c68 combined skip was
an absent `exceptiongroup` backport; its separate compatibility check exercised
the real backport under Python 3.12, not a Python 3.10 runtime. None of these CPU
results replaces GPU qualification.

The c1 `fused_qkv_rope` task-level PASS retains **one outer bytecode-inventory
exception**: a parent audit imported frozen source and created interpreter caches.
Tracked source bytes were verified unchanged; the original outer FAIL is retained,
not rewritten as a clean pipeline pass. This exception is separate from the WARN
and five semantic FAIL reports.

The corrected 9c task-applicability aggregate binds **438** validator PASS
reports: e8-qualified task versions (383), c1 (48), 400f (six) and 9c (one).
These are applicability checkpoints, not execution-revision counts. The aggregate
retains each recorded worker revision and distinguishes 383 exact task trees,
54 documentation-only changes and one task-local test change with unchanged
runtime code. It does not claim a fresh 438-task run on the final shared framework
or a later GEAK adapter.

## Repair and saved-candidate evidence

The 400f Pack FAIL exposed missing all-empty coverage. An unmodified-wrapper
GPU probe passed the 2-D zero-grid path but failed the 3-D input at
`reshape(0, -1)` before JIT. The 9c `flatten(start_dim=1)` repair preserves the
native launch, all nine prior manifest rows and all five scored rows, and adds
two unscored all-empty controls. Its fresh full validator passed **11 correctness
/ 5 scored cases**. Original failures and the separate CPU fixture repairs remain
in their source-bound reports.

Five archived candidates in the accepted113 checkpoint required reevaluation on
changed tasks: Codex, Forge and Claude AWQ dequantize, and Codex and Forge
apply_write. All **35 formal actions passed** on 400f, with those task packages
byte-equivalent at 9c. This adds **zero searches or distinct matrix pairs** and
does not relabel the original agents or source revisions.

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

## Final agent matrix

The independently verified final matrix is **180 completed / 173 accepted**.
Each pair retains its actual execution source. A bounded native capability or
configured-budget terminal may count as completed after real task-specific model
work, even when the agent failed. Candidate acceptance depends independently on
the submitted bytes and formal task checks. Budget exhaustion alone neither
establishes nor removes acceptance. Native success, candidate acceptance and
speedup remain separate fields.

| Agent | Completed | Accepted candidate | Required completed |
| --- | ---: | ---: | ---: |
| Codex | 45 | 45 | 45 |
| Claude Code | 45 | 42 | 45 |
| Forge | 45 | 42 | 45 |
| GEAK | 45 | 44 | 45 |
| **Total** | **180** | **173** | **180** |

The independent structure check confirms **180 unique selected pairs = four
agents × nine families × five tasks**, with no duplicate IDs. The final receipt
reconciles all 180 completed pairs against that selection: every one of the 36
agent/family groups contains five completed tasks. Execution credit comes from
the source-bound terminal audits, not the structure check alone.
The parent-adopted canonical report confirms **180 completed / 173 accepted**.
The final independent set reconciliation also reports `coverage_complete: true`
and **zero remaining pairs**; their locators are listed below.

Retries, probes, saved-candidate reevaluations, provider interruptions and
pre-model infrastructure failures add no distinct completion credit. This is a
dated final matrix, not a claim of 180 accepted candidates or speedups. Older
checkpoints remain retained.
Older GEAK source-39 results have typed SDK completion metadata but
lack full original tool-ID/session correlation; they do not qualify the later
strict completion path.

The September 16 Claude continuation completed all **23 selected tasks** using
Sonnet 5 / medium: **20 native successes and three CLI budget terminations**,
with **21 independently accepted candidates**. Its deduplicated CLI list-price
estimate is **USD 22.8283**, not a subscription invoice. Per-invocation controls
were one outer attempt, 1,800 seconds and a CLI USD 2 threshold; thinking is
included in output-token accounting. These explicit campaign settings do not
change global defaults or historical model identities.

Recent GEAK attempts include real task operations and correlated native returns
containing structured-output failures, while their retained candidates passed
independent formal checks. Native failure and unknown director statuses remain
recorded; these are bounded attempts, not successful search claims. An unchanged
candidate can pass without demonstrating improvement. Search-credit units and
per-response output limits are not aggregate token or USD caps. Where full
provider requests and inner tool arguments were not retained, exact
model-versus-native transport attribution remains limited.

The bounded attribution review of d601 GELU and c68 SiLU found native
`StructuredOutput` errors for missing required fields and an incorrectly typed
`baseline_ms`, respectively. GELU Profile/Report roles succeeded through the same
path. No explicit AKA adapter unpacking defect was found, but complete child role
payloads and provider requests were not retained. The evidence neither attributes
all GEAK failures to model capability nor rules out every CLI/provider transform;
it supplies no positive basis for a parser repair or a capability-failure retry.

The last two GEAK outcomes retain these distinctions. Decode job **143004**
delivered a changed FlyDSL candidate that passed all **6 correctness / 6
performance cases**, with an observed arithmetic speedup of **0.0117×**. Native
Plan/Report structured-output failures remain `FAILED`. Softmax job **143005**
delivered candidate `954a7665…`, which passed all **18 correctness / 9 performance
cases**, with speedup **0.8548×**. Its native author failed to return the required
`authored`/`correctness` fields; this does not negate the independently checked
candidate. An earlier `68bf362e…` candidate failed all 18 correctness cases, and
another intermediate candidate failed performance. Those failures remain tied
to their source hashes within the same original attempt; no separate retry was
launched to replace them. Neither final accepted candidate demonstrates a speedup.

### Unaccepted completed attempts

The seven unaccepted pairs remain legitimate completed optimization outcomes:

| Agent and task | Observed outcome |
| --- | --- |
| Claude, SIKL GEMM n128/k6144 | Generated code calls `Float32.truncf`, which the installed FlyDSL API does not provide. Candidate compilation fails. |
| Claude, Triton-to-FlyDSL SGLang fused MoE | Native agent completed; compilation and correctness passed, but all eight formal performance cases failed with `Invalid device latency`. Candidate rejected. |
| Claude, Triton-to-FlyDSL SGLang decode attention | Native agent completed; candidate correctness exited with SIGSEGV (`-11`) and no result envelope. Performance was not run; candidate rejected. |
| Forge, Torch-to-FlyDSL batched BF16 GEMM | The archived candidate delegates computation to AITER's `flydsl_hgemm`. Its earlier numerical PASS does not satisfy the task's independent implementation contract. |
| Forge, Triton-to-FlyDSL SGLang fused MoE | The selected dot4 kernel uses a block stride of 256 with 128 threads, leaving half the output columns unwritten and subsequently reading uninitialized allocations. The same run's baseline passed all eight cases with matching protected harness bytes. |
| Forge, image-backed HIP paged-attention decode | The native implementer exhausted its 1,200-second session budget without an archived candidate. Later failures before optimization add no completion credit. |
| GEAK, SIKL MXFP4 MoE e257/i128 | A real private FlyDSL candidate passed compilation and all 13 standalone correctness cases, but all 13 performance cases failed. Retained numerical diagnostics include SQNR −12.75 and −∞ against the required ≥13 dB. Delivery failed; the public workspace retained its stub, and final evaluation rejected the missing builder entrypoint before candidate actions. |

The Claude MoE rejection must not be equated with a fully recovered failure
chain. The retained source review records a separate timing-stream-contract finding, while
the harness discarded the original benchmark exception. Decode's internal
candidate/compiler/runtime fault has not been isolated. Preserve the reviewed
root-cause evidence and its limits; these outcomes are not rerun to force green.

The Forge MoE diagnosis is a source-level indexing proof bound to the actual archived
candidate and original actions, not a new GPU run. The original combined
output/reference error did not retain separate finite-state flags, so the exact
historical NaN side remains unknown. That limitation does not remove the
confirmed candidate defect. No task repair or repeated optimization is required
to turn this result into a success.

The GEAK MoE author also returned an invalid structured-output `baseline_ms`
type. That error is separate from the private candidate's numerical rejection
and the final missing-entrypoint gate; it is not the sole failure explanation.
The private bridge retained validated case records and bounded diagnostics, not
full raw subprocess output. The root numerical defect has not been isolated,
and the public rejection must not be described as failed compilation of that
private candidate. SIKL has completed its five distinct GEAK tasks, including
this rejected outcome; no replacement run is needed to make the family green.

### Decode evidence gap and bounded recovery

GEAK decode-attention job **142907** is **not counted**. It matched one native
Workflow call, but no original correlated child terminal could be recovered.
The runtime recorded `native_return_missing`. Separately, the collector parsed
the model-writable delivery file before archiving native evidence; extra XML
after its JSON caused collection to stop. The retained file, model text, outer
SDK completion and candidate PASS do not establish the child outcome. This is
an evidence gap, not a proven capability or provider failure.

The reviewed **collector-only** correction has **81 CPU passes**. It saves the
original selected call, session/line hashes, results, notifications, dispatcher
bindings and raw native output before parsing delivery/output JSON. Malformed
delivery bytes and their parse diagnostic remain separate. It does not strip
XML, accept a JSON prefix, synthesize a return, weaken strict native success or
change production source, model, task or budget. This CPU result adds no matrix
credit.

The single v4 replacement, job **143004**, completed under the original
task/model/budget. Its original selected tool call, session, asynchronous completed
notification and raw child output were independently correlated; retained delivery
bytes parse completely and match that native return. The source is `5895a99b`,
whose tracked tree equals published `68ccbaac`. Its accepted candidate and failed
native outcome are recorded above. Only this evidenced successor receives the
pair's completion credit; job 142907 remains preserved and uncounted. No strict
success requirement was relaxed, and other completed capability failures were
not retried to obtain acceptance.

## Qualification limits

The original matrix target is complete; no additional optimization attempts are
needed to fill its task or family coverage. The final GEAK continuation used
Sonnet 5 / medium with the declared search and time budgets. Historical runs keep
their own models, sources and runtimes. Runtime CI, actual v4 object-path readiness
and the two-task quality-loop smoke retain their separate evidence scopes.

SGLang 18/19 evidence covers the recorded representative workloads, not every
runtime/task combination. Neither matrix completion nor CPU success establishes
180 accepted candidates, universal speedups, sanitizer coverage, whole-final-head
GPU qualification or promotion of a new default scoring image.

## Retained primary proofs

Review locators below are relative to the campaign evidence bundle's
`logs/task-schema-refactor-20260915/`. Generated logs and earlier failed runs remain
preserved outside Git; these files link to their underlying reports and hashes.

| Scope | Primary proof |
| --- | --- |
| Parent-adopted final canonical matrix | `verified-matrix-completed180-accepted173.json` |
| Last two terminal audits | `pasteur-geak-final180-173-receipt.json` (includes prior canonical summaries, final raw audits and 36-group reconciliation) |
| 438 task applicability | `verified-initial-validation-438-e8.json`; `verified-final9c-tasks438-v2.json` |
| Original c1 outcomes and QKV exception | `verified-main-c1-all55-original-outcomes.json`; `verified-main-c1-audit-cache-incident.json` |
| Final task repairs | `verified-final400f-five-gpu.json`; `verified-final400f-moe-gpu141183.json`; `verified-final9c-pack-gpu.json` |
| Saved candidates and paired MoE timing | `verified-final400f-saved5.json`; `moe141240-parent-raw-audit.json` |
| Full local CPU | `verified-final-f65-cpu-parent.json` |
| Quality-loop execution and reviewer evidence | `verified-quality-loop-27f-smoke.json`; `verified-quality27f-review-locators.json` |
| Current runtime CI | `verified-68cc-github-ci.json` |
| Transport CPU/review | `verified-798f-geak-transport-v3-cpu.json`; `verified-68cc-geak-fixed-dispatch-v4-cpu.json` |
| Actual v3 object-path qualification | `pasteur-geak-awq-image-padecode-162-156-receipt.json` (AWQ/pa_decode rows; the image GEMM row retains c68 source) |
| Actual v4 object-path qualification | `pasteur-geak-fp8-v4fused-166-159-receipt.json` (fused row; the FP8 image row retains v3 source) |
| GEAK SIKL MoE rejection | `pasteur-geak-applywrite-siklmoe-164-157-receipt.json` (MoE row) |
| Uncounted decode, collector correction and completed successor | `pasteur-decode142907-native-terminal-gap.json`; `verified-collector-retain-first-r2.json`; `pasteur-geak-final180-173-receipt.json` (decode row) |
| Prior CI checkpoints | `verified-d601-github-ci.json`; `verified-798f-github-ci.json` |
| Earlier compact-dispatch CPU | `verified-c68-geak-cpu-integration.json` |
| CLI and runtime scope | `verified-early-cli-smoke-evidence.json`; `verified-runtime18-19-representatives.json` |

Two additional owner bundles retain the structure-only review at
`logs/matrix180-structure-final-20260916/REVIEW.json` and the bounded native-role
attribution at `logs/geak-role-schema-attribution/REVIEW.md` and `REVIEW.json`.
Their scopes are set reconciliation and two recorded failures, respectively;
neither adds execution credit.

The final independent set reconciliation is retained in the
`geak-doc-qualification-e344` bundle at
`logs/matrix180-closeout-20260916/results/20260916T061353.459012Z-completed180-accepted173/matrix.json`.
It reports `PASS_SUMMARY_SET_RECONCILIATION`, complete 180-pair coverage and no
remaining tasks. The same directory contains `agent-family.csv`, `MAPPING.md`
and the empty `remaining-geak-task-ids.txt`. This is a reconciliation of adopted
outcome summaries, not a new GPU run or a historical raw-evidence re-audit.

The full CPU bundles retain failed 400f/9c results and the passing f65 log/XML
separately. Matrix receipts retain actual native outcomes and candidate bytes;
earlier pre-workflow failures are not upgraded by a later successful attempt.
Pending evidence is not counted as PASS.

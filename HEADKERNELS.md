# Consolidated top-five head kernels

`head-kernels-top5` is the single source branch for **23 SG0.5.20 task mappings across five models**. It contains the refreshed task sources, shared framework, five historical ready-subset guides, and workload archive links. The [runtime catalog](tools/headkernel-runtime-targets.json) and [unified validator selection](example_configs/headkernel_sg520_current_mi355x.yaml) enumerate the same 23 tasks.

**Qualification is still incomplete.** Five subsets retain qualification at their original pinned commits; the other 18 tasks remain unfinished. The merged framework also needs fresh validation for all 23 tasks. Matching a historical task tree does not qualify changed scoring, guards, report handling, or GPU binding. No current merged-checkout verification, complete model-workload readiness, or serving/E2E gain is claimed.

Start with the [consolidated task and evidence index](docs/reference/headkernel-ready-subsets.md). It links the five guides in this checkout, their immutable historical versions, the remaining gates, and the portable OCI workload archives. The original `ready/*/INPUT-PINS.json` and verification scripts remain strict historical records. They are expected to reject this merged checkout because shared framework and documentation files differ; do not rewrite those pins or treat their historical receipts as current validation.

| Model | SG0.5.20 source tasks | Historical ready subsets | Other unfinished tasks |
| --- | ---: | ---: | ---: |
| DeepSeek V4 Pro | 6 | 2 | 4 |
| GLM 5.3 Flash | 3 | 0 | 3 |
| Kimi K3 | 5 | 1 | 4 |
| MiniMax M3 | 4 | 1 | 3 |
| Qwen3.8 2.4T | 5 | 1 | 4 |
| **Total** | **23** | **5** | **18** |

## Sources brought into this branch

The consolidation base is `ba5b1c90877ca826901f2f56c0f1c02a4600d8b0`. These source commits are preserved in its history; their evidence keeps its original scope.

| Source | Commit | Included content and evidence scope |
| --- | --- | --- |
| Refreshed task/framework line | `57fb634c69047120f6ee5b1cf6cd6c9d3df43047` | Current non-Qwen tasks, shared evaluation and protection changes; source integration is not a new GPU result. |
| DeepSeek MLA prefill handoff | `55c8f713b06b206e0ae1d389e6726e65140078c0` | [Two-case receipt](ready/mla-prefill/READY.json), original input pins and guide. |
| DeepSeek quantization handoff | `23cb3067ab81f03264ec6027b97be17156f291a9` | [Three-shape receipt](ready/ds-quant/READY.json), original input pins and guide. |
| Kimi Lean handoff | `c39e94c8e148a171173eb276a9e495ab821c1649` | [Sampled-distribution receipt](ready/kimi-lean/READY.json), original input pins and guide. |
| MiniMax FP4 handoff | `ef7474dd9e29e74428ca09b4c488832db1fd220e` | [Fourteen-case receipt](ready/minimax-fp4/READY.json), original input pins and guide. |
| Full Qwen source refresh and RMSNorm handoff | `f501ef65403bdf76a122918c405f6494bc6eab38` | All five Qwen SG0.5.20 source mappings, including the four unfinished tasks; [historical RMSNorm receipt](ready/qwen-rmsnorm/READY.json) applies only to its explicit trusted/protected qualification route. |
| Published top-five documentation | `8fa0d73c81e55100d8e32d1963d6b247764fe56a` | Earlier published task index; superseded current-status wording is reconciled here without changing historical evidence. |

All five historical ready task trees matched their qualified trees at the consolidation base. Shared framework pins differ, including the timing-quality module for Qwen. The RMSNorm authoritative timing-quality adapter is integrated, with current GPU validation pending. The other four Qwen tasks still need benchmark repairs and authoritative admission. Historical manual trusted-route qualification does not discharge these current checks.

## Prepare and validate the current sources

Use the [portable setup guide](docs/how-to/headkernel-upstream-runtime.md) before running an agent. `prepare_headkernel_run.py` now selects **all 23 tasks** through explicit catalog protocols. The 18 manifest/native contracts retain `trusted_task_v1`; the five Qwen tasks use `headkernel_artifacts_v1`, pinned to the original whole-object registry. Three Qwen tasks require original tensor fixtures; dense GEMM and RMSNorm are explicitly registered as fixture-free. The helper verifies each selected input contract before generating the validator config. Preparation does not qualify the tasks or replace their capture provenance.

The runtime is the exact SGLang 0.5.20 MI355X image in the [catalog](tools/headkernel-runtime-targets.json). Use local NVMe for fixtures, caches, and results. Isolated task validation does not require a Crusoe allocation, shared model tree, model server, or whole checkpoint. A complete serving-workload reproduction has separate model and TP8 resource requirements described in the [workload archive index](docs/reference/headkernel-ready-subsets.md#workload-archives-in-oci).

After preparing each selected task's inputs, use the normal Docker workflow with `CONFIG=example_configs/headkernel_sg520_current_mi355x.yaml`. Set `AKA_SKIP_DEV_MEM=1`, install and authenticate the selected validator CLI, and run `make docker-check-agents` for that configuration before cache preparation or long validation; the [setup guide](docs/how-to/headkernel-upstream-runtime.md#run-the-standard-arena-validator) gives the command order and backend instructions. These host startup settings do not change task cases or timing policy. Require source-matched, framework-finalized reports and complete trusted comparisons before asserting current qualification. Preserve raw samples, source controls, failed attempts, and timing-quality decisions.

## Capture provenance and scope

The recorded non-Qwen workload recipe is 64 requests, ISL8192, OSL1024, concurrency 64, TP8, context 9218 and seed42, with OSL16 warmup followed by a prefix-cache flush. Task manifests remain authoritative about rank-local layouts, controls, occurrence counts and references. Sampled profiler counts must not be extrapolated into full-workload frequencies.

Qwen's sources target SG0.5.20, while its retained shapes and fixtures originate from the documented **SG0.5.18 TP8/EP8** capture, with page 1 attention and FP32 model-default Mamba. Later measured metadata record 103483 describes **TP8/EP1, context 11264, page 16 attention and BF16 Mamba**; the full image/raw-log record is absent. Those settings do not establish equivalent task dispatch. A complete SG0.5.20 Qwen workload capture and matching task provenance remain pending.

Kimi residual aggregation preserves the score/combine logical head through `kimi-k3__attn_residual_aggregate_hip`. Its fixtures contain the first `min(4,T)` actual token rows as labeled parity samples, not full native tensor dumps; full-shape independent checks remain required. GLM replacement ports retain their task-specific native-parity limits. A port-versus-port ratio is not a serving-runtime gain.

Historical implementations retain explicit [dispositions](docs/reference/headkernel-superseded-attention.json): DeepSeek sparse MLA is superseded by unified paged prefill/decode for this recipe; Kimi grouped attention is a native low-batch branch while served batch 64 uses Lean; GLM scale-layout copies produced aliases. The two inactive configured packages are excluded from the 23-task current selection.

The [refresh chronology](docs/reference/headkernel-sg520-refresh.md), [reported workload figures](docs/reference/headkernel-reported-results.md), [upstream README](HEADKERNELS_UPSTREAM.md), [capture-era catalog](tools/headkernel-public-runtimes.json) and [original publication record](tools/headkernel-publication.json) preserve their original dates and scope. Their older runtime settings and reported gains are historical evidence, not defaults or qualification for this merged branch.

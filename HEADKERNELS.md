# Consolidated top-five head kernels

`head-kernels-top5` is the single source branch for **23 SG0.5.20 task mappings across five models**. It contains the refreshed task sources, shared framework, five historical ready-subset guides, seven completed scoped qualification handoffs, and workload archive links. The [runtime catalog](tools/headkernel-runtime-targets.json) and [unified validator selection](example_configs/headkernel_sg520_current_mi355x.yaml) enumerate the same 23 tasks.

**The full 23-task branch remains unqualified.** [MLA prefill](ready/current/mla-prefill/README.md) and [MiniMax FP4](ready/current/minimax-fp4/README.md) have completed trusted and framework evidence at `7bc14698`, with fully verified OCI archives and separately named compatible revisions. [Current Qwen RMSNorm](ready/current/qwen-rmsnorm/README.md) combines the fresh `18bb4d67` framework pass with explicitly compatible trusted evidence from `294e65a0`. [Current Qwen paged attention](ready/current/qwen-attention/README.md) joins fresh trusted evidence at `18bb4d67` with the explicitly compatible framework execution at `8956e210`. [Current Qwen recurrent GDN](ready/current/qwen-recurrent/README.md) combines fresh trusted evidence at `eed4df57` with retained framework evidence at `18bb4d67` under explicit compatibility review. [Current Qwen MoE](ready/current/qwen-moe/README.md) joins the fresh `096452d9` framework PASS with the complete `eed4df57` trusted comparison under exact task/runtime compatibility. [Current Kimi Lean](ready/current/kimi-lean/README.md) connects the recovered `eed4df57` framework PASS with retained `20c1a949` trusted evidence under an explicit CPU-recovery approval. [Current Qwen Dense GEMM](ready/current/qwen-dense/README.md) joins the complete `eed4df57` framework run 209670 and fresh trusted run 210191, with all five cases and no partial-phase reuse. Those scoped results remain valid within their recorded limits. A later checkout needs matching evidence or explicit compatibility for changed task sources, scoring, guards, report handling and GPU binding. No complete model-workload readiness or serving/E2E gain is claimed.

Start with the [consolidated task and evidence index](docs/reference/headkernel-ready-subsets.md). It links the completed qualification handoffs, five historical guides, remaining gates, and portable OCI workload archives. Each current guide identifies its evidence revisions and any strict verifier input sets. The original `ready/*/INPUT-PINS.json` and verification scripts remain strict historical records and may reject this merged checkout because shared framework and documentation files differ. Preserve those pins and the original identities of every report.

Status-documentation corrections do not broaden those accepted input sets. A verifier that pins a README or guide may reject its updated text even when executable inputs are unchanged. Use the guide's accepted snapshot until a separate compatibility review explicitly covers the new revision; documentation edits alone do not establish qualification.

| Model | SG0.5.20 source tasks | Historical ready subsets | Tasks without a historical subset receipt |
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

All five historical ready task trees matched their qualified trees at the consolidation base. Subsequent integration includes all five Qwen timing-quality adapters, the maintained four-task [Qwen trusted evaluator](docs/how-to/trusted-qwen-retest.md), the closed RMSNorm source guard, native graph-event timing and quantization lifetime repairs, and [complete-case GPU groups](docs/how-to/whole-case-parallelism.md) for Kimi aggregation and MiniMax attention, including explicit verified Spur reservation admission. Source integration does not complete their remaining GPU qualification gates.

The framework also enforces [post-validator input integrity](docs/how-to/task-validator.md): current completion binds the normalized report, marker, integrity receipt and actual input observations. Older framework-only passes and the RMSNorm run with a post-run observation gap retain their original scope. Prepared replacement packages require their planned complete runs; an older ready-docs branch is not a substitute for those results.

## Prepare and validate the current sources

Use the [portable setup guide](docs/how-to/headkernel-upstream-runtime.md) before running an agent. `prepare_headkernel_run.py` now selects **all 23 tasks** through explicit catalog protocols. The 18 manifest/native contracts retain `trusted_task_v1`; the five Qwen tasks use `headkernel_artifacts_v1`, pinned to the original whole-object registry. Three Qwen tasks require original tensor fixtures; dense GEMM and RMSNorm are explicitly registered as fixture-free. The helper verifies each selected input contract before generating the validator config. Preparation does not qualify the tasks or replace their capture provenance.

The runtime is the exact SGLang 0.5.20 MI355X image in the [catalog](tools/headkernel-runtime-targets.json). Use local NVMe for fixtures, caches, and results. Isolated task validation does not require a Crusoe allocation, shared model tree, model server, or whole checkpoint. A complete serving-workload reproduction has separate model and TP8 resource requirements described in the [workload archive index](docs/reference/headkernel-ready-subsets.md#workload-archives-in-oci).

After preparing each selected task's inputs, use the normal Docker workflow with `CONFIG=example_configs/headkernel_sg520_current_mi355x.yaml`. Set `AKA_SKIP_DEV_MEM=1`, install and authenticate the selected validator CLI, and run `make docker-check-agents` for that configuration before cache preparation or long validation; the [setup guide](docs/how-to/headkernel-upstream-runtime.md#run-the-standard-arena-validator) gives the command order and backend instructions. These host startup settings do not change task cases or timing policy. Require source-matched, framework-finalized reports and complete trusted comparisons before asserting current qualification. Preserve raw samples, source controls, failed attempts, and timing-quality decisions.

## Capture provenance and scope

The recorded non-Qwen workload recipe is 64 requests, ISL8192, OSL1024, concurrency 64, TP8, context 9218 and client seed42, with OSL16 warmup followed by a prefix-cache flush. The [recorded serving configurations](docs/reference/headkernel-serving-configs.md) give model-specific settings, separate client/server seeds, overlay requirements and portable evidence sources, including Qwen's distinct recorded and proposed recipes. Task manifests remain authoritative about rank-local layouts, controls, occurrence counts and references. Sampled profiler counts must not be extrapolated into full-workload frequencies.

Qwen's sources target SG0.5.20, while its retained shapes and fixtures originate from the documented **SG0.5.18 TP8/EP8** capture, with page 1 attention and FP32 model-default Mamba. Later measured metadata record 103483 describes **TP8/EP1, context 11264, page 16 attention and BF16 Mamba**; the full image/raw-log record is absent. Those settings do not establish equivalent task dispatch. A complete SG0.5.20 Qwen workload capture and matching task provenance remain pending.

Kimi residual aggregation preserves the score/combine logical head through `kimi-k3__attn_residual_aggregate_hip`. Its fixtures contain the first `min(4,T)` actual token rows as labeled parity samples, not full native tensor dumps; full-shape independent checks remain required. GLM replacement ports retain their task-specific native-parity limits. A port-versus-port ratio is not a serving-runtime gain.

Historical implementations retain explicit [dispositions](docs/reference/headkernel-superseded-attention.json): DeepSeek sparse MLA is superseded by unified paged prefill/decode for this recipe; Kimi grouped attention is a native low-batch branch while served batch 64 uses Lean; GLM scale-layout copies produced aliases. The two inactive configured packages are excluded from the 23-task current selection.

The [refresh chronology](docs/reference/headkernel-sg520-refresh.md), [reported workload figures](docs/reference/headkernel-reported-results.md), [upstream README](HEADKERNELS_UPSTREAM.md), [capture-era catalog](tools/headkernel-public-runtimes.json) and [original publication record](tools/headkernel-publication.json) preserve their original dates and scope. Their older runtime settings and reported gains are historical evidence, not defaults or qualification for this merged branch.

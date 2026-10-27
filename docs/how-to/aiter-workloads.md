# AITER tasks for fixed model configurations

The `aiter_workloads` suite starts from upstream AITER and defines each task
for a model variant, parallelism setting and finite workload policy. It uses
the standard Arena task interface, so the same cases and evaluator can be used
with GEAK, GEAK-EVO or another supported agent/model.

Changing TP, EP, quantization, request lengths, batching or speculative decoding
can change both the legal inputs and their frequencies. A model's complete
upstream test sweep is therefore not the optimization objective. Each task's
`workloads.json`, `frequency_provenance.json` and `config.yaml` define its scope.

## Task selection

| Model and configuration | Selected AITER operation | Case basis |
| --- | --- | --- |
| DeepSeek V4 Pro, TP8/EP1, concurrency64, input8192/output1024 | mHC, FP8 block-scale linear, fused MoE | Four authentic input states per task; measured selected-site prefill/decode counts |
| GLM5.2-MXFP4, TP4/EP4, AgentX concurrency8 | [Local fused MoE](../../tasks/aiter_workloads/glm_5_2_mxfp4_tp4_ep4_agentx_c8_m48_fused_moe/README.md) | One explicit M48 scenario; not an inferred EAGLE verification histogram |
| GLM5.3-Flash FP8, TP8, concurrency64, input8192/output1024 | [MLP down GEMM](../../tasks/aiter_workloads/glm53_flash_tp8_c64_8k1k_fp8_mlp_down/README.md) | Fixed prefill M8192/decode M64 scenario, N4096/K1536 |
| Kimi K3, TP8, concurrency64, input8192/output1024 | [Recurrent KDA decode](../../tasks/aiter_workloads/kimi_k3_tp8_c64_8k1k_kda_decode/README.md) | Full 64-request, one-token decode scenario; 12 local heads, width128, explicit state checks |
| MiniMax M3-MXFP4, TP8, concurrency64, input8192/output1024 | [BF16 MLP down GEMM](../../tasks/aiter_workloads/minimax_m3_tp8_c64_8k1k_bf16_mlp_down/README.md) | Fixed prefill M8192/decode M64 scenario, N6144/K1536; the checkpoint exempts this projection from MXFP4 |

The new scenario tasks use fresh generated values with declared distributions.
Their model/configuration bindings do not establish a complete live scheduler
trace, learned-weight distribution or serving dispatch. For example, the
recorded Kimi runtime used FLA; the KDA task evaluates the corresponding AITER
operation as an alternative requiring serving integration. GLM5.3 means the
explicitly selected **Flash** variant here, not IQ2R or the full GLM5.3 model.

Each task keeps measured facts, configuration derivations and assumptions
separate. EP/DCP fields remain unknown where not recorded and irrelevant to the
selected local dense operation. They must be resolved before claiming a full
distributed-workload reproduction.

## Objective and boundaries

The shared evaluator selects candidates by:

```text
sum(case_weight * upstream_AITER_time)
-------------------------------------
sum(case_weight * candidate_time)
```

For a declared 64-request fixed-batch scenario, a selected site can have 64
prefill calls and 1024 full-batch decode calls. Those counts are a scenario
definition unless a task binds a measured trace. A one-case task scores that
single call; its weight does not manufacture knowledge of the request mix.
Correctness-only cases add coverage without timing weight.

The task protects its inputs, reference and timing policy, while allowing the
implementation tree to change. A public operation may launch multiple kernels.
Candidate-specific conversions and preparation stay inside the measured
boundary; externally prepared operands and communication remain outside only
when the task contract explicitly says so. Both roles receive the same
boundary. Read the task README before comparing scores across different tasks.

An operator speedup is a local proxy. Confirming request throughput or latency
requires integrating the delivered implementation and rerunning the same
complete serving workload, including checking that the selected code executes.

## Run and validate

Use the [pinned SGLang0.5.20, AITER and FlyDSL setup](../install/install.md#pinned-aiter-and-flydsl-on-sglang-0520).
The new tasks target gfx950. A freshly built image needs its own qualification;
matching the dependency versions is not a GPU test result.

The maintained scenario validation config selects the four new model tasks:

```bash
AKA_DOCKER_IMAGE=aka-aiter:2cb3d175-flydsl0.4.0 \
  make docker-run CONFIG=example_configs/aiter_workloads_validator_mi355x.yaml
```

Require each task's framework-finalized `validation_report.yaml` to report
`overall_status: PASS` before optimization or publication as GPU-qualified.
CPU oracle checks, successful task discovery and source verification establish
different facts and do not replace that report.

To use an optimization agent, copy an existing configuration for that agent and
select a task directory relative to `tasks/`, for example:

```yaml
tasks:
  - aiter_workloads/glm53_flash_tp8_c64_8k1k_fp8_mlp_down
target_gpu_model: MI355X
```

Agent/model choice and search budget remain in the run configuration. The task
does not contain agent-specific prompts or provider credentials. See
[agent configuration](agents.md) and [task authoring](add-task.md).

The existing DeepSeek tasks additionally require their declared immutable
input and native-library assets on read-only mounts. Those tasks retain their
empirical input policy; the generated scenario tasks need no old captured
kernel corpus or tensor download.

# DeepSeek V4 Pro: complete AITER fused MoE call

This task package is under development and **unqualified**. Its recorded workload settings are selected, but its actual call cases, invocation frequencies, numerical acceptance and composed-runtime binding are incomplete. `validate-task` and all evaluation actions currently exit nonzero with an `ARENA_EVAL_RESULT` failure. Empty cases are deliberate: no generic M64/M8192 sweep or old saved-kernel fixture is presented as this workload.

The workload is DeepSeek V4 Pro, TP8/EP1, 64 ShareGPT-derived requests, concurrency64, input8192/output1024. The protected workload file identifies the selected model revision, dataset and request policy. Workload-derived cases and frequencies must be collected from that exact reproduced workload before this task can be enabled. Concurrency is not a guarantee of instantaneous batch size.

## Operation and implementation freedom

The input boundary is the public `aiter.fused_moe.fused_moe` call: hidden states, already-selected expert IDs/weights, prepared expert weights/scales and explicit caller arguments. The output is the complete local routed-expert result. The operation includes its internal sorting, activation quantization, both projection stages, activation and output accumulation as selected by upstream dispatch. External expert selection and the subsequent TP collective are outside this boundary. The exact serving caller remains to be bound: if it performs per-call, MoE-only activation quantization outside the public call, the measured/editable boundary must widen to include that work before the task is frozen.

`source/kernel.py` initially delegates the whole public call to the installed pinned upstream AITER implementation. The full upstream repository is acquired at an immutable revision under `upstream/` as protected source context. The optimizing agent may replace the delegation and add implementation helpers anywhere under `source/`, using the GPU backends available in the bound runtime. It may reorganize the complete computation; internal launches, layouts and intermediate buffers are not fixed targets. For this first task, externally supplied weights retain the serving loader's fixed preshuffle. There is no candidate load-time weight-transform step; candidate-specific conversions must execute inside the measured call.

Preserve the public function signature and returned tensor semantics. Inputs are read-only. New per-invocation preparation, conversions and intermediate work belong inside the timed call. Do not compute outputs or reusable input-dependent intermediates while importing the implementation. Static compilation is allowed. Do not alter the installed AITER package, protected upstream source, case data, reference, numerical policy, timing helper or evaluation scripts. Helper imports must remain within the allowed implementation/dependency boundary.

The baseline is separately provided by the installed pinned AITER public call. The numerical reference calls the protected installed upstream PyTorch stages with independently frozen quantization/rounding choices. It never imports editable candidate code. Both roles use the same public inputs, correctness contract and complete-call timing policy.

## Existing Arena interface

All seven standard actions are implemented in `scripts/evaluate.py`:

```text
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Each emits one `ARENA_EVAL_RESULT=` JSON envelope and exits nonzero on failure. Compilation syntax-checks the candidate module and exercises the actual callable/JIT path for the bound cases. Performance uses Arena's canonical `_aka_benchmark.py`, materialized by the normal workspace setup; no copied timing implementation is maintained here. Each case explicitly selects eager GPU-event timing (`cuda_event_fallback`) or graph timing (`cuda_graph`) through its protected `params.benchmark_method`. This permits source-supported prefill/eager and decode/graph regimes without assuming their case coverage. The returned timing method must match the declaration. The harness checks the actual measured outputs and their replay against the oracle. Public inputs are snapshotted once per case/action and checked for mutation afterward, outside measured intervals.

Agent/model choices belong to the Arena run configuration. This package uses backend-neutral `candidate.language: gpu`; a native run through any particular agent adapter requires that adapter's independent qualification.

## Binding the task data

Keep `workloads.json.binding_status` pending until these records are complete:

1. **Runtime.** Set `runtime_binding.image_ref` to the composed scoring image's immutable Docker-inspected image ID (`sha256:...`), as supplied by Arena in `AKA_SCORING_IMAGE_RUNTIME_REF`, with `runtime_binding.build_evidence: {path, sha256}` pointing to a protected task-relative source/build/dispatch record. The build recipe may separately record a registry manifest digest. The runner requires exact runtime image ID agreement, the declared build-record hash, selected installed AITER source hashes, and gfx950. These checks supplement native qualification; a Python source hash alone does not establish native build identity. Setup performs no package installation or unlisted download.
2. **Inputs.** Add fresh workload-derived case records with `test_case_id`, `checks`, `shape`, `dtype`, `params` and `inputs`. The public shape is `[tokens, hidden_width, local_intermediate_width, experts, topk]`; dtype is `bfloat16`. `params.tensors` must list every public tensor's exact `shape`, `stride` and Torch dtype string. The runner compares these against the loaded tensors, preventing a mislabeled case from being scored.
3. **Reference.** Freeze `numerical_contract` with the fields checked by `scripts/reference.py`. The prepared adapter supports the upstream-test a8w4 staged reference structure, but that structure has not yet been accepted as the chosen runtime's numerical contract. A1/A2 quantizers, intermediate rounding, clamping, gate layout and explicit comparator thresholds require source/runtime evidence. Missing values fail; no tolerance defaults are supplied. The prepared allclose adapter and an accuracy-envelope alternative are proposals still under review; neither is an agreed acceptance rule, and the numerical contract remains unbound.
4. **Frequency and timing.** Add a complete positive measured frequency map under `evaluation.aggregation` using `method: weighted_total_time`, with `source: {path, sha256}`. The IDs must exactly cover performance cases. Every case must declare `params.benchmark_method` as `cuda_graph` or `cuda_event_fallback`; it is part of captured case identity. Freeze the shared `workloads.json.timing` fields `warmup`, `repetition`, `target_ms` and `max_graph_repeats`. There is no default timing method. Until those records exist, the runner rejects execution instead of using Arena's unweighted default.

`inputs` contains a task-relative `path`, SHA256, exact `call_kwargs` and `tensor_attributes`. Its file is a tensor-only `torch.save` mapping loaded with `weights_only=True`. Required public keys are `hidden_states`, `w1`, `w2`, `topk_weight`, `topk_ids`, `w1_scale`, `w2_scale`. Required protected oracle keys are `canonical_w1`, `canonical_w2`, `canonical_w1_scale`, `canonical_w2_scale`. Optional bias1/bias2 tensors are supported. The loader verifies that the canonical unshuffled weights/scales reproduce the public shuffled bytes. Only public tensors are passed to the candidate.

The first carrier covers one explicit local a8w4/interleaved family. Shared-expert arguments, EP masks, scatter destinations, caller-prequantized activations and output-destination variants require a corresponding input/reference extension when actual calls establish their relevance. Their current absence is a preparation limit, not evidence against those designs. The actual caller's block/dispatch overrides also require explicit binding before admitting a case; do not silently drop them.

Graph padding also remains to be bound. Record each actual padded bucket, live-row count and padded-row contents/semantics; do not assume all padded rows are dead or silently omit rows that affect downstream state. The current adapter checks the complete returned tensor until a justified live-row/state contract is implemented.

Rank/layer coverage and aggregation remain unresolved. A mean over sampled layers estimates the full layer population unless the sampling and coverage establish otherwise. A maximum of independently measured rank times estimates a bottleneck; it is not the simultaneous TP critical path or an exact E2E latency. Preserve rank/layer measurements and coverage, then freeze any approximation before search. Neither a sampled layer average nor maximum isolated-rank timing may be labeled an exact E2E result.

Use fresh correctness draws and complete workload cases prescribed by the frozen contract; this package does not invent a synthetic workload generator. The final comparison must remeasure upstream and candidate under the same cases and common complete-call timing. A local weighted-time improvement remains a proxy until integration and a complete same-workload E2E run confirm it.

## Source attribution and qualification

Upstream: ROCm/AITER, MIT license, commit `2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`. The source and license are available in the materialized `upstream/` directory. The reference arrangement follows `op_tests/test_moe_2stage.py`, particularly the A1/A2 quantization and `torch_moe_stage1`/`torch_moe_stage2` calls. Its `--kernel` mode excludes parts of the operation and is not this task's objective. The upstream composite accuracy check is not silently substituted for a frozen numerical rule.

No GPU validator pass, fresh AITER baseline, optimization result or E2E gain is claimed by this package. CPU protocol/schema checks establish only that the preparation is structurally valid and rejects unresolved bindings. A fresh compatible-GPU `task_validator` PASS remains required before PR submission.

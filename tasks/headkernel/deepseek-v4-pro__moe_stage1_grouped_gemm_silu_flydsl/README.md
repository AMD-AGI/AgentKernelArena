# DeepSeek-V4-Pro: FlyDSL FP8/FP4 MoE stage 1 decode

The input contract is sealed. Trusted GPU evaluation and framework task-validator qualification remain pending.

This task contains 9 fixed cases from the current SGLang 0.5.20 native dispatch on gfx950. The served run completed 64 requests with 8192 input and 1024 output tokens each. All eight ranks supplied structural case metadata; actual tensor representatives were captured on rank 0.

Six additional fixtures use separately generated legal expert routing at observed decode num_valid_ids values [2400, 64], [2432, 64], and [2496, 64], with seeds 42 and 43. Their paired stage-1/stage-2 outputs passed 16 native repeated graph executions per scenario. These fixtures do not recover the uncaptured tensors from other ranks or retroactively pass the original served work-control gate.

The supplied candidate and immutable reference both contain the explicit async-LDS full-wait synchronization repair. This is a repaired baseline, not a byte-identical stock baseline. See provenance/STAGE1-SYNC-REPAIR.json and provenance/COVERAGE.json.

Only the GPU implementation bodies declared in config.yaml are editable. Native host wrappers, launch decisions, source guards, fixture codecs, oracle logic, timing, and state resets are protected. The contract retains tolerance 0.02, the correctness seeds declared in cases.json, 10 warmup iterations, and 100 checked graph measurements per case. All semantic output components and required negative controls remain mandatory.

The paired FP8/E8M0 output is checked as dequantized values at tolerance 0.02, with exact equality for every meaningful E8M0 scale byte. The protected checker derives live scale positions from the immutable CPU routing snapshot and requires each token/slot exactly once. It preserves the full tensor/storage ABI; uninitialized scale-allocation padding is outside the semantic comparison. Invalid routing or an invalid reference remains a contract failure, not a successful negative control.

Tensor data remain external. The trusted host materializes the exact assets in fixtures/EXTERNAL-MANIFEST.json before task execution; the task does not download or regenerate fixture data. Cases keep shapes, strides, storage offsets, aliases, scalar arguments, and packed scale semantics. See cases.json and provenance/COVERAGE.json for explicit historical logical-M gaps. Native repeatability is not independent source-bound correctness or a framework PASS.

Generated fixture admission is explicit in `ut/fixture_admission.py`. The
`FROZEN_CAPTURE_AND_GENERATED` contract verifies the complete hash-bound coverage
set, captured parent identity, native implementation, replay parameters, metadata
projection, and restored work-control values. Generated rows retain
`generated_native_graph_replay` provenance and a count of one generated fixture;
they do not become served launches or repair the original capture gate.

Both fixture kinds use the same protected payload hash checks, CPU golden
snapshots, source-backed reference execution, full numerical comparison, input
mutation checks, graph timing, and negative controls. Admission alone is not GPU
correctness or framework qualification.

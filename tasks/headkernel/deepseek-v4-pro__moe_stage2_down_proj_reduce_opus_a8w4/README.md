# DeepSeek-V4-Pro: Opus FP8/FP4 MoE stage 2 prefill and decode

The input contract is sealed. Trusted GPU evaluation and framework task-validator qualification remain pending.

This task contains 12 fixed cases from the current SGLang 0.5.20 native dispatch on gfx950. The served run completed 64 requests with 8192 input and 1024 output tokens each. All eight ranks supplied structural case metadata; actual tensor representatives were captured on rank 0.

Six additional fixtures use separately generated legal expert routing at observed decode num_valid_ids values [2400, 64], [2432, 64], and [2496, 64], with seeds 42 and 43. Their paired stage-1/stage-2 outputs passed 16 native repeated graph executions per scenario. These fixtures do not recover the uncaptured tensors from other ranks or retroactively pass the original served work-control gate.

Only the GPU implementation bodies declared in config.yaml are editable. Native host wrappers, launch decisions, source guards, fixture codecs, oracle logic, timing, and state resets are protected. The contract retains tolerance 0.02, correctness seeds 42 and 43, 10 warmup iterations, and 100 checked graph measurements per case. All output components and required negative controls remain mandatory.

Tensor data remain external. The trusted host materializes the exact assets in fixtures/EXTERNAL-MANIFEST.json before task execution; the task does not download or regenerate fixture data. Cases keep shapes, strides, storage offsets, aliases, scalar arguments, and packed scale semantics. See cases.json and provenance/COVERAGE.json for explicit historical logical-M gaps. Native repeatability is not independent source-bound correctness or a framework PASS.

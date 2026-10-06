# DeepSeek-V4-Pro: Unified paged attention decode

The input contract is sealed. Trusted GPU evaluation and framework task-validator qualification remain pending.

This task contains 3 fixed cases from the current SGLang 0.5.20 native dispatch on gfx950. The served run completed 64 requests with 8192 input and 1024 output tokens each. All eight ranks supplied structural case metadata; actual tensor representatives were captured on rank 0.

Every required work extreme for this component has a captured representative. The model-level served work-control gate remains failed because other components needed generated supplements.

Only the GPU implementation bodies declared in config.yaml are editable. Native host wrappers, launch decisions, source guards, fixture codecs, oracle logic, timing, and state resets are protected. The contract retains tolerance 0.02, correctness seeds 42 and 43, 10 warmup iterations, and 100 checked graph measurements per case. All output components and required negative controls remain mandatory.

All three captured queries are BF16 `[64,16,512]`, with BF16 KV storage and
`kv_scales=None`. On the observed 256-CU MI355X, the unchanged native heuristic
selects `block_h=16` and `kv_splits=4`: the split and reduce kernels execute.
Those two bodies are the editable targets. The unused fused kernel remains
packaged byte-for-byte and is protected by the source guard. The runner verifies
the actual candidate and reference dispatch controls before invoking each case;
the evidence is recorded in `provenance/EDITABLE-DISPATCH.json`.

The performance command has a 3,600-second budget for the existing complete
checked replay. The previous attempt passed compilation and all three correctness
cases but exceeded the inherited 600-second performance limit. Legacy logical-M
values 1 and 8192 and the FP8-KV specialization remain explicitly outside this
captured task's coverage; no cases or coverage claims are added.

Tensor data remain external. The trusted host materializes the exact assets in fixtures/EXTERNAL-MANIFEST.json before task execution; the task does not download or regenerate fixture data. Cases keep shapes, strides, storage offsets, aliases, scalar arguments, and packed scale semantics. See cases.json and provenance/COVERAGE.json for explicit historical logical-M gaps. Native repeatability is not independent source-bound correctness or a framework PASS.

# DeepSeek-V4-Pro: Unified paged attention decode

This isolated CPU-reviewed proposal retains the three captured regression cases and adds one generated observed-control distribution case. Native execution and framework qualification of the added case remain pending; no canonical task change has been made.

The three original captured cases are preserved unchanged from the current SGLang 0.5.20 native dispatch on gfx950. The served run completed 64 requests with 8192 input and 1024 output tokens each. All eight ranks supplied structural case metadata; actual tensor representatives were captured on rank 0.

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


## Proposed current-control distribution case

`deepseek_mla_decode-current-kvlen-distribution-192-200` uses the captured length-200
fixture only as immutable parent inputs and for initial unchanged-parent calibration.
Each state takes the first L entries of every parent CSR row, repacks `kv_indices`,
sets `kv_indptr[i] = i * L`, and fills the unused index-buffer tail with -1. Every
selected KV row lies within verified captured parent storage. Q values follow the
unchanged seeded refresh policy. KV and sink values remain captured, and all native
storage extents, strides, offsets, aliases, scalar controls, and source bytes are
preserved. There are no new dense tensor fixtures and no claim to have
recovered the missing original tensor/routing states for lengths 193–199.

Correctness covers all nine lengths 192–200 with seeds 42 and 43 and both negative
controls at each length. Performance retains ten warmups and one hundred measured,
validated graph replays. Each replay selects a state using the exact all-rank
histogram and the private request challenge; paired trusted reference/candidate
legs receive the same challenge and therefore the same ordered draws. Performance
is sampled; correctness coverage is exhaustive for this nine-value domain.

The native public ABI has no maximum-length scalar argument. All-rank metadata and
the pinned Python wrapper establish fixed split/reduce grids and launch controls.
Runtime probes enforce those actual controls on candidate capture and every native
reference invocation, while the independent post-replay reference checks graph
validity for the current data state. No generated golden output is precomputed or
shared with the candidate.

The original three case objects, including their historical occurrence fields,
remain unchanged. The new case's occurrence count describes its distribution group
and is explicitly not additive with the endpoint regression cases. Exact per-length
occurrence weights live in `ut/mla_control_distribution.json`.

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

The performance command has a 3,600-second limit. The previous nine-case GPU
attempt passed compilation and correctness, then reached the default
600-second performance limit before producing a complete timing report.
The longer limit preserves every case, warmup, checked replay, tolerance and
measurement boundary; it does not establish qualification by itself.


This isolated proposal preserves every original fixed case, seed, negative control,
tolerance, warmup, and 100-sample measurement. Additional distribution cases support
all recorded `num_valid_ids` values with legal representative routing and fresh seeded
FP8 activation values. Captured per-token scales are remapped to each generated route.
The original other-rank routing arrays were not recorded or recovered.

New distribution cases default to the reviewed targeted behavior checks in
`provenance/WORK-DISTRIBUTIONS.json`; pending selections fail correctness explicitly.
An optional separate run uses `--distribution-correctness-mode exhaustive_observed_values`
to check every recorded value with the unchanged seeds and controls. Reports label
the executed mode and claim full value coverage only after the complete sweep.
Performance uses 10 checked warmups and 100 checked frequency-weighted draws. Use
the same protected request challenge seed for paired reference/candidate comparisons;
independent standalone runner invocations do not share a sampling schedule.

Each new stage-2 routing receives fresh protected stage-1 reference output and matching
scales, copied into stable stage-2 input buffers outside timing. No new dense golden
files are generated. The original fixed-set COVERAGE provenance and qualification
flags remain unchanged. This proposal needs compatible GPU and task-validator checks
before it can qualify; CPU preparation is not GPU validation.

Performance primary scoring uses the protected reference timed in the same invocation as the candidate. Both graphs are warmed and captured on their own streams. Each of the unchanged 10 checked warmups and 100 measured draws executes the candidate once, freezes its CPU observations, and executes the reference once. The reference output supplies the existing oracle; no additional comparison replay is added. Persistent reference outputs are cleared before another candidate can run.

The paired report binds both sample series to the actual input schedule, request, source identities and reference provenance. Unpaired old/new port timings have no secondary speedup. This FlyDSL baseline is the synchronization-repaired protected reference, not unchanged native production. Ordinary pairing is not an anti-cheat attestation; fresh private trusted-host retesting remains required. Targeted correctness coverage and its existing warnings remain unchanged.

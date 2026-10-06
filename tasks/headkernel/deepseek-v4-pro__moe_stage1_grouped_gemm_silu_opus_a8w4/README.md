# DeepSeek-V4-Pro: Opus FP8/FP4 MoE stage 1 prefill

The task retains 3 fixed cases from the complete served SGLang 0.5.20 workload on gfx950. The workload completed 64 requests at ISL8192/OSL1024, concurrency 64 and TP8. All eight ranks supplied structural metadata and rank 0 supplied actual tensor representatives. All current cases and recorded occurrence counts are unchanged.

Only the group-split `process_tile` body is editable. Every recorded `kernelName` resolves to the same group-split native instance. The unused pair-kwave header remains packaged and hash-frozen; its native implementation is unchanged. The captured dispatch values, native metadata and source hashes are recorded in [provenance/EDITABLE-DISPATCH.json](provenance/EDITABLE-DISPATCH.json). The protected runner rejects cases outside that observed dispatch. No synthetic workload shape was added to exercise an unused implementation.

Host wrappers, launch decisions, source guards, fixture codecs, references, timing and state resets remain protected. The encoded FP8 compatibility bound `0.15` and correctness seeds `[42, 43, 44]` are retained. A mandatory additional oracle compares every physical activation after FP8/E8M0 dequantization with the mixed `0.02 * RMS(reference) + 0.02 * abs(reference)` bound in float64. All allocated scale bytes, including initialized padding, remain exact. Every case retains 10 warmups and 100 measured graph replays, with fresh primary inputs, output initialization, full-storage input immutability checks, native-reference comparison and all required negative controls. Candidate output and input observations are owned CPU snapshots made before reference GPU execution.

Each tensor snapshot now makes one owned CPU copy instead of copying to CPU and immediately cloning it again. Candidate and reference inputs remain separate GPU allocations; the reference reuses the already verified CPU golden instead of restoring a second unused copy. CPU tensor work is capped at eight intra-op threads per process to avoid each concurrent GPU validator starting a node-wide CPU pool; the previous and active counts are reported. All captured golden comparisons and fixture-integrity checks remain mandatory. The timed graph and measured work are unchanged.

Compilation and correctness have 1200-second bounds; performance has a 3600-second bound for the full checked replay protocol. Previous compilation and correctness passed, but performance exceeded the old 600-second limit and emitted no completed timing report. Those failed reports remain evidence; this revision still requires fresh native GPU validation and an authentic framework-finalized PASS.

Tensor data remain external and are materialized through [fixtures/EXTERNAL-MANIFEST.json](fixtures/EXTERNAL-MANIFEST.json). Original shapes, strides, storage offsets, capacities, aliases, scalar arguments and packed-scale semantics are preserved. See [cases.json](cases.json) and [provenance/COVERAGE.json](provenance/COVERAGE.json) for scope and unresolved historical logical-M cases. Model-level served work-control qualification remains separate because other components required generated supplements.

[provenance/OUTPUT-CONTRACT.json](provenance/OUTPUT-CONTRACT.json) documents the semantic output units and native scale layout. Raw FP8 codes from differently scaled groups do not have a meaningful global RMS: a large physical error can pass a raw-code comparison. The new oracle reconstructs token/slot scale associations from immutable CPU routing and requires complete unique route coverage. Captured goldens and the separately built frozen native implementation supply expected outputs; this does not claim an independent GEMM implementation.

The current component has no recorded missing work-control payloads. The preserved model-level capture error belongs to the separate FlyDSL `moe1` decode component. Historical logical-M values remain disclosed as historical scope and are not fabricated as cases in this actual prefill workload.


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

The paired report binds both sample series to the actual input schedule, request, source identities and reference provenance. Unpaired old/new port timings have no secondary speedup. The baseline uses the frozen native production source. Ordinary pairing is not an anti-cheat attestation; fresh private trusted-host retesting remains required. Targeted correctness coverage and its existing warnings remain unchanged.

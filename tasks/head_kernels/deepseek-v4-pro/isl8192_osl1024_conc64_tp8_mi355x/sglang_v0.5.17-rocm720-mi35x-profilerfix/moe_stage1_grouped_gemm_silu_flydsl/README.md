# moe_stage1_grouped_gemm_silu_flydsl: generated-input draft

**Qualification is blocked: inputs cover only 183/256 mandatory sequence calls.**
The unchanged 256-entry ledger requires 73 M=1920 calls at zero-based positions
183–255, but the original and mirrored archives contain only five records:
M2048, M256, M64, M1 and M32768. The original UT filtered unresolved signatures,
so its sequence check covered only the first 183 M2048 calls. That partial check
cannot establish full sequence correctness.

Correctness and performance now fail during CPU contract verification, before
runtime preflight or GPU workers. The failure report and
`build/sequence_coverage_report.json` identify every unresolved signature and
position. All 256 calls remain mandatory; no subset can produce a qualification
PASS. Authentic M1920 routing and tensor-layout inputs must be recovered before
promotion. The five benchmark cases, three random draws, M256-capture/M64-replay
boundary, tolerance, 10 warmups and 100 timing samples remain unchanged.

[ut/sequence_coverage_evidence.json](ut/sequence_coverage_evidence.json) records
the exact missing signature and positions, ledger hash, original UT hash,
archive hashes and recovery receipt hashes. The recovery was limited to the
scoped original and mirrored archives; it does not rule out unrelated captures
elsewhere. A shape descriptor alone is not a usable input record.

This task requires no external tensor archive. [ut/generated_cases.json](ut/generated_cases.json)
retains all 5 source records and their exact shapes, strides,
backing-storage relationships, tensor attributes, scalar arguments, routing and padding.
Numeric activations and weights are generated locally. No original weight values or
reference output values are embedded in the compact contract.

The original correctness/timing case identities, tolerance `0.15`, random draw
count `3`, call sequence, and replay boundaries remain in
[ut/meta.json](ut/meta.json). The canonical timing method, **10 warmups / 100 samples**, state restoration
and actual timed-graph replay verification are retained. Complete raw returns
are transformed for numerical comparison outside the scored device interval.

Generated MXFP4 bytes contain two finite E2M1 values. MXFP8 bytes contain finite normal
E4M3 values. E8M0 weight scales use finite exponents 121..124; activation scales use
exponent 123 so repeated sorted routes for a token retain a coherent scale. Original
routing IDs, sorted slots, sorted route weights, padding sentinels and counts are
losslessly retained. Sparse MLA uses its exact 448-byte FP8 payload, 64 BF16 tail
values, seven E8M0 scale bytes plus one unused scale byte, full cache block stride,
referenced block IDs, sparse lengths and exact partial undefined-output masks.

Correctness compares fresh outputs from separate frozen-source and candidate workers.
The reference worker loads the immutable source under `ut/baseline_ref/`; the candidate
worker loads the editable source in the same original package context. Neither worker
receives expected outputs. Helper/worker functions and output metadata primitives are
attested around candidate execution. All integer and boolean output values compare
exactly. Floating values keep the original mixed tolerance and RMS floor.

All defined return values are now numerically gated. Stage 1 compares both FP8
mantissas and the jointly dequantized activations carried by its consumed E8M0
scales; padding is excluded using the verified producer's valid-row/column rules.
DSA compares attention values and finite LSE values, preserves only the exact
captured LSE NaN mask, and checks signed infinity sentinels exactly. The producer,
consumer and scale-layout evidence is in
[ut/secondary_output_evidence.json](ut/secondary_output_evidence.json).

The shared `_trusted_worker.py` preloads every declared helper alias before
editable code runs and executes the attested module's `main()`. There is no
separate generator guard class. Tensor serialization, Base64/zlib operations,
comparison functions and output metadata primitives are protected by the shared
monitor. All task-local loaders reuse those exact instances and reject alias
path collisions. Source kernel bodies and image declarations are unchanged.

Use the declared runtime and GPU architecture in [config.yaml](config.yaml):

```bash
python3 scripts/generated_task_runner.py compile
python3 scripts/generated_task_runner.py correctness
python3 scripts/generated_task_runner.py performance
```

This is a review draft. It has no fresh GPU correctness, timing or task-validator PASS.
The generated numerical distribution is new and must not be presented as recovered
production tensor values or as a reproduction of historical kernel timings.

All five scored cases retain their captured full signature IDs and M buckets 2048, 256, 64, 1, and 32768. No scored case was removed.

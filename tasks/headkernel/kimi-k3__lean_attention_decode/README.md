This task exposes the actual Lean attention decode body used by Kimi-K3 in the pinned SGLang 0.5.20 runtime. The full served workload completed 64 requests at TP8/C64, with 524,288 input tokens and 65,536 output tokens. The single structural case occurred 24,576 times on each rank, for 196,608 calls across all eight ranks.

The editable function is `_lean_attention_decode_kernel` in [source/decode_attention.py](source/decode_attention.py). The launcher, launch geometry, source imports, helper functions and all harness/reference code are protected. The source guard permits closed Triton device-body edits and rejects host execution, reflection, new runtime capabilities and host-wrapper changes.

[cases.json](cases.json) records the observed physical ABI and all positive case counts. The query has shape `[64,12,576]`; K and V share storage with logical dimensions 576 and 512. Output, partial maxima, partial sums, partial outputs and locks retain their captured shapes, strides, aliases and initial/final semantics. The fixed grid has 256 programs. Every observed sequence length from 8193 through 9216 is represented in the empirical work distribution.

The trusted host materializes the exact assets in [fixtures/EXTERNAL-MANIFEST.json](fixtures/EXTERNAL-MANIFEST.json) from the approved capture mirror or pinned OCI prefix. No tensor blobs belong in the Git task. The runtime verifies every fixture hash and the installed source closure. Paged KV is restored only from recorded read footprints; generated indices remain inside those captured physical rows, and complete storage snapshots check readonly bytes.

Each replay resets fresh legal queries and KV indices, initializes output and scratch, snapshots input truth to CPU, invokes the candidate, and snapshots candidate outputs and post-input storage to CPU before running an independent frozen native reference. The returned GPU reference storage is cleared after its CPU snapshot. Correctness uses the existing 0.02 RMS-relative tolerance, checks all final scratch and aliases, requires exact untouched scratch bytes, and rejects no-op and corrupted-output controls. Performance uses 10 warmups and 100 checked single-call graph replays.

In the pinned gfx950 container, after trusted fixture materialization:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Source and metadata CPU checks pass. GPU correctness, timing and a framework-finalized task-validator `PASS` are still required before publication.

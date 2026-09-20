# qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode

This task optimizes the **Qwen3.8-2.4T-A95B-MXFP4** `fused_recurrent_gated_delta_rule_packed_decode_kernel` callable on MI355X (`gfx950`).
The serving contract is ISL 8192 / OSL 1024 / CONC 64 / TP 8; its captured execution regime is `decode`.

Runtime image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`.
Backend: `Triton`; execute in that image's captured SGLang/AITER/PyTorch environment.

The runtime image, source package, editable files, required symbols, timeouts and architecture gate
are declared in [config.yaml](config.yaml). Starting-source provenance: **stock**.
The source snapshot is preserved as imported; inspect [ut/meta.json](ut/meta.json) for the captured
cases for upstream measurement caveats and live capture provenance.

The required structural input is committed in `ut/generated_cases.json` and verified against its
recorded checksum. Numeric values are generated locally; real routing, state and paging structure
remain frozen. No external tensor archive is required, and case sets and buffer sizes stay intact.

Inside the declared GPU runtime, run from this task directory:

```bash
python3 scripts/generated_task_runner.py compile --timeout 600
python3 scripts/generated_task_runner.py correctness --timeout 3600
python3 scripts/generated_task_runner.py performance --timeout 3600
```

Compilation is a CPU syntax and symbol check. Correctness compares fresh baseline and candidate outputs in a protected parent
with the original tolerances and supplemental replay/value checks.
Performance must cover every declared case using equivalent work and state for both Arena legs.
The Arena baseline is the committed starting implementation; historical GEAK speedups are archival
provenance and do not supply this run's score. Reports are written below the task directory.

Archival source package: `P/gated_delta_decode_task`. Capture-machine locations in metadata are
recorded as `provenance://` identifiers; they are not runtime paths. This import has not been
validated on a compatible GPU. A clean task-validator report is required before PR submission.

The editable symbols include Python launchers or operator dispatch where declared in the config.
Those edits must preserve the complete callable workload, launch dimensions, ABI and output/state
contract. A device-kernel speedup cannot be claimed by deleting operator work or changing tested shapes.

## Generated numeric inputs (draft)

The default config generates numeric operands locally at the unchanged captured
shapes, strides, dtypes, and allocation sizes. `ut/generated_cases.json` retains
the exact structural routing, slot/page indices, scalar ABI, and storage descriptors.
No `reference_io.pt` is required. Original numerical capture metadata and case
loader remain in `ut/archival_meta.json` and `ut/archival_cases.py` as provenance.

Correctness uses separate protected baseline and candidate workers with parent
comparison. Full outputs, required aliases, unchanged inputs/state rows, repeated
state transitions, and required graph replay are checked. The original 10/100
timing and exact timed-output replay checks remain in the common runner.
This draft has CPU controls only and requires fresh matching-runtime GPU qualification.

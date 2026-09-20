# kimi-k3__fwd_grouped_kernel_stage1

This task optimizes the **Kimi-K3** `_fwd_grouped_kernel_stage1` callable on MI355X (`gfx950`).
The serving contract is ISL 8192 / OSL 1024 / CONC 64 / TP 8; its captured execution regime is `decode`.

Runtime image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`.
Backend: `triton (attention)`; execute in that image's captured SGLang/AITER/PyTorch environment.

The runtime image, source package, editable files, required symbols, timeouts and architecture gate
are declared in [config.yaml](config.yaml). Starting-source provenance: **prior-candidate**.
The source snapshot is preserved as imported; inspect [ut/meta.json](ut/meta.json) for the captured
cases and [ut/README.md](ut/README.md) for upstream measurement caveats.

Provision the task's external artifacts using the suite's declared artifact manifest before running.
Captured oracle and timing files must keep their recorded checksums. Missing inputs are environment
errors and must not be replaced by generated routing, placeholder geometry or reduced case sets.

Inside the declared GPU runtime, run from this task directory:

```bash
python3 scripts/task_runner.py compile --timeout 600
python3 scripts/task_runner.py correctness --timeout 3600
python3 scripts/task_runner.py performance --timeout 3600
```

Compilation is a CPU syntax and symbol check. Correctness uses the immutable task-specific oracle
and baseline contract with the original tolerances and supplemental replay/value checks.
Performance must cover every declared case using equivalent work and state for both Arena legs.
The Arena baseline is the committed starting implementation; historical GEAK speedups are archival
provenance and do not supply this run's score. Reports are written below the task directory.

Archival source package: `Z/Kimi-K3_fwd_grouped_kernel_stage1`. Capture-machine locations in metadata are
recorded as `provenance://` identifiers; they are not runtime paths. This import has not been
validated on a compatible GPU. A clean task-validator report is required before PR submission.

The editable symbols include Python launchers or operator dispatch where declared in the config.
Those edits must preserve the complete callable workload, launch dimensions, ABI and output/state
contract. A device-kernel speedup cannot be claimed by deleting operator work or changing tested shapes.

`ut/meta.json:standalone_binding` names the protected adapter used by the unit test
and the common benchmark selector. Its `validate_layout()` method checks the same
baseline and candidate inputs without importing GPU packages during compilation.

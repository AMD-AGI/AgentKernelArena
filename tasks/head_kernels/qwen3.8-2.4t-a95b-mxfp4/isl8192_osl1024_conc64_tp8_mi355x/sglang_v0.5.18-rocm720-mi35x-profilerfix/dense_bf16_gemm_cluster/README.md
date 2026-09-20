# qwen3.8-2.4t__dense_bf16_gemm_cluster

This task optimizes the **Qwen3.8-2.4T-A95B-MXFP4** `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI -> MT256x256x64_MI (ledger hk04 dispatches hipblaslt)` callable on MI355X (`gfx950`).
The serving contract is ISL 8192 / OSL 1024 / CONC 64 / TP 8; its captured execution regime is `prefill`.

Runtime image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`.
Backend: `hipBLASLt / Tensile`; execute in that image's captured SGLang/AITER/PyTorch environment.

The runtime image, source package, editable files, required symbols, timeouts and architecture gate
are declared in [config.yaml](config.yaml). Starting-source provenance: **stock**.
The source snapshot is preserved as imported; inspect [ut/meta.json](ut/meta.json) for the captured
cases for upstream measurement caveats and live capture provenance.

Provision the task's external artifacts using the suite's declared artifact manifest before running.
Captured oracle and timing files must keep their recorded checksums. Missing inputs are environment
errors and must not be replaced by generated routing, placeholder geometry or reduced case sets.

Inside the declared GPU runtime, run from this task directory:

```bash
python3 scripts/dense_task_runner.py compile --timeout 600
python3 scripts/dense_task_runner.py correctness --timeout 3600
python3 scripts/dense_task_runner.py performance --timeout 3600
```

Compilation is a CPU syntax and symbol check. Correctness uses the immutable task-specific oracle
and baseline contract with the original tolerances and supplemental replay/value checks.
Performance must cover every declared case using equivalent work and state for both Arena legs.
The Arena baseline is the committed starting implementation; historical GEAK speedups are archival
provenance and do not supply this run's score. Reports are written below the task directory.

Archival source package: `P/dense_bf16_gemm_task`. Capture-machine locations in metadata are
recorded as `provenance://` identifiers; they are not runtime paths. This import has not been
validated on a compatible GPU. A clean task-validator report is required before PR submission.

The editable symbols include Python launchers or operator dispatch where declared in the config.
Those edits must preserve the complete callable workload, launch dimensions, ABI and output/state
contract. A device-kernel speedup cannot be claimed by deleting operator work or changing tested shapes.

## Frozen dispatch setup

The task retains exactly five authoritative rows in `ut/live_dispatch_rows.csv`.
Its recorded SHA-256 is checked before runtime imports. The task entrypoint sets
`AITER_CONFIG_GEMM_BF16` first, so runtime preflight cannot cache an unrelated
merged/default table. The protected setup also clears AITER's public cached
configuration lookup and verifies the resolved table path before using it.

Only protected configuration getters share the warmed native cache; candidate
backend functions and the callable ABI remain unchanged. The task observes the
actual candidate `solMap` and runs fresh backend/device-kernel selection checks.
Missing kernel catalog support or any backend/solution/split-K/kernel mismatch
still fails. In particular, `torch/0` without the retained `kernelName=native`
field is not an exact match for hk05.

This repair is CPU-tested only; it needs a fresh matching-image GPU run.

## Source binding and optimization freedom

AITER's `torch_compile_guard` reuses an existing globally registered operator
when a copied candidate defines the same name. The trusted overlay therefore
loads `gemm_a16w16` without that duplicate registration decorator; its body,
argument defaults, source file, and all backend functions remain unchanged.
The source body is called directly under the shared integrity monitor.

Recorded configuration and device-symbol checks apply to the retained native
baseline. The candidate must prove source execution and device work and pass
all existing mathematical, mutation, output-layout and graph-replay checks.
A valid candidate may use another backend or kernel symbol. Scored baseline
timing uses the retained configured native operator; the independent numerical
oracle remains `torch.nn.functional.linear`.

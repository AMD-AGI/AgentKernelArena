# glm-5.3-flash__fused_moe_kernel

This task optimizes the **GLM-5.3-Flash** `fused_moe_kernel` callable on MI355X (`gfx950`).
The serving contract is ISL 8192 / OSL 1024 / CONC 64 / TP 8; its captured execution regime is `decode`.

Runtime image: `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`.
Backend: `Triton - sglang srt/layers/moe/moe_runner/triton_utils.fused_moe`; execute in that image's captured SGLang/AITER/PyTorch environment.

The runtime image, source package, editable files, required symbols, timeouts and architecture gate
are declared in [config.yaml](config.yaml). Starting-source provenance: **stock**.
The editable source is the actual Triton `fused_moe_kernel` body. The protected dispatcher still
executes the complete fused-MoE operation against the captured oracle. Inspect
[ut/meta.json](ut/meta.json) for capture metadata and [ut/README.md](ut/README.md) for historical caveats.

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

Archival source package: `H/GLM-5.3-Flash_fused_moe_kernel`. Capture-machine locations in metadata are
recorded as `provenance://` identifiers; they are not runtime paths. This import has not been
validated on a compatible GPU. A clean task-validator report is required before PR submission.

The device module comes from public SGLang v0.5.18, commit
`71de97b264b04dcd514cf904003028aefe9775c8`, with SHA256
`9c3342d3147e7d60a78a2c934111f0fc1becbb8df1d2d32aafc82e6c8a0b2e70`.
The captured dispatcher exactly matches commit `aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`.
Its device module and the v0.5.18 module have an identical 44,623-byte prefix containing the
target kernel, launchers, activation and reduction helpers. Full source provenance and frozen hashes
are in [ut/device_source_contract.json](ut/device_source_contract.json). This source correspondence
does not establish the complete custom runtime image's Git revision.

Only `fused_moe_kernel`'s function body is editable. The task rejects changes to imports, decorators,
signatures, host launchers and other module code before importing a candidate. Both process overlays
bind a frozen full-operation dispatcher; the candidate overlay replaces its device module before the
dispatcher imports it. Worker identities verify the two independent source paths and hashes.
The archival `source/fused_moe.py` and all files under `ut/` remain protected.

[ut/sglang_bootstrap.py](ut/sglang_bootstrap.py) uses SGLang's official
`get_context().override_server_args(...).install()` API to publish the recorded non-model settings
through its supported dummy-model boundary. It checks deterministic inference is disabled,
fused MoE sum/all-reduce is disabled, and the runner is Triton. The original one-rank TP group is
retained: inputs already contain the captured TP8 shard and the serving all-reduce lies outside this
operator. This initialization avoids model-source resolution, so the standalone task needs neither
model configuration/weights nor the historical `glm5_next` architecture patch. Initialization is
still mandatory; an incompatible runtime context fails explicitly.

CPU tests cover source integrity, rejected host edits, independent module binding and context
initialization. They do not establish GPU correctness or performance; fresh validation on the
declared image and architecture remains required.

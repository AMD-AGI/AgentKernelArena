# glm-5.3-flash__elementwise_copy_cluster

This task optimizes the **GLM-5.3-Flash** `elementwise_kernel_manual_unroll<128,4,...direct_copy_kernel_cuda...>` callable on MI355X (`gfx950`).
The serving contract is ISL 8192 / OSL 1024 / CONC 64 / TP 8; its captured execution regime is `decode`.

Use the public runtime pinned in [config.yaml](config.yaml). Its historical serving image is recorded separately under `headkernel.capture_runtime`.
Backend: `ATen direct_copy_kernel_cuda`.

The runtime image, source package, editable files, required symbols, timeouts and architecture gate
are declared in [config.yaml](config.yaml). Starting-source provenance: **prior-candidate**.
The source snapshot is preserved as imported; inspect [ut/meta.json](ut/meta.json) for the captured
cases and [ut/README.md](ut/README.md) for upstream measurement caveats.

This generated-input draft requires no external tensor archives. Its committed
[ut/generated_cases.json](ut/generated_cases.json) is 4,266 bytes and retains the captured
structure for every original case. Numerical scale values are generated from recorded evaluator-selected seeds. Original archive
hashes remain under `ut/meta.json:archival_capture` as provenance, not runtime requirements.

Inside the declared GPU runtime, run from this task directory:

```bash
python3 scripts/generated_task_runner.py compile --timeout 600
python3 scripts/generated_task_runner.py correctness --timeout 3600
python3 scripts/generated_task_runner.py performance --timeout 3600
```

Compilation is a CPU syntax and symbol check. Correctness compares generated inputs against protected independent reference workers
with the original tolerances, layouts, aliases and complete case inventory.
Performance must cover every declared case using equivalent work and state for both Arena legs.
The Arena baseline is the committed starting implementation; historical GEAK speedups are archival
provenance and do not supply this run's score. Reports are written below the task directory.

Archival source package: `H/GLM-5.3-Flash_elementwise_copy_cluster_0913`. Capture-machine locations in metadata are
recorded as `provenance://` identifiers; they are not runtime paths. This import has not been
validated on a compatible GPU. A clean task-validator report is required before PR submission.

The editable symbols include Python launchers or operator dispatch where declared in the config.
Those edits must preserve the complete callable workload, launch dimensions, ABI and output/state
contract. A device-kernel speedup cannot be claimed by deleting operator work or changing tested shapes.

The image version is qualified by the complete frozen source, not only the historical README:
`ut/baseline_ref/fp8_utils.py.orig` is byte-identical to public SGLang v0.5.18 at
`71de97b264b04dcd514cf904003028aefe9775c8` (77,013 bytes; SHA256
`401aaf6a34f76781651dccc339899815e0b32989fc55880091f2136b712b43af`). It differs from the
source at the recorded v0.5.17 commit. The capture did not provide an independently verified image
digest. The task preserves the physical transposed-contiguous scale-layout contract; its callable
needs no model configuration or architecture bootstrap.

## Generated-input draft protocol

The declared entrypoint is `scripts/generated_task_runner.py`. It retains **7
recorded correctness cases, 8 random-value cases, and 8
timed cases**, with the original `tol=1e-06` and three random draws. All source kernel bytes,
ABI declarations, scalar arguments, shapes, strides, storage offsets, input alias groups and dispatch
attributes remain unchanged. Integer and boolean outputs compare exactly.

The trusted parent chooses a seed and launches independent reference/candidate workers. Their only
shared inputs are the seed and committed structure. Expected outputs stay in parent memory and are
never passed to the candidate process or saved as golden-answer files. The copy reference is the
mathematical transposed-contiguous layout operation; MoE uses the independent protected frozen
full-operation dispatcher/device modules. Singleton copy outputs retain their true input aliases.

Timing uses the canonical helper with **10 warmups and 100 device samples**. The worker returns outputs
from that exact timed graph after original/changed/restored numerical inputs. The parent compares
all three against separate reference outputs. Eager deployment correctness remains eager; timed
graph capture/replay is still required, and fallback, missing samples, input mutation or stale outputs
fail explicitly. The original numerical archives are not used by either default entrypoint.

The actual generator, checker, encoder, worker and case-builder functions are loaded and attested
before candidate import. Runtime function/module replacement, altered input metadata and shadowed
output methods fail worker finalization. This monitor is not an operating-system security sandbox.

For manual execution, materialize the canonical benchmark helper first with the repository's
`make materialize-perf-task TASK=<this-task-path>` command; normal Arena workspace preparation does
this automatically. Keep using the selected runtime environment from [config.yaml](config.yaml).
The public registry manifest, OCI config digest and historical capture identity remain separate.
Fresh qualification of the selected public runtime is pending.

The compact input contract and original archive hashes are retained in `ut/meta.json`
and task-local JSON. Isolated execution depends only on this task directory and its declared
runtime libraries. **Fresh GPU correctness, performance and task-validator qualification are pending.**

## Shared boundary and producer-branch qualification

This draft uses the shared worker protocol introduced by `a7bf289b` through
`headkernel.trusted_worker_modules`; private generated guard/bootstrap copies
have been removed. The integrated common guard preserves the newer DeepSeek checks and adds the shared two-phase preflight hook;
native resolution follows helper attestation and overlay installation.
[The layout proof](ut/copy_layout_proof.json) gives the exact row-major stride and
alias derivation and the recorded transpose_scale=False producer call. It also
flags the external input_scale branch and missing per-timed-call branch linkage.
The eight declared timing cases remain source-contract cases, not unconditional
qualification of every served producer branch.

# Portable head-kernel inputs and optional archives

All **18 tasks** in the [head-kernel suite](../../tasks/head_kernels/README.md)
construct their inputs locally and require **zero external tensor fixtures**.
The [artifact manifest](../../tasks/head_kernels/artifacts.json) has an empty
required-artifact list and declares all 18 tasks without persistent fixtures.
This covers MiniMax, Kimi, DeepSeek, GLM and Qwen.

Run from the repository root to inspect that declaration:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --list
python3 src/tools/prepare_head_kernel_artifacts.py --verify
```

The result is `0 fixtures; 0 bytes declared`, followed by the 18 task paths.
No mirror or cache argument is needed. The utility's success checks the external
artifact requirement; it does not run kernels or attest every committed compact
input. Task-local runners perform their own integrity and correctness checks.

## What each task carries

Each task includes source code, protected runners and reference implementations,
case metadata, and its declared input construction. Converted capture tasks keep
compact structural JSON and generate numerical buffers deterministically. Other
tasks already generate inputs directly from their mathematical contract.
Shapes, dtypes, strides, storage aliases, paging and routing controls, quantized
layouts and mutable-state behavior are governed by the individual task files.
The generated numbers are new samples, rather than a claim of original captured
numerical values.

Correctness compares against an independent mathematical reference or a
protected original implementation. Where workers are used, the trusted parent
compares independently produced reference and candidate outputs. A temporary
file of golden answers is not passed to the candidate. Metadata and generators
remain protected evaluation inputs when the task is copied into a workspace.

Self-contained inputs do not imply that every captured serving workload is
fully supported or GPU-qualified. Missing authentic workload controls and native
qualification failures remain explicit gates; they must not silently select a
smaller workload or a different reference. See the current
[validation status](../../tasks/head_kernels/VALIDATION.md) and each task's
metadata for eligibility and limits.

## Select the public runtime

Use a compatible MI355X (`gfx950`) host with Docker/ROCm and the task's pinned
public runtime. The [runtime guide](top5-head-kernels-runtime.md),
[environment matrix](../reference/top5-head-kernel-environments.md), and
[public image catalog](../reference/top5-public-images.md) give the exact
requirements. No model checkpoint, private registry access, original compute
node, or shared cluster filesystem is required by these input contracts.

Container image downloads are separate from task data. The historical names in
capture provenance describe where source evidence originated; the current
`config.yaml` selects the public image used for execution. Run-level task
selectors remain full paths relative to `tasks/`.

## Optional archival evidence

The original capture set contained **17 files totaling 33,139,788,731 bytes**:
14 input/output archives and three MiniMax timing-geometry files. That is a
historical inventory, with **zero bytes required for default task execution**.
Original hashes and sizes remain in the manifest's archival sections and in
`ut/meta.json:archival_capture`; compact task contracts retain the structural
evidence needed by their generators.

The original tensor files are not distributed in Git or fetched automatically.
A developer who already has an authorized local copy can use the
[offline extraction utility](extract-head-kernel-contracts.md) with an explicit
`--task`, `--archive`, and new `--output` path. The utility verifies the archived
SHA-256 before restricted CPU loading. No SSH host, HTTP/S3 endpoint, or private
mirror is built into this workflow. Some tasks expose an explicit archival
comparison command; consult that task's README before selecting it.

The provisioning utility still supports local `--mirror` and `--cache` inputs
for an explicitly supplied manifest that declares required files. It does not
install entries from the current manifest's optional archival sections.
Version-2 declarations separate the nested destination `path` from the original
flat `mirror_path` (`<operation-id>/ut/<filename>`). `--task` selects an exact
suite-relative task path, and `--cache` expects files named by SHA-256. Neither
option is part of the default 18-task setup.

Generated random-baseline caches, archived timing reports, and conversion
reports are not persistent correctness oracles. Keep local research output
outside the shipping task tree. Fresh GPU correctness, performance and
framework validation remain separate requirements under the
[task validator contract](task-validator.md).

# Prepare the head-kernel tensor fixtures

The head-kernel suite imports source code, case metadata, and frozen benchmark
contracts. Its persistent tensor bodies are provisioned separately before the
framework copies a task into an isolated workspace. Run the preparation utility
from the repository root with a trusted local artifact mirror or cache that you
explicitly select. The utility performs no downloads and needs only Python's
standard library.

The declaration in [artifacts.json](../../tasks/head_kernels/artifacts.json)
contains 17 fixtures totaling **33,139,788,731 bytes** (about 30.86 GiB):

| Fixture | Count | Total bytes |
| --- | ---: | ---: |
| `ut/reference_io.pt` | 14 | 33,138,154,092 |
| MiniMax `ut/timing_geometry.pt` | 3 | 1,634,639 |

Every required persistent fixture has a declared SHA-256 in its task's
`ut/meta.json`. Sizes come from the captured source inventory. All 17 bodies
were subsequently copied into an owned mirror and fully SHA-256 verified on
2026-09-20. The preparation utility also checks the supplied bytes against both
declarations when installing them. All three
MiniMax timing geometry files exist in the captured inventory. Earlier source
summaries that reported those files as missing are superseded by that inventory.

The Qwen dense BF16 GEMM and fused add RMSNorm tasks declare deterministic input
generation and frozen runtime reference outputs; they require no persistent
tensor file. The two GLM GEMM tasks use deterministic synthetic operands and
independent FP32 references at their declared shapes; they also need no persistent
tensor fixture. All 18 configured tasks are covered by the manifest.
Generated `_baseline_random.pt` files are outside this fixture
set and are never installed as reference data. The selected source inventory
also lists one 41,953-byte MiniMax `_baseline_random.pt` with no declared hash:
18 tensor files totaling 33,139,830,684 bytes when that generated cache is
included. The manifest records it explicitly as excluded.

## Why this revision needs captured tensors

The 33.14 GB is serialized test input and reference-output data, not a model
checkpoint or an additional runtime dependency. Four tasks already generate
their inputs and reference outputs from code; the other 14 currently replay
the imported capture package's `reference_io.pt` files. MiniMax's three input
and output captures account for approximately 21.03 GB of the total. Its three
separate timing-geometry files together require only 1.63 MB.

Large captures are not inherently required for kernel correctness. A portable
task can generate valid tensors at its declared shapes, dtypes, strides and
layouts, then compare the candidate against an independent mathematical
reference or a protected original implementation. That is the approach used
by the older `head_kernels` tasks and the four fixture-free tasks here.

Converting the remaining tasks requires more than randomizing each tensor by
shape: sparse indices, ragged lengths, expert routing and padding, packed
quantization scales, dispatch attributes and mutable state must remain valid
and representative. Any generated-input contract must be checked against the
captured runtime before claiming equivalent workload coverage. This revision
does not yet implement that conversion. The captured files are still required
for its 14 capture-dependent tasks; missing files must not silently select a
different workload or skip correctness checks.

The captures provide replay evidence, not protection against benchmark
tampering. Input generation, reference computation, timing and comparison must
remain evaluator-controlled in either design.

## Select a local source

For the branch owner's runs on Crusoe, the fully verified mirror is available
at this shared-NFS location:

```text
/shared_nfs/sapmajum/aka-top5-artifacts-20260920-2824de3e65244ec68722b0915cb72b0b/tasks/head_kernels
```

Access was checked on `crsuse2-slog-003` on 2026-09-20 as `sapmajum`
(UID 50090896): all 17 files opened read-only and matched their declared sizes.
The namespace is private to that UID. Another node must mount the same NFS and
run as that user to access this mirror. This is an existing local source, not
a public download service; no OCI, HTTP or S3 endpoint has been published.

From a checkout on a node with that access, provision the required files with:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py \
  --mirror /shared_nfs/sapmajum/aka-top5-artifacts-20260920-2824de3e65244ec68722b0915cb72b0b/tasks/head_kernels
```

Add the repeated `--task` selections described below to avoid copying fixtures
for unselected tasks. Other users need their own authorized mirror or cache;
the tool cannot fetch missing payloads from GitHub.

A mirror preserves the original flat operation IDs, independently of the
suite's model/workload/image/kernel directories:

```text
artifact_mirror/
└── minimax-m3__decode_score_kernel/
    └── ut/
        ├── reference_io.pt
        └── timing_geometry.pt
```

The version-2 [manifest](../../tasks/head_kernels/artifacts.json) gives each
fixture an explicit `mirror_path` in that namespace and a separate `path`
relative to `tasks/head_kernels` for its nested task destination. Its
`operation_id` preserves the imported identifier; its `task` is the full
suite-relative task path. Existing reviewed flat mirrors require no rename.
The utility also accepts version-1 manifests with their original flat
task/path contract. Both versions enforce the same byte and path checks.

Obtain the declared files from the source package owner using your separately
authorized transfer workflow. The manifest records source package names and
relative source paths as provenance; the utility never uses those records as
runtime paths. No public download endpoint is assumed.

Inspect the declaration without reading tensor files:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --list
```

Install every declared fixture from an explicitly chosen mirror:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --mirror artifact_mirror
```

Repeat `--task` to provision only the tasks selected for an experiment:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py \
  --mirror artifact_mirror \
  --task minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/decode_score_kernel \
  --task qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/paged_attention_decode
```

Here `--task` matches an exact path relative to the suite root, as declared in
the manifest. It does not accept an ambiguous kernel basename or treat a
stable operation ID as an alias for a version-2 destination.

Alternatively, `--cache artifact_cache` reads each file from
`artifact_cache/<sha256>`, where `<sha256>` is the 64-character digest in the
manifest. Cache and mirror sources are mutually exclusive. Directory and file
symlinks are rejected, including links within either explicitly selected root.
Supply the real directory path if your normal cache path is a symlink.
Existing hardlinks between public source packages are allowed. Installation
creates independent copies and never adds a link to a source inode.

The destination defaults to `tasks/head_kernels` in this checkout. Use
`--suite-root copied_suite` to prepare an existing copy of the suite, and
`--manifest tasks/head_kernels/artifacts.json` when that copy does not include the
suite manifest. The task and `ut` directories must already exist; preparation
never creates or replaces task source directories.

## Verify and run

Verify installed files before launching an experiment:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --verify
```

`--verify` also supports repeated `--task` selection. It returns nonzero on a
missing fixture, a size or hash mismatch, a metadata disagreement, or an unsafe
path. A successful verification proves fixture byte integrity; it does not run
the kernels or establish GPU correctness or performance.

Installation verifies the complete file before atomically publishing it in the
task's `ut` directory with read-only permissions. Existing valid files keep their
bytes, inode, permissions, and modification time. An existing invalid or unknown
file causes an error and is preserved for inspection. There is no force or
overwrite option. Interrupted copies remove their temporary file; a process
killed without cleanup may leave a hidden `.partial` file. Publication is atomic
per fixture, so fixtures completed before a later error remain available.
Atomic publication links the verified private temporary output to the final
name, failing if that name already exists, then removes the temporary alias.
The source inode is never linked. The completed destination is an independent
copy with one link. This works on NFS without requiring `renameat2` support.

Provisioned fixtures are ordinary local files, never symlinks or hard links into
the source mirror. The framework's task copy includes them as protected test
inputs. The copied task therefore needs no mirror, cache path, or repository
setup utility at runtime. Keep tensor binaries, temporary files, generated
baseline caches, and evaluation outputs out of Git.

After provisioning and selecting the compatible GPU runtime, follow the normal
Docker workflow in [the task validator guide](task-validator.md). New tasks still
require a fresh, framework-finalized `validation_report.yaml` with
`overall_status: PASS` on compatible GPU hardware before PR submission.

The focused provisioner regressions use tiny synthetic bytes and run on CPU:

```bash
python3 -m unittest discover -s tests -p test_head_kernel_artifacts.py -v
```

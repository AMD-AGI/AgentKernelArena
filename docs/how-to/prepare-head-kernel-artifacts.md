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
`ut/meta.json`. Sizes come from the captured source inventory. Fixture bodies
were not fetched or rehashed during this import; the preparation utility checks
the supplied bytes against both declarations when installing them. All three
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

## Select a local source

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

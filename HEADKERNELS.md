# Upstream head-kernel setup

This branch restores the original `headkernel_ut_0914_full` tasks for the five
requested model families, as retrieved on 2026-09-22. Commit `a16f2203` records
that restoration, including the task sources, unit tests, configuration,
baseline packages, internal aliases and upstream runners. The adapted
generated-input suite is replaced. Subsequent portable fixes are described below.

The [reported workload results table](docs/reference/headkernel-reported-results.md)
records the supplied E2E gains, head-kernel speedups and roofline figures, with
links to the corresponding tasks and explicit provenance limits.

The canonical tasks are in [`tasks/headkernel/`](tasks/headkernel/). Their original
flat directory names are retained:

| Model | Configured tasks | `NOT_BUILT` placeholders |
| --- | ---: | ---: |
| DeepSeek V4 Pro | 3 | 0 |
| GLM 5.3 Flash | 2 | 2 |
| Kimi K3 | 3 | 2 |
| MiniMax M3 | 3 | 1 |
| Qwen3.8 2.4T | 5 | 0 |
| Total | 16 | 5 |

The two GLM GEMM tasks added by the earlier adaptation are upstream placeholders
again. The previous adapted suite and its eight native-verified results remain
in Git history at commit `8b57ca074e7fd28fcba96b1d0097e772b6054c22`; those results
do not qualify the restored setup.

## Original setup and inputs

Read the [setup guide](docs/how-to/headkernel-upstream-runtime.md) and the
[verbatim upstream README](HEADKERNELS_UPSTREAM.md). Each task's `config.yaml`
retains its original capture-image declaration. The setup guide lists the
corresponding public HyperLoom images and their compatibility limits; it records
the custom Kimi build separately.

The original numerical input contract is restored. The 16 configured tasks need
17 original tensor files totaling 33,139,788,731 bytes: 14 reference archives and
three MiniMax timing-geometry files. Two tasks use their original deterministic
runtime baselines instead of persistent reference archives. The tensor payloads
are external to Git. Their exact names, sizes, and SHA-256 hashes are declared in
the [artifact manifest](tools/headkernel-artifacts.json); the setup guide explains
how to download them from OCI with rclone and verify each original SHA-256.
Generated substitutes are not provided. The OCI root is
`oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922`.
The original patch, GPU-lock script, and GLM configuration/tokenizer files are
available under its `setup/` prefix; see the
[setup manifest](tools/headkernel-setup-artifacts.json).

Task-local `ut/meta.json`, `ut/cases.py`, and the original UT documentation govern
shapes, values, layouts, references, and scoring. The adaptation's generated
contracts, shape catalogs, scoring restrictions, and native-verified selectors
are not part of this restored set.

## Restoration scope

The upstream package also contains GLM 5.2 entries outside the five requested
families. Its original full manifest and maintenance documentation are retained
as source evidence; the task selection above contains the requested families.
Generated build output, caches, temporary overlays, and run logs are not shipped.

The upstream README is preserved verbatim and contains historical statements.
For example, its older MiniMax missing-geometry warning predates the three files
present in the retrieved inventory. Preserving an upstream result or statement
does not turn it into a new verification result.

No new GPU validation was requested or performed for the original restoration. File-copy
and setup checks are separate from upstream GPU results. This branch does not
claim a new framework `task_validator` PASS, universal resistance to benchmark
cheating, or complete HyperLoom end-to-end equivalence.

## Portable execution fixes

These corrections address shared execution behavior. They do not introduce
optimized kernel implementations, substitute captured inputs, or change the
reference math, test shapes, tolerances, warmups, sample counts or timing method.

- **Keep evaluation connected to editable source.** Workspace, quality-loop,
  held-out and inspection copies preserve relative task symlinks such as
  `ut/kernel_src/kernel.py -> ../../source/kernel.py`. The harness guard records
  the original editable-source aliases and rejects missing, detached or
  redirected aliases. Noneditable support files remain protected. Task packages
  must contain their symlink targets; old workspaces with detached copies must
  be recreated from the task rather than treated as valid candidate evaluations.
- **Separate candidate and production MoE registration.** The Qwen two-stage
  MoE candidate registers its guarded operation under a module-qualified name,
  so AITER cannot reuse the production operation's registration for the candidate.
  Duplicate candidate registration fails explicitly.
- **Keep benchmark failures visible.** The common runner, copied into all 16
  configured tasks, permits the existing UT timing fallback only when replay
  reconstruction is unsupported. Failed candidate calls, failed case builders,
  timeouts, stale output and invalid or partial measurements cannot become a
  successful score through fallback. Successful CUDA-event timing retains the
  original measurement procedure.

Focused regression coverage is in `tests/test_task_source_aliases.py`,
`tests/test_headkernel_moe_registration.py` and
`tests/test_headkernel_benchmark_failures.py`. CPU tests establish these control
paths; they do not establish real GPU correctness, AITER dispatch or performance.

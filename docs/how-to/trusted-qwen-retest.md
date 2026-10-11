# Trusted Qwen graph retests

The maintained `src/tools/trusted_task_eval.py` evaluator supports the explicit
quality contracts for Qwen dense GEMM, MXFP4 MoE, recurrent decode and paged
attention. It executes reference compile, correctness and performance, followed
by candidate compile, correctness and performance, in six fresh containers.
The older RMSNorm qualification driver remains separate.

Run the evaluator from the trusted host after the scheduler grants a GPU:

```bash
python3 -I -B src/tools/trusted_task_eval.py \
  --repo /path/to/AgentKernelArena \
  --commit FULL_GIT_COMMIT_HASH \
  --task tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster \
  --candidate-workspace /path/to/candidate/task \
  --render-device /dev/dri/renderD129 \
  --scratch-dir /path/to/owned/nvme-scratch \
  --output /path/to/fresh/evidence-directory
```

Use the render device actually assigned by the scheduler. The existing GPU
binding helper resolves its UUID and PCI address and verifies one matching
device inside every phase container. Host phase caps and container limits use
the evaluator's existing `--compile-timeout`, `--correctness-timeout`,
`--performance-timeout`, `--container-cpus` and `--container-memory-gib` options.

For stock qualification, supply the original source files from the selected Git
commit as the candidate workspace. For an optimized candidate, only the declared
regular source files are copied. Candidate reports, caches, config, fixtures and
harness files are ignored. Closed task source guards and the full framework
harness guard run before GPU access. Literal routing selectors are parsed as
data; their Python is never executed.

Original fixtures come from `tools/headkernel-artifacts.json` at that same Git
commit. Supply `--fixture-local-mirror` using the registry's original paths, or
explicitly approve its exact OCI root with `--fixture-oci-prefix`. Fixture-free
tasks require no download. Whole files are checked against the committed byte
counts, SHA-256 and task metadata before installation. This preserves the
documented SG518 TP8/EP8 capture; it does not assert a new serving capture.

Task inputs remain read-only and phase containers have no network, agent auth,
or Docker socket mount. The old recurrent and attention correctness runners write
declared outputs under `ut/`, so those directories receive fresh writable output
areas with the committed inputs and source-alias directories mounted read-only
inside them. They cannot replace the protected inputs. Original Git source
aliases are accepted only when explicitly declared and contained in the task;
new candidate-source symlinks are rejected.

Each phase's small trusted bridge invokes the existing task runner and binds its
self-generated report to the host request and current source digests. The host
checks the original correctness ledgers, complete CPU-truth reports, native
routing reports and current performance quality adapters. Nested raw reports,
logs, GPU proof and container exit evidence are retained. The six
`*.phase_receipt.json` records are included by hash in `trusted_measurement.json`.
Missing, stale or incomplete phases cannot produce a complete measurement.

All original shapes, tolerances, warmups, samples, reset logic and numerical
checks remain task-owned and unchanged. Timing-quality rejection preserves all
raw samples. Stock sources and equivalent native routes remain controls with no
accepted gain. A completed trusted retest does not substitute for the separate
framework-finalized `task_validator` PASS.

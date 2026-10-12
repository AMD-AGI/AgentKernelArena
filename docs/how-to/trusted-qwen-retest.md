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
areas with every protected top-level file and complete subtree mounted read-only
at its own entry. Protected parent directories are mount points and cannot be
renamed and replaced. The writable backing tree stays outside all other container
mounts while the phase runs; it moves into the evidence build only after container
removal succeeds. Original Git source
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

The native image cache copier defaults to batches of 64 files. Trusted hosts may
opt into larger local batches with `AKA_AITER_JIT_COPY_BATCH_SIZE=512` (range
1–512), or `--batch-size` on `seed_aiter_jit_cache.py`. Each copier caps its
request using its own soft `RLIMIT_NOFILE`: 64 descriptors reserved and 4 budgeted
per offered file. A limit of 1024 therefore caps a 512 request to 240. The seed
container applies its own limit, and the effective size and observed limit are
logged when enabled. No process resource limits are raised.

Only the number of files offered to each rclone process changes. Every copy
retains `--transfers 64000`, `--progress`, zero buffers, disabled multi-thread
streams, `GOMAXPROCS=1`, fresh private destinations, all three full SHA passes,
path/link parity and executable-mode checks. Containers, cases, seeds, input
checks, warmups and samples remain unchanged. Use a reviewed helper version;
do not change active packages or infer GPU speedup from preparation savings.

Batches above64 also require readable current PID/thread accounting and an
explicit `AKA_AITER_JIT_COPY_THREAD_ALLOWANCE` assigned by the trusted caller
**exclusively to that copy for its lifetime**. It is a reserved task-slot count,
not a count of momentarily free shared slots. The allocator must cover shared
or namespace-hidden ancestor/UID limits; do not broadcast the same allowance
to concurrent copies without reserving their combined demand. The copier
reads applicable visible cgroup-v2 `pids.max`/`pids.current` ancestors and
`RLIMIT_NPROC` with visible real-UID thread usage. Unknown accounting or no
allowance caps at64; known lower headroom reduces it further. Two blocking-I/O
threads per offered file plus64reserved runtime slots bound admission.512files
therefore require at least1088exclusively allocated additional task slots and
sufficient observed headroom, as well as the descriptor budget. No process
limit is raised. This is conditional admission, not a universal safety claim.

Use `python3 src/tools/seed_aiter_jit_cache.py --show-copy-budget --batch-size 512`
to inspect the actual caller context without copying or GPU work. The seed
container rechecks its own ceilings. Every integrity pass and copy flag remains
unchanged, including disabled multi-thread streams.

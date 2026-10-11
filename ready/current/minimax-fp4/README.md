# Current qualification: MiniMax FP4 GEMM

The qualification covers all **14 captured cases** of
`headkernel/minimax-m3__gemm_afp4wfp4_kernel`: all six trusted phases and all
12 framework checks passed, with **4,200 raw samples and 168 submitted-source
controls**. Recomputed timing quality passes without trimming. This is an
unchanged-source control; **no optimization or serving/E2E gain is accepted**.

The GPU-qualified framework and task checkout is
`7bc14698e1044931a3c893b3dbabc9676b0e57e6`. The task tree is
`795611e8f0c602084338e47bd5f3a601da8b8744`, and `source/kernel.py` has SHA-256
`66d8611eae70089b20639eae8dc94ae4df1d7a5cf537317575ea89adf9944fbf`.
[READY.json](READY.json) binds the reports, published archive and runtime pins.
The [historical entry](../../minimax-fp4/README.md) retains its earlier receipts
and limitations.

The [compatibility review](COMPATIBILITY-5854b2ac.json) separately approves
`5854b2ac7a55fb287fc1b5d4d6fd4741f0b51476` for this task. The task tree is
identical. Both checkouts produce identical validation and quality results
from all nine retained raw reports, and 179 focused CPU regressions pass.
The review covers the shared evaluator changes and matched performance
protocol; task-tree equality alone is not its approval basis. The GPU reports
keep their original `7bc14698` identities. Later revisions require a new pin
or compatibility review.

The exact qualified checkout has **255 pinned files**; the compatible checkout
has **257**. Both inventories cover every tracked file under `src/`, `agents/`,
`tools/`, `docker/`, this task, all root files, and the two scoped configurations.
They include `src/quality_measurements.py`, `src/benchmark_quality.py`, the
quality CLI, and the framework report finalizer. The verifier defaults to the
GPU-qualified pins; `--compatible` selects the separately reviewed pins.

## Completion and recovery

The framework process exited successfully at **2026-10-11 12:49:51.507309 UTC**.
Source-parity exit/completion receipts and ownership retirement followed.
`FINAL.json` was written at **12:49:54.362495 UTC**, about 0.638 seconds before
allocation cancellation at **12:49:55 UTC**. The retirement receipt records
seven exited containers with exit code 0 and zero remaining owned containers.

The last monitor snapshot preserved an earlier agent YAML and every raw report,
but missed the finalized YAML, completion marker, and `MONITOR-COMPLETE.json`.
The retained YAML and raw reports, processed through the pinned deterministic
finalizer logic, reproduced the finalized report and marker **byte for byte**:
both SHA-256 values match the pre-existing `FINAL.json`. The recovered files are
stored separately under `evidence/independent-recovery-20261011/` in the archive,
with their recovery receipt. No new GPU execution or qualification was performed.

The archive preserves the earlier YAML, intermediate states and all monitor
snapshots. `MONITOR-COMPLETE.json` remains absent. The NVMe source-parity log was
not checkpointed; its successful exit and phase-completion receipts survive.
The later cancellation receipt has a stale `scheduler_receipt_sha256` field;
its explicit `scheduler_raw_sha256` matches the archived scheduler file. Both
fields remain unchanged and the discrepancy is recorded in `EVIDENCE-STATE.json`.

## Prepare a pinned run

Use Linux amd64, MI355X/gfx950 and the pinned SGLang 0.5.20 image. From the
repository containing this guide, prepare a fresh checkout and local NVMe space:

```bash
FP4_GUIDE="$PWD/ready/current/minimax-fp4"
FP4_STORAGE=/path/to/local-nvme/minimax-fp4-current
FP4_COMMIT=7bc14698e1044931a3c893b3dbabc9676b0e57e6
mkdir -p "$FP4_STORAGE/scratch" "$FP4_STORAGE/tmp"
git worktree add --detach "$FP4_STORAGE/qualified" "$FP4_COMMIT"
python3 "$FP4_GUIDE/verify.py" --repo "$FP4_STORAGE/qualified"
export TMPDIR="$FP4_STORAGE/tmp"
export GOMAXPROCS=1
export RCLONE_CONFIG=/path/to/your/rclone.conf
export AKA_SKIP_DEV_MEM=1
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
python3 "$FP4_STORAGE/qualified/tools/prepare_headkernel_run.py" \
  --repo "$FP4_STORAGE/qualified" --commit "$FP4_COMMIT" \
  --output "$FP4_STORAGE/prepared" --scratch-dir "$FP4_STORAGE/scratch" \
  --task headkernel/minimax-m3__gemm_afp4wfp4_kernel \
  --use-manifest-oci-prefixes
```

The existing preparer verifies exactly **74 external fixture objects /
2,044,395,146 bytes**, using the manifest and prefix in `READY.json`. Fixture
transfers retain `GOMAXPROCS=1` and
`--transfers 64000 --progress --buffer-size 0`. No model weights are required.
A preparation receipt means staged inputs; it does not create GPU qualification.

Configure the chosen agent normally and use the single-task configuration:

```bash
cd "$FP4_STORAGE/prepared/arena"
make docker-check-agents CONFIG=example_configs/ready_minimax_fp4_mi355x.yaml
make docker-run CONFIG=example_configs/ready_minimax_fp4_mi355x.yaml
```

The existing lane controller must bind the assigned physical GPU UUID and render
device, keep caches on NVMe, honor the allocation, and retire only owned
containers. Only `_gemm_afp4wfp4_kernel`'s body in `source/kernel.py` is editable.
Preserve all 14 cases, seeds `[0,1,2]`, the independent packed-FP4 oracle,
10 warmups, 100 measured replays, fresh inputs, output poisoning, every oracle
check, and all no-op/wrong-output controls in eager and graph modes.

A changed candidate needs a fresh complete trusted comparison from the immutable
qualified checkout after the optimization worker stops. Use
`src/tools/trusted_task_eval.py` with `--commit "$FP4_COMMIT"`, the separate
candidate workspace, an assigned render device, explicit NVMe scratch,
`--fixture-max-files 74 --fixture-max-bytes 2044395146`, and `--timeout 1800`.
Retain all six phases and the authoritative quality decision. A rejected or
partial run stays rejected or partial; no automatic repeat selects a better run.
The published same-source raw ratio is **1.0310508755769854**, with no accepted gain.

## Published evidence

The OCI archive contains the complete shared qualification record and exact-hash
recovery, runtime compatibility checks, cancellation evidence, and historical
scope receipts. It contains no fixture tensors or compiled JIT caches. Every
published object was downloaded; every archive member was checked against the
manifest. The original 228 evidence files are preserved.

```bash
export GOMAXPROCS=1
FP4_EVIDENCE=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261011/qualification-evidence/minimax-m3__gemm_afp4wfp4_kernel/7bc14698-207740/
rclone copy "$FP4_EVIDENCE" /path/to/local-nvme/fp4-evidence \
  --transfers 64000 --progress --buffer-size 0
cd /path/to/local-nvme/fp4-evidence
sha256sum -c SHA256SUMS
tar -xzf minimax-fp4-current-complete-207740.tar.gz
```

[READY.json](READY.json) pins the archive, manifest and readback receipt.
The qualification covers this task's 14 captured cases only. Other MiniMax
heads, other models, whole-model readiness and serving/E2E gains remain outside
this entry.

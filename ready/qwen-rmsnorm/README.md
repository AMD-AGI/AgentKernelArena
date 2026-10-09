# Scoped starter: Qwen fused add + RMSNorm, both live shapes

This entry selects only
`headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm`. Its native source is
[`source/minimax_m3_rmsnorm.py`](../../tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm/source/minimax_m3_rmsnorm.py).
Keep the callable signature, both fresh output tensors, input immutability,
protected inputs/reference, and correctness/timing harness intact.

The two required cases are prefill `[8192,8192]` and decode `[64,8192]`, with
BF16 x/residual/weight, activation strides `[8192,1]`, weight stride `[1]`,
`eps=1e-6`, and the existing mixed tolerance `0.02`. The source and frozen
reference are identical to the pinned SGLang 0.5.20 implementation. The two
shapes come from the documented SGLang 0.5.18 serving capture; this qualification
does not claim a new SGLang 0.5.20 serving capture or an E2E improvement.

The trusted comparison retains measurement commit
`294e65a0b68a316de424d83ccfa4334ddc088456` and measurement task tree
`d94df4d6fe1705e66e00b5077d335d86eb090fc4`. The protection-only successor uses
`a4d4fe4a73e831b29c4467978534e21b24faa0c8` and task tree
`6339dec51c41c218c42b3e3e5c447c899bfc7ae2`. Its GPU source, numerical
harness, oracle, cases, and timing helper are byte-identical; the reviewed
compatibility receipt explicitly binds the two versions. Old measurement
receipts are not relabeled as measurements of the new task tree. The native file SHA-256 is
`e7e81662d1989eacd46b757b0240111ef9caae31a77fb3035ca2b9196307198d`.
[READY.json](READY.json) records the exact qualification and evidence bindings.
Preparation-time status fields inside the task package remain frozen with that
tree; the external qualification receipt records the completed run status.
This entry covers this one task and both cases. Other Qwen heads and other
models are not made ready by this entry.

## Replay environment

Use Linux amd64, MI355X (`gfx950`), ROCm-compatible Docker access, Git, Python 3
with PyYAML, and sufficient local NVMe space for private caches and workspaces.
No full-model checkpoint or external tensor fixture download is needed: inputs
and CPU truth are generated during each run. rclone is required by the standard
image-cache preparation helper and for optional evidence readback.

Expose only the GPU authorized for this run. A scheduler GPU number, a render
node number, and a process-local PyTorch ordinal are not interchangeable.
Resolve the assigned device to its physical UUID/PCI identity and verify that
the runtime sees exactly that device before running a shared-node benchmark.
Configure the scheduler's Docker device binding to expose `/dev/kfd` and only
the assigned render node. The commands below assume that binding is already
in place; the standard Docker runner's `/dev/dri` argument must not expand to
unassigned devices through a host Docker daemon.

From the root of this checkout:

```bash
python3 ready/qwen-rmsnorm/verify.py
RMSNORM_STORAGE=/path/to/local-nvme/qwen-rmsnorm
mkdir -p "$RMSNORM_STORAGE/tmp"
export TMPDIR="$RMSNORM_STORAGE/tmp"
export GOMAXPROCS=1
export RCLONE_CONFIG=/path/to/your/rclone.conf
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
export AKA_VISIBLE_GPU="GPU-<assigned-stable-UUID>"
export AKA_LOGICAL_GPU=0
export AKA_SKIP_DEV_MEM=1
docker pull "$AKA_DOCKER_IMAGE"
make docker-smoke
make docker-check-agents CONFIG=example_configs/ready_qwen_rmsnorm_mi355x.yaml
make docker-run CONFIG=example_configs/ready_qwen_rmsnorm_mi355x.yaml
```

The [optimization config](../../example_configs/ready_qwen_rmsnorm_mi355x.yaml)
selects only this task and the existing Claude Code template. Configure that
agent normally, or change only the agent block to a supported integration.
The [validator config](../../example_configs/validate_ready_qwen_rmsnorm_mi355x.yaml)
selects the backend used for the archived full framework review:

```bash
make docker-run CONFIG=example_configs/validate_ready_qwen_rmsnorm_mi355x.yaml
```

A later candidate still needs its own complete correctness, performance, timing
quality, and ordinary review. The stock-source qualification is not an accepted
optimization gain for a later source edit.

## Fresh inputs, CPU truth, and graph timing

The kernel forms `s = float32(x) + float32(residual)`. It normalizes using this
unrounded FP32 sum, then independently casts the normalized output and residual
sum to BF16. Neither output aliases the inputs or the other output.

Before every warmup, capture preparation, and replay, the evaluator generates
new BF16 x, residual, and weight values using a continuous CPU RNG. Ordinary
performance inputs retain the original uniform domains: x `[-0.75,0.875]`,
residual `[-0.625,0.5]`, and weight `[-0.125,0.125]`. Expected outputs are computed
entirely on the CPU and remain there. Only the three input tensors are copied
to the GPU; both public output buffers are poisoned with NaN before replay.

The materialized canonical graph helper records events around one graph replay.
Input generation/copies, CPU oracle work, completed-output checks, and poisoning
are outside the event interval. The preceding completed invocation is checked
before its inputs or outputs are overwritten. Capture failure or invalid timing
fails the run; these graph-capable cases have no event fallback.

Each case retains exactly **10 warmups and 100 ordered raw graph samples**.
Validation covers 114 timing-related invocations. Four additional correctness
challenges run on the same measured graph after timing: zero residual, exact
cancellation, near cancellation, and small finite amplitudes where epsilon
matters. The resulting report records 118 checked invocations and 120 generated
input sets, including capture setup. The correctness command also runs ordinary
values and those four challenges at both shapes after the unchanged frozen UT.
A private cache of the first outputs is rejected once inputs change, even when
it rewrites both poisoned public outputs.

## Timing quality and evidence

After a complete protected performance run, inspect its actual report and raw
sample file with the mandatory single-run quality check:

```bash
python3 ready/qwen-rmsnorm/check_timing.py \
  --report /path/to/workspace/build/performance_report.json \
  --raw /path/to/workspace/build/_bench_raw.json \
  --output "$RMSNORM_STORAGE/timing-quality.json"
```

This checks the complete two-case contract, every raw sample and reported mean,
CPU-truth/challenge metadata, and the canonical catastrophic-instability policy.
It does not trim samples, replace a failed run, attribute a submitted source, or
approve a speedup. Both source legs of the published trusted comparison use the
same native source, so that comparison is an acceptance-path control and has
no accepted gain.

[INPUT-PINS.json](INPUT-PINS.json) binds the protected task and canonical timing
helper in this checkout. [READY.json](READY.json) identifies the complete trusted
comparison, all 72 compiled controls, the separate full framework validation,
all raw timing-quality checks, OCI readback, and independent publication
approval. Interrupted runs and earlier event/fixed-input methods are preserved
as history and are not used as completed framework phases.

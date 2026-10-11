# Ready starter: Kimi Lean attention decode

The ready task is `headkernel/kimi-k3__lean_attention_decode`. Optimize only
`_lean_attention_decode_kernel` in
[`source/decode_attention.py`](../../tasks/headkernel/kimi-k3__lean_attention_decode/source/decode_attention.py).
The source guard freezes signatures, decorators, imports, helpers, launchers,
host controls, protected reference and harness.

This entry covers **one structural case**,
`_decode_lean_attention_fwd-616b10700cd40ae2986c1283`: batch 64, 12 query heads,
Q/K dimension 576, value dimension 512, page size 1, shared latent K/V storage,
and a persistent grid of 256 programs. It retains the full observed
sequence-length histogram from **8193 through 9216**. Replays draw fresh legal
inputs and work amounts from that histogram; 100 measured draws do **not**
exhaust all 1024 lengths. Each realized work schedule is recorded and checked.

The task and evaluated framework code match
`20c1a949b07e0d2530315ee666cae8a178edad69`; the task tree is
`4c3c322f6970a887e0ec5559da76b5f11ae000cf`. Full framework validation passed
**12/12 checks**, all six trusted GPU phases completed, and separately compiled
no-op and wrong-output sources were rejected after their GPU invocations
completed. The unchanged-source calibration passed the captured case and
correctness seeds `[0, 1, 2]`. The bad sources were rejected during captured
parity, before seed tests or timing; the no-op failed scratch-state parity and
the wrong-output source failed output parity.

The trusted primary paired ratio was **0.9912336585x** with unchanged source.
That is a control result, not an optimization gain. The archive retains all
400 trusted inner-leg samples and the 200 framework inner-leg samples. The
later timing-quality analysis passed, but 92 of 96 measured sequence-length
classes were singletons, so repeated stability for every work amount is not
established. No other Kimi task, whole-model readiness, or end-to-end serving
gain is claimed.

[READY.json](READY.json) records the completed scope and proof hashes;
[INPUT-PINS.json](INPUT-PINS.json) records the unchanged inputs. Earlier
provisional status text in the task and broad runtime catalog is preserved to
retain the qualified identities. Use this entry's one-task configs.

## Prepare the runtime and captured inputs

Use a Linux amd64 host with a MI355X (`gfx950`), ROCm-compatible Docker access,
Git, Python 3 with PyYAML, and rclone. Configure the host's `oci` remote for the
fixture prefix below. No full model checkpoint is required. The immutable
SGLang 0.5.20 image is recorded in the
[`sglang_v0520` runtime entry](../../tools/headkernel-runtime-targets.json).
Keep Docker data, fixture storage, preparation output and private caches on
local NVMe. The unchanged standard Docker helper initially seeds AITER under
host `/tmp`; that location also needs local storage and sufficient space.

From this handoff checkout:

```bash
python3 ready/kimi-lean/verify.py
LEAN_REPO="$PWD"
LEAN_HANDOFF_COMMIT=$(git rev-parse HEAD)
LEAN_QUALIFIED_COMMIT=20c1a949b07e0d2530315ee666cae8a178edad69
LEAN_STORAGE=/path/to/local-nvme/kimi-lean-ready
mkdir -p "$LEAN_STORAGE/scratch" "$LEAN_STORAGE/tmp"
export TMPDIR="$LEAN_STORAGE/tmp"
export GOMAXPROCS=1
export RCLONE_CONFIG=/path/to/rclone.conf
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
docker pull "$AKA_DOCKER_IMAGE"
python3 tools/prepare_headkernel_run.py \
  --repo "$LEAN_REPO" --commit "$LEAN_HANDOFF_COMMIT" \
  --output "$LEAN_STORAGE/prepared" --scratch-dir "$LEAN_STORAGE/scratch" \
  --task headkernel/kimi-k3__lean_attention_decode \
  --use-manifest-oci-prefixes
```

Use a new preparation directory each time. The existing helper clones the
handoff, fetches only this task's declared objects, verifies their sizes,
hashes and references, and installs them before workspace copying. Its
`staged_not_evaluated` receipt means preparation itself did not repeat the
qualification.

The fixture manifest contains **15 objects / 632,810,514 bytes** under:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/kimi-actual-v4-no-stack-194550/
```

Exact object names, hashes, storage offsets and codecs are in
[`EXTERNAL-MANIFEST.json`](../../tasks/headkernel/kimi-k3__lean_attention_decode/fixtures/EXTERNAL-MANIFEST.json).
Transfers use `GOMAXPROCS=1` and
`rclone --transfers 64000 --progress --buffer-size 0`. Each evaluation phase
uses a private, byte-verified image AITER cache.

## Start optimization

The [ready config](../../example_configs/ready_kimi_lean_mi355x.yaml) selects
only Lean decode and the existing Claude Code template. Configure normal agent
authentication or select another supported agent by changing its `agent`
block. That choice does not alter this task's qualification.

```bash
cd "$LEAN_STORAGE/prepared/arena"
make docker-smoke
make docker-check-agents CONFIG=example_configs/ready_kimi_lean_mi355x.yaml
make docker-run CONFIG=example_configs/ready_kimi_lean_mi355x.yaml
```

Preserve the frozen 0.02 RMS-relative/relative comparisons, all output and
scratch checks, integer locks, storage aliases, metadata, seeds, 10 warmups
and 100 samples. Candidate observations and immutable input truth are captured
on CPU before the independent frozen reference runs.

For an optional new full framework review, apply the exact archived
[validator profile](validator-profile.yaml) **only to the prepared checkout**:

```bash
cd "$LEAN_REPO"
rclone copyto ready/kimi-lean/validator-profile.yaml \
  "$LEAN_STORAGE/prepared/arena/agents/task_validator/agent_config.yaml" \
  --transfers 64000 --progress --buffer-size 0
cd "$LEAN_STORAGE/prepared/arena"
make docker-run CONFIG=example_configs/validate_ready_kimi_lean_mi355x.yaml
```

The profile retains the archived 1800/3600/3600-second command limits and
12000-second backend allowance. It changes runtime timeout configuration,
not task or framework code. Reserve the complete lifecycle before starting;
the qualified framework used a conservative 14,400-second bound, plus fixture
staging. This portable entry does not require a particular cluster scheduler.

## Retest a stopped candidate

Run the trusted evaluator from the handoff checkout after the optimization
worker has stopped. Choose a physical render device for the current machine;
render numbers do not identify the same GPU on every host.

```bash
cd "$LEAN_REPO"
LEAN_CANDIDATE=/path/to/stopped/task-workspace
LEAN_RENDER_DEVICE=/dev/dri/renderD128
python3 src/tools/trusted_task_eval.py \
  --repo "$LEAN_REPO" --commit "$LEAN_QUALIFIED_COMMIT" \
  --task tasks/headkernel/kimi-k3__lean_attention_decode \
  --candidate-workspace "$LEAN_CANDIDATE" --render-device "$LEAN_RENDER_DEVICE" \
  --output "$LEAN_STORAGE/trusted-candidate" --scratch-dir "$LEAN_STORAGE/scratch" \
  --fixture-oci-prefix oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/kimi-actual-v4-no-stack-194550/ \
  --fixture-max-files 15 --fixture-max-bytes 632810514 --timeout 7200
python3 ready/kimi-lean/check_timing.py \
  --measurement-dir "$LEAN_STORAGE/trusted-candidate" \
  --output "$LEAN_STORAGE/trusted-candidate-quality.json"
```

The evaluator accepts only the declared source submission, reconstructs the
protected task from the qualified commit, binds the selected GPU, and runs all
six phases in fresh containers. It mounts no agent or object-store credentials
into the source-only evaluator. A timeout is a per-phase cap; reserve a complete
six-phase and cache-preparation budget for the chosen candidate.

The mandatory postchecker validates all six reports, source/request/GPU and
fixture bindings, exact cases, paired input receipts and raw means. Its primary
score is the candidate invocation's protected native reference divided by the
candidate; the outer source-leg ratio remains a secondary diagnostic. It then
uses the unchanged timing policy from
`47aaa88342ba07ef67f95ba7b3c348eb20f945e1`, supplied as
[quality_policy.py](quality_policy.py), as a separate CPU-only admission step.
The task and measurement code remain at `20c1a949`.

The check retains all samples, writes a new receipt and exits nonzero for
invalid evidence or rejected timing quality. Same-source runs never receive an
accepted gain. A `pass` is an outlier-screen result, not statistical proof of
speedup; review the complete paired series and singleton-work-class limitation
before accepting an improvement. Do not trim samples or automatically retry
failures. Any authorized repeat must preserve the earlier result and rerun the
complete comparison.

To reproduce the postchecker tests with the archived `trusted/` directory:

```bash
python3 ready/kimi-lean/test_timing.py --measurement-dir /path/to/archive/trusted
```

## Audited evidence

The complete proof archive and per-file SHA-256 manifest are under:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261008/qualification-evidence/kimi-k3__lean_attention_decode/20c1a949-201533-r2/
```

[READY.json](READY.json) identifies the final summary, independent publication
approval, archive manifest, and readback evidence. The archive includes trusted
reports, the finalized framework checkpoint, compiled-control reports and
source variants, physical bindings, retirement receipts and historical failures.

The original trusted launcher reported a CPU verifier compatibility failure
because it expected the older outer-only meaning of `reference_ms`. The GPU
reports were complete; an independently approved corrected consumer verified
the paired `20c1a949` semantics. That original failure remains in `history/`.
The archive also preserves the failed image-name comparison and a pre-GPU
missing-directory setup attempt; neither repeated a numerical test. The final
compiled controls used the reviewed exact Docker Hub image-name equivalence
and completed under a fresh bound allocation.

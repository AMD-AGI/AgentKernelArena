# Ready starter: DeepSeek MLA prefill, two captured cases

The ready task is `headkernel/deepseek-v4-pro__unified_paged_attention_prefill`.
Optimize `pa_prefill_16mx1_16nx4_kernel` in
[`source/pa_sparse_prefill_opus.h`](../../tasks/headkernel/deepseek-v4-pro__unified_paged_attention_prefill/source/pa_sparse_prefill_opus.h).
Only its declared GPU body is editable. The source guard freezes the host
wrappers, launch decisions, other dispatch branch, references and harness.

This handoff selects exactly the two captured **M=8192, H=16** cases:

- `deepseek_mla_prefill-2800f2114cb8548cf5fdd0ba`
- `deepseek_mla_prefill-fff723d1365f8696979e5779`

The task and evaluated framework inputs match qualified commit
`20c1a949b07e0d2530315ee666cae8a178edad69`; the task tree is
`b659474a77efd6d2e4b795011b669d06e634d595`. Full framework validation passed
12 checks, all six trusted phases passed, and the no-op/wrong-output source
controls were numerically rejected for both cases and seeds 42/43 (eight
checks). All 600 raw timing samples were retained. The same-source mean ratio
was 0.995592x, with no sample above twice its series median. This is a correct,
measurable starter; it establishes no optimization or serving gain.

The [scoped readiness receipt](READY.json) and [input pins](INPUT-PINS.json)
record this later completed qualification. Earlier provisional text in the
unchanged task and broad runtime catalog remains historical. **Other tasks are
excluded from this recommended entry and its readiness claim.** Historical
prefill M=1/M=64 coverage and model-level qualification remain unclaimed. Use
the two configs linked below instead of an all-head-kernels selection.

## Prepare the exact environment and inputs

Use a Linux amd64 host with a MI355X (`gfx950`), ROCm-compatible Docker access,
Git, Python 3 with PyYAML, and rclone. Configure the host's `oci` remote to read
the committed fixture prefix. The replay needs no full model checkpoint.
The runtime is SGLang 0.5.20; the verified tag and immutable image identity are
recorded in the [`sglang_v0520` runtime entry](../../tools/headkernel-runtime-targets.json).
Keep image caches, fixture data, preparation output and result directories on
local NVMe storage. The unchanged standard Docker helper creates its initial
AITER seed under host `/tmp`; use a host whose `/tmp` and Docker data directory
are backed by local storage with enough space for the image and cache copies.
The explicit scratch path below controls fixture and trusted-evaluator staging.
Supply ordinary authentication for the chosen agent; the
source-only trusted evaluator mounts no agent or object-store credentials.

From the root of this handoff checkout, first verify its unchanged starter:

```bash
python3 ready/mla-prefill/verify.py
MLA_REPO="$PWD"
MLA_HANDOFF_COMMIT=$(git rev-parse HEAD)
MLA_QUALIFIED_COMMIT=20c1a949b07e0d2530315ee666cae8a178edad69
MLA_STORAGE=/path/to/local-nvme/mla-ready
mkdir -p "$MLA_STORAGE/scratch" "$MLA_STORAGE/tmp"
export TMPDIR="$MLA_STORAGE/tmp"
export GOMAXPROCS=1
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
docker pull "$AKA_DOCKER_IMAGE"
python3 tools/prepare_headkernel_run.py \
  --repo "$MLA_REPO" --commit "$MLA_HANDOFF_COMMIT" \
  --output "$MLA_STORAGE/prepared" --scratch-dir "$MLA_STORAGE/scratch" \
  --task headkernel/deepseek-v4-pro__unified_paged_attention_prefill \
  --use-manifest-oci-prefixes
```

Choose a new `prepared` directory for each preparation. This existing helper
clones the pinned handoff, fetches only the selected task's declared objects,
checks every size/hash and reference, and installs the verified fixtures before
workspace copying. Its receipt says `staged_not_evaluated`: staging itself does
not rerun qualification. The handoff commit adds entry metadata/configs while
preserving the qualified task and actual evaluated framework dependencies.

The exact fixture prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/deepseek-final-v7-mixed/
```

The manifest declares **25 objects / 1,413,573,052 bytes**. Individual object keys,
SHA-256s, storage offsets and codecs are in the pinned
[`EXTERNAL-MANIFEST.json`](../../tasks/headkernel/deepseek-v4-pro__unified_paged_attention_prefill/fixtures/EXTERNAL-MANIFEST.json).
The helper copies with `GOMAXPROCS=1` and
`rclone --transfers 64000 --progress --buffer-size 0`. Docker is digest-pinned;
the runner seeds each private AITER cache from the image, with byte verification.

## Start optimization

The [ready entry config](../../example_configs/ready_ds_mla_prefill_mi355x.yaml)
selects only this task and the existing Claude Code agent template. Configure
that agent normally, or change only the `agent` block to another supported
agent. Agent authentication/model choice does not alter the task's qualification.

```bash
cd "$MLA_STORAGE/prepared/arena"
make docker-smoke
make docker-check-agents CONFIG=example_configs/ready_ds_mla_prefill_mi355x.yaml
make docker-run CONFIG=example_configs/ready_ds_mla_prefill_mi355x.yaml
```

The [validator config](../../example_configs/validate_ready_ds_mla_prefill_mi355x.yaml)
is available for a fresh full framework review. It records the backend settings
used for the archived PASS. The task keeps its 1200/1200/3600-second
compile/correctness/performance limits, seeds `[42, 43]`, 10 warmups and 100
measured graph replays per case. Configure a compatible allocation and complete
run budget before any GPU work; no cluster-specific scheduler is required.

## Retest a stopped candidate

After the optimization worker stops, run the existing evaluator from the trusted
handoff checkout. Set `MLA_CANDIDATE`, `MLA_RENDER_DEVICE` and a new result path
for your machine; a render number is not a portable physical-GPU identity.

```bash
cd "$MLA_REPO"
MLA_CANDIDATE=/path/to/stopped/task-workspace
MLA_RENDER_DEVICE=/dev/dri/renderD128
python3 src/tools/trusted_task_eval.py \
  --repo "$MLA_REPO" --commit "$MLA_QUALIFIED_COMMIT" \
  --task tasks/headkernel/deepseek-v4-pro__unified_paged_attention_prefill \
  --candidate-workspace "$MLA_CANDIDATE" --render-device "$MLA_RENDER_DEVICE" \
  --output "$MLA_STORAGE/trusted-candidate" --scratch-dir "$MLA_STORAGE/scratch" \
  --fixture-oci-prefix oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/deepseek-final-v7-mixed/ \
  --fixture-max-files 25 --fixture-max-bytes 1413573052 --timeout 3600
```

This admits only the declared submitted source, reconstructs reference/harness
inputs from the qualified Git commit, binds the selected GPU, and executes six
fresh-container phases. Reserve the complete six-phase and cache-preparation
budget; the archived bounded run used a conservative 23,400-second admission
budget. A candidate's speedup requires its own complete reports.

## Mandatory timing-quality review before accepting a gain

After every trusted comparison, run this frozen post-evaluation check before
using a candidate speedup:

```bash
python3 ready/mla-prefill/check_timing.py \
  --measurement-dir "$MLA_STORAGE/trusted-candidate" \
  --output "$MLA_STORAGE/trusted-candidate-quality.json"
```

It checks all six native phase reports, exact case/source/request/GPU bindings,
fresh compiled operators, fixture and report hashes, and the raw means before
calling the unchanged `benchmark_quality.assess_comparison` implementation from
commit `47aaa88342ba07ef67f95ba7b3c348eb20f945e1`. The measurement path and task
remain at `20c1a949`; this is a separate mandatory admission step. It retains
all samples and original reports, writes a new receipt, and exits 2 when timing
quality rejects the comparison. Same-source controls never receive an accepted
gain, even when their raw ratio is favorable.

Reject a comparison with `status: reject`; consider a gain only when the new
receipt has `accepted_gain: true`, the executable source changed, and ordinary
review supports the claimed improvement. The shared module rejects dominant
extreme replays and extreme dispersion. This is a catastrophic-instability
gate, **not a statistical proof of a speedup or a universally noise-proof
framework**. Review the retained series for smaller noise effects too. Do not
trim samples or automatically retry failures. Any authorized repeat must be a
fresh complete comparison, with every earlier result retained.

The CPU checks use the authentic archived six-phase prefill proof and inject a
68 ms reference replay into a copy while preserving all 100 samples and
recomputing the raw summary. The stable proof remains a no-gain control; the
injected outlier is rejected. To reproduce after downloading `positive/` from
the archive below with the documented rclone transfer flags:

```bash
python3 ready/mla-prefill/test_timing.py \
  --measurement-dir "$MLA_STORAGE/positive-proof/proofs/trusted/trusted"
```

## Audited proof archive

[READY.json](READY.json) lists exact proof URIs and SHA-256s. The archive root is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261008/qualification-evidence/deepseek-v4-pro__unified_paged_attention_prefill/20c1a949-200712-r2/
```

`positive/` contains 107 files, `source-negatives/` contains 48, and `metadata/`
contains 13. All three packages and their archive receipts were verified by
readback. The final receipt is `metadata/FINAL-QUALIFICATION.json`, SHA-256
`67e77cd74781f61162750c8eca093987ae6e8f98c5303072faa230d52696ca6a`.

[PUBLISHED-COMPARISON.json](PUBLISHED-COMPARISON.json) records why the supplied
GitHub commit `373cf51f5601d88e49c04058e984cde4134eb883` cannot be used for this
entry: it lacks the task and differs in required framework/trusted inputs.
Use this dedicated handoff and its one-task configs when sharing the starter.
The broader collection in the repository is still undergoing qualification.

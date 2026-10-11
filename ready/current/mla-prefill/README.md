# Current qualification: DeepSeek MLA prefill

The fresh qualification covers the two original **M8192/H16** cases of
`headkernel/deepseek-v4-pro__unified_paged_attention_prefill`. All six trusted
phases and all 12 framework checks passed. Strict timing review accepted all
400 trusted samples and 200 framework samples without trimming. The source
was unchanged, so **no optimization gain is accepted**.

Framework checkpoint 67 finalized at **2026-10-11 12:47:05 UTC**, including
input parity, complete publication and verified container retirement. The
allocation was subsequently cancelled at **12:49:55 UTC**. The completed
qualification predates that cancellation. All 37 final checkpoint files were
rehashed, and the complete proof archive was downloaded and verified.

The qualified framework and trusted runtime commit is
`7bc14698e1044931a3c893b3dbabc9676b0e57e6`, with task tree
`b659474a77efd6d2e4b795011b669d06e634d595` and source SHA-256
`a71cbe21053ba78fc09d0ecc4e34bb03f0b2bd5d95da906b7fd6b8ec171b1883`.
This entry verifies **505 pinned inputs** from that exact checkout, including
the quality-admission module and its command-line gate.

The separate [compatibility approval](COMPATIBILITY-0df9540d.json) covers
`0df9540d1db9462cf0e2d108f7f653f34c2ae94f` for these two MLA cases. Its review
found identical task inputs and MLA runtime behavior, with 158 CPU regressions
passing. The GPU reports retain their original `7bc14698` identities. The
verifier continues to enforce the qualified snapshot's pins, and the setup
below uses that snapshot. The compatibility approval covers only its named
revision; later branch tips require their own identity or compatibility review.
The [historical entry](../../mla-prefill/README.md) retains its original evidence
and pins.

Only `pa_prefill_16mx1_16nx4_kernel` in `source/pa_sparse_prefill_opus.h` is
editable. Preserve both cases, seeds 42/43, 10 warmups, 100 measured replays,
fresh inputs, output initialization, every oracle check and the no-op and
wrong-output controls. Historical M1/M64 coverage, other tasks and model-level
readiness remain outside this entry.

From the repository containing this guide, prepare a new checkout and storage
directory on verified local NVMe:

```bash
MLA_GUIDE="$PWD/ready/current/mla-prefill"
MLA_STORAGE=/path/to/local-nvme/mla-current
MLA_COMMIT=7bc14698e1044931a3c893b3dbabc9676b0e57e6
mkdir -p "$MLA_STORAGE/scratch" "$MLA_STORAGE/tmp"
git worktree add --detach "$MLA_STORAGE/qualified" "$MLA_COMMIT"
python3 "$MLA_GUIDE/verify.py" --repo "$MLA_STORAGE/qualified"
export TMPDIR="$MLA_STORAGE/tmp"
export GOMAXPROCS=1
export AKA_SKIP_DEV_MEM=1
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
python3 "$MLA_STORAGE/qualified/tools/prepare_headkernel_run.py" \
  --repo "$MLA_STORAGE/qualified" --commit "$MLA_COMMIT" \
  --output "$MLA_STORAGE/prepared" --scratch-dir "$MLA_STORAGE/scratch" \
  --task headkernel/deepseek-v4-pro__unified_paged_attention_prefill \
  --use-manifest-oci-prefixes
```

The existing preparer fetches and verifies exactly **25 fixture objects /
1,413,573,052 bytes**, using the manifest prefix in [READY.json](READY.json).
No model weights are needed. Fixture copies retain `GOMAXPROCS=1` and
`--transfers 64000 --progress --buffer-size 0`. Staging produces a
`staged_not_evaluated` receipt; it does not create another qualification.

Use Linux amd64, MI355X/gfx950 and the pinned image. Configure the selected
agent normally. The existing lane controller must bind the assigned physical
GPU UUID and render device, keep caches on NVMe, enforce the allocation bounds,
and retire only owned containers. In the prepared checkout:

```bash
cd "$MLA_STORAGE/prepared/arena"
make docker-check-agents CONFIG=example_configs/ready_ds_mla_prefill_mi355x.yaml
make docker-run CONFIG=example_configs/ready_ds_mla_prefill_mi355x.yaml
```

A changed candidate needs a fresh complete trusted comparison after the
optimization worker stops. Use the immutable qualified checkout's
`src/tools/trusted_task_eval.py`, the candidate's separate task workspace,
an explicit assigned render device and NVMe scratch, the same fixture manifest,
and the original 3600-second container timeout. Preserve all six phases and
the existing quality decision. Then run the supplemental source, native-binding,
fixture and raw-timing check from this guide:

```bash
python3 "$MLA_GUIDE/check_timing.py" \
  --repo "$MLA_STORAGE/qualified" \
  --measurement-dir /path/to/fresh/trusted-results \
  --output /path/to/new/quality-review.json
```

A timing rejection remains rejected. No samples are removed, and no automatic
repeat selects a better run. The unchanged-source control has a raw ratio of
0.9855112941, which is a diagnostic and carries no accepted gain.

The immutable proof prefix is:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261011/qualification-evidence/deepseek-v4-pro__unified_paged_attention_prefill/7bc14698-207740/
```

| Object | SHA-256 |
| --- | --- |
| `mla-current-complete-207740.tar.gz` | `cc3159c484c69d8be39d19eef06b94c80e5953bcc40dbf87801d8dc8f172a329` |
| `PRESERVED-MANIFEST.json` | `911492278e6c11abc48a25fde77f7a21ffecb1d2101698bcda731dfb8d691587` |
| `ARCHIVE-VERIFIED.json` | `559b8ba425de7c195fdf592530413b3c6323a859887b06cc10b83f058877f183` |

All five published objects were downloaded and hash-verified. Every one of the
116 payload files and the archive's manifest member was rehashed. The bundle
contains complete phase reports, final framework artifacts, physical-GPU
preflight receipts, ownership retirement, and the later cancellation receipt;
it contains no fixture tensors, compiled binaries or bulk caches.

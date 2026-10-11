# Scoped starter: MiniMax FP4 GEMM, all 14 captured cases

This entry selects only
`headkernel/minimax-m3__gemm_afp4wfp4_kernel`. Optimize the body of
`_gemm_afp4wfp4_kernel` in
[`source/kernel.py`](../../tasks/headkernel/minimax-m3__gemm_afp4wfp4_kernel/source/kernel.py).
Imports, decorators, signatures, configuration lookup, native wrappers, references,
and the correctness/timing harness remain protected.

The task preserves all 14 native case identities from the completed MiniMax
64-request ISL8192/OSL1024, concurrency64, TP8 capture. These include M1/M64 decode
and M8189/M8192/M16365/M16384 prefill, packed weight shapes `[768,3072]` and
`[6144,192]`, the recorded scale strides and output aliases, and the four-way
split-K M1/N768 case. The manifest and full-workload provenance remain the source
of truth for case identities and occurrence counts. No profile-only or generated
case replaces a captured case.

The tested task/framework pin is
`0ed45cdd4cbff6cf305adffc50f937d549314884`; its task tree is
`795611e8f0c602084338e47bd5f3a601da8b8744`.
The [qualification receipt](READY.json) records six completed trusted phases and
a finalized 12/12 PASS framework review. Publication requires an independent
approval receipt matching this branch head and its archive manifest, including
explicit acceptance of the framework binding limitations described below.

Qualification is scoped to this task and its admitted manifest. Other MiniMax
heads, other models, unobserved source branches, and serving E2E performance are
not covered. Broad collection status and historical figures elsewhere in the
checkout are unchanged. Use the one-task configs below instead of a whole-suite
selection.

## Environment and immutable inputs

Use Linux amd64, a MI355X (`gfx950`), ROCm-compatible Docker access, Git, Python 3
with PyYAML, and rclone. The verified runtime is **SGLang 0.5.20**, identified by the
immutable image below and the `sglang_v0520` entry in the
[runtime catalog](../../tools/headkernel-runtime-targets.json). No full-model
checkpoint or Crusoe-specific scheduler is required to replay this isolated task.

Configure an `oci` remote with access to the task's declared fixture prefix. Keep
fixtures, private caches, preparation output, and results on local NVMe. The
standard Docker helper also uses host `/tmp` for its initial image cache; ensure
`/tmp` and Docker storage have suitable local backing and sufficient free space.
Use an explicit rclone configuration path when the host execution environment
does not retain your usual home/configuration directory.

From the root of this checkout:

```bash
python3 ready/minimax-fp4/verify.py
FP4_REPO="$PWD"
FP4_HANDOFF_COMMIT=$(git rev-parse HEAD)
FP4_QUALIFIED_COMMIT=0ed45cdd4cbff6cf305adffc50f937d549314884
FP4_STORAGE=/path/to/local-nvme/minimax-fp4
mkdir -p "$FP4_STORAGE/scratch" "$FP4_STORAGE/tmp"
export TMPDIR="$FP4_STORAGE/tmp"
export GOMAXPROCS=1
export RCLONE_CONFIG=/path/to/your/rclone.conf
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
docker pull "$AKA_DOCKER_IMAGE"
python3 tools/prepare_headkernel_run.py \
  --repo "$FP4_REPO" --commit "$FP4_HANDOFF_COMMIT" \
  --output "$FP4_STORAGE/prepared" --scratch-dir "$FP4_STORAGE/scratch" \
  --task headkernel/minimax-m3__gemm_afp4wfp4_kernel \
  --use-manifest-oci-prefixes
```

Use a fresh preparation directory. The existing preparation helper copies the
pinned checkout and materializes only the selected task's declared objects,
checking their sizes, hashes, codecs, and references before workspace copying.
Its staging receipt is not a new GPU qualification result.

The task declares **74 fixture objects, totaling 2,044,395,146 bytes**, at:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/minimax-original-head-v3-194969/
```

The pinned [fixture manifest](../../tasks/headkernel/minimax-m3__gemm_afp4wfp4_kernel/fixtures/EXTERNAL-MANIFEST.json)
contains every object key and SHA-256. Copies use `GOMAXPROCS=1` and
`rclone --transfers 64000 --progress --buffer-size 0`, with bounded file groups.

## Start optimization or framework validation

The [optimization config](../../example_configs/ready_minimax_fp4_mi355x.yaml)
selects this task and the existing Claude Code agent template. Configure that
agent normally, or change only the `agent` block to another supported agent.
Authentication/model choices do not alter the task qualification.

```bash
cd "$FP4_STORAGE/prepared/arena"
make docker-smoke
make docker-check-agents CONFIG=example_configs/ready_minimax_fp4_mi355x.yaml
make docker-run CONFIG=example_configs/ready_minimax_fp4_mi355x.yaml
```

The [validator config](../../example_configs/validate_ready_minimax_fp4_mi355x.yaml)
records the backend settings used for the archived full framework review. The
task retains all 14 cases, correctness seeds `[0,1,2]`, 10 warmups, 100 measured
graph replays per case, and the fixed independent CPU packed-FP4 oracle with its
0.02 mixed-RMS limit. Compile, correctness, and performance each retain their
1800-second task timeout.

The ownership correction at the qualified pin tracks every floating allocation
made by the private frozen wrapper. Eager writable allocations are poisoned
before native use. Capture records and retains the writable buffers; initialization
poisons them outside capture before each replay and outside device timing.
Read-only input storage is excluded. Both source legs use the same allocation
and reset path, and the original kernel math is unchanged.

Each correctness phase also runs no-op and wrong-output submitted sources for
every case in both eager and graph modes: **56 source controls**. Each requires
successful candidate compilation/native engagement, calibrated reference output,
and numerical rejection. This includes the earlier eager split-K no-op acceptance,
whose original evidence remains in the archive. Diagnostic controls collect no
performance samples and replace none of the positive cases.

## Retest a stopped candidate

Run the existing trusted evaluator from this trusted checkout after the
optimization worker stops. Select the physical GPU for your host; a render-device
number is not a portable physical-GPU identity.

```bash
cd "$FP4_REPO"
FP4_CANDIDATE=/path/to/stopped/task-workspace
FP4_RENDER_DEVICE=/dev/dri/renderD128
python3 src/tools/trusted_task_eval.py \
  --repo "$FP4_REPO" --commit "$FP4_QUALIFIED_COMMIT" \
  --task tasks/headkernel/minimax-m3__gemm_afp4wfp4_kernel \
  --candidate-workspace "$FP4_CANDIDATE" --render-device "$FP4_RENDER_DEVICE" \
  --output "$FP4_STORAGE/trusted-candidate" --scratch-dir "$FP4_STORAGE/scratch" \
  --fixture-oci-prefix oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261005/minimax-original-head-v3-194969/ \
  --fixture-max-files 74 --fixture-max-bytes 2044395146 --timeout 1800
```

This admits only the submitted source, reconstructs the frozen reference/harness,
binds the actual GPU, and performs six independent container phases. Reserve the
complete six-phase, fixture/cache preparation, and cleanup budget. A partial or
interrupted run is not a completed comparison and must not be spliced into one.

## Timing admission and evidence limits

After every complete trusted comparison, run the mandatory post-evaluation check:

```bash
python3 ready/minimax-fp4/check_timing.py \
  --measurement-dir "$FP4_STORAGE/trusted-candidate" \
  --output "$FP4_STORAGE/trusted-candidate-quality.json"
```

It validates all six reports, the 14 case identities, source/request/GPU and fixture
hashes, native compilation evidence, both sets of 56 source controls, and all raw
means before applying the pinned timing-quality policy. Every original sample
and report is retained. It writes a new receipt and exits nonzero for rejected
timing quality; it does not trim, retime, or automatically retry a comparison.

Same-source measurements are acceptance-path controls and never receive an
accepted gain. A changed candidate needs its own full correct comparison, passing
timing admission, and ordinary review before a speedup is accepted. The timing
gate detects catastrophic instability; it is not a statistical proof of gain.
No isolated kernel ratio is an E2E or serving-performance claim.

The completed same-source comparison contains **2,800 raw trusted samples**
(14 cases × 100 samples × two source legs), with 10 warmups per case. Its raw
arithmetic mean reference/candidate ratio is **0.991258107**.
The pinned timing-quality policy passes the retained samples; unchanged source
is ineligible for a gain claim. The framework report separately retains 1,400
raw samples and 56 numerically rejected source controls.

The archive contains 565 manifest-bound files. Every archived file
and all **74 fixture objects (2,044,395,146 bytes)** were read back and SHA-256
verified. [READY.json](READY.json) links the archive manifest, preservation receipt,
six-phase audit, all raw reports, timing admission, historical false acceptance,
interrupted comparison, and retirement evidence. [INPUT-PINS.json](INPUT-PINS.json)
binds the unchanged task and framework inputs plus this entry's post-evaluation
checks and configs.

The framework review completed on job 201262 before that allocation was cancelled.
Its finalized report, completion marker, checkpoint, source/fixture parity evidence,
and retirement receipt were preserved. Its image record was recovered from existing
bytes matching the original checkpoint SHA-256. **The raw GPU-admission receipt,
full transfer plan, and staged fixture-receipt bytes were not recovered.** Their
absence remains recorded in the binding disclosure. The separately recorded content
hash audit compares the exact 41 canonical task files and all 74 pinned fixture
objects with the framework's three request hashes; it does not recreate those
missing operational records.

The new six-phase trusted run on job 201533 has its own complete image, GPU, fixture,
request, container, and retirement evidence. No phase from an interrupted attempt
was spliced into it. Independent publication approval must explicitly accept the
disclosed framework limits and bind this exact ready branch head and archive
manifest. Its external receipt is identified by `publication_gate` in READY.json.

# MI355X runtime and compatibility

The MI355X default is the immutable SGLang 0.5.20 / ROCm 10 manifest
selected in [`docker_benchmark.sh`](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/scripts/docker_benchmark.sh).
The `gfx942` and `gfx1201` defaults are unchanged. Explicit image overrides and
task-specific runtime requirements still apply. GPU ASan is **unqualified** on
the tested hosts and remains unavailable when its required controls fail. The
scoring-runtime migration does not qualify this optional analysis tool. Earlier
image qualifications do not qualify a changed source/image combination.

All six optional tool runtimes use the same scoring base and distinct
`gfx950-rocm10` tags. The scoring-image verifier rejects mismatched images and
freezes accepted aliases to their local image IDs. See the maintained
[tool capability matrix](../how-to/use-evaluation-tools.md) for startup and
candidate-analysis boundaries. GPU ASan needs working GPU access to ordinary
host allocations; a successful build or an `HSA_XNACK=1` setting alone does not
establish that prerequisite. Required safe and known-bug probes remain mandatory.

Ordinary runs now propagate their selected image into materialization and session
identity. Start a fresh run after changing the image, including for workspaces
created before image identity was recorded. Custom mutable tags must be frozen
to a digest. Keep earlier artifacts and compare baseline and candidate within
the same runtime; do not treat an image change as a kernel optimization.

The new runtime contains Python 3.12, PyTorch 2.11, Triton 3.8 and FlyDSL 0.3.2.
ROCm SDK and HIP component versions are distinct. AITER is installed from source
without distribution metadata, so retain the image identity and materialized
source hashes. Image-acquired AITER trees can change even when task config and
candidate bytes do not. Tasks that require a vLLM installation still need their
explicitly documented runtime; the SGLang default does not provide it.

## Known issue: hipBLASLt bfloat16/float16 GEMM kernels

The image's hipBLASLt (`100401-8d1ae90e`, from ROCm/rocm-libraries `8d1ae90e`) has the race reported in
[rocm-libraries#14129](https://github.com/ROCm/rocm-libraries/issues/14129) and fixed by
[PR #14147](https://github.com/ROCm/rocm-libraries/pull/14147) (merge commit `46a10bc9cd28`), which postdates this build.
As of 2026-11-27, no ROCm (TheRock) release up to and including 10.2 contains the fix. Backport request:
[rocm-libraries#16919](https://github.com/ROCm/rocm-libraries/issues/16919).

- **Affected kernels:** bfloat16/float16 TN GEMMs with macro tile 256x160x64, the custom main-loop schedule (`_CMS_`
  in the kernel name) and DirectToLds loads (`DTLA1`/`DTLB1`). In the kernel's naming, Tensile's A is torch's `b` (the
  `[n, k]` weight) and Tensile's B is torch's `a` (the activations). The prologue's last prefetch of `a` can still be
  in flight when the main loop first reads it from LDS, so rows of the output tile use stale values.
- **Sizes the default heuristic sends to them (no TunableOp):** hipBLASLt's exact-size table for MI355X maps bfloat16
  (M, N, K) = (1280, 7680, 3840), (1919, 5120, 4096), (2056, 8192, 2048), (1792, 5120, 256) and (150000, 1024, 1024),
  and float16 (6144, 1536, 3072), to these kernels. At the first three, with two alternating input sets,
  `torch.matmul` returned wrong results in 13,818 of 13,824 calls in this image; the other three sizes were not run in
  this image. Other sizes use other kernels; PyTorch TunableOp can also select the affected kernels at other sizes.
- **Kernel tasks are not affected.** For the 384 schema-v3 kernel tasks, a GPU scan of every reference at every
  case found only float64 matrix products, which PyTorch sends to rocBLAS, not hipBLASLt. The QA calibration
  probe evaluates references in float32, which selects none of the affected kernels. No task case has an affected
  size. Candidates cannot call library GEMMs, baselines are vendored kernels, and AgentKernelArena does not enable
  TunableOp.
- **End-to-end serving tasks are not affected.** `qwen3_0_6b_sglang` and `qwen3_next_80b_sglang` run in their own
  locked ROCm 7.2.0 images (hipBLASLt `1.2.1.70200`), whose hipBLASLt library contains no kernel of the affected
  schedule. A Magpie catalog workload runs in the image its run config binds; a binding to an image with this
  hipBLASLt build could reach the affected kernels if a model GEMM has one of the sizes above.
- **Outside scoring:** a bfloat16/float16 `torch.matmul` or `F.linear` in this image (for example an agent's own test)
  can hit the affected kernels at the sizes above. Do not use a reduced-precision PyTorch GEMM as an oracle, and do
  not set `PYTORCH_TUNABLEOP_ENABLED` in this image.
- **Image update:** no qualifying release image exists yet. When one whose hipBLASLt contains PR #14147 is
  available, check the sizes above against a float64 reference before switching the `gfx950` default, then
  requalify as described in the last section of this guide. The
  [earlier runtime qualification record](runtime-upgrade-qualification.md) shows the procedure used for a previous
  image change.

## SDK and profiling

The runner selects SDK core libraries before starting workloads and routes
`rocprofv3` through that same tree. Loading both the SDK developer and core copies
of COMGR can abort a Python workload with duplicate LLVM registration, including
when vLLM loads a PyTorch library before importing PyTorch. `rocprof-compute` is
optional and absent from the base image; profiler availability is separate from
successful trace/counter collection and from ordinary graph/event timing.
GEAK's SDK cache is separated by Python ABI to prevent reuse of Python 3.10
extension modules in Python 3.12.

## Tasks and analysis tools

FlyDSL tasks use task-local compatibility adapters for removed buffer/vector
APIs and retain their existing cases, numerical gates and timing contracts.
The gfx950 MLA decode path uses four waves per block to retain its two-stage
pipeline. The two-wave pipelined configuration produced incorrect results on
the new runtime; the four-wave configuration preserves asynchronous loads and
double-buffered shared memory.

The FP8 fused-MoE task exceeded its historical 1% numerical gate on both runtimes.
It now uses the maintainer-approved 1.3% gate and has been requalified against
that revised contract. This tolerance change is not a compiler accuracy fix.

Triton AOT extraction resolves named signature keys against function argument
names and rejects unrepresented global scratch. FlyDSL extraction accepts the
asynchronous launch form while rejecting additional launches. GPU ASan candidate
invocations use the same attested HIP/HSA and library paths as startup probes;
those paths cannot be supplied or overridden by task configuration.

Keep current validation reports, source/config fingerprints, scheduler outcomes
and old/new comparison data outside the repository. Report completed checks and
remaining limitations in the PR. Each changed task needs a fresh full
framework-finalized QA auditor `qa_report.yaml` with `status: PASS`; initial actions, CPU tests and interrupted
runs do not replace that requirement.

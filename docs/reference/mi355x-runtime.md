# MI355X runtime and compatibility

The proposed MI355X default is the immutable SGLang 0.5.20 / ROCm 10 manifest
selected in [`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh).
The `gfx942` and `gfx1201` defaults are unchanged. Explicit image overrides and
task-specific runtime requirements still apply. The migration remains
**unqualified for promotion** while GPU ASan cannot pass its required controls.
Earlier image qualifications do not qualify this source/image combination.

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
The gfx950 MLA decode path uses a single-stage schedule because its two-stage
pipeline produced incorrect results with the new compiler. This fallback can
increase latency. Numerical failures in the FP8 fused-MoE task have also been
reproduced on the previous runtime and remain unresolved; they must not be
counted as passing qualification or attributed to this upgrade without evidence.

Triton AOT extraction resolves named signature keys against function argument
names and rejects unrepresented global scratch. FlyDSL extraction accepts the
asynchronous launch form while rejecting additional launches. GPU ASan candidate
invocations use the same attested HIP/HSA and library paths as startup probes;
those paths cannot be supplied or overridden by task configuration.

Keep current validation reports, source/config fingerprints, scheduler outcomes
and old/new comparison data outside the repository. Report completed checks and
remaining limitations in the PR. Each changed task needs a fresh full
framework-finalized validator PASS; initial actions, CPU tests and interrupted
runs do not replace that requirement.

# GLM fused-MoE portable semantic task

**Workload scoring is disabled.** The original nine archives retain M19/M1/M8192
initialization/post-call payloads only. M64 and M16384 are counter observations,
not retained routing records. The source capture used CAPTURE_MAX=2, REPEATS=0,
PROFILE=0, disabled CUDA graphs, and saved tensors after the in-place operation.
The archives predate the outer 64-prompt warmup by about 1004.7 seconds.

The [fidelity disposition](ut/workload_fidelity.json) requires a new authentic
served pre-call capture before workload scoring can be enabled. The performance
entrypoint exits nonzero before runtime initialization and never emits a proxy or
partial aggregate. The prior score IDs `moe_m64_decode`, `moe_m8192_prefill` and
`moe_m1_decode` are removed from the enabled score set; enabled count is **zero**.

The portable semantic checks remain useful: all three prior oracle-shape cases
and four random/robustness cases are retained, with the original `tol=0.02`.
M64 pool resampling is explicitly correctness-only. Three additional in-place
semantic probes use the retained routing arrays verbatim and permit only the
recorded hidden-state write/alias behavior. They are not served-workload replay.

The committed [compact contract](ut/generated_cases.json) is 867,430 bytes. It
retains exact routing, shapes, strides, offsets and scalar metadata; numerical
activations and expert values are generated from recorded seeds with finite FP8
values and positive block scales. No external tensor archive is required. The
original archive hashes remain provenance under `ut/meta.json:archival_capture`.

The editable target remains `fused_moe_kernel` in
[source/fused_moe_triton_kernels.py](source/fused_moe_triton_kernels.py), pinned to
SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8` and SHA256
`9c3342d3147e7d60a78a2c934111f0fc1becbb8df1d2d32aafc82e6c8a0b2e70`.
The full dispatcher and source boundary remain protected; source/ABI bytes are
unchanged. Model-free configuration and one-rank TP initialization remain required.

The configured entrypoint is `scripts/generated_task_runner.py`. Reference
outputs remain in the parent process and are never sent to candidate workers.
The shared worker protocol introduced by `a7bf289b` loads the declared aliases
before candidate access and executes the same attested module objects. The branch includes the shared two-phase preflight hook: native target resolution
follows helper attestation and overlay installation. No private worker/guard
remains. Native source/ABI checks stay enabled.

No GPU qualification is claimed. Task-local metadata and provenance record source
recovery, the retained structural inputs, and integration requirements.

Use the public runtime pinned in [config.yaml](config.yaml). The registry manifest,
OCI config digest and historical `headkernel.capture_runtime` metadata remain
separate. The integrated shared guard retains the newer DeepSeek checks.

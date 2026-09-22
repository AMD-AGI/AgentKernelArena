# pa_decode_swa_kernel: unified task contract

The task starts with an implemented FlyDSL candidate. Arena freezes that initial
implementation in a separate workspace for baseline evaluation. Candidate actions
always use this workspace's declared source; they never search other workspaces.

Optimize the FlyDSL partitioned paged-attention decode kernel for sliding-window
attention on AMD CDNA GPUs.
You MUST keep the kernel in FlyDSL — do NOT rewrite it in HIP, CUDA, or Triton.
You MUST NOT add FastLauncher, ctypes dispatch bypass, _call_state_cache extraction,
or any wrapper that bypasses JitFunction.__call__. Only optimize the GPU kernel
computation itself.

Only `candidate.editable` files in `config.yaml` may be changed. Preserve the declared
entrypoints and their call signatures as exercised by `test_kernel_harness.py`.
The final operator computation must execute FlyDSL GPU kernels. Python/PyTorch may
allocate, reshape, pack inputs and launch kernels; calling PyTorch, Triton, AITER,
reference/model/harness code to compute the submitted operator is not allowed.
Bundled `kernels/` modules are protected implementation utilities, not reference
solutions. Do not introduce dynamic imports, subprocess kernels or native dispatch
bypasses. Passing numerical tests alone does not waive the FlyDSL requirement.

`cases.json` declares 5 correctness cases and 5 performance cases
before candidate execution. Original correctness order/seeds, all numerical gates,
all output checks, warmups and benchmark sample counts are retained. Performance
cases absent from the old correctness list receive additional correctness checks.
`test_kernel_harness.py` remains the source of truth for numerical comparisons.
The task-owned runner returns `arena-eval-v1` evidence on stdout; Arena alone computes
scores and writes final result files. No agent-specific driver is required.

From a materialized task workspace, the seven public commands are:

```sh
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Compilation syntax-checks every declared candidate source. Correctness executes
real case-specific GPU compilation/launches and the original numerical comparisons.
The canonical `_aka_benchmark.py` helper must be materialized by Arena before GPU
actions. GPU execution requires the architecture declared in `config.yaml` and the
container's FlyDSL/ROCm/runtime dependencies. CPU protocol tests are not GPU validation.
Any older validation reports in this directory predate this migration and do not
qualify the v2 runner. The parent integration schedules new GPU validation.

Upstream source: {"commit": "28a18d328b4882c999864b2df2f8f9fe3fcc8b47", "date": "2026-06-01", "path": "kernels/pa_decode_swa.py", "repo": "https://github.com/ROCm/FlyDSL"}.


The task retains its original FNUZ FP8 caches and independent FNUZ reference.
On gfx950, the GPU kernel converts loaded K/V register bytes to the OCP FP8
encoding consumed by that architecture's MFMA instructions, using normal
round-to-nearest-even conversion. The gfx942 path uses its original bytes. No
harness input encoding, stored cache contents, dequantization scales, cases or
tolerance is changed. The conversion executes inside the original timed launch.
See [AMD's FP8 format support](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html).

The bundled `flydsl_compat/` package adapts removed buffer/vector APIs; its source
pin and license are included. Performance checks the actual measured decode
output, changes query values in place, poisons output and replays the same graph
against the unchanged independent reference. All original shapes, tolerance,
warmups, samples, allocations and timed stage/reduction calls are preserved.

Every declared correctness and performance case passed on MI355X gfx950 with
measured-output and same-graph replay checks. The original gfx942 path remains
supported and was not revalidated by this port. Support is scoped to `cases.json`.

Candidate dependency enforcement runs before candidate import for compile,
correctness and performance. AITER package/operator imports are forbidden,
including `aiter.ops.flydsl` implementations; a FlyDSL runtime call from an
imported operator is not candidate-owned arithmetic. Import aliases and
`from ... import ...` do not change this rule. External backend/native dispatch
(`ctypes`, subprocesses, or `torch.ops`) and dynamic implementation loading are
also forbidden. Ordinary Python utilities, PyTorch allocation/layout operations,
and the task's bundled `kernels/` helpers remain available under the existing
numerical and timing contract. Baseline checks retain their declared initial
backend; the final candidate must use FlyDSL.

# preshuffle_gemm_v2_kernel: unified task contract

The task starts with an implemented FlyDSL candidate. Arena freezes that initial
implementation in a separate workspace for baseline evaluation. Candidate actions
always use this workspace's declared source; they never search other workspaces.

Optimize the FlyDSL preshuffle GEMM v2 (layout API, fp8/fp16/bf16) for AMD CDNA GPUs.
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

`cases.json` declares 4 correctness cases and 4 performance cases
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

Upstream source: {"commit": "28a18d328b4882c999864b2df2f8f9fe3fcc8b47", "date": "2026-06-01", "path": "kernels/preshuffle_gemm_v2.py", "repo": "https://github.com/ROCm/FlyDSL"}.

Scored outputs must satisfy the original numerical rule after the actual timed
invocation. Outside timing the harness also changes the activation tensor in
place, poisons output storage, and checks replay against the protected reference.
Logical and preshuffled input buffers are protected from candidate mutation.
Output shape/dtype/device and complete case coverage remain required. Unsupported
replay collection or any failed case is a failure, never a successful skip.

The initial layout implementation uses the typed `Vector.from_elements` API.
Its preload policy lives in protected `kernels/preload.py`, extracted unchanged
from the bundled legacy preshuffle source. This avoids importing an unrelated
legacy kernel solely to read its constant table. The algorithm, tile lookup,
scaling order, supported cases and numerical gate are unchanged. Import checks
alone do not establish GPU compilation or numerical correctness.

# softmax_kernel: unified task contract

The task starts with an implemented FlyDSL candidate. Arena freezes that initial
implementation in a separate workspace for baseline evaluation. Candidate actions
always use this workspace's declared source; they never search other workspaces.

Optimize the FlyDSL Softmax kernel for AMD MI300X and MI355X GPUs.
The kernel computes numerically stable softmax using exp2(x * log2e)
for fast exponentiation and float32 accumulation.
You MUST keep the kernel in FlyDSL — do NOT rewrite it in HIP, CUDA, or Triton.

Only `candidate.editable` files in `config.yaml` may be changed. Preserve the declared
entrypoints and their call signatures as exercised by `test_kernel_harness.py`.
The final operator computation must execute FlyDSL GPU kernels. Python/PyTorch may
allocate, reshape, pack inputs and launch kernels; calling PyTorch, Triton, AITER,
reference/model/harness code to compute the submitted operator is not allowed.
Bundled `kernels/` modules are protected implementation utilities, not reference
solutions. Do not introduce dynamic imports, subprocess kernels or native dispatch
bypasses. Passing numerical tests alone does not waive the FlyDSL requirement.

`cases.json` declares 10 correctness cases and 10 performance cases
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

Upstream source: {"commit": "28a18d328b4882c999864b2df2f8f9fe3fcc8b47", "date": "2026-06-01", "path": "kernels/softmax_kernel.py", "repo": "https://github.com/ROCm/FlyDSL"}.

## Architecture compatibility

The gfx950 port has executed every declared correctness and performance case on
MI355X with unchanged operator semantics, inputs, tolerances, and timing boundaries.
The original gfx942 declaration and legacy import path are preserved; gfx942 has
not been revalidated by this port. Support is scoped to the cases in `cases.json`.

FlyDSL releases that moved vector helpers into `flydsl.expr.typing` use that API;
older releases retain `flydsl.expr.vector`. The bundled common helper imports its
unused legacy pointer-conversion dependency only when that conversion is requested.
These compatibility changes do not replace any GPU operator with a host/reference
implementation. A fresh framework-finalized task-validator report is required for
each supported runtime before publishing qualification results.

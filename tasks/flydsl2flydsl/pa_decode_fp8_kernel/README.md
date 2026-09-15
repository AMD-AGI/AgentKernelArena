# pa_decode_fp8_kernel: unified task contract

The task starts with an implemented FlyDSL candidate. Arena freezes that initial
implementation in a separate workspace for baseline evaluation. Candidate actions
always use this workspace's declared source; they never search other workspaces.

Optimize the FlyDSL Paged Attention Decode FP8 kernel for AMD MI300X GPU.
The kernel implements paged KV-cache attention decode with FP8 quantized
keys/values, MFMA-based dot products, online softmax, multi-partition
reduce, and supports both one-shot and split-reduce modes.
You MUST keep the kernel in FlyDSL — do NOT rewrite it in HIP, CUDA, or Triton.

Only `candidate.editable` files in `config.yaml` may be changed. Preserve the declared
entrypoints and their call signatures as exercised by `test_kernel_harness.py`.
The final operator computation must execute FlyDSL GPU kernels. Python/PyTorch may
allocate, reshape, pack inputs and launch kernels; calling PyTorch, Triton, AITER,
reference/model/harness code to compute the submitted operator is not allowed.
The exact existing PA metadata/reduce support exception is documented below.
Bundled `kernels/` modules are protected implementation utilities, not reference
solutions. Do not introduce dynamic imports, subprocess kernels or native dispatch
bypasses. Passing numerical tests alone does not waive the FlyDSL requirement.

`cases.json` declares 8 correctness cases and 8 performance cases
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

Upstream source: {"commit": "28a18d328b4882c999864b2df2f8f9fe3fcc8b47", "date": "2026-06-01", "path": "kernels/pa_decode_fp8.py", "repo": "https://github.com/ROCm/FlyDSL"}.

The actual measured output and a replay of the measured launch must satisfy the
same finite-output, BF16, maximum-absolute-error `5e-3` rule as correctness.
Outside timing, the harness negates the query, recomputes the protected attention
reference, poisons the measured output, and replays that same launch. KV caches,
scales, indices and lengths are read-only; implementation-owned metadata and
scratch retain their existing contract. Input cloning, checks, reference work
and restoration are excluded from device timing for both roles. Warmups, sample
counts, cases and numerical thresholds remain unchanged. The original secondary
launch timing is diagnostic; Arena scores against the separate frozen baseline.

Per-tensor quantization scales retain their original expanded, zero-stride views.
Replay restoration writes each shared storage location once; it does not replace
those views with contiguous tensors or change the captured kernel arguments.
Read-only checks still compare every logical element before and after replay.
All eight original cases, seeds, absolute-error gate and graph timing remain.

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

This task retains three existing AITER helpers, declared precisely in protected
`scripts/dependency_policy.json`: `get_pa_metadata_info_v1` and
`get_pa_metadata_v1` inside `get_pa_metadata`, and `pa_reduce_v1` inside
`pa_decode_ps_launch`. These prepare scheduling metadata and combine the
candidate's FlyDSL partial outputs; their original arguments, semantics and
allocation/timing boundaries remain in the kernel and harness. Use exact named
imports from `aiter.ops.attention`; import aliases are allowed. The imported
helpers may only be called directly in those functions in `kernel.py`, not
exported, introspected, or used as access to another AITER operator. Other AITER
operators and namespace imports remain forbidden. This preserves the existing
metadata/reduce glue and does not permit delegating attention computation.

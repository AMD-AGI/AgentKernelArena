# gemm_a8w8: Triton to FlyDSL task contract

The initial implementation is real **Triton**, not an empty FlyDSL starter.
Arena freezes it in a separate baseline workspace. The required final backend
is **FlyDSL**. Finishing with the original Triton implementation is not accepted,
even when it passes the numerical tests. No agent-specific driver is required.

Edit only `candidate.editable` paths in `config.yaml`. Keep the declared callable
interfaces and all outputs/state changes exercised by the protected harness.
Names containing `triton` are historical public API names; preserve those names
while replacing their implementation with FlyDSL. Private GPU function names can
change unless they are explicitly declared or called by the protected harness.

The final candidate-owned arithmetic must execute FlyDSL GPU kernels. Python and
PyTorch may prepare layouts, allocate storage and launch kernels. Candidate code
must not use PyTorch/AITER/Triton/reference/model code as a replacement operator,
or import task runners or protected reference functions. Do not introduce dynamic
module loading, external native kernels, subprocess dispatch or launch bypasses.
The protected harness's existing glue operations and allocation/reset boundaries
remain identical for baseline and candidate.

`cases.json` contains the 18 original correctness identities, four additional
correctness-only optional-argument cases, and the nine original performance
identities. The new cases exercise FP16 output, vector row/column scales,
preallocated output (including a strided view), and a valid non-split-K config
override. Performance variants retain their original correctness coverage. Input generators,
explicit seeds, numerical gates, output-contract checks, warmups, sample counts,
state reset and graph/event benchmark calls remain in `test_kernel_harness.py`.
Where the original suite allowed an environment dtype override, the manifest now
pins its original default; selecting a different suite requires updating both the
protected manifest and case definition. The small independent known answers in
`scripts/reference_controls.py` supplement the complete original GPU suite.

The source and module docstrings retain upstream operator semantics/provenance.
The task's public actions, from a materialized workspace, are:

```sh
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

The compile action syntax-checks actual role sources. Correctness performs real
GPU compilation/launches and the original comparisons. Every action emits one
`ARENA_EVAL_RESULT=` + `arena-eval-v1` JSON envelope. Failed or incomplete timings
cannot be scoreable. Arena owns score aggregation and final result files.

The runtime image provides ROCm, Triton (initial baseline) and FlyDSL (candidate),
plus any operator dependencies stated by the source. Arena must materialize the
canonical `_aka_benchmark.py` helper before GPU execution. CPU controls/protocol
checks do not qualify these GPU kernels. Existing legacy reports are historical;
the parent integration schedules new GPU validation.


All 18 original correctness and 9 performance cases remain: FP8 per-row/per-column scaling with both bias states. The original cases return [M,N] BF16 on the input device; API controls also check FP16 and supplied output storage. The original allclose gate remains atol=0.02, rtol=0.01, with finite outputs. Inputs, weights, scales and optional bias are read-only.

Every one of the 100 measured outputs must meet the same FP32 reference and
numerical rule. The ten external warmups retain the original seeded inputs. Every prepared
invocation, including capture/calibration and all 100 scored samples, independently
redraws and quantizes both X and W, preserving their shapes, FP8 format and
per-row/per-column scale semantics. Actual preparation indices, operand seeds
and full input-byte hashes are recorded for the scored draws. Preparation, oracle computation,
output checking and input-immutability checks stay outside device timing for
both baseline and candidate. Output storage is checked before a later invocation
can reuse it.

Timing remains graph-first with 10 external warmups and 100 reported samples.
The input-preparation callback makes each captured replay contain one logical
GEMM invocation (`benchmark_effective_repeats: 1`), replacing the previous
adaptive repetition of unchanged operands within a replay. This is a timing
protocol change: historical latencies are not directly comparable. Any graph
capture fallback must preserve the same per-sample input and output checks.

After timing, the exact measured invocation is replayed on three unseen draws:
X only, W only, and both operands changed, with their corresponding scales.
Outputs are poisoned before each replay; all operand values are restored after
validation. These checks reject a cached accumulator that only reapplies scales.
The original nine performance shapes, input generator, default output dtype,
tolerances and frozen Triton implementation are unchanged. Final FlyDSL calls retain their separate
execution audit and cannot substitute protected references or another backend.
The changed harness requires fresh full GPU validation; CPU regression checks
alone do not qualify it.

The four API controls use the same independent FP32 scaled-dot-product oracle,
finite-value checks and atol=0.02/rtol=0.01 rule. They check caller input/config
immutability, the identity of supplied output storage, and untouched padding in
strided output. They do not add timed workloads or change the original nine
performance shapes, warmups, samples, or aggregation.

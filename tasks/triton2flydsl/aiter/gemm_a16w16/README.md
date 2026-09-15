# gemm_a16w16: Triton to FlyDSL task contract

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

The evaluated workload is `gemm_a16w16(x, w)` with contiguous BF16 inputs,
FP32 accumulation and a newly allocated BF16 output: `Y = X @ W.T`, without
bias, using the source's default configuration and non-split-K path. All eight
original matrix shapes in `cases.json` have both correctness and performance
coverage. The numerical gate is finite output and element-wise
`torch.allclose(..., atol=1e-1, rtol=1e-2)`; measured outputs and perturbed
replays use the same gate.

The upstream source also documents FP16, bias, a caller-provided output `y`,
and explicit configuration arguments. Those describe the source's broader API;
this task does not claim to validate or score those additional modes. Preserve
the declared entrypoint and the calling convention exercised by this workload.
Adding other modes requires corresponding protected manifest, reference and
benchmark cases, rather than inferring coverage from the source signature.

Input generators,
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

The scored invocation must also pass its task-specific numerical comparison. The
benchmark exposes the last measured output through the canonical `TimedRun`,
then checks that output before changing inputs. Outside all timed regions it
perturbs inputs in place, poisons the output, replays the measured unit, and
compares against the original numerical policy again. Read-only inputs must
remain unchanged by either execution. Output shape, dtype and device are part
of the contract. Unsupported replay collection fails; it is never a PASS/SKIP.

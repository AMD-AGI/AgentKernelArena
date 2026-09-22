# fp8_mqa_logits: Triton to FlyDSL task contract

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

`cases.json` contains 5 independent correctness identities and 5
performance identities, including every original dtype, bias, activation, routing
and shape variant. Performance variants have correctness coverage. Input generators,
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


The effective workload covers all five original full, causal and sliding-band
windows, including ragged KV lengths and both MFMA size branches. Input generation
keeps seed 20260617 and the original FP8 magnitudes, positive scales and weights.
Return FP32 `[query_length, KV_length]` logits on the input device. In-window
values must be finite; the exact out-of-window mask must be negative infinity.
The original normalized maximum error <= 0.05 AND cosine difference <= 0.01
are required. The printed 0.05 allclose result remains a diagnostic, not an
additional acceptance gate. `clean_logits=True` is the declared correctness and
benchmark mode (the function default). Export `e4m3_dtype` for input preparation.

All six input tensors are read-only. Final operator calls are audited separately
from reference/baseline and timing; the operator computation must use FlyDSL.
The original 10 external warmups and 100 graph samples remain. Actual measured
logits and poisoned-output replay are compared with the same numerical and mask
rules after changing the two positive scale/weight inputs. Controls and restoration
run outside timing. The task runner returns all five individual timing records;
the legacy geomean field is only a standalone diagnostic. No scores are exported
by the task and no missing/failed case is accepted.

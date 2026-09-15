# dsv4_fp4_indexer: Triton to FlyDSL task contract

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

`cases.json` contains 6 independent correctness identities and 6
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


The six original N/dtype cases and seed42+i remain. Quantization must return
INT8[N,64] packed codes and INT32[N] packed scales on the input device, both
bit-exact to the original reference. Input x is read-only. The second required
store entry is still checked on every original case: original random locations,
pages and bit-exact full cache (including unchanged zero padding). It must not
modify x or locations. The source quantization thresholds/rounding and reference
bit expressions are unchanged.
The original performance scope is quantization only; cache store remains part
of correctness, and is not silently added to timed work. Check both actual
measured outputs, then negate x and multiply by4 outside timing so both signed
codes and scale exponents change; poison both output buffers and replay under
bit-exact comparison. Restore x before continuing. A cached code or stale scale
cannot qualify merely because the other output changes.
The original10external warmups and100samples remain, with the same graph-first
collector and allocation boundary for both roles. Final candidate FlyDSL calls
are audited individually during correctness, outside timing and oracle calls.
Both declared interfaces (where present) must run FlyDSL operator computation;
host allocation/layout preparation is allowed, substitute PyTorch/AITER/Triton
operator execution and protected reference imports are rejected.

Timing metadata separates the actual ten external task warmups from the
collector's zero additional warmups: benchmark_external_warmup=10,
benchmark_warmup=0 with scope collector_only, and benchmark_total_warmup=10.
The existing first successful setup/JIT invocation is separate from those
warmup loops. These fields describe the unchanged execution; no warmup or
sample was added, removed, moved or reclassified as scored work.

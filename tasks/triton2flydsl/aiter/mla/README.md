# mla: Triton to FlyDSL task contract

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


All six original BF16 paged-latent attention cases remain, including128query
heads/lora512 and the four-query non-ALL_DECODE path. The result must write and
alias the supplied [total_queries,query_heads,lora_rank] BF16 output buffer.
Q, KV pages, block tables, cumulative query offsets and sequence lengths are
read-only. The original normalized maximum error<=0.01 remains the sole numerical
gate; allclose at0.01 remains diagnostic. No reference or baseline algorithm changed.

The actual measured output and the same captured graph replay must both pass.
Untimed replay negates Q and KV together: QK logits stay the same, latent values
and the attention output negate. Page addresses and lengths stay unchanged.
Output poisoning, independent reference work and restoration are outside timing.
All original seeds42+i, ten external warmups and100graph samples remain.

Runtime qualification must bind the exact image. The unchanged source's
multi-stage pipeline has shown non-finite output for the128-head/lora512case in
a newer Triton runtime; the original pinned runtime passed all six diagnostic
cases. Diagnostics alone do not qualify the task. Use a full validator report
for the selected runtime; do not waive that case, alter its tolerance, or select
a different baseline implementation after a session has frozen it.

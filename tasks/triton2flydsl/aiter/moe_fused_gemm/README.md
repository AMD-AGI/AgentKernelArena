# moe_fused_gemm: Triton to FlyDSL task contract

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

`cases.json` contains 10 independent correctness identities and 5
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

The prepared MoE host launch ABI accepts `torch.dtype` for `compute_type`. The
initial Triton host wrapper converts it to its internal Triton dtype. Final
FlyDSL code receives the same neutral dtype; it need not expose `mod.tl`. Keep
routing preparation outside timing and preserve all fixed intermediate buffers.


All ten original correctness cases (both routing-weight states for five shapes)
and five performance cases remain. The output is BF16[M,top_k,N] on the input
device. The normalized maximum-error gate remains0.01; elementwise allclose0.01
is diagnostic only. Activations, weights, route IDs and route weights are read-only.
Both declared entrypoints, moe_align_block_size and fused_moe, must use FlyDSL
for their operator computation in a final candidate. Sorting and padding remain
prepared once outside timing; output.zero_ remains the original prepare_fn,
with its exact benchmark boundary preserved. Replay negates activations while
keeping routing and prepared sort metadata valid. The reference retains all
per-slot expert selection and optional route-weight computation.

Both the actual measured output and same captured graph replay must pass the
original numerical rule. Oracle work, perturbation, poisoning and restoration
are outside timing. Originalseeds,tenwarmups and100samples are retained.
Capture failure cannot bypass validation. The original frozen Triton source
is unchanged; final candidate execution is audited outside measurement windows.

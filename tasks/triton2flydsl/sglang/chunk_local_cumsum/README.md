# chunk_local_cumsum: Triton to FlyDSL task contract

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

`cases.json` contains 19 independent correctness identities and 10
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


All ten original FP32 scalar/vector, prefix/suffix and scaled variants remain, including the partial final chunk. These ten scored workloads use the regular layout (`[B,T,H]` or `[B,T,H,S]`, `head_first=False`, no `cu_seqlens`), chunk size 64 and atol=rtol=0.001. Output has the input shape and FP32 dtype. Inputs are read-only. The `head_first` and variable-length (`cu_seqlens`/`chunk_indices`) branches are also part of the evaluated contract through nine correctness-only controls; see [Head-first and variable-length controls](#head-first-and-variable-length-controls).

The actual measured output and the same captured graph replay on changed inputs
must satisfy that same numerical rule. Negative log-gates stay negative during
Cumsum replay. Checking, perturbing, poisoning and restoring inputs/outputs occur
outside timing. Both roles retain seed 42+case index, 10 external warmups and
100 graph samples. A graph-capture failure cannot bypass measured-output checks.
The frozen original Triton source is unchanged; final candidate operator calls
are audited separately for FlyDSL computation and cannot call protected references
or AITER/Triton/PyTorch operator implementations.

## Head-first and variable-length controls

The public operator is `chunk_local_cumsum(g, chunk_size, reverse=False,
scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float,
chunk_indices=None)`. These evaluated semantics apply to both roles:

- `head_first=True` takes `[B,H,T]` or `[B,H,T,S]` input and returns that layout;
  each chunk still runs along the time axis.
- `cu_seqlens` holds N+1 int32 or int64 offsets for `B=1`. Chunks restart at each
  sequence start, so a sequence can end with a partial chunk, including a
  one-token sequence. No sum crosses a sequence boundary.
- `chunk_indices`, when supplied, has one `(sequence, local chunk)` row per chunk
  in order and the dtype of `cu_seqlens`. When it is omitted, the operator derives
  the same chunks from `cu_seqlens`.
- `reverse` and `scale` behave as in the scored cases.

Nine untimed identities in `scripts/semantic_controls.py` call the actual
operator. They become correctness records 11-19:

| Control | Layout | Shape | Sequence lengths | `cu_seqlens` / `chunk_indices` | reverse, scale |
| --- | --- | --- | --- | --- | --- |
| `control_head_first_scalar` | head-first 3D | B2 H3 T200 | - | - | no, none |
| `control_head_first_scalar_reverse_scale` | head-first 3D | B1 H5 T129 | - | - | yes, 0.5 |
| `control_head_first_vector` | head-first 4D | B2 H2 T100 S24 | - | - | yes, none |
| `control_varlen_scalar` | varlen 3D | B1 T301 H4 | 1, 64, 100, 129, 7 | int32 / omitted | no, none |
| `control_varlen_scalar_chunk_indices` | varlen 3D | B1 T269 H3 | 65, 1, 200, 3 | int32 / supplied | yes, 0.5 |
| `control_varlen_vector` | varlen 4D | B1 T232 H2 S40 | 70, 1, 128, 33 | int64 / omitted | no, 0.5 |
| `control_varlen_vector_chunk_indices` | varlen 4D | B1 T193 H3 S16 | 1, 63, 64, 65 | int64 / supplied | yes, none |
| `control_head_first_varlen_scalar` | head-first varlen 3D | B1 H4 T150 | 150 | int32 / omitted | no, none |
| `control_head_first_varlen_vector` | head-first varlen 4D | B1 H2 T97 S8 | 97 | int64 / supplied | yes, 0.5 |

Inputs are seeded FP32 negative log-gates. Controls use chunk size 64 and the
default FP32 `output_dtype`, as the scored cases do. Before each call, the
expected output is computed on CPU in FP64 with plain PyTorch from pristine input
clones. Each identity reuses one input allocation for a base, a changed and a
second base draw. After every call, `g`, `cu_seqlens` and `chunk_indices` must be
unchanged. The output must match the input shape, FP32 dtype and device, contain
only finite values and pass the existing atol=rtol=0.001 comparator. The final
candidate's control calls pass the same per-call FlyDSL launch and PyTorch
preparation audit as its scored correctness calls. None of the 27 calls is timed
or scored. The ten scored cases, seeds, gates, warmups, samples and timing
method are unchanged.

`head_first=True` with `cu_seqlens` is evaluated for one sequence only. The
retained kernels address sequence n as its own `[H,T_n]` block at
`bos*H + h*T_n`, which matches the logical `[1,H,T]` layout only when N=1.
Upstream FLA v0.2.0 `chunk_gated_delta_rule` rejects variable-length
head-first inputs, and SGLang's `chunk_gated_delta_rule` rejects `head_first`.
Multi-sequence head-first varlen input is therefore outside this contract. The initial source
docstring says the harness exercises only the regular path; that note predates
these controls.

On CPU, `validate-task` checks the control oracle against hand-computed answers.
It also checks every control draw against the protected FP32 harness oracle,
applied per sequence after a head-first transpose. These CPU checks do not
qualify any GPU kernel.

## Candidate execution policy during performance

The candidate-only PyTorch preparation policy remains active on every actual
operator call made by canonical benchmark warmup, graph capture and eager event
timing. Tensor arithmetic, reductions and library operators must execute as
FlyDSL kernels even when candidate behavior changes after earlier calls. A
probe of the actual performance closure observes FlyDSL invocation outside
clocks and capture; Python profiling is disabled inside canonical measurement.
Allocation and view preparation retain the existing allowlist. Baseline and
reference operators keep their original policy and all original device timing
parameters. One additional untimed audit probe of each performance closure runs
before canonical benchmarking; stateful closures use the existing prepare
callback before and after that probe. This probe is separate from the retained
benchmark warmup/sample counts. A forbidden operation remains a failure even if
a capture fallback or candidate code catches its exception. These checks are
implementation-contract evidence, not a Python
security sandbox.

Candidate sources and their normal local-import helpers must not access the
evaluator's module registry, stack frames or profiling controls through `sys`,
including aliases and dynamic attribute lookup. Ordinary `sys` metadata remains
available, and installed FlyDSL library internals are not scanned as candidate
code. Protected reference imports remain prohibited.

## Protected evaluator state

Candidate sources may not import the materialized timing helper `_aka_benchmark`,
assign or delete attributes of imported modules, read frame or function globals,
or disable the evaluator's PyTorch dispatch mode. When the local import closure
is resolved, a same-name `.py` module takes precedence over a namespace-package
directory, matching Python. These structural checks are not a security sandbox.

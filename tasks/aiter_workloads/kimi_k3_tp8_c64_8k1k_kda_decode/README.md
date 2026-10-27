# Kimi K3 TP8: recurrent KDA decode

Optimize the complete public `aiter.ops.triton.attention.kda.fused_recurrent_kda`
call at AITER commit `2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`. The initial
candidate delegates to that upstream call. Baseline actions always execute the
separately installed, source-checked upstream package and its shipped gfx950
Gluon tuning table. Candidate code may use any GPU backend available in the
scoring runtime. Keep implementation changes under `source/`.

## Fixed workload and scope

The model is `moonshotai/Kimi-K3`, revision
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`. The protected `model_config.json`
matches the recorded checkpoint config hash. Its 96 KDA heads, head dimension
128 and gate lower bound -5 give 12 heads per TP8 rank. The model has separate
full-attention layers; this task covers one KDA recurrence site.

The recorded recipe has 64 requests, concurrency 64, 8192 input tokens, 1024
output tokens, BF16 serving, context limit 9218, and seed 42. This task selects a
**config-derived, fully admitted decode scenario**: all 64 requests contribute
one token at each of 1024 steps. Its weight is 1024 calls at one KDA site. This
is not a reconstruction of instantaneous batches from the historical scheduler.
EP and DCP were not explicitly bound in the recorded recipe and remain unknown;
they are not inferred as one. No collective is inside this rank-local operator.

The recorded serving implementation used FLA for KDA. The pinned AITER routine
is an alternative implementation of the same recurrence. Serving installation,
dispatch engagement and end-to-end benefit remain unverified. The upstream
26-call TP8+DCP8 prefill fixture is not enrolled: even its first sequence exceeds
the recorded recipe's context limit. Prefill, projections, short convolution,
output gating/norm, MoE and communication are outside this operation.

## Interface and data

`source/kernel.py:fused_recurrent_kda(**kwargs)` receives the public API's
explicit keyword arguments. Q/K/V are contiguous BF16 `[64,1,12,128]`; raw G is
FP32 at that shape; scalar raw beta is FP32 `[64,1,12]`. `A_log` is FP32 `[12]`
and `dt_bias` FP32 `[1536]`. Q/K normalization, bounded gate activation and beta
sigmoid are enabled. Scale is `128**-0.5`. There is no speculative token path,
convolution, output gate, custom tuning override or compact state encoding.

`initial_state` is a caller-owned FP32 V-first pool `[192,12,128,128]`.
`ssm_state_indices` is an int32 vector of 64 distinct slots. The call updates
exactly those state slabs and writes the provided BF16 `out`. It returns
`(out, initial_state)` using the same storage and layout. All other input bytes,
unselected slots and any slot padding must remain unchanged at completion.
The correctness-only case adds 128 FP32 padding elements between state slabs.
Internal scratch and the implementation's decomposition may change.

Operands use the pinned upstream tests' declared distributions: Q/K/V and raw
beta uniform `[0,1)`, raw gate/bias and initial state standard normal,
`A_log=log(U[1,16))`, permuted state slots, and the upstream `1e30` sentinel in
unused slots. These are generated operator tests parameterized by the model
configuration, **not captured activations, learned weights, or reconstructed
8192-token prefix states**. The selected recurrence must work for arbitrary
valid incoming state. No old saved programs or old case corpus are used.

## Oracle and timing

The protected `scripts/kda_ref.py` is the pinned upstream token-wise FP32
reference. The adapter gathers each sequence's initial state and compares both
output and updated state against it. Each sequence must have RMS relative error
below 0.005, matching the upstream decode limit while applying it per sequence.
NaN/Inf, wrong output/state aliasing, changed inputs, changed unselected slots
or changed padding fail. Correctness uses seeded and fresh draws, consecutive
state transitions with changed operands and routing, and zero incoming state.
Between calls the harness moves each request's state to a different pool slot
and updates the existing index tensor in place. Old slots become sentinels;
the incoming request state and physical-slot padding are preserved.

The canonical Arena graph benchmark times one complete public call per replay.
Its `prepare_fn` restores incoming state and poisons output **outside** every
sample's device interval; the helper forces one logical call per replay when
preparation is supplied. Both roles use 10 warmups and 100 samples. Every timed
sample's outputs and complete observable state are checked after its end event
and before the next restoration. The same captured graph must also work after
inputs, incoming state and routing change in place. Graph failure is a failure, not an
unreported change of timing method. Restoration is test isolation, not part of
the serving operation. Any candidate-required copies/resets remain inside the
candidate call and its measured cost.

The framework owns scores. The weighted total-time ratio is an operator proxy,
not a serving throughput estimate. GPU task-validator qualification on gfx950
is required before submission or use as a qualified benchmark. CPU checks do
not establish numerical accuracy or performance of the GPU implementation.

## Runtime and commands

The runtime needs ROCm, gfx950, PyTorch, Triton/Gluon, PyYAML and an installed
AITER package matching the declared source/config hashes. Workspace setup
acquires the pinned AITER source for inspection; it does not replace the
installed package. Arena materializes the canonical `_aka_benchmark.py` helper
beside the runner. Runtime versions, architecture, source checks and the
provided scoring-image identity are reported with each action.

```text
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Source attribution: the AITER implementation, test input policy and FP32
reference are from ROCm/aiter under its MIT license; see `UPSTREAM_LICENSE` and
the retained copyright notice in `scripts/kda_ref.py`. Exact source hashes and
links are in `source_identity.json`; case and frequency derivations are in
`workloads.json` and `frequency_provenance.json`.

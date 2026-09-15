# mi355x-kimi-k3-kda-linear-attn-20260728

`image_kernel` harness for **Kimi-K3 KDA (Kimi Delta Attention)** — the Triton-JIT
gated-delta-rule path used by K3's 69 linear-attention layers. Built from Hyperloom
session `20260728T091437Z` on MI355X/gfx950.

## Source and qualification status (2026-09-15)

The original session used custom vLLM build
`0.1.dev19253+g5f76ae224.d20260727`; its exact kernel sources remain unavailable.
The retained task now explicitly uses public vLLM commit
[`000c7df9ffd3e470980fd4cd6b8ec1b0585500ff`](https://github.com/vllm-project/vllm/tree/000c7df9ffd3e470980fd4cd6b8ec1b0585500ff).
That revision contains the AMD KDA implementation, including the same packed
non-speculative decode and chunk-with-fused-gate entrypoints. This is a public
source port, not recovery of the custom build's bytes.

`workspace.sources` acquires the pinned Git repository into `upstream/vllm`.
Protected setup stages its Python package into `vllm` before the framework
freezes the initial candidate. A receipt binds that staging to the source
identity. Repeating setup retains candidate edits; unknown collisions or changed
upstream content fail. The existing nine nested editable paths remain unchanged.
Other package files are protected dependencies, including reference/input code.

The task creates package namespaces for the parent Kimi model packages and the
shared FLA `ops` package to avoid importing unrelated inference-server components.
Real KDA/FLA leaf modules still execute from the declared local source; no kernel
is replaced by a shim. External imported package paths are rejected. No vLLM
server or its precompiled C++ operators are used by this Triton task.

A bounded MI355X probe of this public source passed actual compilation and the
original output gates for **all five full-size cases**, including 32,768 tokens.
The probe also checked FP32 final states, BF16 output contracts and state aliasing.
It used the original vLLM 0.24.0 runtime pinned by digest
`sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`
(Python 3.12.13, PyTorch 2.11.0, Triton 3.6.0), job 139332 on node100.
This is operator feasibility evidence, **not a full task-validator PASS**.
The current timing guard additionally requires the shared measured-state
observer described below; full qualification remains pending.

The readable original calling-wrapper backup has SHA-256
`e126de56567249973b92b29b445edb0bd016267f84c6b022da3c9baf79612dc8`.
It confirms these entrypoints and that chunk returns a final state which the
model caller subsequently scatters into its cache. The original archive lacks
the kernel sources. Historical timings and file line numbers below describe that
session; no before/after performance equivalence with the unavailable custom
implementation is claimed. New comparisons must use a baseline frozen from the
same public revision and the same protected harness as the candidate.

## Where the kernels actually live

The KDA kernels are vendored **per GPU vendor**; `kimi_gdn_linear_attn.py:399`
selects the AMD copy on ROCm:

```
vllm/models/kimi_k3/amd/ops/third_party/kda/{chunk,chunk_intra,chunk_intra_token_parallel,fused_recurrent}.py
vllm/third_party/flash_linear_attention/ops/{chunk_delta_h,cumsum,index,l2norm,op,utils,solve_tril,wy_fast}.py
```

This is confirmed by the session's own call stack (tracelens
`unified_perf_summary.csv`):

```
kimi_gdn_linear_attn.py(381): _forward
 -> kda/chunk.py(774): chunk_kda_with_fused_gate
    -> kda/chunk.py(694): chunk_kda_with_fused_gate_fwd
       -> kda/chunk_intra.py(573): chunk_kda_fwd_intra
```

They are Triton-JIT, which is why the trace reports `launcher = Not found`.

## Hot kernels covered

- **k007** `fused_recurrent_kda_packed_decode_kernel` (decode; 48.4 ms, 2.11% GPU)
  → the `packed_decode` case.
- **prefill chunk group** `chunk_kda_fwd_*`, `chunk_gated_delta_rule_fwd_h_*`,
  `kda_gate_chunk_cumsum_*`, `chunk_gla_fwd_kernel_o`, `recompute_w_u_*`,
  `l2norm_*`, `layer_norm_gated_*` (~43.7 ms, grouped with no individual k-IDs)
  → the `chunk` cases.

### The decode entry point matters

k007 is launched **only** by `fused_recurrent_kda_packed_decode`
(`fused_recurrent.py:596`), called from the non-spec decode branch
(`kimi_gdn_linear_attn.py:609`). The sibling `fused_recurrent_kda` launches a
*different* kernel, `fused_recurrent_kda_fwd_kernel`, on the speculative-decode
branch — and K3 sets `num_nextn_predict_layers=0`, so that kernel appears **0
times** in every trace artifact of this session. Targeting it would benchmark code
the model never runs.

## Config (from K3 `config.json linear_attn_config`)

`num_heads=96`, `head_dim=128` (`d_k = d_v`), `chunk_size=64`,
`gate_lower_bound=-5.0`, `short_conv_kernel_size=4`, `use_full_rank_gate=true`,
69 KDA layers + 24 full-attention layers. The trace is rank0 of TP=8, so the cases
use the per-rank shape `num_heads = 96/8 = 12`.

Shape evidence: `aten::fill_` under `chunk_kda_fwd_intra` carries
`Input Dims (1, 7211, 12, 64)` and `(1, 1080, 12, 64)` bf16 — i.e. per-rank H=12
and packed prefill token counts 7211 / 1080. Note that ISL=1024 is the
*per-request* input length, not the packed batch size, so it is not a kernel shape.

## Two contract details that are easy to get wrong

Both were read off the kernel sources, not assumed:

1. **`gate_lower_bound = -5.0` is not a clamp — it selects a different gate
   function.** With the bound set the kernel computes
   `gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`; without it,
   `gate = -exp(A_log) * softplus(raw_g + dt_bias)`
   (`fused_recurrent.py:513-521`, `chunk.py:507-515`). K3 always takes the first
   branch, so both the kernel call and the golden use it.

2. **`raw_beta` is passed pre-sigmoid.** Both kernels apply `sigmoid` internally
   (`fused_recurrent.py:525`, `chunk.py:470`), so pre-applying it in the harness
   would square the gate.

Also: `A_log` is 1-D of length `local_num_heads` and `dt_bias` is
`local_num_heads * head_dim` (`kimi_gdn_linear_attn.py:238,266`);
`state_indices` entries must be `> 0` because `<= 0` is the NULL slot and makes the
kernel emit zeros for that row (`fused_recurrent.py:481`).

## Cases

| id | mode | seqs x len | source |
|---|---|---|---|
| `kda-decode-packed-k007` | packed_decode | 62 x 1 | reconstructed (slice concurrency conc62) |
| `kda-prefill-chunk-t7211` | chunk | 1 x 7211 | trace |
| `kda-prefill-chunk-t1080` | chunk | 1 x 1080 | trace |
| `kda-long-chunk-t16384` | chunk | 1 x 16384 | extrapolated headroom |
| `kda-long-chunk-t32768` | chunk | 1 x 32768 | extrapolated headroom |

## Correctness — numerical parity vs a float64 golden

FLA ships no naive torch reference in-tree, so `scripts/task_runner.py:_golden` is
an independent **float64** transcription of the recurrence, taken directly from
`fused_recurrent_kda_packed_decode_kernel` (`fused_recurrent.py:504-533`):

```
g_t = -5.0 * sigmoid(exp(A_log) * (raw_g + dt_bias))     # safe-gate branch
per token (state S = [H, d_v, d_k], one segment per sequence):
  q = l2norm(q_t) * scale ;  k = l2norm(k_t) ;  v = v_t
  S  = S * exp(g_t)          # decay per k-column
  v  = v - S @ k             # delta-rule "remove old value"
  v  = v * sigmoid(raw_beta_t)
  S  = S + outer(v, k)
  o_t = S @ q
```

`chunk_kda_with_fused_gate` computes the same recurrence blockwise, so one
reference covers both modes. Gate: `cos > 0.999` and normalized max error `< 0.03`.

Historical measurements on MI355X: packed decode `cos = 0.999999`, `rel_max_err = 0.0027`;
chunk `cos = 0.999992`, `rel_max_err = 0.0071`.

Packed decode updates the selected cache slots in place and returns the same
state storage. Chunk leaves its input state unchanged and returns a new final
state; the model caller's cache scatter is outside this operator's timing scope.
The independent reference starts from private FP64 state copies. Both output
and final state use the original cosine and normalized-maximum-error gates;
output shape/BF16/device and state shape/FP32/device are also checked. Unselected
decode cache slots must remain unchanged.

Correctness covers every full scored shape, including all 32,768 tokens. The
old prose about a 320-token cap did not describe the runner. Original case seeds
23, 124, 225, 326 and 427 remain unchanged. `session_cases.json` is preserved as
historical evidence, including its older state-aliasing annotation; the executable
contract and this section state the verified current behavior.

## Edit surface and JIT freshness

`_configure()` puts the workspace-seeded `vllm` copy first on `sys.path`, so an
agent's kernel edits shadow the in-image install; Triton keys its cache on kernel
source, and `TRITON_CACHE_DIR` is additionally pinned inside the workspace so no
run can serve a binary compiled from another workspace's source.

Historical edit-propagation evidence: scaling `b_q` by 1.5 in the workspace copy of
`fused_recurrent.py` changes the decode output norm by exactly 1.5x
(0.214733 -> 0.322133), and the same edit in `chunk_intra.py` moves the chunk
output (1.473997 -> 2.210493).

## Run

```
python3 scripts/evaluate.py candidate compile       # smoke: one KDA call
python3 scripts/evaluate.py candidate correctness   # float64 parity, both modes
python3 scripts/evaluate.py candidate performance   # CUDA-graph timed -> build/performance_report.json
```

## Effective task instructions

Optimize Kimi-K3 KDA (Kimi Delta Attention) linear attention on MI355X/gfx950. KDA is the Triton-JIT gated-delta-rule path used by K3's 69 linear-attention layers. The kernels are vendored per GPU vendor; the ROCm copy that kimi_gdn_linear_attn.py selects lives in models/kimi_k3/amd/ops/third_party/kda/ (chunk.py, chunk_intra.py, chunk_intra_token_parallel.py, fused_recurrent.py) plus the shared FLA ops under third_party/flash_linear_attention/ops/. Two entry points are timed: fused_recurrent_kda_packed_decode (decode hot kernel k007, fused_recurrent_kda_packed_decode_kernel) and chunk_kda_with_fused_gate (the prefill chunk-KDA kernel group). Do NOT retarget anything at fused_recurrent_kda: that is the speculative-decode entry and K3 never executes it (num_nextn_predict_layers=0). Dims are the session's per-rank TP=8 shapes: num_heads=12, head_dim=128 (d_k=d_v), chunk_size=64, gate_lower_bound=-5.0. That lower bound selects the safe-gate branch gate = -5.0 * sigmoid(exp(A_log) * (raw_g + dt_bias)) rather than the softplus branch, and raw_beta arrives pre-sigmoid because both kernels apply sigmoid internally. session_cases.json carries the session's real packed prefill token counts (7211 and 1080) plus long-sequence headroom up to T=32768. Correctness is numerical parity against an independent float64 golden (cos > 0.999 and normalized max error < 0.03). Preserve all correctness cases and improve the CUDA-graph measured performance, especially at long sequence length. Keep the public signatures of fused_recurrent_kda_packed_decode and chunk_kda_with_fused_gate unchanged.

## Arena v2 contract

The candidate is the implementation staged from the declared pinned Git source.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing declared source is an error, never permission to use the installed copy.

Setup runs `scripts/materialize_source.py` and then `scripts/setup_task.py`
after Git acquisition and before baseline capture. Both are declared argv
commands in `config.yaml`; they stage and validate the source before evaluation.
Do not edit `scripts/`, workload files or references. Additional source files
outside `candidate.editable` are dependencies, not editable implementation.
Preserve the original numerical gates, seeds, layouts, dispatch, state handling
and CUDA graph/event timing. `workloads.json` enumerates the complete manifest
independently of reported timings; `session_cases.json`, when present, retains
its original session provenance. Cases marked correctness-only are not scored.

Use the agent-neutral commands:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Baseline commands run in the framework's frozen workspace. Each command emits
one `ARENA_EVAL_RESULT=` envelope. A failed dependency, dispatch or output contract
is a failure, not an accepted baseline numerical diagnostic. The original
`task_runner.py` remains the protected operator implementation of these checks;
its generated performance region must be materialized by Arena. Optional
profiling does not supply final evaluation evidence. Agent CLI adaptation belongs
to the agent integration; use the declared v2 runner for task evaluation, with
the task's full numerical and workload checks.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.


## Stateful timing verification

Timing retains the original warmups, samples, repeat caps and evolving decode
state across graph replays. It does not reset the state or force one invocation
per graph. Before candidate execution, the harness saves private copies of all
input tensors. The canonical helper must call the optional collector
`before_sample(calls_per_replay)` on the measurement stream, before the start
event of each actual measured sample. The task clones decode state or chunk
`v` there;
it does not execute the reference or alter input state inside the timed interval.

After timing, the harness evaluates the independent recurrence for the observed
number of invocations from that private starting state and checks the original
measured output and final state. The pinned public chunk implementation passes
`o=v` to its final output kernel: output aliases and overwrites the BF16 value
input, while the initial state remains read-only. Its reference therefore feeds
the BF16-rounded output into the next invocation's value input for the actual
reported repeat count. The harness requires this output alias and validates its
complete numerical contents; it does not poison that alias before replay,
because doing so would corrupt the new input. The separate returned state is
still poisoned. This preserves the original graph batching and allocation work.
A second check perturbs Q, gates, state and chunk values from original private
copies, then validates the same captured replay. All other input mutation fails
and all input tensors are
restored in `finally`, including on replay or comparison failures. Missing
observer support fails explicitly; a replay-only check cannot certify the
original stateful measured path.

For state and replay comparisons, two exactly zero vectors match. If only one
vector is zero, comparison fails. Nonzero vectors are rescaled before computing
cosine to preserve their angle when their norms are very small. This avoids the
cosine library's epsilon floor; the original `cos > 0.999` and normalized maximum
error `< 0.03` gates remain unchanged. Shape and finite-value checks still apply.
Task validation exercises these rules with exact-zero, tiny-vector, opposite
direction and incorrect-magnitude controls.

Job 139802 failed the original timed chunk-state check. Diagnostic job 139981
reproduced it on MI355X with the unchanged cases and timing: at T=7211, after
four calls in the observed graph replay, both the actual FP32 final state and
the independent FP64 reference were exactly zero. The previous cosine check
returned zero for that pair and rejected it. A fresh single invocation of the
same full-size case passed both output and state checks. The comparator correction
above addresses that false rejection; the diagnostic is not a task-validator
PASS, and fresh full validation remains required.


Job 139646 passed all five full-shape correctness cases and the actual timed
decode output/state check, but failed chunk performance because the harness
incorrectly treated `v` as read-only. That failure remains recorded. The alias
contract above follows the explicit `o=v` call in the declared public source,
not a general exemption from input checking. CPU scalar controls independently
verify repeated BF16 value updates and reject a single-invocation answer and an
output that does not alias `v`. Fresh full GPU validation is required for this
correction; the passing decode helper before/after experiment is separate.

# AgentX workloads

Derive the performance cases of a kernel task from real agentic serving
traffic. The tool serves a model with SGLang, replays the AgentX corpus
(agentic coding traces, `semianalysisai/cc-traces-weka-062126`) against it,
records which kernels run at which shapes and how often, and proposes a
`workloads.json` whose performance cases carry **weights**: how often each
case's shapes run in that workload.

A weighted task scores the time the workload spends in the operator before and
after, `Σ wᵢ·t_baselineᵢ / Σ wᵢ·t_candidateᵢ`, instead of the geometric mean.
Tasks without weights are unaffected. The full reference (method, legs,
log weighting, crash handling) is
[`docs/how-to/agentx-workloads.md`](../../../docs/how-to/agentx-workloads.md);
the weights format is in
[`docs/how-to/add-task.md#case-weights`](../../../docs/how-to/add-task.md#case-weights);
`python3 -m src.tools.agentx_workloads <command> --help` documents every option.

## Quick start

```bash
# 1. Capture (on a GPU node, about 1-2 h): serve, replay, profile.
python3 -m src.tools.agentx_workloads capture \
  --model-path <HF snapshot or model directory> --model-name <org/model> --served-model-name <org/model> \
  --corpus <AgentX corpus: HF dataset cache directory or traces.jsonl> \
  --image <SGLang ROCm image> --out <capture root> --capture-id <model>-tp<N>-mi355x \
  --tp <N> --context-length 262144 --chunked-prefill-size 16384 \
  --page-size 16 --kv-cache-dtype fp8_e4m3 --server-args "<other recipe flags>" --env KEY=VALUE

# 2. Census (CPU, minutes): kernel time and calls per shape over the steady state.
python3 -m src.tools.agentx_workloads census <capture root>/<capture id>

# 3. Propose (CPU, seconds): a weighted workloads.json for one task, written for review.
python3 -m src.tools.agentx_workloads propose tasks/<language>/<category>/<operator>/<scenario> \
  --census <capture root>/<capture id>/kernel_census.json [--baseline-us baseline.json]
```

Then review `<capture>/proposals/<task>.report.json` and rerun with `--write`
to replace the task's `workloads.json` (see "Quality assurance" below).

`capture` needs Docker, the GPUs and the serving image on the node; run it
under your scheduler's exec command and detach it (it runs for hours).
`census` and `propose` need only Python.

## Choose the deployment before you capture

The proposal is only as real as the deployment you capture. Decide these per
model, and record anything unusual in the capture ID:

| Setting | Why it matters | Guidance |
|---|---|---|
| Parallelism (`--tp`, DP attention) | Task shapes are **per-GPU** shapes: a 5120-wide projection is N = 640 at TP8 | Use the parallelism that produces the task's shapes; check the task's `task.yaml` const axes against the model config |
| `--chunked-prefill-size` | Caps prefill GEMM M; with 8192 no M above 8192 is recorded | Use the model's production value (16384 or 32768) |
| `--context-length` | Requests that do not fit are skipped (`corpus_coverage` in `capture.json`) | The largest window the model serves; 262144 matches the official capped corpus |
| Server flags and `--env` | Page size, KV-cache dtype, attention/MoE backends and environment variables choose the kernels | Copy them from the model's production MI355X SGLang recipe |
| Speculative decoding | MTP/EAGLE/DSpark change decode shapes (verify runs r tokens per sequence) and add draft-model kernels | Enable it when production does (`--server-args "--speculative-algorithm ..."`); the census handles it |
| `--lanes` | Concurrency sets the decode batch sizes and so the decode share | Keep 32 for comparability unless you target another load; it is recorded |

The settings the server actually used (from SGLang's `/server_info`) are
recorded in each leg's `server_info.json` and carried into the proposal's
`weight_provenance.deployment`.

## What each step writes

```
<capture root>/<capture id>/
  capture.json            arguments, corpus coverage, per-leg windows, crashes and restarts
  eager/  graph/          one directory per leg: logs/server*.log, logs/replay_requests.jsonl,
                          server_info.json, traces/w<n>/*-TP-<r>*.trace.json.gz
  kernel_census.json      census (machine-readable)       <- census
  census.md               census (readable summary)       <- census
  proposals/<task>.report.json      every matched shape, grouping, warnings  <- propose
  proposals/<task>.workloads.json   the proposal (only when valid)            <- propose
```

- **Two legs.** The eager leg (`--disable-cuda-graph`) records decode kernels
  with their shapes; the CUDA-graph leg gives the production step counts.
  Both legs record prefill.
- **Shape hooks.** `--shape-hooks` (default: the `default` preset of
  [`shape_hooks.py`](shape_hooks.py)) wraps Python kernel wrappers (FP8
  block-scale GEMM, Gated DeltaNet and causal conv1d, Triton extend attention,
  DeepSeek-V4 sparse MLA, indexer and top-k) so their kernels are linked to a
  `pyhook::` op with input shapes; `none` turns them off. `capture.json`
  records the hooked targets and `census.md` the share of device time linked
  through them.
- **Census phases.** `prefill`, `decode`, and with speculative decoding
  `verify`, `draft` and `draft_prefill`. Token-indexed GEMMs are rebuilt over
  the whole steady state from the scheduler log: logged forwards × calls per
  forward × per-call cost interpolated in log M from the profiler windows.
- **Proposal.** Each performance case has `weight` (calls per second of the
  steady state), `observed` (phases, share of the operator's GPU time, merged
  shapes) and `source`. `weight_provenance.coverage` (≥ 0.8 required) is the
  share of the operator's recorded time the cases represent.

## Adding an operator

`propose` needs a matcher for the task's operator family in
[`matchers.py`](matchers.py): it selects the census rows whose shapes match
the task's const axes and maps them to the task's var axes. A matcher reports
the const axes it could not check; `propose` lists them in the report, so
confirm those by hand. Without a matcher `propose` stops with an error naming
the operator. Existing matchers:

| Operators | Census rows | Var axis from | Const axes checked |
| --- | --- | --- | --- |
| `gemm_*`, batched and skinny GEMM | GEMM rows (log-weighted when present) | GEMM M, N, K (and batch) | `n`, `k`, `m` or `tokens`, `batch`; input dtype |
| `paged_attention_decode`, `decode_attention` | decode attention rows | decode batch, logged KV tokens per sequence | query heads and head size when recorded |
| `rms_norm`, `fused_add_rms_norm` | norm ops | rows of the first input | hidden size |
| `qk_rmsnorm_rope` | ops that fuse a norm and RoPE, with q `[T, heads, head_dim]` and kv `[T, head_dim]` | q tokens | heads, head dim; RoPE width and table length when the cos/sin table is an input |
| RoPE, gated activations | RoPE or activation ops | rows of the first input | the width axis |
| `dynamic_mxfp4_quant`, `dynamic_mxfp8_quant` | quantization ops with FP4 (or FP8 and E8M0) types or an MX op name | rows of the `[T, hidden]` input | hidden size; 32-wide scale blocks when the scale tensor is an input |
| `per_token[_group]_quant_{fp8,int8}`, other `*quant*` | quantization ops of the output type (FP8 with float32 scales, INT8; any quantized type for the others) | rows of the `[T, hidden]` input | hidden size; scale groups per row (`num_groups`, `hidden_size / group_size`, or 1 per token) |
| `biased_grouped_topk`, `topk_softmax` and routing | router ops with logits `[T, experts]` (and a correction bias `[experts]`) | logits rows | experts, top-k; group counts are scalars the profiler does not record |
| `moe_sorting` | sorting ops, and sorting kernels of the fused MoE op | rows of `topk_ids` | top-k, experts and block size from the buffer sizes, output width |
| `fused_moe_fp8_blockscale`, `fused_moe_mxfp4` | ops holding the expert weights `[E, 2I, H]` and `[E, H, I]` in the task's weight format, the layer's sorting op and inter-stage requantization, summed per token count | tokens of the layer input | experts, hidden, intermediate size, top-k; weight format |
| `causal_conv1d_fwd`, `gdn_chunk_fwd_h`, `gdn_fused_recurrent_decode` | conv1d / gated-delta-rule ops or kernels | packed tokens, sequence tokens or decode batch | channels and width; key and value heads and head sizes |

A census row counts kernel launches; where an op launches several kernels per
call, matchers that combine rows (`fused_moe`) count op calls. Kernels
launched from Python without a recorded op (Triton, Gluon and FlyDSL wrappers)
get their shapes from the capture's shape hooks (`--shape-hooks`): the hooked
wrapper is an op named `pyhook::<package>.<function>` whose dims are the
wrapper's arguments, so the GEMM rows and the matchers read it like any other
op. Kernels of wrappers that are not hooked carry no shapes; a matcher reports
their time in its notes, and the state-space matchers read the step's token
count or batch from such a row when the census records it (`step_tokens`,
`step_batch`). A const axis counts as checked only when every matched row
showed it.

## Quality assurance

A proposal changes a task's workloads, so it needs the full task workflow:

1. Review the report: the matched shapes, the representative of each merged
   case, the unchecked const axes, the clamped share and the coverage.
2. Update the task README's workload section (model, capture, coverage, what
   became correctness-only).
3. Run `python3 -m src.tools.check_task tasks/<...>/<scenario>`.
4. Run the QA auditor on MI355X. It enforces the 10 µs floor with the
   baseline's cold-cache timing; a case below it must become correctness-only,
   and the proposal must be regenerated with `--baseline-us`.

Keep captures outside the repository; record the capture ID and its location
in the PR.

## Pitfalls

- **Stop a capture by its PID**, not `pkill -f <pattern>` inside a
  `bash -lc` command that contains the pattern (it kills its own shell).
  `capture` refuses to start when its port already answers, so a stale server
  cannot be mistaken for the new one.
- **Shared HF caches.** Snapshots whose files link into a shared blob store
  outside the repo directory are mounted automatically; for any other path the
  server must see (for example a separate draft model), add `--mount`.
- **Kernel tables.** A shape the production deployment never runs can hit a
  missing kernel config (for example an FP4 MoE at a TP the recipe does not
  use). Prefer the recipe's parallelism over forcing TP8.
- **One capture per port.** Several small captures can share a node with
  distinct `--gpus` and `--port` (keep ports ~1000 apart; SGLang uses
  port+100..+1000 internally).
- **Large checkpoints** load slowly from network storage; raise
  `--health-timeout` (default 1800 s) for 0.5 TB and larger.

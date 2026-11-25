---
myst:
    html_meta:
        "description": "Derive weighted performance cases for AgentKernelArena kernel tasks from the AgentX agentic serving corpus: capture a model's serving workload, build its kernel census and propose a task's workloads.json."
        "keywords": "AgentKernelArena, AgentX, workloads, case weights, kernel census, SGLang, profiler, MI355X"
---

# Derive task workloads from AgentX serving traces

`src.tools.agentx_workloads` turns a model's real agentic serving workload into
[weighted performance cases](add-task.md#case-weights) for kernel tasks. It
serves the model with SGLang, replays the AgentX corpus against it, records
which kernels ran with which shapes and for how long, and proposes a task's
`workloads.json` whose case weights are the recorded GPU-time shares.

It is a maintenance tool. Task code does not import it, it is not materialized
into workspaces, and it changes a task only when asked to (`--write`).

| Step | Where | Command | Output |
| --- | --- | --- | --- |
| Capture | GPU node with Docker | `python3 -m src.tools.agentx_workloads capture ...` | `<out>/<capture-id>/{capture.json, eager/, graph/}` |
| Census | CPU | `python3 -m src.tools.agentx_workloads census <out>/<capture-id>` | `kernel_census.json`, `census.md` |
| Propose | CPU | `python3 -m src.tools.agentx_workloads propose tasks/<...>/<scenario> --census ...` | `proposals/<task>.workloads.json`, `.report.json` |

## Capture

Capture one model in one deployment (image, parallelism, context length,
server settings) per capture ID. Use the deployment that produces the task's
shapes and matches production:

- **Parallelism.** Task shapes are per-GPU shapes. Pick the tensor (or data)
  parallelism whose per-rank shapes are the task's (a GEMM with N = 5120 on a
  model whose projection is 5120 wide is a TP1 shape; at TP8 it would be 640).
- **Server settings.** Follow the model's InferenceX MI355X SGLang agentic
  recipe: chunked prefill (16384 or 32768), page size, KV-cache dtype,
  speculative decoding. `--chunked-prefill-size` sets the largest prefill
  GEMM M; with 8192 no prefill shape above M = 8192 is recorded.
- **Context window.** Requests that do not fit `--context-length` are
  skipped; use the largest window the model serves (the official capped corpus
  uses 256k). `capture.json` records the share of requests and input tokens
  the window drops (`corpus_coverage`).

```bash
python3 -m src.tools.agentx_workloads capture \
  --model-path <HF snapshot directory> --served-model-name <name> --model-name <org/model> \
  --corpus <directory or traces.jsonl of the AgentX corpus> \
  --image <SGLang ROCm image> --out <capture root> --capture-id <model>-tp8-mi355x \
  --tp 8 --context-length 262144 --model-override-args '<JSON, for example a YaRN rope_scaling>'
```

Run it on the GPU node itself (under Slurm or Spur, through the scheduler's
exec or run command for your job); it uses the node's Docker daemon and GPUs
(`--gpus`). The serving image must already be on the node.

Every leg starts a fresh server container, waits for it to load, replays the
corpus from a client container, takes `--window-steps` forward steps of
torch-profiler trace at each `--window-at` offset, then stops both
containers. The server uses `--chunked-prefill-size` (16384), `--mem-fraction`
(0.80; SGLang may lower it), and `--page-size` / `--kv-cache-dtype` when given;
other SGLang flags go in `--server-args`. `capture.json` records the
arguments; each leg's `server_info.json` records the settings the server
actually used, and the census and proposals carry those.

### Shape hooks: shapes for kernels launched from Python

The census reads a kernel's shapes from the CPU op that launched it (the
trace's `External id` and the op's `Input Dims`). Triton, Gluon and FlyDSL
kernels that a Python wrapper launches directly have no such op, so without
help their rows carry only a kernel name. `--shape-hooks` wraps such wrappers
in the server: while the profiler records, each call runs in a recorded
function scope named `pyhook::<package>.<function>` (the same mechanism
TorchInductor uses around its Triton launches) whose inputs are the call's
arguments. With `record_shapes` on, the trace holds it as a CPU op with
`Input Dims` and `Input type`, and the kernels the wrapper launches carry its
`External id`. The census treats it like any other op: GEMM wrappers become
GEMM rows (M, N, K from the activation and weight), state-space and attention
wrappers become ops whose dims the matchers read. Each kernel is still
counted once, linked to its innermost op, so a kernel that a hooked wrapper
launches through an ordinary op keeps that op.

```bash
--shape-hooks default                                     # the preset (the default)
--shape-hooks none                                        # off
--shape-hooks default,my.pkg.module:my_launcher[=label]   # the preset plus a wrapper
```

The `default` preset (`PRESETS` in `src/tools/agentx_workloads/shape_hooks.py`)
covers, for the SGLang 0.5.23 ROCm images: aiter's Triton FP8 block-scale GEMM
and MLA batched FP8 GEMM; SGLang's Gated DeltaNet wrappers (chunk state pass,
fused recurrent, replay-SSM commit, gating, q/k/v split) and
causal conv1d (SGLang's and aiter's); SGLang's Triton extend and prefill
attention and MLA state merge; and the DeepSeek-V4 sparse MLA (aiter
`pa_decode_sparse`, `sparse_mla_fwd`), FP4 indexer logits, top-k and split
reduce. A wrapper whose module the deployment never imports stays inert. To
add one, find the Python function that launches the kernel (`grep -rn
"<kernel name>\[" ` in the image's `sglang` and `aiter` sources) and name its
defining module.

How it is loaded: the capture mounts the tool directory read-only into the
server container, prepends its `shape_hooks_site` directory to `PYTHONPATH`
and sets `AKA_SHAPE_HOOKS` to the targets; that directory's `sitecustomize.py`
installs an import hook in every server process (scheduler subprocesses
included) and then runs the image's own `sitecustomize`. A target is patched
in the module that defines it, right after that module runs (every time it
runs: aiter's old import paths, such as `aiter.ops.triton.gemm_a8w8_blockscale`,
run the module a second time), so modules that import the function by name get
the hooked one. Nothing is patched when
`AKA_SHAPE_HOOKS` is unset.

Cost and safety: with the profiler off a hooked call costs one Python frame
and a flag test (about 0.1 µs); the wrapper calls the original function with
the original arguments, so results are unchanged. CUDA-graph replays run no
Python and are not hooked; during graph capture the profiler is off and the
wrapper only forwards (the same under `torch.compile` tracing). The server log
lists the hooks installed per process and the first recorded call of each
hook (`[shape_hooks ...]`), which shows which wrappers the deployment used.
`capture.json` records the targets (`shape_hooks`), the census copies them
and reports each window's `hooked_share` of device time.

The model's HF repo directory is mounted into the server; when the snapshot's
files resolve into a shared blob store outside the repo (a deduplicated cache,
for example a cache shared between several model directories), that store is
mounted too.

If the server dies during a leg, `capture` records the crash (time, fault
line and the scheduler batches before it) in `capture.json`, starts a new
server (log `logs/server.<n>.log`), and resumes the replay for the remaining
`--duration`, up to `--max-restarts` times. The steady state resumes
`--restart-warmup` seconds after the restart, while the radix cache refills;
profiler windows that fall before then, or whose trace was cut by the crash,
are skipped and left out of the census. The census reads each leg's
steady-state segments with their own logs and lists crashes and restarts in
`census.md`. A server that stops answering during a profiler window ends the
window with status `server exited` (or `server unreachable` when the container
still runs; it is then stopped) and takes the same crash-and-restart path.
Containers are named `aka-agentx-<capture-id>-<leg>-{server,client}`
and removed afterwards; outputs are given back to the calling user. The
orchestrator needs only Python and Docker on the node; the replay client runs in
the serving image.

The replay keeps what shapes kernel work in the AgentX recipe: closed-loop
`--lanes`, trajectories shuffled with `--seed` and started at a random point
between 25% and 75% of their turns, think time capped at `--think-cap`
seconds (300, aiperf's idle-gap cap), and output length from the trace
(`ignore_eos`, at most 16384 tokens). Prompts are sent as
token IDs rebuilt from the corpus `hash_ids` (one fixed 64-token block per
trajectory and hash ID), so input lengths are exact and the radix cache sees the
corpus's prefix sharing. A request that does not fit `--context-length` is
skipped and the trajectory continues (agentic traces compact their context, so
later turns can fit again), as in the official capped corpus variants. The
steady state starts `--warmup` seconds after the replay starts. The load
(`--lanes`, 32) changes the decode batch sizes and therefore the decode share;
it is recorded with the deployment.

### Two legs: why decode is measured eagerly

| Leg | Server | Used for |
| --- | --- | --- |
| `graph` | Production configuration, decode CUDA graphs on | The observed workload (scheduler log) and the captured graph batch sizes. Its profiler windows are summarized for comparison only. |
| `eager` | The same server with `--disable-cuda-graph` | The census: every kernel's shapes and GPU time. |

On ROCm the torch profiler records the device kernels of only a few HIP-graph
replays per window, and none of them is linked to the CPU op that would give
its shapes. Rescaling the recorded replays guesses both. In the eager leg every
decode kernel is an ordinary launch with its CPU op's input dims, so decode
shapes and times are measured. A kernel's device time does not depend on graph
replay, but eager decode runs the batch unpadded, so `propose` pads decode
shapes to the graph leg's captured batch sizes, which is what production runs.
The eager leg also runs slower on the host side, so its batch sizes can differ
from production; `census.md` lists the batch-size and KV-length distributions
of both legs for review. Eager collectives stage through copies and take a
larger share of device time than with graphs. Proposals compare each operator's
shapes only with its own time, so this does not change their weights.

The serving image also ships `rocprofv3`, which has an attach mode for a
running server. Its kernel records carry no CPU-op shapes or step boundaries,
so it is not used for the census.

## Census

```bash
python3 -m src.tools.agentx_workloads census <out>/<capture-id> [--rank 0]
```

The census is the device time of one TP rank. Decode kernels come from the
eager leg's windows, prefill kernels from the windows of both legs (prefill
runs eagerly in both). Each phase is scaled from the steps the windows recorded
to the steps the graph leg's scheduler log counts over its steady state
(`phase_weights`; `--no-log-weighting` keeps the raw sums).

Profiler windows are short: a prefill chunk the log shows hundreds of times
(16384 tokens) can be missing from all of them. So GEMMs are also rebuilt over
the whole steady state (`gemm_log_weighted`), as the Qwen3-32B probe did:

- every logged prefill batch is one forward with M = its new tokens (split at
  the chunked-prefill size), every logged decode iteration one forward with M =
  its running batch padded to the captured CUDA-graph size;
- calls per forward are the GEMM's calls per recorded step (one per layer);
- the cost per call is piecewise-linear in log M through the per-call times
  measured in the windows. Below the smallest measured M it is that point's
  cost; above the largest it grows in proportion to M. Both are marked
  extrapolated, and `propose` reports the share of the operator's time that
  rests on extrapolated costs.

This applies to the GEMMs whose recorded calls run over every token of their
step (M equals the step's tokens or batch); others, such as the LM head on
prefill, keep their window rows. The GEMM matcher uses these rows when present,
and `weight_provenance.description` records the method. The census also lists
every kernel with its family (GEMM, attention, norm, RoPE, quant, MoE,
communication, ...), the CPU op and input dims that launched it, its phase
(prefill or decode, from SGLang's step annotations), calls and time. GEMMs are
aggregated by (M, N, K, dtype, batch, phase). Attention records batch, query
tokens and KV tokens: prefill from the op's ragged-index dims, decode from the
scheduler log line nearest in time with the same batch size (mean KV tokens per
sequence, in 512-token buckets). `census.md` is the readable summary;
it reports the share of device time linked to shapes, and the part linked
through shape hooks.

## Propose

```bash
python3 -m src.tools.agentx_workloads propose tasks/<...>/<scenario> \
  --census <out>/<capture-id>/kernel_census.json [--baseline-us baseline.json]
```

1. The operator's matcher selects the census shapes that match the scenario's
   const axes and input dtype (below); their time is the operator's recorded time.
2. Shapes whose var-axis sizes are within `--merge-ratio` (1.5) of each other
   merge into one case at the member with the most time, until at most 16
   remain. A case must take at least 10 µs per call (`--min-case-us`); a
   shorter shape merges into an eligible neighbour or stays correctness-only.
   Without `--baseline-us` the floor uses the served kernel's time, which can
   be faster than the task's baseline; pass the baseline's measured times per
   var-axis size (JSON `{"size": us}`, interpolated in log-log) when known.
   Groups below `--min-share` (0.5%) are dropped while more than four remain.
3. Each case's `weight` is how often its shapes run: their calls per second
   of the production leg's steady state on TP rank 0. AKA scores a weighted
   task as `Σ wᵢ·t_baselineᵢ / Σ wᵢ·t_candidateᵢ`, the time the workload
   spends in the operator before and after (the serving-weighted speedup of
   GEAK's kernel extraction), so the weights apply to AKA's own baseline
   rather than the served kernel's times. The case's `observed` record keeps
   its phases, its share of the operator's recorded time (`time_share`) and
   the recorded shapes merged into it (`members`). `weight_provenance` records
   `kind: agentx`, the corpus and revision, the capture ID, the model,
   `coverage` (the sum of the time shares), `weight_unit`, the effective
   `deployment` (a draft model's local path is recorded as `org/name@revision`
   for a Hugging Face cache snapshot, else as its directory name) and the
   `corpus` coverage under the context window.
4. Every existing case stays as a correctness case, as do the recorded shapes
   that could not be performance cases; missing boundary cases are added.
5. The proposal is validated with the task contract
   (`src.task_contract.validate_task_contract`): 4 to 16 performance cases and
   coverage of at least 0.8.

`<task>.report.json` lists every matched shape with its share and per-call
time, the share of the time whose shapes fell outside the task's axis bounds
and were clamped to them (`clamped_to_axis_bounds_share`), the grouping, the dropped shapes and the const axes the matcher could
not check. `<task>.workloads.json` is written only for a valid proposal;
`--write` also replaces the task's `workloads.json`.

### Matchers

| Operators | Census rows | Var axis from | Const axes checked |
| --- | --- | --- | --- |
| `gemm_*`, batched and skinny GEMM | GEMM rows (log-weighted when present) | GEMM M, N, K (and batch) | `n`, `k`, `m` or `tokens`, `batch`; input dtype |
| `paged_attention_decode`, `decode_attention` | decode attention rows | decode batch, logged KV tokens per sequence | query heads and head size when recorded |
| `rms_norm`, `fused_add_rms_norm` | norm ops | rows of the first input | hidden size |
| `qk_rmsnorm_rope` | ops that fuse a norm and RoPE (such as the fused norm + RoPE + cache store), with q `[T, heads, head_dim]` or `[T, heads * head_dim]` and kv `[T, head_dim]` | q tokens | heads, head dim; RoPE width from the cos/sin table; its length when equal |
| RoPE, gated activations | RoPE or activation ops | rows of the first input | the width axis |
| `dynamic_mxfp4_quant`, `dynamic_mxfp8_quant` | quantization ops with FP4 (or FP8 and E8M0) types or an MX op name | rows of the `[T, hidden]` input | hidden size; 32-wide scale blocks when the scale tensor is an input |
| `per_token[_group]_quant_{fp8,int8}`, other `*quant*` | quantization ops of the output type (FP8 with float32 scales, INT8; any quantized type for the others) | rows of the `[T, hidden]` input | hidden size; scale groups per row (`num_groups`, `hidden_size / group_size`, or 1 per token) |
| `biased_grouped_topk`, `topk_softmax` and routing | router ops with logits `[T, experts]` (and a correction bias `[experts]`) | logits rows | experts, top-k; group counts are scalars the profiler does not record |
| `moe_sorting` | sorting ops, and sorting kernels of the fused MoE op | rows of `topk_ids` | top-k, experts and block size from the buffer sizes, output width |
| `fused_moe_fp8_blockscale`, `fused_moe_mxfp4` | ops holding the expert weights `[E, 2I, H]` and `[E, H, I]` in the task's weight format, the layer's sorting op and inter-stage requantization, summed per token count | tokens of the layer input | experts, hidden, intermediate size, top-k; weight format |
| `causal_conv1d_fwd`, `gdn_chunk_fwd_h`, `gdn_fused_recurrent_decode` | conv1d / gated-delta-rule ops or kernels; the chunk state pass also from the chunk pipeline op (`ChunkGatedDeltaRuleFunction`) at the state-pass kernel's share of its time; under speculative decoding the recurrent update is the verify step's replay-SSM kernel (r draft tokens per sequence) | packed tokens, sequence tokens or decode (verify) batch | channels and width; key and value heads and head sizes when the op records shapes |
| `paged_mqa_logits` | indexer logits of decode and verify steps (not the prefill indexer): q `[T, heads, head_size / 2]`, cache `[blocks, 1, 4, block_size, 16]`, block table `[B, max_seq_len / block_size + 4]` | `batch_size` = B, `next_n` = T / B, `ctx` from the step's `kv_tokens_per_seq` | heads, head size, block size, logits width |
| `flash_mla_sparse_decode_fp8` | sparse MLA over the FP8 sliding-window pool (and the compressed pool when the task has `extra_*` axes) | `batch` = query rows T (verify and draft tokens are one row each); `ctx` from `kv_tokens_per_seq` | heads, head dim, page size, slot width, index-table widths, presence of a compressed pool |
| `sparse_mla_prefill` | sparse MLA prefill over a bf16 latent pool: q `[T, heads, head_dim]`, indices `[T * topk]` | query tokens | heads, head dim, top-k |
| `topk_transform_paged` | indexer top-k over paged score rows `[T, width]` | score rows T | score width, page-table width, top-k |

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
showed it. A const axis that only sizes a pool or table the kernel
indexes into (`pages`, `num_kv`, `max_position`) does not reject a row: when
the capture's differs, the notes say so and the axis stays unchecked. The notes
also give the time of the operator's rows each mismatch rejected, with the
capture's values (for instance heads per tensor-parallel rank).

Register a matcher for a new operator family in
`src/tools/agentx_workloads/matchers.py`. A matcher reports the const axes it
could not check; confirm them by hand before using the proposal.

## Quality assurance

A proposal changes a task's workloads, so it needs the full task workflow:

1. Review the report: the matched shapes, the representative of each merged
   case, the unchecked const axes and the coverage.
2. Update the task README's workload section (model, capture, coverage, what
   became correctness-only).
3. Run `python3 -m src.tools.check_task tasks/<...>/<scenario>`.
4. Run the [QA auditor](qa-auditor.md) on MI355X. It enforces the 10 µs floor
   with the baseline's cold-cache timing; a case below it must become
   correctness-only and the proposal must be regenerated with `--baseline-us`.

Keep captures outside the repository; record the capture ID and its location
in the PR.

---
myst:
    html_meta:
        "description": "Compute a theoretical lower bound on the time of every AgentKernelArena schema-v3 performance case and compare it with measured baselines and candidates."
        "keywords": "AgentKernelArena, roofline, lower bound, floor, headroom, MI355X, GPU kernel"
---

# Estimate roofline floors

The optional `src.roofline` module computes a **floor** for every performance
case of the schema-v3 kernel tasks: a time no correct implementation can beat
on the target GPU. Joined with measured times it shows how much headroom a
kernel has left, helps choose tasks, and flags results that are faster than
physically possible.

It is an evaluation-side tool. It reads task packages and never changes them,
task code does not import it, it is not materialized into workspaces, and it
does not affect prompts or scores. It runs on the CPU and needs no GPU.

## The floor

```text
t_floor = max(sum_p F_matrix[p] / P_matrix[p],  F_vector / P_vector,  B_min / HBM_bandwidth)
```

Every counting rule errs low: fewer bytes, fewer FLOPs and higher peaks. A floor
can therefore understate the remaining headroom but never claim that none is
left when there is.

- **Bytes.** Rule G1 (every input read once, every output written once) applies
  when an operator has no mutable, state or index tensors; the evaluator poisons
  outputs before each call and compares them whole. Otherwise rule G2 counts
  weight, state, index and mutable tensors as zero bytes, because paged caches,
  routed experts and gathered rows may be touched only in part. Both rules leave
  out inputs whose declared extent is rounded up (`(m + 31) // 32 * 32`, or an
  axis derived with `ceil_div` or `max`), since padding need not be read.
- **FLOPs.** Operator models, registered by `operator.name`, count only the
  matrix products every correct implementation must perform. Epilogues are
  ignored, and algebraic shortcuts that a kernel may take are counted instead of
  the direct product (pooled weights, reassociation, merged linear maps).
  Convolutions use the direct count with the task's kernel geometry, less the
  taps that fall in zero padding; a 16-bit 3x3 stride-1 convolution is counted
  at the Winograd F(4x4, 3x3) multiply count (see `operators/conv.py`).
  Causality is not part of the task contract: attention models count causal
  pairs unless the task description says "non-causal" (a test pins which tasks
  match). MoE models
  rely only on the routed pair count `n_tokens * top_k` and on at least `top_k`
  distinct experts being used.
- **Peaks.** Each product is charged at the datasheet peak of its
  lowest-precision operand: bfloat16 activations against fp8 weights are
  charged at the fp8 peak. That is the faster peak, so the floor stays a lower
  bound whichever format a kernel computes the product in. `uint8` operands are treated as packed fp4. The
  bundled peaks are in `src/roofline/arch/mi355x_datasheet.yaml`, with their
  source. Products of float32 operands are marked
  `precision_substitution_sensitive`: a kernel can split each operand into bf16
  or fp16 parts and multiply those on the 16-bit matrix cores, and it is correct
  when the result still passes the task's tolerance. Which split passes is a
  measurement, so it stays outside the repository: `floors --precision-probe
  <file>` reads the results of a precision probe and charges each covered case
  at the cheapest arithmetic that passed for it (for example key
  `matrix_bf16x3`, three 16-bit products). A case without a probe result is
  charged at one bf16 product, the fastest arithmetic, because nothing has ruled
  it out; for most float32 tasks that floor is loose, and a probe tightens it.
  The file format is in `src/roofline/precision_policy.py`.
- **Timing.** Only measurements made with `cuda_graph_cold_cache` are joined:
  that method evicts the caches before every call, so every byte comes from HBM.

## Confidence tiers

| Tier | Meaning | Use |
|---|---|---|
| `verified` | The model's lower-bound argument was reviewed (listed in `REVIEWED_MODELS` in `src/roofline/floor.py`), it has unit tests and no measurement beats it | Reports, task selection, flagging |
| `estimated` | A generic rule or an unreviewed model | Reports and task selection, stated as an estimate |
| `none` | No work model for a compute-bound or data-dependent operator, or no bound at all | Not used |

Operators listed as memory-bound in `src/roofline/operators/tiers.py` use the
generic byte rules; any other operator without a model gets `none`, because a
bytes-only floor of a compute-bound operator would suggest headroom that does
not exist. A baseline measurement more than 5% faster than its floor means the
model is wrong: every floor of that model is downgraded to `none` and the case
is recorded in `violations.jsonl`. A candidate that fast is recorded there with
`role: candidate` as a suspect result, and the floor that flagged it stays. Floors
marked `precision_substitution_sensitive` can be beaten by a kernel that
computes in a lower precision and still passes the tolerance; review those by
hand instead of treating them as invalid.

## Run it

Every output directory must be outside the repository; the command refuses one
inside it.

```bash
python3 -m src.roofline floors --arch mi355x_datasheet --out <OUT> [--generate] \
    [--precision-probe <PROBE.json>]
python3 -m src.roofline join --floors <OUT>/floors.jsonl --evidence <EVIDENCE_DIR> --out <OUT> \
    [--practical-arch <MEASURED_PEAKS.json>]
python3 -m src.roofline report --joined <OUT>/joined.jsonl --out <OUT>
```

- `floors` writes `floors.jsonl` (one performance case per line) and
  `floor_failures.jsonl`. `--tasks <task-id> ...` limits it to some tasks.
  `--arch` also accepts the path of a peak file with the same keys. The floor
  must use datasheet peaks; measured peaks go to `join --practical-arch`.
- `floors --generate` also counts data-dependent operators (paged and ragged
  decode attention, MLA decode, paged and varlen prefill and extend attention,
  sparse MLA, MQA logits, routed MoE and LoRA GEMMs, masked or gathered
  rows, KV-cache scatters, recurrent-state decode, varlen chunk states, paged
  top-k, and the rows of a loss that are not ignored) from the inputs the evaluator would generate
  for each case: the framework seeds, the built-in distributions and the task's
  `generator`. The evaluator draws its inputs on the GPU, and torch's CPU and
  GPU generators give different streams for the same seed, so run it on a GPU
  inside the runtime image to reproduce the timed inputs. A CPU draw can hold
  more work than the timed one, so with `--device cpu` (the fallback when no GPU
  is visible) no data-model floor is emitted: the case keeps its shape-only floor,
  or tier `none` when there is none, with the note
  `data_model_requires_gpu_generation`. Of a case's two seeds the smaller floor is kept. When
  generation fails the case falls back the same way (its shape-only floor, or tier
  `none`), and the error is recorded in its notes. Each generated floor records its
  `generation` provenance: the runtime identity a task session records (torch, HIP,
  GPU, library versions and scoring-image references) and the digest of the task
  package.
- `join` searches `<EVIDENCE_DIR>` for task-session action records
  (`action-NNNN-baseline-performance.json`, `action-NNNN-candidate-performance.json`)
  beside a `session.json`, and writes `joined.jsonl`, `violations.jsonl` and
  `skipped_measurements.jsonl`. When the session's `runtime_identity.json` names a
  different torch build or GPU than a generated floor's `generation`, the joined
  record (and any violation) carries `generation_mismatch_<role>`: those inputs
  need not be the ones that were timed. `--practical-arch` takes a measured-peak file in
  the TraceLens arch JSON layout (the output of TraceLens's
  `PerfModel.benchmarking.microbench`, kept outside the repository) and adds
  `t_practical_us`, the same counts at the measured peaks, and
  `practical_fraction_baseline` to each case. A good kernel can beat measured
  peaks, so they never change the floor, the tier or the violations; a case
  that needs a peak the microbench could not measure (recorded as 0) gets no
  practical time.
- `report` writes `report.md` and `report.csv`: one row per task with its tier,
  bound, and the median and maximum of `floor / measured baseline` (datasheet)
  and, after `--practical-arch`, of `t_practical / measured baseline` side by
  side, with the baseline violations and the suspect candidates of each task.

Keep generated floors, joined measurements and measured peaks out of the
repository, as with [held-out evaluation](held-out-evaluation.md) data.

## Check it

```bash
python3 -m pytest -q tests/roofline
```

Set `TRACELENS_ROOT` to a TraceLens checkout in an environment that can import
it to also run `tests/roofline/test_tracelens_crosscheck.py`, which takes one
real performance case per Tier 2 operator family (GEMM, convolution, attention
and the composite blocks) and checks that our FLOP and byte counts equal
TraceLens's PerfModel formulas up to a stated, recomputed difference (scale
tensors, the causal diagonal, vector work and algebraic shortcuts). TraceLens is
not an AgentKernelArena dependency.

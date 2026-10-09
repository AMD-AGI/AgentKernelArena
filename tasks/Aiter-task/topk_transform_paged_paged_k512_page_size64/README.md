# Paged top-k transform to FlyDSL

Implement the DeepSeek-V4 decode top-k selection with page-table transform,
`k = 512` and `page_size = 64`. For each row `b`, with valid length
`L = seq_lens[b]`:

- if `L <= 512`, the selected positions are `0, 1, ..., L - 1` in this order;
- otherwise they are the 512 positions of `scores[b, :L]` with the largest
  scores, in any order;
- each selected position `p` is written as the slot
  `page_tables[b, p // 64] * 64 + p % 64`, and the remaining output slots are `-1`.

Only the valid prefix may influence the result. Scores outside it, page-table
entries beyond the pages it covers and stale destination contents must not.
Qualification scores are tie-free; NaN scores are unsupported.

The operator is destination-passing: the caller owns `out_page_indices`. The
configured builder signature is `builder(batch, width, pages, k, page_size) ->
launch`, invoked with keyword arguments. Launch takes the definition's inputs in
declared order followed by the destination, writes every element of the
destination and returns `None`:

```python
launch(scores, seq_lens, metadata, page_size, page_tables, out_page_indices) -> None
```

| Argument | Shape / value | Dtype |
| --- | --- | --- |
| scores | `[batch, 262208]` | float32 |
| seq_lens | `[batch]`, valid lengths in `[0, 262208]` | int32 |
| metadata | `[batch + 1, 2]`, v2 routing plan | int32 |
| page_size | `64` | Python int |
| page_tables | `[batch, 4097]` | int32 |
| out_page_indices (destination) | `[batch, 512]` | int32 |

`metadata` row 0 holds `(cluster_threshold, N)` and rows `1..N` hold the
`(batch_id, seq_len)` of exactly the rows longer than the threshold. Every
input carries the bundle initializer's plan, which routes no row
(`cluster_threshold = 2^31 - 1`, `N = 0`) and therefore matches every legal
length; the runner verifies that it matches each checked or timed input. A
candidate may read or ignore the plan, but must produce the result above for
whatever lengths `seq_lens` holds on that call.

The baseline entry is the sglang paged top-k v2 kernel,
`sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2`, with
materialized sources at `sglang_source/kernels/ops/attention/dsv4/topk.py`,
`sglang_source/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` (the kernels and host
dispatch) and `sglang_source/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`
(the selection algorithm, with the `sgl_kernel` headers it includes). The bundle's baseline
names it `topk_transform_512_v2`, its name before an sglang rename that did not
change it. `scripts/task_baseline.py` binds whichever of the two names the
installed sglang exports, preferring the current one, and leaves the bundle's
call unchanged; the runner records the executed entry point. The runtime must
provide `sglang.kernels.ops.attention.dsv4.topk` with one of these names.

`task_initialize.py` draws each score row as a random permutation of
`0, 1/width, ..., (width - 1)/width`, each page-table row as a random
permutation of the pages, lengths of 0, a quarter, half or all of the capacity,
and the plan above. `task_reference.py` selects by a stable descending sort and
maps through the page table. `task_compare.py` requires, per row, **exactly**
the reference's set of slots, `-1` in exactly the reference's padding
positions, and the reference's order for rows with padding; it also requires
matching shape, dtype and device.

## Cases

The workload's 13 batch sizes (1 through 4096, powers of two) keep their
capacities: score width 262208, 4097 pages of 64 tokens, 512 output slots. The
valid lengths are values inside `seq_lens`, so the cases vary them per batch
size while every shape stays fixed:

- `batch_<B>_bundle`: the initializer's own lengths for the workload seed;
- `batch_<B>_len_<L>`: every row has length `L`, for each of the boundary
  lengths 0, 1, 63, 64, 65, 511, 512, 513, 1024, 2048, 4096, 8192, 65536,
  131072, 262207 and 262208 (empty, short, around the page and output size,
  intermediate, long, near capacity and full);
- `batch_<B>_mixed` (B > 1): row `r` has boundary length `(5r + 7) mod 16`, so
  each batch mixes all regimes across its rows.

All 233 cases are checked for correctness and timed. Case rows carry the batch
size, the length descriptor, the seed and the shapes; reports give each case's
correctness and both roles' latencies.

## Task contract

`config.yaml` is the only task configuration. `evaluation.workloads` locates the
protected case data; `candidate.entrypoints` declares the candidate file and the
exact builder symbol. The symbol is not derived from the operator identity.
The initial candidate is an empty module. The agent must implement the operator
in FlyDSL in the declared editable file. All other task files are protected.

The builder receives shape arguments only and returns a callable launch. The
runner builds one launch per shape and calls it for every case of that shape,
with lengths changing between calls. Prepare compilation, shape-dependent tile
choices and reusable scratch in the builder. Every launch must compute the
complete operator on the supplied current tensor contents, including the
current `seq_lens`; it must not cache answers or alter input tensors. No state
kept across launches may be derived from `scores`, `seq_lens`, `metadata`,
`page_tables` or the destination. Recognizing inputs seen before and returning
a stored or partial result games the measurement; it is not an optimization.
Tiling, split reductions and per-length dispatch are implementation choices.

## Baseline numerical policy

The reference is the bundle's FP32 PyTorch computation and its comparator. The
production baseline is the performance reference and is not required to meet
that comparison: `config.yaml` declares `correctness_policy: diagnostic` for the
deepseek-v4-flash tasks. Baseline correctness still reports every case's actual
PASS/FAIL, and baseline timing keeps the full comparison of its timed outputs.
Only a completed finite numerical mismatch is accepted; crashes, missing cases,
invalid outputs and input mutation remain failures. Candidates have no
exception: candidate correctness and every checked timed invocation must pass
the comparator.

## Baseline, reference and dependencies

The production baseline calls the **installed sglang package**. The framework
materializes the image's `sglang/kernels/ops/attention/dsv4` and
`sglang/kernels/jit/csrc/deepseek_v4` subtrees under `sglang_source/`, excluding
`__pycache__`. This is read-only explanatory code, not an installation or build
input; it must not shadow the installed package. The selected Docker runtime
must provide sglang (with the entry point above), FlyDSL and ROCm PyTorch on
gfx950. The runner records runtime versions, the executed baseline module and
kernel source hashes and the entry point. The original initialize, reference
and compare callbacks are included unchanged under `scripts/`.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator or its selection, including
PyTorch `topk`, `sort`, `argsort` or `kthvalue`, is forbidden. Importing sglang,
sgl_kernel, AITER, protected task modules, other local modules or another GPU
computation library is forbidden. Loading code or task files dynamically is
also forbidden. The AST guard checks direct imports, imported members/aliases
and torch-rooted references to these selections. It is a guard against
ordinary violations, not a proof against arbitrary Python reflection or
tensor-method calls. Numerical and timed-path checks and evaluator review also
remain required.

## Evaluation

From a materialized task workspace, the public CLI supports:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Each command emits exactly one `ARENA_EVAL_RESULT=` JSON line and exits nonzero
on failure. All case rows retain the same ID, shape, dtype and semantic params.
`validate-task` enumerates the complete protected manifest before executing any
candidate. For every case it builds the inputs, verifies the lengths, the plan,
the page mapping and tie-free finite valid scores, and checks the reference's
output contract; in framework `task_validation` phase it verifies the actual
initial stub and reports `metadata.candidate_state`. An absent candidate,
missing builder or `NotImplementedError` always fails candidate actions, in
both phases; only the framework can defer candidate checks for an initially
empty task.

Before every invocation, outside timing, the destination is filled with `-2`,
a value no legal output contains, so every invocation has to write all of its
output itself. Compilation executes each case once. Correctness uses the
task's original comparison, independently for the requested role. Each batch
size's cases run in manifest order on one set of buffers and one launch, so
the lengths a launch sees grow from call to call on the same storage. Every
case is checked on the workload seed and on two further data draws seeded from
the operating system; each `mixed` case additionally checks one set of per-row
lengths drawn from the operating system's entropy (uniform, log-uniform and
boundary-adjacent), never fixed in advance.

Both roles use the materialized canonical `_aka_benchmark.py` helper with the
unchanged workload warmup, repetition and target duration. Input generation,
input validation, reference computation and plan construction occur outside
timing; the timed callable is the complete operator invocation writing the
destination. CUDA-graph timing is used; event fallback is recorded and the
framework checks that baseline and candidate timing methods match. Each sample
times one logical invocation; calls are never batched into one capture. Before
each sample, outside timing, the scores and page table are overwritten in place
with one of several draws from the bundle's initializer, while the case's
lengths and plan stay fixed. Draw seeds come from the operating system when the
case is timed. After the samples, the timed unit runs once over each of several
further draws it has never read, timed like a sample, and the fastest of those
may take at most `UNSEEN_DRAW_MARGIN` (in `scripts/task_measure.py`) times the
reported mean. The outputs of randomly chosen reported samples and of every
unseen-draw invocation are compared, with the original comparator, against the
reference on the draw each one consumed, and the inputs must be unchanged
afterwards. Input mutation, wrong selections, missing writes or runtime
failures fail the case. No timing from an instrumented sanitizer build may
become an official score.

A task does not require any agent-specific driver or environment variable.
Forge and other adapters invoke these same commands; their private search
protocols do not change the task's comparison or measurement policy.

## Export

The common Arena post-processing stage invokes `exports[].command` after final
acceptance. `scripts/export_solution.py` reads the framework-owned
`task_result.yaml`, the declared candidate and workload paths, and the protected
`solution.json` template; it writes only the declared artifact output. It
requires compilation, correctness, tool policy and complete comparable timings.
It rejects failed/stub candidates and never computes or writes Arena scores.
The artifact includes the candidate and a destination-passing tensor-call
wrapper with the baseline's signature, binding the declared builder per shape.
There is no external publication or sync. Consumers must support the
artifact's `flydsl` language; exporting alone does not prove compatibility or
acceptance by a separate SIKL installation.

## Validation controls

`validate-task` executes independent known-answer fixtures and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls` and do not add or remove scored cases. A failed
control fails task validation. Expected answers come from scalar Python
selection and page mapping written from the rule above, not from the reference.

- A `k = 4`, page size 4 fixture with lengths 0, 3, 4 and 11 checks the empty
  row, prefix order with padding, the exactly-`k` row and a long row spanning
  several permuted pages.
- A fixture at the task's `k = 512` and page size 64, with lengths 512 and
  700, checks the bound reference at the output-size boundary.
- The comparator must accept the exact answer and a reordered long row, and
  reject a wrong long-row selection, a reordered short row, moved or missing
  padding, a duplicated slot, a wrong dtype and a wrong shape.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation.

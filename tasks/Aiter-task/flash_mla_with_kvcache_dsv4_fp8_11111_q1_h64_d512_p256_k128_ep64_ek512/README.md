# DSv4 sparse flash MLA to FlyDSL

Implement DeepSeek-V4 sparse decode attention over packed FP8 shared-KV pools,
with one query token, 64 heads of dimension 512, attention sinks and the
log-sum-exp. Every definition has a main sliding-window pool (`kv_cache`,
`sparse_indices`, `sparse_lens`); a definition that declares `extra_*` inputs
also has an extra pool with its own page size and index width. The protected
workload JSON gives the dimensions, capacities, scalar value and cases.

For each batch row `b`, the selected KV entries are, in each pool, the
nonnegative indices among the first `lengths[b]` entries of the row's index
table; an index is a slot `page * page_size + position` of that pool. Negative
indices inside a prefix select nothing, and entries past a prefix must not
influence the result. With `S` the selected slots of both pools, decoded to
512-dimensional vectors `kv[s]` that serve as both key and value:

```text
logit[h, s] = (q[b, 0, h, :] . kv[s]) * sm_scale
output[b, 0, h, :] = sum_s exp(logit[h, s]) * kv[s] / (sum_s exp(logit[h, s]) + exp(sinks[h]))
lse[b, 0, h]      = log(sum_s exp(logit[h, s]))           # the sink is excluded
```

A row with no selected entry has a zero output and `lse = +inf`. The output is
BF16 and the LSE FP32.

**Packed pool layout.** A cache is `[pages, page_size, 1, 584]` bytes. Each
page stores `page_size` payloads of 576 bytes, then `page_size` scale records
of 8 bytes. A payload holds 448 E4M3 codes followed by 64 BF16 values; a scale
record holds 7 exponent bytes, one per 64 codes. Decoding is a literal BF16 bit
assembly, not FP8 multiplication: a code's sign bit, its four exponent bits
plus `scale - 7` as the BF16 exponent, and its three mantissa bits as the top
BF16 mantissa bits. A zero code therefore decodes to `2^(scale - 134)`, not to
zero. The 64 BF16 values are the last 64 dimensions unchanged.

The configured builder signature is `builder(**axes) -> launch`, invoked with
the case's dimensions as keyword arguments (`batch`, `query_length`, `heads`,
`head_dim`, `pages`, `page_size`, `packed_width`, `topk`, and for an extra pool
`extra_pages`, `extra_page_size`, `extra_topk`). Launch takes the definition's
inputs as positional arguments in declared order and returns `(output, lse)`:

```python
launch(q, kv_cache, sparse_indices, sparse_lens,
       [extra_kv_cache, extra_sparse_indices, extra_sparse_lens,]
       sm_scale, sinks) -> (output, lse)
```

| Argument | Shape / value | Dtype |
| --- | --- | --- |
| q | `[batch, 1, 64, 512]` | bfloat16 |
| kv_cache / extra_kv_cache | `[pages, page_size, 1, 584]` packed bytes | float8_e4m3fn storage |
| sparse_indices / extra_sparse_indices | `[batch, 1, topk]` / `[batch, 1, extra_topk]` | int32 |
| sparse_lens / extra_sparse_lens | `[batch]`, prefix lengths within the index width | int32 |
| sm_scale | `1/sqrt(512)` | Python float |
| sinks | `[64]` | float32 |
| output (out) | `[batch, 1, 64, 512]` | bfloat16 |
| lse (out) | `[batch, 1, 64]` | float32 |

The baseline entry is
`sglang.kernels.ops.attention.dsa.tilelang_kernel.dpsk_v4_fp8_attention_fwd`,
with materialized source at
`sglang_source/kernels/ops/attention/dsa/tilelang_kernel.py`. The runtime must
provide this entry point and its tilelang backend.

`task_initialize.py` draws queries at a moderate logit scale, random E4M3
payloads, BF16 rope values and scale bytes 124 to 127 per pool page, uniform
legal slot indices, prefix lengths (row 0 full, row 1 empty, others random)
padded with -1, and normal sinks. `task_reference.py` decodes the selected
slots and computes the attention in FP32. `task_compare.py` requires the
output within additive `atol = rtol = 1e-2`, the finite LSE within additive
`1e-3`, `+inf` LSE exactly on the reference's empty rows, and matching shapes,
dtypes, device and finite values.

## Cases

The workload's 13 batch sizes (1 through 4096, powers of two) keep every
capacity: pool pages, page sizes and index widths. Valid lengths are values
inside the length vectors, so the cases vary them per batch size; the length
grid in the workload JSON holds the boundary lengths of each pool (main: 0, 1,
63, 64, 65, 127, 128; extra: 0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513,
1024, 2048, 4096, 8192, 8255, 8256, limited to the extra index width). For each
batch size:

- `bundle`: the initializer's own lengths;
- `s<S>[_e<E>]`: every row has main length `S` (and extra length `E`): each
  main boundary length with the extra prefix full, each extra boundary length
  with the main prefix full, both prefixes empty, and for the C128 definition
  (`ep2_ek8256`) the decode trajectory `(min(N, 128), max(N // 128, 1))` for
  `N` = 4096, 8192, 16384 and 32768;
- `holes`: full prefixes with every third entry `-1`;
- `short_history` (extra pool): main length 64 and an extra length of 1 whose
  only entry is `-1`, a history too short for any extra KV entry;
- `tail`: partial prefixes followed by legal slots instead of `-1`, which must
  not affect the result;
- `mixed` (batch > 1): row `r` takes combination `(5r + 7) mod n` of the `n`
  grid combinations, so rows mix every regime and large batches cover the full
  grid product.

All cases are checked for correctness and timed. Case rows carry the batch
size, the length descriptor, the seed and the shapes; reports give each case's
correctness and both roles' latencies.

## Inputs of a case

One full initializer run per batch shape provides the base draw. A case writes
its lengths and then updates the index tables that depend on them: a prefix
position the initializer padded with `-1` (its own random length was shorter)
receives a legal slot drawn by the initializer's index rule, uniform over the
pool's slots; positions past a prefix are `-1`, except in a `tail` case; a
`holes` case then sets its declared positions to `-1`. The runner verifies
these properties for every checked or timed input.

A further draw of the call-varying operands (queries, the main pool and both
index tables) runs the initializer without its optional extra pool, which
redraws the queries, the main pool and its indices; the extra indices are drawn
by the same uniform slot rule, and the case's lengths and index pattern are
then applied. The extra pool and the sinks stay those of the batch shape's base
draw, like a deployed KV cache and model parameter.

## Task contract

`config.yaml` is the only task configuration. `evaluation.workloads` locates the
protected case data; `candidate.entrypoints` declares the candidate file and the
exact builder symbol. The symbol is not derived from the operator identity.
The initial candidate is an empty module. The agent must implement the operator
in FlyDSL in the declared editable file. All other task files are protected.

The builder receives shape arguments only and returns a callable launch. The
runner builds one launch per shape and calls it for every case of that shape,
with lengths and index tables changing between calls. Prepare compilation,
shape-dependent tile choices and reusable scratch in the builder. Every launch
must compute the complete operator on the supplied current tensor contents; it
must not cache answers or alter input tensors. No state kept across launches
may be derived from the queries, either pool, the index tables, the lengths,
the sinks or outputs. Recognizing inputs seen before and returning a stored or
partial result games the measurement; it is not an optimization. Tiling, split
reductions and per-length dispatch are implementation choices.

## Baseline, reference and dependencies

The production baseline calls the **installed sglang package**. The framework
materializes the image's `sglang/kernels/ops/attention/dsa` subtree at
`sglang_source/kernels/ops/attention/dsa`, excluding `__pycache__`. This is
read-only explanatory code, not an installation or build input; it must not
shadow the installed package. The selected Docker runtime must provide sglang
with the entry point above, tilelang, FlyDSL and ROCm PyTorch on gfx950. The
runner records runtime versions and the executed baseline module's source hash.
The original initialize, reference, compare and baseline callbacks are
included unchanged under `scripts/`.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator or its parts, including
PyTorch matrix products, softmax, log-sum-exp, exp or scaled dot-product
attention, is forbidden. Importing sglang, sgl_kernel, tilelang, flash_mla,
AITER, protected task modules, other local modules or another GPU computation
library is forbidden. Loading code or task files dynamically is also
forbidden. The AST guard checks direct imports, imported members/aliases,
common matrix-product forms and torch-rooted references to these operations.
It is a guard against ordinary violations, not a proof against arbitrary
Python reflection or tensor-method calls. Numerical and timed-path checks and
evaluator review also remain required.

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
candidate. For every case it builds the inputs, verifies their lengths and
index tables, and checks the reference's output contract, including zero
output and `+inf` LSE on rows without KV entries; in framework
`task_validation` phase it verifies the actual initial stub and reports
`metadata.candidate_state`. An absent candidate, missing builder or
`NotImplementedError` always fails candidate actions, in both phases; only the
framework can defer candidate checks for an initially empty task.

Compilation executes each case once. Correctness uses the task's original
comparison, independently for the requested role. Each batch size's cases run
in manifest order on one set of buffers and one launch, so the lengths a launch
sees change from call to call on the same storage. Every case is checked on the
base draw and on two further draws seeded from the operating system; each
`mixed` case additionally checks one set of per-row lengths per pool drawn from
the operating system's entropy (uniform or boundary-adjacent), never fixed in
advance.

Both roles use the materialized canonical `_aka_benchmark.py` helper with the
unchanged workload warmup, repetition and target duration. Input generation,
input validation and reference computation occur outside timing; the timed
callable is the complete operator invocation, including its output
allocation. CUDA-graph timing is used; event fallback is recorded and the
framework checks that baseline and candidate timing methods match. Each sample
times one logical invocation; calls are never batched into one capture. Before
each sample, outside timing, the call-varying operands are overwritten in place
with one of several draws, while the extra pool, sinks and lengths stay fixed.
Draw seeds come from the operating system when the case is timed. After the
samples, the timed unit runs once over each of several further draws it has
never read, timed like a sample, and the fastest of those may take at most
`UNSEEN_DRAW_MARGIN` (in `scripts/task_measure.py`) times the reported mean.
The outputs of randomly chosen reported samples and of every unseen-draw
invocation are compared, with the original comparator, against the reference
on the draw each one consumed, and the inputs must be unchanged afterwards.
Input mutation, wrong results or runtime failures fail the case. No timing from
an instrumented sanitizer build may become an official score.

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
The artifact includes the candidate and a tensor-call wrapper with the
baseline's signature, binding the declared builder per shape. There is no
external publication or sync. Consumers must support the artifact's `flydsl`
language; exporting alone does not prove compatibility or acceptance by a
separate SIKL installation.

## Validation controls

`validate-task` executes independent known-answer fixtures and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls` and do not add or remove scored cases. A failed
control fails task validation. Expected answers are literal format facts and
scalar Python attention written from the rules above, not reference outputs.

- A packed slot with known codes, scale bytes and BF16 values must decode
  exactly, including scaled codes and unset codes under their group's scale.
- A six-row fixture checks attention over the main pool, a `-1` inside a
  prefix, legal entries past both prefixes, a row with no entry (zero output,
  `+inf` LSE), both pools together, and a short history whose one extra entry
  is `-1`; definitions without an extra pool use its main-pool rows.
- The comparator must accept a nonzero error within its gate in both outputs
  and reject larger output or LSE errors, finite LSE on an empty row, `+inf` on
  a nonempty row, NaN output or LSE, a wrong dtype or shape, and a missing LSE.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation. The fixture's rounding bounds do not alter the candidate tolerance.

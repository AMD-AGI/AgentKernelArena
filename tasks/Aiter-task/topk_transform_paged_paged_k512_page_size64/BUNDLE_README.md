# Paged Top-k with K=512 and page size 64

Implement `run(**kwargs)` in [source/kernel.py](source/kernel.py) using Triton.
The [workload manifest](scripts/workload.json), [reference](scripts/reference/main.py),
and [comparator](scripts/compare/main.py) define the protected contract. Only
files listed in [config.yaml](config.yaml) are editable during optimization.

## Interface and semantics

`scores` is FP32 `[batch, 262208]`; `seq_lens` is int32 `[batch]`;
`page_tables` is int32 `[batch, 4097]`; `metadata` is int32 `[batch + 1, 2]`.
`page_size` is the scalar 64. Write the preallocated int32 destination
`out_page_indices[batch, 512]` and return `None`. Do not mutate inputs.

For each row, consider only `scores[row, :seq_lens[row]]`. Valid lengths range
from zero through 262208. For lengths at most 512, emit every valid logical
index in ascending logical order; this includes **length exactly 512**. For
longer rows, select the 512 highest scores in any order. Map each selected
logical index `i` to `page_tables[row, i // 64] * 64 + i % 64`. Fill all unused
output positions at the end with `-1`. Tie-free inputs make the selected set
unambiguous. NaN scores are unsupported.

The comparator uses the actual `seq_lens` to decide whether logical ordering
is required. Inferring short rows solely from `-1` padding misses the length-512
boundary. Long rows are compared as sets, with exact integer values.

## Runtime-length coverage

All 13 original batch sizes and workload rows remain. Additional manifest cases
cover these lengths while score and page-table capacities stay fixed:

```text
0, 1, 63, 64, 65, 511, 512, 513,
1024, 2048, 4096, 8192, 65536, 131072, 262207, 262208,
37, 65537
```

Batch 1 exercises every length independently. Larger batches use uniform empty,
length-512 and full rows, plus mixed patterns that collectively include every
listed length for each batch size. One additional case per batch uses a
reproducible pseudorandom length pattern whose seed and exact values are retained
in the manifest. `runtime_lengths.seq_lens` is a repeating
pattern across batch rows; its values are explicit in both workload rows and
case parameters. The 13 original cases retain their original input draws and
initial length distributions.

The protected input adapter applies each declared length after initialization.
It rebuilds the conservative non-cluster metadata in the existing buffers.
Correctness checks use two data seeds per case. Performance measures every
manifest case, checks the exact timed output, then validates the captured graph
after a second data draw and after changing the length pattern in place.
`runtime_lengths.replay_seq_lens` declares that final transition; lengths grow
by one, or reset from full capacity to zero. Neither buffers nor strides change.

## Timing and validation

Input generation, metadata setup, reference computation and output allocation
remain outside kernel-only timing. Both roles use the same warmup, repetition,
case manifest, and canonical device-timing helper. Each measured input is checked
before timing. Outputs are poisoned before replay validation, and input mutation
is rejected. Per-case latency, timing method, length pattern, and replay checks
are emitted through the task runner.

The numerical reference is independent of the production baseline. The baseline
and initial candidate compile separate copies of the same SGLang production HIP
kernel with a task-local overflow fix. The upstream bounded candidate scratch
loses valid long-row selections for the original input distribution; overflowing
bins now use exact FP32 radix refinement over the complete valid prefix.
[scripts/baseline/main.py](scripts/baseline/main.py) defines the production loader.
See [UPSTREAM.md](UPSTREAM.md) for the immutable public source, retained license,
local changes, compiler dependencies, cache isolation and timing boundaries.
There is no fallback to the reference or another backend. The final candidate
must implement its own Triton computation; the initial Python-to-HIP wrapper is
not a completed rewrite.

Source inspection and CPU regression tests do not qualify the task; a fresh
framework-finalized GPU validator PASS is required.

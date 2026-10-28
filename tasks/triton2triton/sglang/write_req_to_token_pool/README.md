# Request-to-token pool writer

Optimize `write_req_to_token_pool_triton` in `kernel.py`. The supplied Triton
implementation is the frozen initial-candidate baseline. All other files are
protected. Keep the entrypoint, argument order and in-place behavior.

For request `i`, write the first `pre_lens[i]` int64 values from the device
pointer `prefix_tensors[i]` to the selected int32 table row, then append
`seq_lens[i] - pre_lens[i]` values from `out_cache_loc`. Its offset is the sum of
earlier requests' `extend_lens`; `extend_lens == seq_lens - pre_lens`. Row indices
are unique, lengths are valid, and token slot values fit int32. Preserve every
other cell, including unselected rows, row suffixes and physical row padding.
All inputs and pointed-to prefix buffers are read-only. Empty prefixes and
extensions are valid. The last argument is the physical row stride in elements.

The scored cases retain the original c2/c32/c64 request counts, 2048-by-202756
table, sequence length 1024, prefix length 16, seed and RNG order. Additional
correctness-only cases cover mixed lengths, zero lengths, prefix-only and
extension-only requests, and 511/512/513 boundaries. The independent CPU oracle
requires exact equality of the full backing table; no floating-point tolerance
applies. CPU known-answer and deliberately incorrect-output controls also run.

Performance uses the shared cold-cache graph protocol: ten warmups, 100 samples,
one complete operator per graph replay, 1 GiB read-only cache eviction, and
same-stream GPU Events. Before every invocation, preparation restores untouched
cells and poisons all required writes. Preparation, eviction, compilation and
oracle work are outside timing. Each measured output is checked. The captured
graph must also pass three changed-input replays, including prefix contents,
cache values, row mapping and lengths. Graph capture failures are fatal.

This replaces the original warm Event benchmark and therefore establishes a new
benchmark identity. Old timings are not comparable. Both baseline and candidate
must be measured with this harness. This task measures table writes, not an
end-to-end serving workload.

The task-owned runner implements `validate-task` and both roles' `compile`,
`correctness` and `performance` actions through the shared schema-v2 evaluator.
`compile` syntax-checks all Python files and loads the Triton entrypoint; the
GPU correctness action compiles and executes the JIT for the declared cases.
See [SOURCE.md](SOURCE.md) for source attribution and [LICENSE](LICENSE).

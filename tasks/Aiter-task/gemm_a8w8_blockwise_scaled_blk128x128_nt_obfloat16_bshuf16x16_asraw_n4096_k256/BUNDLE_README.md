# Flash MLA and Top-k: Accuracy and Performance Tests

**Kernels must generalize across valid lengths, not just tensor shapes or the supplied samples.** As the KV cache grows during decode, effective lengths change even when buffer capacities stay fixed.

| Operator | Runtime length values to vary | Capacity stays fixed within a test sweep |
| --- | --- | --- |
| `topk_transform_paged` | `seq_lens[b]` | Score width, page-table width, output `k` |
| `flash_mla_with_kvcache` | `sparse_lens[b]`, plus `extra_sparse_lens[b]` when present | KV pools and sparse-index widths |

These are **values inside tensors**, not changes to the length tensors' shapes. Length growth alone does not require changing constant axes such as `extra_topk=8256` into variable axes.

## Top-k example

Case: [`topk_transform_paged_paged_k512_page_size64`](definitions/topk_transform_paged/topk_transform_paged_paged_k512_page_size64.json), using its [batch-1 workload](workloads/topk_transform_paged/topk_transform_paged_paged_k512_page_size64.jsonl).

Keep these shapes fixed:

```text
scores:           [1, 262208]
page_tables:      [1, 4097]     page_size = 64
metadata:         [2, 2]
out_page_indices: [1, 512]
```

Vary `seq_lens[0]`, for example:

```text
0, 1, 63, 64, 65, 511, 512, 513,
1024, 2048, 4096, 8192, 65536, 131072, 262207, 262208
```

For a request with current visible token length `N`, this decode path uses `seq_lens = N // 4`. Thus `N = 4096 → 8192 → 16384 → 32768` gives `seq_lens = 1024 → 2048 → 4096 → 8192`, while the output remains 512 slots. **Do not clamp the candidate length to 512.**

- Keep `0 <= seq_lens[b] <= min(width, pages * page_size)`.
- Build or validate matching `metadata` for every length; never reuse a stale length-specific plan.
- Vary tie-free scores and legal page mappings. NaN scores are unsupported.
- Select only from the valid score prefix. Short rows preserve logical order and end with `-1` padding; long-row selections need not be sorted. Use the definition's `compare` callback.

## Flash MLA example

Case: [`flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256`](definitions/flash_mla_with_kvcache/flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256.json), using its [batch-1 workload](workloads/flash_mla_with_kvcache/flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256.jsonl).

Keep these shapes and both KV pool capacities fixed (`pages=2019`, `extra_pages=201836`):

```text
q:                    [1, 1, 64, 512]
sparse_indices:       [1, 1, 128]
extra_sparse_indices: [1, 1, 8256]
```

In this C128 case, `sparse_lens = min(N, 128)` and `extra_sparse_lens = max(N // 128, 1)`, within the configured legal capacity:

| Current visible tokens `N` | `sparse_lens[0]` | `extra_sparse_lens[0]` |
| ---: | ---: | ---: |
| 4096 | 128 | 32 |
| 8192 | 128 | 64 |
| 16384 | 128 | 128 |
| 32768 | 128 | 256 |

The main sliding window saturates at 128; the extra prefix continues growing. **8256 is index capacity, not the effective length of every call.**

Also test legal boundary combinations independently, including both branches empty, either branch empty, and both branches nonempty:

```text
sparse_lens:       0, 1, 63, 64, 65, 127, 128
extra_sparse_lens: 0, 1, 2, 31, 32, 33, 127, 128, 129,
                   511, 512, 513, 1024, 2048, 4096, 8192, 8255, 8256
```

- Keep `0 <= sparse_lens[b] <= 128` and `0 <= extra_sparse_lens[b] <= 8256`.
- Apply each length to its index prefix, then ignore negative indices inside that prefix. For short histories, `extra_sparse_lens=1` with index `-1` means no effective extra KV entry.
- Vary Q, correctly packed KV data, legal indices, negative padding, and sinks. Test that indices outside the selected prefixes cannot affect outputs.
- Check both BF16 output and FP32 LSE with the definition's `compare` callback: additive `atol=rtol=1e-2` for output and `1e-3` for finite LSE. Fully empty rows require zero output and `+inf` LSE.

## Required test coverage and reporting

- **Accuracy:** use the exported `reference` and operator-specific `compare` callbacks for every tested input. Cover empty, short, intermediate, near-capacity, and full lengths; add values immediately around kernel tile/partition boundaries and random lengths not used during tuning. The lists above are starting points, not an exhaustive whitelist.
- **Data and batch generalization:** use multiple data seeds per length. Repeat across supported workload batch sizes, including mixed per-row lengths. Reuse buffers across growing-length calls to catch stale length or plan assumptions.
- **Performance:** benchmark the same length regimes, including irregular lengths and mixed batches—not only full capacity. Validate each measured input first, warm up, and use repeated GPU-timed measurements under identical conditions for candidate and baseline. Keep input generation, reference checks, and metadata construction outside kernel-only timing; report setup cost separately if measured.
- **Report per case:** batch, length values/distribution, seed, accuracy result, baseline latency, candidate latency, and speedup. Include short-, medium-, and long-length results and regressions, not just a best case or one aggregate speedup. Record hardware, timing method, and repetition count.

**Changing seeds alone is insufficient.** The Top-k initializer selects only `0`, one-quarter, one-half, or full capacity. For batch 1, the MLA initializer always sets both prefixes to full width. Explicitly set the test lengths after initialization and update dependent inputs before checking or timing.

These are test requirements and example inputs, not measured accuracy or performance results.

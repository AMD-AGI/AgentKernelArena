# Ragged paged decode attention, BF16, 8 query heads / 1 KV head, head size 128, one-token pages

Single-token (decode) grouped-query attention over a token pool addressed by ragged (CSR) page
lists. For each sequence `s` and query head `h`,

```text
out[s, h] = softmax(scale * q[s, h] . K_s^T) . V_s
```

where the tokens of sequence `s`, in order, live in the pool slots
`kv_indices[kv_indptr[s] : kv_indptr[s + 1]]`, all eight query heads share the single KV head and
the result is rounded to bfloat16.

## Production origin

AITER's `paged_attention_ragged` (`csrc/cpp_itfs/pa/pa_ragged.py`) is the decode attention of
SGLang's AITER attention backend (`python/sglang/srt/layers/attention/aiter_backend.py`) for
non-MLA models. SGLang calls it with its token pool viewed as one-token pages
(`k_cache.view(-1, 1, kv_heads, head_dim)`, NHD layout, `block_size` 1), the per-request token
slots as `kv_indices` with `kv_indptr` offsets, a bfloat16 cache (`kv_cache_dtype` "auto") and a
partition size of 256. The scenario is 8 query heads and 1 KV head of size 128 per tensor-parallel
rank, the geometry of Llama-3.1-70B or Qwen2.5-72B served at TP=8.

The source task (`image_kernel/mi300x_sglang_hip_pa_ragged`) drove the same kernel through AITER's
own test layout: 16-token HND pages with `kv_last_page_lens`. This task uses SGLang's production
call instead (one token per page, so there is no last-page length input); the ragged context
lengths, their partition boundaries and the source task's shapes are kept in the workloads. SGLang
never engages the 16-token-page path (its call hard-codes one-token pages), and the source task's
float16 correctness cases with 4 query heads, 2 KV heads and head size 64 are AITER unit-test
geometries; dtype, head counts and page size are scenario constants, so those cases are not part
of this scenario.

## Interface and layouts

- `query[num_seqs, 8, 128]`: one decode token per sequence.
- `key_cache[num_slots, 1, 128]`, `value_cache[num_slots, 1, 128]`: the token pool,
  `num_slots = num_seqs * max_seq_len + 1`. Slot 0 is the pool's padding slot.
- `kv_indptr[num_seqs + 1]`: CSR offsets; sequence `s` has `kv_indptr[s + 1] - kv_indptr[s]`
  tokens.
- `kv_indices[num_seqs * max_seq_len]`: the pool slot of every token, sequence after sequence, in
  token order. The buffer has room for every sequence at `max_seq_len`, as a CUDA-graph decode
  buffer does; entries past `kv_indptr[num_seqs]` are 0.
- `scale`: `128 ** -0.5`. `max_context_len`: the longest context in the batch (it equals the
  `max_seq_len` axis in every case), a host-side bound with which the baseline sizes its partition
  grid, `ceil(max_context_len / 256)` partitions per sequence. SGLang instead passes the partition
  count of the model's maximum context length (512 partitions for a 128K-context model) and lets the
  workgroups past each sequence's end exit at once; this task follows vLLM's ROCm paged-decode path,
  which sizes the grid from the batch's longest sequence (`max_seq_len`), so the timing measures
  the batch's work rather than the model's context limit.

Context lengths are ragged: most sequences hold between half of `max_seq_len` and all of it, about
one in sixteen is a short request of 1 to 512 tokens, and one sequence holds exactly `max_seq_len`.
Sequences take distinct slots in random order, so their tokens are scattered through the pool.
Every slot that holds no token of any sequence, slot 0 included, is NaN: unwritten pool memory may
hold anything, and an implementation must not let it reach the result. (The baseline never reads
such a slot; for positions past a sequence's end it rereads the sequence's last token and masks
it.)

## Workloads

Batches span 1 to 1024 sequences and the longest context 1 token to 128K tokens. Timed cases go
from about 60 thousand cached tokens per step, the smallest expected to stay above the 10 us
timing floor, to about six million, close to what a 70B-class model's bfloat16 KV cache can hold
per GPU at TP=8 on MI355X. The source task's timed
shape (`s128_l4097`, 128 sequences up to 4097 tokens) is kept as a timed case and its other shapes
(`s128_l128`, `s16_l129`) as correctness cases, together with the boundary shapes (one-token
contexts, contexts at and one past a 256-token partition, partition counts past 64 that need two,
three and six reduce loops, and a 128K context at the kernel's 512-partition limit).

## Baseline

`baseline/kernel.hip` is AITER's HIP ragged paged-attention decode: `pa_ragged.cuh` and
`pa_kernels.cuh` hold the kernels (one workgroup per sequence, 256-token partition and KV head,
QK and PV on bfloat16 MFMA instructions, then a reduce over the partitions) and `kernel.hip` is the
host code of `pa_ragged.cpp.jinja` rendered for this scenario, with SGLang's argument handling from
`pa_ragged.py`. AITER builds these headers against its own and Composable Kernel's include trees;
`pa_compat.h` supplies the few definitions the kernel takes from them (the wavefront size, the
`Fp8KVCacheDataType` enum, and a non-masking `ck_tile` mask with the plain softmax-scale attention
variant), so the baseline needs only HIP. AITER JIT-compiles one library per partition-loop count;
the vendored host code instantiates the reduce for 1 to 8 loops and selects one at run time.
The reduce launch's last template argument (`ENABLE_LAST_PAGE_LENS`) is written as the literal
`false`, as `pa_ragged.cpp.jinja` renders it for one-token pages; a `static_assert` ties it to the
block size. torch's extension builder runs hipify on HIP sources, and hipify rewrites every
triple-chevron launch into `hipLaunchKernelGGL`. Its parser cannot handle a comparison operator
inside a launch's template arguments, and an expression such as `(kBlockSize > 1)` there fails the
build before the compiler runs. The kernels themselves are unchanged.

The AITER code is MIT-licensed; `baseline/LICENSE.h` holds that license as a comment-only header,
because a HIP implementation directory may contain only sources and headers. `pa_ragged.cuh` also
carries the vLLM team's Apache-2.0 notice.

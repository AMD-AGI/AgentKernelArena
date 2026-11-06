# Fused MoE LoRA delta, bfloat16, Mixtral-8x7B gate/up, rank 16

vLLM serves LoRA adapters on mixture-of-experts models by adding, per routed token and expert, a
low-rank update to the expert projections. This task is the gate/up (w13) update computed by
`fused_moe_lora` from `vllm/lora/ops/triton_ops/fused_moe_lora_op.py`: for each route (token `t`,
top-k slot `k`) whose token uses LoRA slot `l` and expert `e`,

```
lora_delta[t, k, s*N:(s+1)*N] = (hidden_states[t] @ lora_a[s, l, e]^T) @ lora_b[s, l, e]^T
```

for the gate (`s = 0`) and up (`s = 1`) slices, rounded to bfloat16 (shrink to rank 16, expand to
`N = 7168`). Routes whose token is a base-model token or uses an adapter without MoE weights get zeros.

## Scenario

Mixtral-8x7B (hidden 4096, 8 experts, top-2, expert intermediate 14336) at tensor parallel 2, so
`N = 7168` per slice; four resident adapter slots, maximum LoRA rank 16, bfloat16 weights. These are
the `mixtral` preset of vLLM's own kernel benchmark (`benchmarks/kernels/benchmark_fused_moe_lora_one_shot.py`),
whose LoRA weight scales (`randn / sqrt(hidden)`, `randn / sqrt(rank)`) and standard-normal activations
the task uses. The batch size `n_tokens` varies.

The routing arrives as vLLM's `moe_lora_align_block_size` tables with block size 64 (the benchmark's
`block_size_m`): per LoRA slot, `sorted_token_ids` lists that slot's routes grouped by expert, each group
padded to a multiple of 64 with the value `n_tokens * top_k`; `expert_ids` gives one expert per 64-entry
block (-1 past `num_tokens_post_padded`); `lora_ids` is vLLM's `active_lora_ids` (the sorted distinct
slots of the batch, -1 first when base-model tokens are present, padded with -1 to `max_loras + 1`),
and `adapter_enabled` marks slots whose adapter has MoE weights. Table sizes follow vLLM's
`moe_lora_align_block_size` (`max_num_tokens_padded = min(routes * 64, routes + 8 * 63)`).

`inputs.py` draws top-2 distinct experts per token, splits the batch into up to eight requests of
contiguous tokens (as the vLLM benchmark does), gives each request a random adapter slot or none, and
disables one random slot (an adapter that only targets attention, which vLLM's `reset_lora` leaves with
`adapter_enabled = 0`); the first request always uses an enabled slot so even one-token cases carry a
delta. Within an expert group the route order is random, as the CUDA alignment kernel places routes with
atomics. The alignment kernel writes only the rows of listed, enabled slots; vLLM allocates
`sorted_token_ids` and `num_tokens_post_padded` uninitialized and presets `expert_ids` to -1, so the other
slots' rows hold garbage token ids and -1 experts. The generator is stricter: it fills every other slot's
rows, `expert_ids` included, with a stale but in-range alignment of random routes, so an implementation
must honor `lora_ids` and `adapter_enabled` (as the vLLM kernel does) rather than walk every slot or rely on
-1 expert ids.

The output is the delta alone, the form vLLM's dual-stream MoE LoRA path computes (it hands the kernel a
`torch.zeros_like` buffer with `add_inputs=False` and adds the delta to the base projection afterwards);
the default single-stream path adds the same values into the base output in place.

## Baseline

vLLM's `_fused_moe_lora_one_shot_kernel` with its host wrapper `_run_fused_moe_lora_one_shot`, the
path `_fused_moe_lora` takes for block-sorted routing when LoRA is not fully sharded: one launch per call
whose programs each own a 64-route block of one (slot, expert), compute the rank-16 shrink in fp32 and
round it to bfloat16, then stream the expand over their share of the 7168 columns. `run()` zeroes the
delta buffer first, as the dual-stream path's `torch.zeros_like` does. Modifications are listed in
`task.yaml`; the kernel body is unchanged apart from addressing the stacked LoRA slices by stride
(upstream receives a tuple of per-slice tensors and loads their data pointers from a device table).

The previous task package used the older two-kernel shrink/expand path (`_fused_moe_lora_kernel`, still
used upstream for fully sharded LoRA) at toy sizes without a recorded upstream revision; this package
vendors the current production fast path at a fixed vLLM commit instead.

## Workloads and comparison

Batch sizes come from the vLLM benchmark's `M` range (16 to 16384) plus 128, 512, 2048 and a
32768-token batched prefill. The performance cases span 256 to 32768 tokens (128x). Decode-sized
batches up to 128 tokens and odd sizes are correctness-only: at 128 tokens the baseline moves about
33 MB (the active slots' LoRA weights, the delta and its zero fill), which puts it at the 10 us timing
floor. vLLM itself routes batches of at most two tokens through its unsorted ("naive") assignment;
the task keeps the block-sorted tables for every size, so `t1` checks the sorted path at its minimum.
`lora_delta` uses the default baseline-relative bound against the float64 reference, which does not
round the rank-16 intermediate; untouched routes must be exactly zero.

## Coverage relative to the previous task

The previous package exercised the generic `fused_moe_lora` wrapper at toy sizes: float16 scored cases
(bfloat16 in a control), the unsorted (naive) assignment, routing-weighted outputs (`mul_routed_weight`),
accumulation into a nonzero output at a column offset, odd ranks and split-K. This scenario fixes what
one production call fixes: Mixtral is served in bfloat16, the w13 projection never applies routing weights,
and its rank, widths and block size are constants. The routing-weighted path belongs to the down (w2)
projection, a separate scenario (`K = 7168`, `N = 4096`, one slice, hidden states per route) that this
package does not cover; the naive assignment and in-place accumulation are discussed above. The
correctness cases keep the routing edge cases that apply here: empty expert groups (`t1`, `t3`), multi-block
groups, unaligned and odd batches, base-model tokens, a disabled slot and unlisted slots.

# GLM-5.3 indexed decoder layer, tensor parallelism 8 (multi-GPU)

One launch per rank of sgl-amd-mono's `glm5_monokernel` runs a whole GLM-5.3 decoder layer that runs the DSA
indexer, for one decode or MTP-verify step (2, 4 or 8 rows), on 8 GPUs:

input RMSNorm -> block-FP8 q_a | kv_a | k_pe -> q_a RMSNorm -> q_b (+ interleaved RoPE) -> kv RMSNorm and k_pe
RoPE -> MLA cache write (bf16, ATOM layout) -> indexer (index query: wq_b, RoPE on 64 dims, 128-point Hadamard;
index key: wk, LayerNorm, RoPE, Hadamard, E4M3 with a power-of-two scale into SGLang's paged index cache; head
weights: weights_proj) -> score = scale x sum_h w_h relu(q_h . k) over every key up to the row's position ->
exact top-2048 -> absorbed sparse MLA over the selected keys (8 local heads, softmax scale 1/16) -> W_UV ->
o_proj -> TP sum + residual (in-kernel, peer memory) -> post-attention RMSNorm -> sigmoid router, top-8 of
256 with bias, weights x 2.5, plus the shared expert -> FP8 experts (per-128 E4M3 activations, SiLU(gate) * up,
per-128 E4M3 mid, down) -> TP sum + residual (in-kernel).

Each rank holds 8 of the 64 heads and 256 of the 2048 expert intermediates (SGLang's TP 8 shard); everything
else (q_a | kv_a, the indexer, the router, the caches) is replicated, and every rank writes the same outputs.

## Interface

`task.yaml` declares one rank's call; `distributed` (world size 8) adds `rank`, `world_size`, `peers` and
`symmetric`. The reference reads the *logical* inputs (the whole layer, row-major SGLang checkpoint layouts,
see `inputs.py`); `shard_inputs` slices them to the rank and applies the kernel layouts (AITER (16, 16)
shuffle of the in-place FP8 weights, the kernel's packed 16 x 64 tiles for W_UK, W_UV, the router and
weights_proj). `rank_expected` returns the full tensors (every output is replicated); `ranks_agree` checks
`out` and `topk` bitwise across ranks.

Outputs: `out` (bf16 [rows, 6144]) and `topk` (int32 [rows, 2048]: each row's selected cache slots, compared
as a set: `element_order: any_in_full_rows`, the kernel's order is its radix select's). Mutable: `kv_cache`
(the new rows) and `index_cache` (the new keys and scales).

Persistent-kernel state follows the multi-GPU contract: the guard block `step_guard` is a read-only input
(epoch 1, so every mailbox tag is 2; a 600 s wait deadline in 100 MHz ticks). The peer mailboxes and the
kernel's hand-off scratch live in the framework's symmetric buffer (zeroed before every call), sized
`rows * 11854336 + 4096` bytes (the layout's peer regions and scratch at index_max_seq 2^20: 90.4 MiB at 8
rows; most of it is the per-row index score buffer).

## Reference and decision margins

`reference.py` (math in `glm53_math.py`, shared with the generator) is float64 PyTorch over the whole layer,
with the quantizations the operator defines applied exactly (dequantized block-FP8 weights, the index key's
E4M3 / power-of-two-scale rule, per-128 E4M3 expert activations and mid, the bf16 MLA cache as the attention
reads it). It does not model the kernel's bf16 intermediates or its bf16 TP partials.

Two discrete decisions must not flip between the reference and a kernel; `inputs.py` plants margins and the
generator raises if they are missing:

* **Top-2048.** weights_proj is planted so every row's head weights are positive; each request's cached keys
  are tiers of per-token power-of-two scales so that every row's 2048th score is at least twice its 2049th:
  2048 - L keys far above everything, the request's own new keys (computed by the layer) next, L - 1 keys
  ordered 2x apart below them (a verify row that sees fewer new keys takes more of them), the rest below.
  Each verify row thus selects its visible new keys too, which exercises the kernel's path that scores and
  attends keys written in the same launch.
* **Router.** The router is planted on the rows' float64 post-attention states: each row's 8 experts (random)
  get logits 2..3, every other logit <= 0.6; the 8th/9th gap of sigmoid + bias is checked (>= 0.05).

## Compare bounds (QA-BOUND-OVERRIDE)

* `out`: `baseline_relative` with `k: 4`, `max_violation_fraction: 0.001`. The output passes two E4M3
  quantizations (expert input and mid) whose code flips land on different elements for different correct
  implementations, plus the kernel's bf16 attention residual and bf16 TP partials; the same relaxation as
  this repository's FlyDSL A8W4 MoE task.
* `kv_cache`: default `baseline_relative`. `topk`: exact (as a set).
* `index_cache`: default `baseline_relative` on decoded values, with
  `decode: {file: inputs.py, symbol: index_cache_values, precision: float8_e4m3fn}`. The hook maps each
  token's 128 E4M3 codes and f32 scale to the float64 key values (code x scale), and the evaluator compares
  the candidate's, the reference's and the baseline's decoded caches. Bytes are the wrong criterion: the
  kernel computes the index key at lower precision than the float64 reference, so a code can round one step
  the other way, and a scale that rounds to the other power of two rewrites every byte of the token without
  changing its values (on MI355X the baseline failed the byte-exact compare in every case, with about 10% of
  the new keys' bytes different). With `precision: float8_e4m3fn` the bound allows one E4M3 code step on top
  of twice the baseline's own per-token error, so such a code passes, while an unwritten, zeroed or wrong key,
  a wrong scale or a cached key changed by more than one code step fails. `max_violation_fraction` stays 0: almost all of the cache is
  unchanged state, so any fraction of it would exceed every newly written key.

## Workloads

Derived from corpus statistics; not yet confirmed by a live capture. Contexts are AgentX decode-step-weighted
quantiles (semianalysisai/cc-traces-weka-062126; p5 11K, p10 52K, p25 108K, p50 220K, p75 426K, p90 632K, p99
890K), clipped to GLM-5.3's 1M-position context (no AgentX request exceeds it). Rows are SGLang's widths 2
(two decode requests), 4 (one 4-token MTP chain) and 8 (two chains); only those three widths compile.
Other requests of a step hold ctx/2 .. ctx (at least 2048, so every row selects exactly 2048 keys). Ten
performance cases span 224x in rows x ctx; correctness adds the minimum (2 x 2048), a non-power-of-two
(4 x 3072), an unaligned (8 x 5000) and the cap (8 x 1048000) case. Prefill does not apply: the monokernel
serves decode and verify steps only.

Estimated baseline time per perf case (bytes at ~5 TB/s, one rank; not measured): weights ~47 MB of attention +
4.7 MB a distinct expert (17 / 33 / ~60 experts at 2 / 4 / 8 rows), the index cache scan (132 B a key per
request) and the per-row index scores (8 B a key written and read back): about 30 us (2 x 11K) to 130 us
(8 x 632K); every case is above 10 us.

## Provenance

sgl-amd-mono 52486703 (`sgl_amd_mono/glm53/gfx950/kernel.py`, adapted there from ROCm/aiter#6173), Apache-2.0,
with ROCm/ATOM-derived helpers (MIT); see `baseline/LICENSE` and `task.yaml` for the files and modifications.
SGLang ROCm images pin sgl-amd-mono e52803f9 (11 commits older).

## Unverified (no GPU was available when this task was written)

* GPU compile of the vendored kernel on FlyDSL 0.3.2: the FlyDSL >= 0.3.4 entry points are replaced by llvm
  dialect forms in `baseline/compat.py` (lds atomics, global loads, the guard's stop store); other API gaps,
  if any, appear only at compile time. The signature change (seven indexer pointers as arguments) and the
  `reload_args` rewrite are untested.
* Numerics against the reference, the planted margins as the kernel computes them, timing, and the 10-minute
  budget (three FlyDSL compiles of a 7000-line kernel plus 9.7 GB of logical expert weights a draw).
* The decoded `index_cache` bound on the GPU: only the byte-exact compare ran on MI355X (and failed); the
  decoded bound was checked on the CPU against simulated E4M3 key rounding differences.

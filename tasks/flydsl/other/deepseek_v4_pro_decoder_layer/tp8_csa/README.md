# DeepSeek-V4-Pro decoder layer, CSA, TP8 decode (`deepseek_v4_pro_decoder_layer/tp8_csa`)

One whole DeepSeek-V4-Pro decoder layer of a **CSA** layer (compress ratio 4, lightning indexer with an
exact top-1024) for 1 to 8 decode rows, on **8 GPUs** (tensor parallelism 8), both TP all-reduces
included. The baseline is sgl-amd-mono's monokernel `dsv4_kernel`: one persistent FlyDSL launch per
rank (256 CTAs x 512 threads) that runs the mHC mixers, the attention with its KV compressor and
indexer, the o projections, the MoE (router, 1 shared + 6 routed MXFP4 experts) and pushes BF16
partials to every peer's symmetric buffer over XGMI, summing them in rank order. `task.yaml` gives the
dataflow; `reference.py` its exact quantization and rounding points.

## Provenance

* Kernel: sgl-project/sgl-amd-mono `52486703fbf27e77f05b573e4d931d4eec87118d`,
  `sgl_amd_mono/deepseek_v4/gfx950/{kernels/*,kernels/stages/*,config.py,layout.py}` and
  `sgl_amd_mono/common/{ops,guard,sync,plan}.py` (Apache-2.0; adapted from ROCm/ATOM#2559, MIT). SGLang's
  ROCm images pin e52803f9 (11 commits older). FlyDSL helpers from ROCm/aiter (Apache-2.0): `dpp_utils.py` from 652f2d4b, `buffer_ops.py` from
  dbd8bf5bd (FlyDSL 0.3.2's rocdl raw buffer ops take the cache modifier as an attribute, not 652f2d4b's i32 operand).
* `run()` follows `runner.py`'s `Dsv4MonoRunner.launch`; weight packing (`packing.py`) is in
  `inputs.py`'s shard. All modifications are listed in `task.yaml` (`baseline.provenance.modifications`)
  and in each file's header. The device code is numerically unchanged; three FlyDSL >= 0.3.4 APIs are
  bridged for the runtime's FlyDSL 0.3.2 on llvm-dialect ops (`stop_launch`, `global_ptr`/`global_load`,
  the top-k histogram's atomic add). One read is clamped: the indexer score stage bounds its entry
  index by the row's live entries, not `N_COMP - 1`. Upstream relies on SGLang's block-table rows
  spanning `max_seq`; here `block_tables` holds only the case's pages, so the dead lanes of a row's last
  score tile read past it (`r1_c4100`: entries 17-23 of 17) and turned stale bytes into i_cache
  addresses. Dead lanes stay masked, so no value changes.
* The repository has no numeric test of this kernel (compile-only); the reference was derived from the
  kernel source. Assumptions are listed below.

## Interface (one rank's call)

* Weights are rank `r`'s shard (heads 16r..16r+15, o_a groups 2r and 2r+1, o_b / expert-down K columns
  and expert intermediates 384r..384r+383) in the kernel's packed tile layouts. The logical inputs the
  reference reads are the whole layer (`inputs.py` docstring). The indexer, `w_qkv_a`, the compressor
  projections, the mHC mixers and the router are replicated, as in production.
* Caches are SGLang's: the two-pool FP8 unified KV (`kv_cache` NoPE plane read as e4m3, `kv_rope`), the
  per-request compressor and indexer-compressor state rings (`c_state`, `i_state`, rows `[kv | score]`),
  the FP4 index-key pages (`i_cache`, `i_cache_s`), block tables and the window index list.
* `symmetric` (framework-owned, zeroed before every call) holds the two all-reduce regions
  (`2 * world_size * rows * 7168` tagged pairs of 8 bytes), the guard's 256 flag bytes and then the
  kernel's mailbox scratch (at most 4,243,456 bytes a row): `symmetric_bytes = world_size * rows *
  114688 + rows * 4243456 + 256`. Production keeps the scratch in ordinary device memory and advances
  an epoch instead; here the epoch buffer is a read-only input (step 1, a 20 s wait deadline).
* `out` is the next residual (4 mHC streams), bit-identical on every rank (`ranks_agree`). The step
  writes its token's KV ring row and its compressor state rows in place (mutable inputs).

## Workloads

Derived from corpus statistics (AgentX `semianalysisai/cc-traces-weka-062126`, decode-step weighted:
p10 52K, p25 108K, p50 220K, p75 426K, p90 632K, max 990K); **not yet confirmed by a live capture.**
Rows are the monokernel's decode widths 1, 4 and 8 (three compiled row counts), one request a row; the
other rows of a step decode at contexts drawn in [ctx/2, ctx]. Eight performance cases span 152x in
rows x ctx. Short contexts (<= 4K, 4.4% of decode steps), the indexer threshold (1025 live entries),
mixed batches across it and the 1M cap are correctness-only. Only decode applies (no prefill); DSpark
verify (width 7) is a different build and not covered.

Input design: both discrete decisions are planted clear, so neither can flip between the reference
and a correct kernel.

* Router: a planted component of every hidden row along 6 random experts' router directions keeps
  the top-6 clear of the 7th.
* Indexer top-1024: in every row with more than 1024 live compressed entries, 1024 random live
  entries hold planted FP4 index keys (`inputs.py` `_plant_index_keys`). Each comes from the dual
  basis of the row's 64 index-query heads, computed exactly as `reference.py` computes them: it scores
  high through the heads of positive weight, with relu off on the others. It has the random keys'
  norm, so neither its norm nor its e8m0 bytes mark it. Random keys scoring above 60% of the row's
  lowest planted score are redrawn (on synthetic queries at 262K entries: none in most rows, at most
  25). The generator then rescores every live entry from the stored bytes and raises unless the
  planted set leads every other entry by 35% of its lowest score (40-60% in practice). Only the set
  matters: the kernel and the reference both place the selected entries in index order.
* The planted entries' KV rows stay random. K = V here, so a planted high-logit direction would be
  common to every planted row and hide which of them are attended. With random rows every attended
  entry carries an independent value at a near-uniform weight, so a wrong entry moves every head's
  output: before planting, the baseline's `out` fell to 19-23 dB SQNR at the dense long-context
  boundaries, against 25-26 dB where no boundary was dense (MI355X).

FP8 block scales are arbitrary positive floats (the checkpoint's are powers of two). Caches hold random
normalized rows quantized exactly as the kernel stores them.

## Correctness bounds

`out`, `kv_cache` and `kv_rope` use `baseline_relative` with `k: 4` (default 2): the layer chains about
twenty quantization and bf16 rounding points (FP8 activations, FP4 indexer query, MXFP8 expert inputs,
per-rank bf16 all-reduce partials), at each of which a correct implementation with another accumulation
order can round the other way. The indexer's selection adds no error: its planted margin (at least 35%
of the lowest selected score) is far above the score change of an FP4 index query with hundreds of
codes one step off, head weights one bf16 ulp off and fp32 sums (under 10% on synthetic queries), so a
correct implementation attends to the reference's keys. Mutable state is compared against the
reference's values before the kernel's own rounding (the KV codes as `clamp(nv / scale)`, the RoPE
plane and the compressor projections unrounded), so the kernel's rounding is within the bound. The
e8m0 bytes are exact.

`out` also sets `baseline_min_sqnr_db: 21` (default 25), which **loosens** only the sanity floor that the
baseline itself must clear against the float64 reference, not the candidate bound. Measured on MI355X with the
planted indexer selection, over every correctness case, both seeds and all 8 ranks on two nodes (224 baseline
outputs a node): `out` is 24.9-29.5 dB from the reference (lowest 24.87 dB in r8_c220k, on both nodes; 25.1-26.0
dB in the other long-context cases), so the default 25 dB floor rejected the production kernel by 0.1 dB. The
floor is that value rounded down, minus 3 dB.

Run-to-run evidence (MI355X, two nodes): the baseline is bitwise deterministic (4 eager re-runs and 40 cold-cache
graph replays per case, both seeds, all 14 cases on all 8 ranks: 5376 comparisons a node, every one identical to
the stored baseline output), so the `k: 4` on `out`, `kv_cache` and `kv_rope` (which **loosens** the default 2, see
above) is for other correct implementations, not for the baseline's own variation. Single-output mutants of a
baseline run, judged by the evaluator's comparison in every case on every rank (112 each): zeroing the first
eighth of the last dimension is rejected 112/112 on every output; Gaussian noise of 1% of the RMS is rejected
112/112 on `kv_cache`, `kv_rope`, `c_state` and `i_state` but only 35/112 on `out`, and a 1% relative shift only
on `kv_rope`, `c_state` and `i_state`: `out` is itself about 25 dB (6%) from the float64 reference, so errors of
1% on it are below what its bound can resolve.

## Reference assumptions (from the source; not checked against a GPU run)

1. The TP reductions are exact sums; the kernel rounds each rank's partial to bf16 before summing.
2. The 6th-ranked routed expert's down input is FP8 1x128-quantized and the shared expert's MXFP8
   (1x32): the kernel's down stage applies the 1x128 rule to slot `MOE_SLOTS - 1` (pick 5) although its
   comment names the shared expert (slot 0). Kept as the kernel does it; possibly an upstream slip.
3. e4m3 and e2m1 conversions round to nearest even; e2m1 ties go to the even code.
4. The mHC mixers use the fp32 checkpoint values (the kernel's bf16 hi + lo split is within 2^-16).
5. Hardware exp2 / rcp / rsq approximations are not modelled (float64 exact functions).
6. The RoPE tables (ctx + 4096 positions, like SGLang's precomputed cache) are plain RoPE at compress_rope_theta 160000 (any table is valid input; YaRN not
   modelled). The layer index (tag only) is 10; it is a scored, not hash-routed, layer.

## Not covered / unverified

* **No compressor boundary**: no row ends a compressor window ((p + 1) % 4 = 0), so the pooled KV
  entry, the FP4 index-key write and the in-launch rescoring of a new entry are not exercised. Those
  writes are packed uint8 (FP4 codes, e8m0 bytes), which the framework compares exactly, and a float64
  reference cannot match a fp32 kernel bit for bit there. The reference raises on such rows.
* The planted indexer selection is checked at generation (the generator raises below its margin), but
  the baseline's `out` SQNR on the planted inputs has not yet been measured on a GPU.
* Nothing here has run on a GPU: FlyDSL 0.3.2 compile of the vendored kernel, baseline numerics against
  the reference, the 8-rank launch through the framework's peer buffers, timing and compile time.

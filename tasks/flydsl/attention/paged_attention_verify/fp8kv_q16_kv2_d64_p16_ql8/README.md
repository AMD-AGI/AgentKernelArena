# Muse Spark verify attention: paged FP8 GQA, 16 / 2 heads, head size 64, 8 rows a request

The attention of one Muse Spark full-attention layer for a DFlash2 speculative-verify step, on one rank of
the model's TP8 group: `requests` requests x 8 draft rows, the rank's 16 query heads over its 2 KV heads
(GQA 8), head size 64, over a paged FP8 (E4M3) KV cache in 16-key pages. Row j of a request attends the
request's committed context and the step's rows 0..j (causal; the step's K / V are already in the cache).
This is sgl-amd-mono's K-attn launch (`muse.kattn`), one persistent 256-CTA FlyDSL kernel.

**Model support is semi-experimental.** Muse Spark is registered in sgl-amd-mono's `SPECS` but absent from
the repository README's model table; it landed in one PR (#26), has no end-to-end server test, and its
default served widths are 8 and 16 rows (`gfx950/config.py`: "32 and 64 rows are under test").

**Relation to `paged_attention_decode`.** The tasks under `tasks/flydsl/attention/paged_attention_decode/`
are the same attention family but a different operator scenario and baseline: single-token decode (one
row a sequence, no causal rows), 1024-token pages in AITER's split K / V layouts with per-token or
per-tensor scales, head size 128, AITER's persistent-scheduling worklist inputs, and the FlyDSL PA
baseline. This task is a verify step (QL = 8 causal rows a request) over a page-16 NHD cache of unscaled
E4M3 codes with the sgl-amd-mono baseline. Under the schema-v3 target-language rules a scenario with a
different baseline and contract is its own task; since the semantics differ (multi-row causal verify), it
is filed as a new operator, `paged_attention_verify`.

## Semantics (`reference.py`, float64)

Request r has `keys = seq_lens[r] + 8` keys; key i lives at slot `page_table[r, i // 16] * 16 + i % 16` of
`k_cache` / `v_cache` (`[slot, kv_head, 64]` float8_e4m3fn, exact values, no scale). For row t = 8 r + j
and query head h:

    out[t, h] = softmax_i(q[t, h] . K[i, h // 8] / 8) @ V[:, h // 8],   i <= seq_lens[r] + j

The scale is `RadixAttention.scaling` = 64 ** -0.5; `q` arrives per-head RMS-normed and multiplied by the
checkpoint's query scale 5.473 (the layer's K1 does both). The layer is a full-attention layer (window 0),
whose work grows with the context; Muse Spark's sliding layers (2047-key window) read at most 2055 keys
and are not part of this task.

## Baseline and provenance

`baseline/` is vendored from sgl-project/sgl-amd-mono at `52486703fbf27e77f05b573e4d931d4eec87118d`
(Apache-2.0; `common/ops.py`, `plan.py`, `sync.py` adapted from ROCm/ATOM, MIT): `muse_spark/gfx950/layer.py`
(the K-attn build only), `stages/attention.py::run_kattn` (unchanged apart from imports, the
`SGLANG_AMD_MONO_ATTN_RING_WAIT0` default and the mask fix below), `config.py`, `layout.py`, `device.py`, the FlyDSL 0.3.2 bridge
`compat.py`, and `common/{guard,epoch,ops,plan,sync}.py`, in the same flattened form as
`tasks/flydsl/other/muse_spark_layer/tp8` (that task's modifications: relative imports, frozen dataclasses /
a slotted class instead of `msgspec`, the 0.3.2 bridge called directly instead of `compat.install()`
patching modules, environment reads replaced by their defaults, no source-file digest). `buffer_ops.py` and
`dpp_utils.py` are ROCm/aiter `dbd8bf5bd` (the copies of `tasks/flydsl/other/qwen38_flash_next_decoder`,
whose descriptor flags do not read the environment). SGLang's ROCm images pin sgl-amd-mono `e52803f9` (11
commits older); the baseline is vendored from `52486703`.

**One kernel change: a fix of the causal mask.** `run_kattn`'s chunk loop applies the mask only on a
task's last chunk and the request's first ("a chunk with no masked key" otherwise). A verify step's
masked keys are the step's later rows (up to 7 keys before the end), so when a request's key count is
1-6 past a multiple of 256 they reach into the chunk before the last; when that chunk and the last share
a partition (every multi-chunk partition: contexts above ~16K), the step's earlier rows attended up to 6
future draft keys. A simulation of the kernel's partition sizing shows it, e.g. at 25,859 keys (two-chunk
partitions) rows 0-3 see 4, 3, 2, 1 future keys; with this task's planted inputs those rows are off by
29-59% (10.6 dB SQNR), which would fail the baseline's own sanity check. The repository's fixed test
lengths avoid it (2049, its one straddling length, ends a 4-chunk partition) and its random-length cases
use unplanted data, where a few leaked keys among thousands rarely move a row past the test's tolerance.
The fix adds one scalar compare a chunk: a chunk skips the mask only if it also ends at or before the
step's first row's last key. The sliding-window path has the same pattern at the window's start (the
later rows' window starts up to 7 keys later, possibly in the chunk after the first); this task's layer
has no window. `r1_c25859` is the regression case. The same fix is applied to the attention stage of
`tasks/flydsl/other/muse_spark_layer/tp8`; D-attn (`draft_attn.py`) shares the decision but attends its
block without a causal mask.

`baseline/kernel.py` is new. It makes `MuseMonoRunner.kattn_kernel(rows, 8)`'s build and passes
`MuseMonoRunner._attn_args`' arguments for a target verify of a full-attention layer (`extra` = 8, window 0,
the partition sized by the kernel, the scale above, int32 `seq_lens`, the page table's row stride). The
runner keeps one scratch for all launches and moves the step epoch on after every step, so no mailbox flag
of an earlier step carries the current tag. Here the scratch and the epoch / guard words (epoch 0, the
30 s wait deadline, clean error words) share one device buffer per row count, created zeroed at the first
(eager) call; before every launch `run()` zeroes the scratch's flag span (task flags, level-2 records and
their flags: 0.28 / 0.56 / 2.2 MB at 8 / 16 / 64 rows), one memset inside the timed graph. The large
partial-record region (4.3 MB a request) is not cleared: the merges read only records whose flags this
launch raised, and weigh the records they skip by zero, which needs them finite (zeroed at creation, then
only this kernel's finite records).

## Inputs (`inputs.py`)

Request 0 holds `ctx` keys; the others draw theirs in [ceil(ctx / 2), ctx]. Each request owns distinct
pages in random order; page-table entries past a request's pages are random stale page ids, and the last
page's slots past its keys hold stale codes, so reading past `seq_lens + 8` keys is visible. The attention
is peaked, as real heads are: every (request, KV head) has a direction c that its query heads share
partly, and planted keys along c (logits about 15-45 against a random key's standard deviation of about
5.5) sit at the step's 8 positions (so the causal mask among the step's rows changes every row), at key 0,
mid-context and the last committed key (maxima in different partitions, a late rescale); stronger planted
keys sit in the masked tail of each request's last page. Softmax is continuous, so no discrete decision
needs a margin.

## Workloads

Derived from AgentX corpus statistics (`semianalysisai/cc-traces-weka-062126` rev 23f152f6, 98,827 requests,
decode-step-weighted context p10 51,891, p25 108,326, p50 219,840, p75 426,387, p90 631,824, p99 889,977)
and the per-session in-flight distribution (one request in flight 90% of the busy time, two 5.4%); **not
yet confirmed by a live capture**. Monokernels serve decode / verify only; prefill shapes do not apply. The
model's maximum context is not in the source; the cap is assumed to be 1,048,576 (AgentX's maximum is 990K).

Rows: 8, 16 and 64 (R = 1, 2, 8 requests; three compiled widths for the 10-minute budget; 8 and 16 are the
default served widths, 64 the largest). Performance (9 cases, var-axis product requests x ctx spanning
104x): R = 1 at 32K and the p25, p50, p75, p90 and p99 contexts; R = 2 at p50; R = 8 at p25 and p75.
Correctness only: ctx 8 (minimum: the step's rows only), 3000 (non-power-of-two; 12 partitions, one-level
merge), R = 2 at 4099 (not a multiple of 16; 17 partitions, a two-level merge with an empty level-1 group),
R = 8 at 2049 (short ragged contexts), 25,859 (the step's rows straddle a chunk boundary inside a
two-chunk partition, the mask fix's regression case), 16K (near the 10 us floor) and the 1M cap. The production-range span
exemption is not used: the context axis gives the 100x span.

Estimated baseline time (sgl-amd-mono's own figures for this kernel: ~12 us a task beside its chunks of
~1.8 us; plus ~3 us of merges; the partition is sized so a step has about one task a CTA): 1 chunk a CTA
at 16K (~17 us, kept correctness-only), 2 at 32K (~19 us), then ~28, 40, 64, 85 and 114 us for R = 1 at
108K .. 890K, ~58 us for R = 2 at 220K, ~91 and ~310 us for R = 8 at 108K and 426K. The KV bytes alone at
5 TB/s give lower bounds of 2 .. 136 us. The smallest performance case is the one closest to the 10 us
floor.

## Comparison

`out` uses the default `baseline_relative` bound. On the CPU, a float32 emulation of the kernel's numerics
(fp32 scores, exp2-domain partition softmax, P rounded to bf16, fp32 merges, bf16 output) reaches 56-57 dB
SQNR against the reference on r1_c8, r1_c3000, r2_c4099, r8_c2049 and r1_c16k, and 55-56 dB on the
long-context r1_c108k, r2_c220k and r8_c108k (run on a CPU host). Taking it as the baseline,
the same emulation with 3-chunk partitions and one with float32 P both pass the bound with no violations,
while attending without the causal mask (47-74% of elements over the bound), with the step's mask off by
one key (40-50%) and through the unmasked tail of the last page (87-100%) all fail.

## Unverified

Nothing here has run on a GPU: FlyDSL 0.3.2 compilation of the flattened package (the same vendoring as
`muse_spark_layer/tp8`, also not yet GPU-qualified at the time of writing) and of the mask fix, the mask
bug itself on hardware (found by reading the kernel and simulating its partition schedule), the
persistent-buffer scheme in `kernel.py`, numerics against the reference, the bound with the real
baseline, compile time for three widths, and timing (in particular whether r1_c32k takes at least
10 us). The CPU checks covered the contract (`check_task`), the baseline's candidate-rule audit, the
inputs and reference of every case (finite, declared shapes; the largest, r8_c426k, generates in 3 s
and references in 4 s on 4 CPU cores), and the mutant / bound emulation above.

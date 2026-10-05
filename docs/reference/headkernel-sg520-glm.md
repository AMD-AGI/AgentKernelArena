# GLM 194292: final native trace review

Both workload and native finalizer gates passed. All 64 requests have 8192 input and 1024 output tokens, with zero request errors. There are 16 complete stable traces, with TP ranks 0–7 in both phases. Context 9218; requested C64/TP8.

The baseline is SGLang 0.5.20 plus 18 approved overlays, with original native AITER JIT and the image-precompiled baseline preserved. It is not an unpatched stock-image profile.

| Stage | GPU kernel occurrences |
| --- | ---: |
| Prefill | 153,016 |
| Decode | 13,427 |

Leading observed prefill symbols include generic `main_kernel` (1,408 calls), cross-device reduction (5,824), and `aiter::fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256` (2,688). The latter appears 336 times in decode. The old `fused_moe_kernel` name is absent. A generic `main_kernel` symbol is not a unique callable.

CPU prefill steps have batch size 1 and 8192 tokens. Decode CPU steps have batch size 64. Of eight CPU graph launches per rank, only one has explicitly correlated GPU graph kernels. Do not multiply or rescale observed kernel counts.

| Actual per-group quant input | Calls per rank | FP32 scale allocation |
| --- | ---: | --- |
| `[8192,1536]` BF16 | 88 | `[8192,12]` |
| `[8192,2048]` BF16 | 88 | `[8192,16]` |
| `[8192,4096]` BF16 | 784 | `[8192,32]` |

Output is FP8 E4M3FN. Recorded group size is 128 and transpose-scale is true; the physical scale storage permutation must be preserved. The fused-MoE CPU record also exposes activation `[8192,4096]`, FP8 expert weights `[288,512,4096]` and `[288,4096,256]`, and routing weight/ID shapes `[8192,8]`. Tensor values were not saved.

No task fixtures were created and no UT qualification was performed. Tensor values, reference outputs, page/routing state, and many graph kernel arguments remain absent. Counts and summed durations are raw profiler observations; they establish neither correctness nor speedup.

The [progress catalog](../../tools/headkernel-sg520-refresh.json) records external evidence keys and verified OCI archive locations. `CURATED-SUMMARY.json` contains selected evidence, `SUMMARY.json` the full per-rank groups, `stream/` every GPU occurrence and its links, and `VALIDATION.json` the independent count/hash checks; archive membership is recorded separately.

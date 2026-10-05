# minimax-nvme-194299: final native trace review

Both workload and native finalizer gates passed. There are 16 complete stable traces, with TP ranks 0–7 in prefill and decode. All 64 requests have 8192 input and 1024 output tokens, with zero request errors. Context 9218, requested C64/TP8.

SGLang 0.5.20 exact image; original native AITER JIT and image-precompiled baseline preserved.

| Stage | GPU kernel occurrences |
| --- | ---: |
| prefill | 108,640 |
| decode | 94,817 |

Counts are unscaled raw occurrences. Summed GPU durations overlap across ranks and streams; they are not latency or speedup.

Observed existing attention targets:

| Symbol | Phase | Calls across eight ranks |
| --- | --- | ---: |
| `_gqa_share_sparse_fwd_kernel` | Prefill | 3,648 |
| `_decode_score_kernel` | Decode | 1,856 |
| `_gqa_share_sparse_decode_kernel` | Decode | 3,648 |

CPU step annotations record prefill bs 1–3 with 8192/16384 tokens, then decode bs 63 for all eight sampled steps. Global concurrency 64 does not establish a physical 63- or 64-row graph tensor contract.

Decode attention kernels have runtime graph links without their original CPU tensor arguments. Eight CPU graph launches per rank have linked GPU occurrences, but the native receipt still makes no claim that all workload graph replays were captured.

Actual prefill quantization arguments include BF16 inputs `[8192,6144]`, `[16384,6144]`, `[32768,512]`, and `[65536,512]`; output is packed FP4, scales E8M0, group size 32, shuffle false. Actual FP4 GEMM scale views include strides `[1,8192]` and `[1,16384]`. Full scalar/type/stride evidence and per-rank counts remain in the JSON report.

No task fixtures were created and no UT qualification was performed. Tensor values, reference outputs, page/routing state, and many graph kernel arguments remain absent. This intake does not qualify a task or establish correctness or speedup.

The [progress catalog](../../tools/headkernel-sg520-refresh.json) records external evidence keys and verified OCI archive locations. `CURATED-SUMMARY.json` contains selected evidence, `SUMMARY.json` the full per-rank groups, `stream/` every GPU occurrence and its links, and `VALIDATION.json` the independent count/hash checks; archive membership is recorded separately.

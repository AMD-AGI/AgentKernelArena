This task exposes Kimi-K3's actual stage-2 GPU implementations in the pinned SGLang 0.5.20 runtime. Prefill uses FlyDSL-v2 GEMM2 followed by the native BF16 reduction. Decode uses the native Opus A8W4 kernel with explicit kernel ID 2005. The full served workload completed 64 requests at TP8/C64, ISL8192/OSL1024.

The editable bodies are `gemm2_body_v2` in [source/flydsl/mxmoe_gemm_v2.py](source/flydsl/mxmoe_gemm_v2.py) and `opus_moe_stage2_a8w4_decode_kernel_gfx950` in [source/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh](source/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh). The host wrappers, import graph, signatures, launch controls, build recipe, native reduction and all other code are protected. The HIP source is freshly compiled into a task-owned extension. Each FlyDSL leg uses a distinct private dispatcher and emitter package with a source-bound GPU symbol; neither leg replaces the installed dispatcher module.

The three exact cases are:

| Stage | Tokens | Calls per rank | Calls across eight ranks |
| --- | ---: | ---: | ---: |
| Prefill | 8192 | 184 | 1472 |
| Prefill | 16384 | 2852 | 22816 |
| Decode | 64 | 94208 | 753664 |

All cases preserve packed FP4 model weights, FP8 activations, native tiled E8M0 scales, topk 16, intermediate dimension 384, model dimension 3584, captured physical storage and aliases. Decode accumulates into caller-zeroed BF16 output; prefill overwrites BF16 output. Complete empirical work distributions, including dynamic valid sorted-row counts, remain in [provenance/WORK-DISTRIBUTIONS.json](provenance/WORK-DISTRIBUTIONS.json).

The trusted host materializes the exact assets in [fixtures/EXTERNAL-MANIFEST.json](fixtures/EXTERNAL-MANIFEST.json). No tensor blobs belong in the Git task. Every replay creates fresh legal distinct-expert routes at an actually observed valid-row work amount, remaps captured quantized activation payloads and their exact scale layout, and generates normalized routing weights. Model-weight bytes remain readonly.

Before candidate execution, the harness owns CPU input truth. After execution, it freezes candidate outputs and complete post-input storage on CPU before invoking the separately bound frozen reference. Reference GPU output storage is cleared after its CPU snapshot. Correctness retains the 0.05 RMS-relative tolerance, exact metadata and alias checks, and no-op/wrong-output negative controls. Performance uses 10 warmups and 100 checked single-call CUDA graph replays.

In the pinned gfx950 container, after trusted fixture materialization:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Source and metadata CPU checks pass. GPU correctness, timing and a framework-finalized task-validator `PASS` are still required before publication.

The protected loader reads and hashes each raw fixture blob once per case into an immutable CPU byte cache. Tensor construction copies those bytes into separate CPU storage, and each replay restores separate device input storage before fresh routing and activation generation. The callback layer owns each CPU input snapshot, avoiding a redundant full-storage host copy while preserving every immutable-input and oracle check. The declared performance timeout is 1800 seconds because these full-storage checks run outside all 100 timed replays for each of the three cases.


Fresh routing uses bounded expert block populations that permit skew and varying active-expert counts. Each expert receives between `(blocks-1)*tile_m+1` and `blocks*tile_m` live routes, capped at the token count; the resulting token ring gives every token 16 distinct experts. The padded-row sampler, its observed frequencies, request seeds, correctness seeds 0/1/2, 10 warmups and 100 timed replays are unchanged. The conditional population sampling law is synthetic test data, not an inferred empirical routing frequency.

[retained_route_populations.json](ut/retained_route_populations.json) also pins all 27 distinct retained expert-ID/block-count populations for this task, nine per case, to the complete eight-rank `kimi-actual-v4-no-stack-194550` manifests. During correctness, [RoutingCallbacks](ut/routing.py) first runs the existing fresh-seed checks and negative controls, then checks each retained population once using the same request challenge seed and the existing graph/deferred native oracle. These additional checks carry no timing weight. Their IDs appear in `retained_route_populations` in the correctness report; work logs record population IDs, active-expert counts, largest block counts and whether a draw was targeted or fresh.

The retained examples include prefill `num_valid_ids=[290304,16384]` with 4536 blocks, 690 active experts and 167 blocks for one expert, which the previous even-block generator could not realize. Exact expert populations are preserved in these targeted inputs; token-to-expert assignments and within-expert live counts are generated anew because full routing histories and population frequencies were not recorded. This establishes representation of all retained populations and all histogram bins, with sampled and targeted correctness checks. It does not establish exhaustive execution of every numeric bin or coverage of uncaptured routing patterns. A fresh GPU/framework qualification is required for the revised generator.

CPU routing checks run with `python3 -m unittest discover -s ut -p 'test_routing.py' -v`. They realize every retained population, verify top-16 uniqueness and exact block padding, check degree feasibility for every histogram bin at all three correctness seeds, and preserve the existing deferred-oracle and 100-sample callback behavior.


Ordinary performance scoring now uses one candidate graph replay followed by one timed protected-reference replay on the exact same private CPU input snapshot. Candidate outputs and complete post-input storage are captured on CPU before reference restoration or execution. Both legs have three setup warmups and one capture on their respective warmed streams, followed by the unchanged 10 policy warmups and 100 measured pairs. The reference result supplies the original oracle comparison and its complete returned-output storage is scrubbed before the next candidate; failure paths also scrub bound reference outputs. The original correctness seeds, negative controls, targeted retained-population checks where applicable, formats, aliases and tolerances are unchanged.

The paired report records every realized integer work/address control hash and input seed. The shared scorer invokes the pinned task-local CPU validator in [paired_input_validation.py](ut/paired_input_validation.py) before accepting timing: it rebuilds all weighted work draws and integer routing or KV-index arrays from the exact case histogram, request seed and [work registry](provenance/PAIRED-WORK-REGISTRY.json). Reported schedules, private storage sizes and raw-sample means must agree. The original request sampler and generated inputs are unchanged. This proves pairing for each measured call; it does not invent missing routing frequencies, recover unrecorded routing histories or claim exhaustive timing of every histogram bin.

[PAIRED-REFERENCE.json](provenance/PAIRED-REFERENCE.json) identifies the scoring baseline and pins its source and CPU receipt validator. Stage 1's synchronization-repaired implementation is labeled `protected_reference`; native Stage 2 and Lean implementations use `native_production`. Historical/private-path qualification does not qualify this revised ordinary scorer. Fresh GPU, framework-finalized and independent trusted-host qualification is required for the new paired path. Ordinary paired timing is not an anti-cheat attestation, and the scorer suppresses secondary cross-run port ratios when their input schedules differ.

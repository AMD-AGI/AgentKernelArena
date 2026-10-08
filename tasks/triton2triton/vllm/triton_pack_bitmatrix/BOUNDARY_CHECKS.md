# Top-k boundary checks

The packing kernel must include every selected expert, including assignments
after slot 31. It keeps 32-assignment tiles and combines all tiles before
writing each output word. Masked tile-padding slots do not set any bits.

Two correctness-only cases in `workloads.json` use top-k 33 and 65 with 64
experts. They place expert 63 only after the first 32 assignments, covering
repeated IDs, partial final tiles, and accumulation into the second word.
The native correctness action executes both cases. A candidate that packs
only its first tile fails these controls.

The original five scored shapes, exact-equality checks, 10 warmups and 100
samples remain unchanged. The existing replay checks still compare the timed
outputs and perturbed-input replay with the independent membership reference.

The existing implementation is adapted from the Apache-2.0-licensed
[vLLM packing kernel](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py).
Copyright contributors to the vLLM project.

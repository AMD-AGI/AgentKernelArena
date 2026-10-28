# Source

The starting kernel is copied from the public
[PR #44 source capture](https://github.com/AMD-AGI/AgentKernelArena/blob/803818cdcb5ea1d8326fff6712ab1b51cf2f06a1/tasks/campaign20/baseline/write_req_to_token_pool_triton/source/triton_write_req_to_token_pool_triton.py).
That task identifies its upstream entrypoint as
`sglang.srt.mem_cache.common.write_req_to_token_pool_triton`, captured through
Triton inspection from [SGLang](https://github.com/sgl-project/sglang).
The capture does not record a precise upstream SGLang revision.

SGLang is distributed under the Apache License 2.0; a copy is included as
`LICENSE`. The source capture contains no per-file copyright notice to preserve.
The protected reference and schema-v2 evaluation harness implement the task
contract independently of the editable kernel.

# fused_qk_norm_rope

This task is extracted from vLLM kernel sources and categorized as `hip2hip`.

- Upstream test command: `python3 -m pytest kernel/vllm/tests/kernels/core/test_fused_qk_norm_rope.py -q`
- Source files copied under `source/`
- Runner script keeps task-level commands self-contained

## Independence

This task is self-contained and does not call upstream `kernel/vllm/tests` during compile/correctness/performance.

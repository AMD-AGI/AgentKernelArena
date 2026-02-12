# reshape_and_cache

This task is extracted from vLLM kernel sources and categorized as `hip2hip`.

- Upstream test command: `python3 -m pytest kernel/vllm/tests/kernels/test_cache_kernels.py -q`
- Source files copied under `source/`
- Runner script keeps task-level commands self-contained

## Independence

This task is self-contained and does not call upstream `kernel/vllm/tests` during compile/correctness/performance.

---
myst:
    html_meta:
        "description": "GPU qualification evidence for the first AgentKernelArena serving task, with the exact source, lock and runtime it applies to."
        "keywords": "AgentKernelArena, e2e, serving, qualification, task_validator, MI355X, SGLang"
---

# E2E serving task qualification — 2026-09-29

This record binds the first serving task's GPU evidence to the exact source and
lock it was produced with. It is not a claim about later revisions of the task.

## Scope

| Item | Value |
| --- | --- |
| Task | `e2e/qwen3_0_6b_sglang` |
| Source revision | `e04dac43` on `codex/e2e-kernel-tasks` |
| Runtime lock SHA256 | `1809c2be77ff415a45022782a69459a6d7a92abafbe588029d8ee24fde2e96d4` |
| Hardware | one AMD Instinct MI355X (`gfx950`) |
| Runtime | pinned `lmsysorg/sglang-rocm` digest from the lock; Python 3.10.12, Torch 2.9.1+rocm7.2, Triton 3.6.0 |
| Workload | Qwen3-0.6B FP16, TP=1, concurrency 8, ISL/OSL 128, 80 requests per measurement |

## Evidence

- **Task validator:** framework-finalized `validation_report.yaml`, schema
  version 4, `overall_status: PASS`, all fifteen checks `PASS`, replay
  validation recorded as not applicable to the serving client timer.
- **Optimization smoke run:** Codex with three iterations completed in about
  22 minutes; the final Triton candidate passed compile, GPU execution and
  model correctness; candidate accepted with identical runtime fingerprints in
  every pair.
- **Final pairs:** 4112.5/4119.0, 4097.8/4096.8 and 4101.5/4119.8 tokens/s
  (baseline/candidate), median ratio 1.0016. A separate three-pair check
  measured 0.9975. Neither shows a stable end-to-end gain; both are inside the
  repeat variation of the 2.5-second windows used at that revision.
- **Controls:** a correct Triton implementation passed the compile and operator
  checks; a numerical mutant was rejected with 121 of 128 mismatched elements.
- **Kernel regression:** the existing `triton2triton/geak_eval/L2/fast_rms_layernorm`
  task passed with the original and the updated evaluator on the same GPU.
- **CPU gate:** 14,045 passed, 125 skipped in pull-request CI.

## Applicability

The lock, workload and task configuration changed after this evidence was
produced: the measurement window grew from 80 to 2,000 requests, `limits` and
a longer final-evaluation minimum were added to the lock, and the tail-latency
threshold was tightened. Those revisions require a fresh framework-finalized
validator PASS on the locked runtime before the task is treated as qualified.
Until then this record certifies the contract and integrity machinery at
`e04dac43`, not the current task package.

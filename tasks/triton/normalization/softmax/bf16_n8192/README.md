# Row softmax, bfloat16, 8192 columns

`out = exp(x - max(x)) / sum(exp(x - max(x)))` along the last axis of a bfloat16 matrix, as used for the vocabulary
distribution or a wide router. The reference is float64; the output is rounded to bfloat16 and compared with the
default baseline-relative bound. Inputs are normal with standard deviation 2, so row maxima differ clearly from the
mean and the largest probabilities are well resolved.

## Baseline

AITER's Triton online softmax `_softmax_kernel_online` and its launcher (`aiter/ops/triton/softmax.py`), ROCm/aiter
commit `652f2d4b768239aefd515c929e84817681ff598d` (MIT, `baseline/LICENSE`): one program per row, a running maximum
and rescaled sum in the first loop, normalization in the second, a single 8192-wide block (one iteration), 8 warps.
The kernel is unchanged; the kernel-name helper and logging are removed and the launcher is reduced to `run()`. The
earlier rocmbench online and naive (three-pass) softmax tasks and the fp16 and fp32 suites are not part of this
scenario.

## Workloads

A row moves 32 KB (16 KB in, 16 KB out), so a case needs about 830 rows to exceed the 10 us timing floor; performance
cases start at 2048 rows and span 100x up to 204800 rows (6.7 GB moved). The largest case needs about 15 GB for the
float64 reference, which works in chunks. Smaller batches, from a single row up, are correctness-only.

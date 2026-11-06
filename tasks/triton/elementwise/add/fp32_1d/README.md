# Element-wise add, float32, 1D

`out = x + y` for two contiguous one-dimensional float32 tensors, rounded to float32. The task is
memory-bandwidth bound (two reads and one write per element).

- **Origin.** The `add_kernel` of the ROCm Triton benchmark suite (`test_add_kernel.py`, AMD, MIT-style
  header from the old task package; its exact upstream revision was not recorded, so the provenance
  revision says so). The baseline keeps the kernel unchanged (1D grid, block pointer load of `x`,
  masked load of `y`) and launches it with block size 1024.
- **Scenario.** float32 only (sibling of `add/fp16_1d`).
- **Workloads.** The old task used 98432 and 1048576 elements, which take well below 10 us on MI355X;
  they are correctness-only here, with the small non-aligned sizes 127 and 129 added. Performance cases are scaled up (2^22 to 2^29 elements) so every case
  moves at least ~50 MB. The 2^29 case needs about 6 GB of inputs/outputs plus the float64 reference.
- **Comparison.** Default `baseline_relative` bound; the reference is the float64 sum.
- **License note.** The old package carried only an AMD copyright header, no licence text; the
  baseline is treated as MIT (ROCm Triton and the upstream Triton tutorial it derives from), and
  `baseline/LICENSE` reflects that.

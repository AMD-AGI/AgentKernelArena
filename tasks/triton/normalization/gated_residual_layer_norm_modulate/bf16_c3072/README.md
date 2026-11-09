# Gated residual + LayerNorm + adaLN modulate with per-token selection, bfloat16, width 3072

The fused step between two sub-blocks of a diffusion-transformer (DiT) block with adaLN-zero conditioning:
the sub-block output is added to the residual stream through a gate, the new stream is layer-normalized
(no affine parameters), and the normalized rows are modulated by `(1 + scale)` and `shift` of the next sub-block.
The block's modulation vector is produced from the timestep embedding. In the image-edit variant of
Qwen-Image the tokens of the edited image use the modulation of the real timestep (`index` 0) and the tokens of the
reference images use the modulation of timestep zero (`index` 1), so the kernel selects, per token, between two
`(scale, shift, gate)` sets of its sample. It also emits the selected gate (`gate_out`), which the next
residual add uses as `residual_gate`.

```
residual_out = residual + residual_gate * x                (float32, stored as bfloat16)
out          = LayerNorm(residual_out) * (1 + scale) + shift      (scale/shift of set index[b, t])
gate_out     = gate of set index[b, t]
```

## Baseline

SGLang's Triton `_fused_residual_layernorm_scale_shift_gate_select01_kernel`
(`sgl-project/sglang@887c401e151e2d1b082d74ef8769a23eef5c465c`,
`python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py`, Apache-2.0), one program per token row,
used by `QwenImageTransformerBlock._modulate` in
`python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` when `zero_cond_t` is set. The kernel body is copied
unchanged; the launcher is specialized to the non-affine LayerNorm and inlined into `run()` (see
`baseline.provenance.modifications`).

- The upstream file has no SPDX or copyright header, so the repository `LICENSE` (Apache-2.0) applies. No other
  vendored file exists. No "adapted from", "copied from" or third-party URL appears in the file. Its git history
  at the pinned checkout ends at a refactor commit (`19aa19b111`); the select01 kernels were added by the
  commit titled "Opt qwen-image-edit with fuse_residual_layernorm_scale_shift_gate_select01_kernel (#20395)"
  (`e00328d1e5`), and no third-party origin is recorded.
- The upstream wrapper falls back to eager PyTorch on HIP (`forward_hip` calls `forward_native`), so on ROCm
  this fused kernel is a task baseline rather than SGLang's AMD serving path; the kernel itself is plain Triton
  and contains a comment about an AMD Triton pass it was written to avoid (`gfx950`).
- The hidden size 3072 is not a power of two: the kernel runs with `BLOCK_N = 4096` and masks the tail.

## Scenario and constants

- `hidden_size` 3072 = `num_attention_heads` 24 x `attention_head_dim` 128 of `QwenImageArchConfig`
  (`python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`, pinned SGLang checkout); the image-edit
  configuration `QwenImageEditPlus_2511_ArchConfig` sets `zero_cond_t = True` there.
- `epsilon` 1e-6 is the default `eps` of `QwenImageTransformerBlock` (`runtime/models/dits/qwen_image.py`); the block builds its
  norms with `elementwise_affine=False`, so the task has no LayerNorm weight or bias.
- Model dtype is bfloat16.

## Workloads

`seq_len` is the image-token count of one sample: each image contributes (height / 16) x (width / 16) tokens
(VAE scale factor 8, from `temperal_downsample` of the Qwen-Image VAE config, and 2x2 patches; see
`configs/pipeline_configs/qwen_image.py`), and an edit request carries the edited image plus one to three reference
images. Text tokens go through a different path and are not part of `seq_len`. The perf cases run from
1152 tokens (two 384x384 images) to 65536 per sample (four 2048x2048 images); `batch` 2 stands for two
samples batched into one step. The SGLang sources do not state which requests the server batches; the
batch axis only covers the kernel's batched code path. Tiny shapes (1, 3, 33, 257, 1000) are
correctness-only boundary cases. Perf cases move 42 MB (3 reads and 3 writes of 3072 bfloat16 per token) to 4.8 GB;
the 114x span is bounded by the float64 reference memory at the top end.

Inputs: `x`, `residual` standard normal; the six modulation vectors normal with std 0.5 and `residual_gate`
built from two gate rows of std 0.5 per sample, repeated over the tokens of each index. These magnitudes are stand-ins for trained
values, not measured from a checkpoint. `index` is the real layout: the edited image tokens (index 0) come first, then
the reference images (index 1), with the image count drawn from 2 to 4 per sample, equal image sizes (so the
split point is `ceil(seq_len / n_images)`, kept below `seq_len` for `seq_len` above 1). A sequence of 1 has
only index 0.

## Comparison

`out` and `residual_out` use the default `baseline_relative` bound. `gate_out` is a copy of a bfloat16 row of
the selected gate, with no arithmetic, so it is compared `exact`; this is stricter than the default, not a
relaxation. The reference keeps the residual sum unrounded in float64, whereas the kernel normalizes its
float32 sum before storing `residual_out` in bfloat16; the baseline's own error against the reference sets the
bound, as for the other fused-add norm tasks.

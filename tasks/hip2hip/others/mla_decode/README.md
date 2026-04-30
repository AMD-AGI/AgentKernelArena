# Baseline HIP MLA decode

A naive but correct hand-written HIP MLA (multi-latent attention) decode
kernel for AMD CDNA-3 / CDNA-4 (gfx942 / gfx950). Companion task for the
[`hip-and-triton-kernel-optimization`](https://github.com/AMD-AGI/GEAK/pull/203)
skill.

## Shape contract

Hardcoded in `mla_decode.hip`; do not change.

| Dim | Value | Source |
| --- | --- | --- |
| `NHEAD` | 128 | Q heads (decode-shaped, GQA-ratio = 128) |
| `BLOCK_H` | 16 | heads per workgroup |
| `HEAD_GROUPS` | 8 | `NHEAD / BLOCK_H` |
| `LK` | 576 | qk dim (512 NoPE + 64 RoPE) |
| `LV` | 512 | v dim (= `kv_lora_rank`) |
| `decode_qlen` | 1 | one query token per request |
| `page_size` | 1 | KV cache one slot per token |

Q dtype: `bf16`. KV dtype: `fp8 e4m3fn` (bias = 7, saturating). O dtype:
`bf16`. Optional LSE (`fp32 [batch, NHEAD]`) supported.

## Files

- `mla_decode.hip` — kernel + main + host fp32 reference + bench loop.
- `Makefile` — `hipcc -O3 --offload-arch=gfx950 --offload-arch=gfx942`.
- `scripts/task_runner.py` — `compile / correctness / performance` modes.
- `config.yaml` — Arena task descriptor (`task_type: hip2hip`).

## Sweep space

Five representative shapes (`batch`, `ctx`):
`(1, 512)`, `(4, 1024)`, `(16, 2048)`, `(64, 4096)`, `(1, 8192)`.

## Bar

- **Correctness:** `max_abs <= 5e-2 OR max_rel <= 1e-1` against the in-binary
  fp32 host reference, on every shape. The baseline meets this.
- **Performance:** mean device-time per shape across 20 measured iterations
  (3 warmup). The baseline is naive (no MFMA, full-FP32 inner loop); the
  optimization headroom is huge.

## Quick test

```bash
make
./applications_mla_decode --batch 4 --ctx 1024
# Check: max_abs=...  PASS
# Perf:  <us> us/launch | ~BW: <gbs> GB/s

python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

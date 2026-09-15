# mla_decode: native HIP task

Optimize only `source/kernel.hpp`. All task inputs, references,
CPU comparisons, wrappers, bindings, compiler flags, launch boundaries, and
benchmark helpers are protected. The initial implementation is present and
written in HIP. Arena freezes that initial candidate in a separate workspace
for every baseline action; final candidate actions always build and execute the
submitted implementation, with no baseline fallback.

The complete independent manifest in `workload.json` includes **5 cases**
from the original 5 shapes, including every measured operation/layout.
The original harness remains `scripts/task_runner.py`; its seed schedule,
numerical tolerances and output-contract checks are retained. Its independent numerical reference is implemented in the protected C++ host
program; the native benchmark also validates graph replay. The Python reference
check module performs structural checks only. Do not reduce these
cases or alter expected outputs to improve a score.

Compilation uses the actual original HIP compiler/extension build, including
both native verification and benchmark binaries where applicable. The original
warmup (10), sample count (100), CUDA-graph/event fallback policy, state reset,
and allocation/timing boundaries remain in the protected harness. Runtime
requirements are the selected ROCm image, a compatible GPU and HIP compiler,
and PyTorch for extension tasks. Generated performance helpers are supplied by
Arena and must not be edited.

Call `python3 scripts/evaluate.py` followed by `validate-task`, or by
`baseline|candidate` and `compile|correctness|performance`. Each action emits one
`ARENA_EVAL_RESULT=` envelope. Missing cases, compiler errors, numerical errors,
and unavailable runtime dependencies are failures, never implicit skips.

The GPU implementation was extracted verbatim into `source/kernel.hpp`. A protected
`candidate_driver.hip` includes the header in a separate translation unit, so
candidate preprocessor definitions cannot alter the host reference or timing. Its launch
interface and constants are part of the fixed task contract; a Python symbol
scope is not used to protect C++ code.

## Original task instructions

Optimize this baseline HIP MLA (multi-latent attention) decode kernel
for AMD MI300X / MI325X / MI355X (gfx942 / gfx950).

Shape contract (do not change):
  - NHEAD = 128, BLOCK_H = 16, HEAD_GROUPS = 8
  - LK = 576 (= 512 NoPE + 64 RoPE), LV = 512
  - decode_qlen = 1, page_size = 1
  - Q dtype: bf16, KV dtype: fp8 e4m3fn, O dtype: bf16

The baseline is correctness-first and uses naive FP32 dot products
instead of MFMA. Expected gaps to address (in priority order):

  1. LDS bank conflicts on KV staging (use ds_read_b64_tr_b16 / _b8
     transposed reads with XOR-swizzled layout).
  2. MFMA pipe under-fed (switch to mfma_f32_16x16x32_fp8_fp8 or
     the K=128 scaled variant mfma_scale_f32_16x16x128_f8f6f4).
  3. Repeated address and setup work (hoist loop invariants within each workgroup).
  4. State held in LDS instead of registers (Q tile + softmax state
     can move to registers where register pressure permits).
  5. s_waitcnt-bound after MFMA pipe is fed (hand-schedule the
     load / MFMA interleave with __builtin_amdgcn_sched_group_barrier).
  6. FetchSize > algorithmic minimum at high L2 hit rate (use
     buffer_load_dwordx4 with a hand-built v-descriptor).

Maintain accuracy: the binary's "Check: ... PASS" line must pass on
every test shape (max_abs <= 5e-2 OR max_rel <= 1e-1).


## Previous task notes

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

- `source/kernel.hpp` — editable device implementation.
- `mla_decode.hip` — protected main, host fp32 reference, and launch interface.
- `Makefile` — `hipcc -O3 --offload-arch=gfx950 --offload-arch=gfx942`.
- `scripts/task_runner.py` — `compile / correctness / performance` modes.
- `config.yaml` — Arena v2 task descriptor with explicit roles and actions.

## Sweep space

Five representative shapes (`batch`, `ctx`):
`(1, 512)`, `(4, 1024)`, `(16, 2048)`, `(1, 4096)`, `(1, 8192)`.

## Bar

- **Correctness:** `max_abs <= 5e-2 OR max_rel <= 1e-1` against the in-binary
  fp32 host reference, on every shape. The baseline meets this.
- **Performance:** mean device-time per shape across 100 measured iterations
  (10 warmup). The baseline is naive (no MFMA, full-FP32 inner loop); the
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


The native timing driver computes the protected CPU attention reference for
its exact deterministic timing inputs, outside measured calls. It validates the
actual measured output before poisoning and the exact captured replay afterward:
the original eager/replay `1e-2` absolute gate is retained, and both outputs also
satisfy the original host-reference rule (`max_abs <= 5e-2` OR `max_rel <= 1e-1`).
Finite-value checks use exponent bits so `-ffast-math` cannot remove them. Host
reference work uses OpenMP outside GPU timing. No case, seed, launch, warmup,
sample, graph-batching or numerical threshold is weakened. A further fresh-input control sets Q to zero with token-varying FP8 values
at nontrivially mapped locations, whose analytical answer is the valid-token
mean described below. It replays the same graph, checks the original reference
rule, and restores original input buffers even on failure. It rejects returning
the old timed answer or zeros; it does not exclude every possible caching strategy.

The observer also reads back every caller-owned Q, KV, token-map and sequence-
length buffer and checks exact bytes after timed execution, replay and the fresh
known-answer probe. It restores the original four buffers on every validation
exit, including exceptions; restoration failure fails the benchmark. These
copies and checks occur outside timed samples.

## Routing and valid-length workload coverage

All five original case IDs, allocation shapes and Q/KV random seeds remain.
Both correctness and timing use the protected `scripts/native/routing_controls.hpp`
policy declared in every manifest row: reverse the physical cache (including
1024 spare slots), and cycle request lengths through `ctx/2+1`, `ctx`,
`max(1,ctx-1)` and `1`. These valid lengths change the amount of operator work;
older full-length timing results are not comparable to this repaired workload.
The same new policy applies to the frozen baseline and candidate. Bandwidth
metadata counts valid tokens; warmups, samples and numerical gates are unchanged.

Every native correctness case additionally runs Q=0 with exactly representable,
token-varying FP8 values at the mapped locations and -4 in unused/padded slots.
The expected output is the analytical mean of valid tokens, independently of
the attention reference. The actual measured graph/Event invocation repeats this
fresh-input control outside timing, under the original accuracy gate. Ignoring
token indirection or using full capacity as the valid length is checked by CPU
negative controls as well as the real GPU output check. All four original input
buffers are restored by the replay guard on success, failure or exception.

## Fixed native launch

The protected host launches `grid=(batch, HEAD_GROUPS)`, 64 threads per
workgroup and `BLOCK_H * LK * sizeof(float)` (36,864 bytes) of dynamic shared
memory. It supplies no persistent work queue or configurable grid. Optimize
within this launch and kernel signature; residency is not fixed at one WG/CU.
Both correctness and benchmark binaries link the separate protected candidate
translation unit. The editable kernel header bytes, original numerical gates,
current routing/valid-length policy, input seeds and timed work are unchanged.

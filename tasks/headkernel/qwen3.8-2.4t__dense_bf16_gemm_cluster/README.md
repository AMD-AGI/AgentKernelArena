# Qwen dense BF16 native routing

This task tunes the existing SG520 image-native GEMM dispatch for five preserved
Qwen geometries. The editable interface is a literal routing policy. Native
Python launchers, GPU implementations, references and timing callbacks are frozen;
this package does not expose an editable GPU kernel body.

The `native_routing` task type describes this policy interface. Compilation
validates its syntax and complete literal mapping; correctness and performance
execute the selected image-native implementations. All twelve framework
validator checks remain required, and this configuration needs fresh GPU
qualification before it can be accepted.

Edit only the body of `select_routes` in `source/routing.py`. Return every existing
case signature exactly once, mapping it to `recorded`, `torch`, or `triton`.
The evaluator parses this function as data and never imports or executes it.
Calls, imports, decorators, additional statements/functions, duplicate or missing
cases, and all other values are rejected before importing the GPU runtime.

`recorded` calls the original backend with the original solution index, split-K
and kernel name from `ut/meta.json`. The other choices call the corresponding
backend already implemented by the pinned image. No replacement port is added.
An unsupported backend fails the complete evaluation; it is never skipped or
replaced by another route.

| Case | M | N | K | Recorded backend | Tolerance |
|---|---:|---:|---:|---|---:|
| hk04 | 16384 | 4608 | 8192 | hipBLASLt | 0.02 |
| hk05 | 64 | 32 | 8192 | Torch | 0.02 |
| hk06 | 64 | 512 | 8192 | FlyDSL split-K8 | 0.03 |
| hk07 | 64 | 4608 | 8192 | Assembly split-K3 | 0.02 |
| hk08 | 64 | 8192 | 256 | FlyDSL split-K1 | 0.02 |

All inputs retain the original BF16 shapes and strides. Bias, scaling and
preshuffling remain disabled. The comparison uses the existing mixed relative
and RMS-scaled tolerance in `ut/harness_lib.py`, including the original hk06
exception; no tolerance is relaxed by this adapter.

The protected evaluator calls the checked image-owned `solMap` backend directly.
It verifies the live source digest, function/code-file identity, route and fixed
configuration. This avoids the `torch_compile_guard` collision in which a later
candidate wrapper silently calls an already registered native custom op. The
legacy source-loader regression remains covered by CPU tests. An overlay log
message is not accepted as source engagement proof.

Correctness retains two fixed-value draws and three random draws per geometry,
adds checked graph replay, and rejects poisoned no-op and corrupted outputs.
Performance uses the materialized canonical graph helper: exactly ten warmups,
one logical invocation per graph replay, and one hundred ordered samples per
case. Each preparation validates the preceding output before replacing inputs,
creates new CPU-owned A/B values, and poisons the output. Validation owns output
and input snapshots on CPU before computing an independent `F.linear` reference
from those CPU inputs. The exact measured graph is replayed and checked again
outside the sample set. Capture failure, missing cases, changed requests,
nonfinite samples, or an unchecked replay fail the complete result.

Reports and per-case diagnostics are written under `build/`, with a fresh request
and retained raw report for every run. Historical `ut/selection_validation.json`,
`ut/negative_check.json`, `ut/runtime_refresh.json`, native source copies and all
other task inputs remain immutable. The source policy's semantic fingerprint uses
the resolved native configurations and native source checksum, separately from
its file hash. Formatting changes and equivalent route names, such as hk05
`recorded` and `torch`, remain unchanged-source controls. The authoritative
quality adapter owns any accepted comparison or gain.

The image and native source pin are recorded in `ut/runtime_refresh.json` and
`ut/meta.json`. The preserved geometry/selection evidence came from the documented
SG518 workload; it is not a new complete SG520 serving capture. The four decode
rows share the historical 21.43% GPU-time bucket and must not be added four times.
CPU checks are preparation evidence only. Fresh compatible-GPU execution, quality
admission and a framework-finalized PASS are still required before qualification.

Run the ordinary task commands after preparing the pinned image and assigned GPU:

```sh
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Ordinary Arena scoring and framework finalization enforce the task's explicit
`benchmark_quality` adapter. The consumer checks the current protected validator,
source and workload bindings, complete raw/published reports, and every ordered
100-sample timing series. Missing, stale or unstable evidence cannot earn an
accepted speedup or a framework PASS. Stable unchanged-source runs remain valid
task controls with no accepted gain. For native routing, equivalent resolved
backend configurations or observed MoE launch arguments remain unchanged
controls even when their policy labels differ.

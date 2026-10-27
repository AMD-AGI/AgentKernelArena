# Pinned FlyDSL compatibility helpers

Derived from ROCm/FlyDSL commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47`,
`python/flydsl/expr/{buffer_ops,vector,meta}.py`, under Apache-2.0 (see LICENSE).
The original buffer descriptor, byte offset, masking and cache policy is retained.
Relative dependency imports now target the installed package. Removed memref
pointer extraction uses the current typed iterator and LLVM pointer conversion.
Legacy runtimes continue using their installed original helpers. No runtime
monkeypatching, external repository imports or downloads are performed.

Current ROCDL load/store operations receive the same cache-policy bits through
their `aux` attribute instead of a removed SSA operand. Vector unwrapping uses
the current signature while retaining the surrounding MLIR location context.

The bundled `meta.py` uses `traceback.extract_stack` for MLIR debug locations,
without retaining frames or importing `inspect` into the candidate dependency
closure. It preserves the selected caller's filename, line and source label;
debug columns consistently use zero. Kernel arithmetic and benchmark policy
are unchanged.

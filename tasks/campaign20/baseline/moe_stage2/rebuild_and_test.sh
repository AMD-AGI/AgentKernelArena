#!/usr/bin/env bash
# Force-rebuild + test wrapper for the MiniMax-M2.5 CK block-scale MoE STAGE-2 GEMM.
#
# WHY THIS EXISTS (the v1 fix):
#   The MoE GEMM is JIT-compiled by aiter into `module_moe_ck2stages_*` from the CK
#   C++ source `gemm_moe_ck2stages_common_blockscale.cuh` (+ the gridwise template it
#   includes). aiter caches the compiled `.so` and will NOT recompile just because the
#   source changed. So if GEAK edits the .cuh but we don't invalidate the cache, the
#   harness keeps running the OLD kernel and any "speedup" is pure noise -- exactly the
#   failure mode of the original example_MOE2 (which also pointed --kernel-path at the
#   Python dispatch moe_op.py and never rebuilt).
#
#   This wrapper deletes the relevant cached .so + build dir BEFORE the run, so the next
#   kernel call recompiles GEAK's edited .cuh from scratch (~80s) and the harness then
#   benchmarks the REAL recompiled kernel. The same module is what the live
#   run_sglang_test_minimax.sh server uses, so a win here applies back to e2e.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER="${GEAK_REPO_ROOT:-/sgl-workspace/aiter}"
JIT="$AITER/aiter/jit"

# Invalidate the block-scale MoE GEMM modules (covers gemm-stage2 = mulWeightStage1 and
# gemm-stage1 = mulWeightStage2; the harness only triggers a rebuild of the one it calls).
rm -f  "$JIT"/module_moe_ck2stages_*silu_per_1x128*.so      2>/dev/null || true
rm -rf "$JIT"/build/module_moe_ck2stages_*silu_per_1x128*   2>/dev/null || true

# Respect the GPU GEAK assigned us; fall back to 6 for manual runs.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6}"

# First call recompiles the edited .cuh (~80s) and gates on numerics. If correctness
# fails, stop here so GEAK discards the candidate instead of timing a wrong kernel.
python "$HERE/harness_test_moe_stage2_runtime.py" --correctness

# Second call reuses the freshly-built .so (no rebuild) and prints GEAK_RESULT_LATENCY_MS.
python "$HERE/harness_test_moe_stage2_runtime.py" --benchmark

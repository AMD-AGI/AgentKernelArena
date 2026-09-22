#!/usr/bin/env bash
# UNRUN. First-hand stage-1 capture -- needs an 8-GPU window, which ut_stage1 was deliberately
# built WITHOUT (the oracle currently reuses the 0817 stage-2 routing offline; see README.md).
#
# What running this buys, concretely:
#   1. routing captured AT THE STAGE-1 SEAM instead of inverted from the stage-2 one, and
#   2. the DECODE launch variant OBSERVED instead of assumed identical to prefill's -- NOTES.md
#      records a decode bimodality (27.8 us vs 122 us layer groups) that a single assumed variant
#      cannot represent.
# After it lands: point _extract_gen1.py at the resulting routing_capture_stage1_*.pt instead of
# ../ut_stage2/reference_io.pt, regenerate, and drop the second-hand caveats from meta.json's
# provenance_note / routing_provenance.
#
# Paths below are the 0828 cycle-1 ones (the 0817 sibling script used e2e_cycle0 under the 0817
# session); re-point EVAL_DIR/TASK at whatever session is live when this is finally run.
set -u
EVAL_DIR=${EVAL_DIR:-/shared_nfs/hyperloom-claw/Kimi-K3/20260828T135708Z-451fa49c/geak/e2e_cycle1}
TASK=${TASK:-$EVAL_DIR/kernels/flydsl_moe_stage1_gate_up_situv2_task}
mkdir -p "$TASK/_capture" "$TASK/_capture_bench"
cp -r "$(dirname "$0")/_capture_overlay1" "$TASK/_capture_overlay1"

BACKEND=sglang \
OUT_DIR="$TASK/_capture_bench" \
GPU=0,1,2,3,4,5,6,7 TP=8 \
MODEL=/shared_nfs/hyperloom/models/Kimi-K3 \
MEM_FRACTION=0.8 \
ISL=8192 OSL=1024 CONC=64 NUM_PROMPTS=64 NUM_WARMUPS=8 SEED=0 \
REPEATS=0 PROFILE=0 BENCH_COLD_FINAL=0 \
HEALTH_TRIES=2400 OVERLAY_HEALTH_TRIES=2400 SERVING_LOCK_WAIT=14400 \
OVERLAY_PYTHONPATH="$TASK/_capture_overlay1" \
EXTRA_SERVER_ARGS="--trust-remote-code --disable-radix-cache --attention-backend triton --dtype bfloat16 --chunked-prefill-size 8192 --cuda-graph-max-bs 64 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --context-length 13312 --watchdog-timeout 1800 --moe-runner-backend aiter" \
EXTRA_ENV="SGLANG_USE_AITER=1 HF_HUB_TRUST_REMOTE_CODE=1 GEAK_MOE_CAPTURE_OUT=$TASK/_capture GEAK_MOE_CAPTURE_MAX=120" \
BENCH_TRUST_REMOTE_CODE=1 HF_HUB_TRUST_REMOTE_CODE=1 \
  bash "$EVAL_DIR/bench_e2e.sh"
echo "CAPTURE_BENCH_EXIT=$?"
ls -la "$TASK/_capture"

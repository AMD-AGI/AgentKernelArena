#!/usr/bin/env bash
EVAL_DIR=/shared_nfs/hyperloom-claw/Kimi-K3/20260817T115910Z/geak/e2e_cycle0
TASK=/shared_nfs/hyperloom-claw/Kimi-K3/20260817T115910Z/geak/e2e_cycle0/kernels/flydsl_moe_stage2_downproj_plus_reduction_task
BACKEND=sglang \
OUT_DIR="$TASK/_capture_bench" \
GPU=0,1,2,3,4,5,6,7 TP=8 \
MODEL=/shared_nfs/hyperloom/models/Kimi-K3 \
MEM_FRACTION=0.8 \
ISL=8192 OSL=1024 CONC=64 NUM_PROMPTS=64 NUM_WARMUPS=8 SEED=0 \
REPEATS=0 PROFILE=0 BENCH_COLD_FINAL=0 \
HEALTH_TRIES=2400 OVERLAY_HEALTH_TRIES=2400 SERVING_LOCK_WAIT=14400 \
OVERLAY_PYTHONPATH="$TASK/_capture_overlay" \
EXTRA_SERVER_ARGS="--trust-remote-code --disable-radix-cache --attention-backend triton --dtype bfloat16 --cuda-graph-max-bs 256 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --context-length 13312 --watchdog-timeout 1800 --moe-runner-backend aiter" \
EXTRA_ENV="SGLANG_USE_AITER=1 HF_HUB_TRUST_REMOTE_CODE=1 GEAK_MOE_CAPTURE_OUT=$TASK/_capture GEAK_MOE_CAPTURE_MAX=120" \
BENCH_TRUST_REMOTE_CODE=1 HF_HUB_TRUST_REMOTE_CODE=1 \
  bash "$EVAL_DIR/bench_e2e.sh"
echo "CAPTURE_BENCH_EXIT=$?"

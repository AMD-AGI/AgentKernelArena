#!/usr/bin/env bash
# Run compile -> correctness -> performance for one task (or all) on a real GPU,
# inside the image that task's config.yaml names. This is the half of the arena
# validation that tools/validate_suite.py has to SKIP.
#
#   tools/run_on_gpu.sh <jobid> <task-id>        # one task
#   tools/run_on_gpu.sh <jobid> all              # every built task, smallest first
#   tools/run_on_gpu.sh <jobid> pending          # only tasks not yet all-green
#   tools/run_on_gpu.sh <jobid> t1,t2,t3         # exactly these, in this order
#   HK_GPU=3 tools/run_on_gpu.sh <jobid> all     # pin a specific GPU
#
# `pending` exists because spur reservations roll over every ~40 minutes and take
# the running sweep with them, while a full sweep takes hours. Each task archives
# its three reports to _results/<task>/ the moment it finishes, so re-running
# `pending` on whatever job is alive now picks up exactly where the last one died.
# Archives survive tools/clean.sh; build/ does not.
#
# <jobid> is a LIVE spur job -- resolve it at point of use by NODE name, not from
# a remembered ID:  nssh -l | awk '$2=="crsuse2-m2m-192"{print $1}'
# Job IDs are renumbered whenever reservations roll over.
#
# Three things here are the scar tissue from a run that wedged for 8 hours:
#
#   1. GPU CHOICE. Defaults to the emptiest card, not card 0. These nodes are
#      shared; landing on a card someone else is already saturating is how a
#      timing leg ends up blocked in uninterruptible sleep.
#   2. TIMEOUTS THAT ACTUALLY KILL. `timeout` alone sends SIGTERM, which a
#      process stuck in a GPU ioctl ignores forever. Every budget is enforced
#      twice: `timeout -k` inside the container so SIGKILL follows, and a
#      `timeout` around `docker run` plus an explicit `docker kill` outside, so a
#      wedged container can never hold up the remaining tasks.
#   3. ORDER. Smallest oracle first, so the good tasks are not stuck behind one
#      heavyweight that misbehaves.
#   4. TVM_FFI_DISABLE_TORCH_C_DLPACK=1. The sglang:v0.5.18 image ships only the
#      CPU variant of tvm-ffi's optional torch<->dlpack addon, so every fresh
#      subprocess tries to compile the ROCm one, fails, and retries. A UT that
#      forks per leg per draw turned that into 637 concurrent compiles and hung
#      the node for 30 minutes. The addon is a host-side conversion fast path; it
#      is not the device kernel under measurement.
#
# Results: <task>/build/*.json, transcripts in <suite>/_runs/<stamp>/, driver log
# under $HOME (readable from the login node).
set -uo pipefail

SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS="$SUITE/tasks/headkernel"
JOB="${1:?usage: run_on_gpu.sh <jobid> <task-id|all>}"
WHICH="${2:?usage: run_on_gpu.sh <jobid> <task-id|all>}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="$SUITE/_runs/$STAMP"
DRIVER="$HOME/hk_run_$STAMP.sh"
DRIVER_LOG="$HOME/hk_run_$STAMP.log"

# Per-mode wall budgets (seconds). correctness is the expensive one: the GEAK
# driver runs the oracle AND a full interleaved measure_legs sweep, one fresh
# subprocess per bucket per leg, each paying a cold torch import.
T_COMPILE="${HK_T_COMPILE:-300}"
# The DSA/MLA tasks carry 2.6-4 GB oracles and spawn a fresh subprocess per bucket
# per leg; 2700s was not enough for glm-5.2 dsa_mla_core_prefill.
T_CORRECT="${HK_T_CORRECT:-5400}"
T_PERF="${HK_T_PERF:-1200}"

# Who the containers must hand the tree back to (they run as uid 0 on a
# no_root_squash export). Resolved here, on the login node, where the real uid is.
OWNER="$(id -u):$(id -g)"

# Persistent KERNEL-COMPILATION cache, shared across nodes and reservations.
#
# These are compiler artifacts, not warmed kernels: the TileLang DSA tasks spend
# 40+ minutes JIT-compiling before the first comparison, and with a per-container
# /tmp cache every retry paid it again - which, on reservations that live minutes,
# meant they never finished at all. Timing is unaffected because _bench.py and the
# GEAK legs both run 10 warmup iterations before measuring, so a cold vs warm
# compile cache changes setup time, not the device time being reported.
#
# NOT cached here: tvm_ffi, which stays container-local (its addon build is what
# fork-stormed a node once, and it is disabled anyway).
CACHE="${HK_CACHE_DIR:-$SUITE/_cache}"
mkdir -p "$CACHE"/{triton,flydsl,comgr,tilelang,torch_ext}

mkdir -p "$LOGS"

# ---- pick a GPU -------------------------------------------------------------
if [[ -n "${HK_GPU:-}" ]]; then
  GPU="$HK_GPU"
else
  probe_img="$(sed -n 's/^  docker:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' \
    "$TASKS"/*/config.yaml 2>/dev/null | head -1)"
  GPU="$(nssh "$JOB" "docker run --rm -u 0 --entrypoint /bin/bash \
      --device=/dev/kfd --device=/dev/dri --group-add video '$probe_img' \
      -c 'rocm-smi --showmeminfo vram --csv 2>/dev/null'" 2>/dev/null \
    | awk -F, '/^card[0-9]/{gsub(/card/,"",$1); print $3+0, $1+0}' \
    | sort -n | head -1 | awk '{print $2}')"
  GPU="${GPU:-1}"
fi
echo "using GPU $GPU on job $JOB"

# ---- task order: smallest oracle first --------------------------------------
RESULTS="$SUITE/_results"
mkdir -p "$RESULTS"
if [[ "$WHICH" == "all" || "$WHICH" == "pending" ]]; then
  mapfile -t LIST < <(
    for d in "$TASKS"/*/; do
      t="$(basename "$d")"
      [[ -f "$d/config.yaml" ]] || continue
      # `pending` skips a task only when all THREE archived legs are status ok.
      # Skipping on mere existence would leave every red leg permanently unretried
      # while the resume workflow cheerfully reported "nothing to do".
      if [[ "$WHICH" == "pending" ]] && python3 - "$RESULTS/$t" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
ok = all(
    os.path.isfile(os.path.join(d, n))
    and (json.load(open(os.path.join(d, n))).get("status") == "ok")
    for n in ("compile_report.json", "correctness_report.json", "performance_report.json"))
sys.exit(0 if ok else 1)
PYEOF
      then
        continue
      fi
      sz=$(stat -c%s "$d/ut/reference_io.pt" 2>/dev/null || echo 0)
      echo "$sz $t"
    done | sort -n | awk '{print $2}'
  )
elif [[ "$WHICH" == *,* ]]; then
  # Explicit comma-separated list, run in the order given. Useful when you want
  # the UNKNOWN tasks first and the known-bad ones last, instead of letting the
  # smallest-oracle ordering spend a short reservation window re-confirming a
  # failure you already understand.
  IFS=',' read -r -a LIST <<< "$WHICH"
else
  LIST=("$WHICH")
fi
if [[ ${#LIST[@]} -eq 0 ]]; then
  echo "nothing to do: every built task already has an archived result in $RESULTS"
  exit 0
fi
echo "${#LIST[@]} task(s) queued"

# ---- emit the driver --------------------------------------------------------
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  for task in "${LIST[@]}"; do
    dir="$TASKS/$task"
    [[ -f "$dir/config.yaml" ]] || { echo "echo 'skip $task (NOT_BUILT)'"; continue; }
    img="$(sed -n 's/^  docker:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$dir/config.yaml" | head -1)"
    [[ -n "$img" ]] || { echo "echo 'skip $task (no image)'"; continue; }
    # Some packages need a framework patch the stock image does not carry. The
    # only one here is GLM-5.3-Flash's fused_moe, which is the single package
    # that boots sglang's ServerArgs against the real checkpoint and dies with
    # "model type 'glm5_next' not recognized" without it.
    patch="$(sed -n 's/^  pre_run_patch:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$dir/config.yaml" | head -1)"
    # NOTE: this text is spliced into the container's  -c '...'  single-quoted
    # argument, so it must contain NO single quotes and no parentheses. An earlier
    # version used both and broke the quoting for the whole driver, which then
    # died at parse time without running a single task.
    prepatch=""
    if [[ -n "$patch" ]]; then
      prepatch="echo \"    applying $(basename "$patch")\"; git -C /sgl-workspace/sglang apply --whitespace=nowarn $patch || echo \"    WARNING: pre_run_patch did not apply cleanly, the run will likely fail\""
    fi
    cname="hk_${STAMP}_${task//[^a-zA-Z0-9]/_}"
    # Outer budget: every mode, plus slack for image load and torch imports.
    outer=$(( T_COMPILE + T_CORRECT + T_PERF + 600 ))
    cat <<EOS
(
echo "=== $task"
echo "    image $img"
if ! docker image inspect "$img" >/dev/null 2>&1; then
  # Pulls fail intermittently on these nodes (registry throttling / transient DNS).
  # A single attempt used to SKIP the task outright and burn the whole window.
  pulled=0
  for attempt in 1 2 3; do
    echo "    pulling $img attempt \$attempt"
    if docker pull "$img" >/dev/null 2>&1; then pulled=1; break; fi
    sleep 20
  done
  [[ \$pulled -eq 1 ]] || { echo "SKIP $task: cannot pull $img after 3 attempts"; exit 0; }
fi
timeout --signal=KILL $outer docker run --rm -u 0 --name "$cname" --entrypoint /bin/bash \\
  --ipc=host --network=host --shm-size 128G \\
  --device=/dev/kfd --device=/dev/dri --group-add video \\
  --security-opt seccomp=unconfined \\
  -e HIP_VISIBLE_DEVICES=$GPU -e PYTHONUNBUFFERED=1 \\
  -e TRITON_CACHE_DIR=$CACHE/triton -e FLYDSL_RUNTIME_CACHE_DIR=$CACHE/flydsl \\
  -e AMD_COMGR_CACHE_DIR=$CACHE/comgr -e TILELANG_CACHE_DIR=$CACHE/tilelang \\
  -e TORCH_EXTENSIONS_DIR=$CACHE/torch_ext -e TVM_FFI_CACHE_DIR=/tmp/c/tvm_ffi \\
  -e TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \\
  -v /shared_nfs:/shared_nfs "$img" -c '
    mkdir -p /tmp/c/tvm_ffi
    $prepatch
    cd $dir || exit 2
    # task_runner.py has its OWN timeout (HK_TASK_TIMEOUT, default 1800s). If that
    # is shorter than the outer budget it fires first and the outer budget is dead
    # letter -- which is exactly what capped glm-5.2 dsa_mla_core_prefill at 1800s
    # while this script thought it had allowed 2700s. Keep the inner one 60s under
    # the outer so python still exits gracefully and writes a report.
    run() { echo "--- \$1"; HK_TASK_TIMEOUT=\$(( \$2 - 60 )) timeout -k 120 \$2 \
              python3 -u scripts/task_runner.py \$1; echo "exit=\$?"; }
    run compile     $T_COMPILE
    run correctness $T_CORRECT
    run performance $T_PERF
    # The container is uid 0 and /shared_nfs is exported no_root_squash, so
    # everything these three legs wrote - build/, __pycache__, ut/_cand_overlay,
    # ut/reports - lands root-owned and the suite owner then cannot delete it.
    # That is how a previous sweep left 365 undeletable files behind and made
    # tools/clean.sh report success while validate_suite.py FAILed 12 of 15
    # tasks. Hand the tree back before leaving.
    chown -R $OWNER "$dir" 2>/dev/null || echo "    (warning: could not chown $dir back to $OWNER)"' 2>&1 | tee "$LOGS/$task.log"
rc=\$?
# The docker client can die while the container lives on (a wedged GPU ioctl
# ignores SIGTERM); tear it down by name so the next task gets a clean device.
docker kill "$cname" >/dev/null 2>&1 && echo "    (force-stopped $cname after outer timeout)"

# Hand the tree back from OUTSIDE the measured container. The chown inside it is
# the last statement of the script, so any SIGKILL -- from the inner timeout, the
# outer timeout, or the spur reservation being reclaimed mid-run -- skips it and
# leaves root-owned files the suite owner cannot delete. This pass always runs.
docker run --rm -u 0 -v /shared_nfs:/shared_nfs --entrypoint /bin/bash "$img" \\
  -c 'chown -R $OWNER "$dir"' >/dev/null 2>&1 \\
  || echo "    (warning: could not chown $task back to $OWNER)"

# Archive the three reports so progress survives tools/clean.sh and the next
# reservation can resume with \`pending\`.
if [[ -d "$dir/build" ]]; then
  mkdir -p "$RESULTS/$task"
  cp -f "$dir"/build/*.json "$RESULTS/$task/" 2>/dev/null
  echo "    archived \$(ls "$RESULTS/$task" 2>/dev/null | wc -l) report(s) to _results/$task/"
fi
echo "    done $task rc=\$rc"
)
EOS
  done
  echo 'echo ALL_DONE'
} > "$DRIVER"
chmod +x "$DRIVER"

# Parse-check the generated driver before shipping it to the node. A quoting bug
# in any task block kills the WHOLE driver at parse time, and the only symptom is
# an empty log on a machine you may no longer have a reservation on.
if ! bash -n "$DRIVER"; then
  echo "ERROR: generated driver $DRIVER has a syntax error (see above); not launching." >&2
  exit 1
fi

nssh "$JOB" "setsid nohup bash $DRIVER > $DRIVER_LOG 2>&1 < /dev/null & sleep 2; echo launched"

echo
echo "driver   $DRIVER"
echo "log      $DRIVER_LOG"
echo "reports  $LOGS"
echo
echo "poll:    nssh \$JOB 'grep -E \"^=== |^exit=|ALL_DONE\" $DRIVER_LOG | tail -20'"
echo "         (read it on the NODE - the login node's NFS view of \$HOME caches stale)"

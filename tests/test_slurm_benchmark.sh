#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/src/scripts/slurm_benchmark.sh"
CONFIG="$ROOT/config_codex_mi355x_spur.yaml"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_has() {
    local expected="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" == "$expected" ]] && return 0
    done
    fail "missing argument: $expected"
}

assert_not_has() {
    local unexpected="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" != "$unexpected" ]] || fail "unexpected argument: $unexpected"
    done
}

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT
TEST_BIN="$TEST_ROOT/bin"
mkdir -p "$TEST_BIN"

for command_name in srun sbatch docker; do
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$@"\n' > "$TEST_BIN/$command_name"
    chmod +x "$TEST_BIN/$command_name"
done

bash -n "$RUNNER"

# The synchronous development path requests one typed GPU and re-enters the
# runner on the allocated node without trying to launch a native Slurm container.
mapfile -t args < <(
    env \
        PATH="$TEST_BIN:$PATH" \
        AKA_SLURM_TIME=00:30:00 \
        bash "$RUNNER" run --config_name "$CONFIG"
)
assert_has "mi355x:1" "${args[@]}"
assert_has "00:30:00" "${args[@]}"
assert_has "_inside" "${args[@]}"
assert_has "run" "${args[@]}"
assert_has "--config_name" "${args[@]}"
assert_has "$CONFIG" "${args[@]}"
assert_not_has "--container-image" "${args[@]}"

# Parallel runs request all eight GPUs on one node. Optional account, QoS, node,
# and exclusive settings are rendered as scheduler arguments.
mapfile -t args < <(
    env \
        PATH="$TEST_BIN:$PATH" \
        AKA_SLURM_ACCOUNT=aka-account \
        AKA_SLURM_QOS=aka-qos \
        AKA_SLURM_NODELIST=gpu-node-7 \
        AKA_SLURM_EXCLUSIVE=yes \
        bash "$RUNNER" parallel-run --config_name "$CONFIG"
)
assert_has "mi355x:8" "${args[@]}"
assert_has "aka-account" "${args[@]}"
assert_has "aka-qos" "${args[@]}"
assert_has "gpu-node-7" "${args[@]}"
assert_has "--exclusive" "${args[@]}"
assert_has "parallel-run" "${args[@]}"

# Batch mode uses sbatch --parsable and writes scheduler output beneath the
# configured log directory.
LOG_DIR="$TEST_ROOT/slurm-logs"
mapfile -t args < <(
    env \
        PATH="$TEST_BIN:$PATH" \
        AKA_SLURM_LOG_DIR="$LOG_DIR" \
        bash "$RUNNER" submit --config_name "$CONFIG" --run-suffix batch-test
)
[[ -d "$LOG_DIR" ]] || fail "batch log directory was not created"
assert_has "--parsable" "${args[@]}"
assert_has "$LOG_DIR/%x-%j.out" "${args[@]}"
assert_has "$LOG_DIR/%x-%j.err" "${args[@]}"
assert_has "run" "${args[@]}"
assert_has "--run-suffix" "${args[@]}"
assert_has "batch-test" "${args[@]}"

# On an allocated Spur node, the assigned physical GPU is forwarded to the
# Docker runtime and optional /dev/mem passthrough is disabled.
mapfile -t args < <(
    env \
        PATH="$TEST_BIN:$PATH" \
        HOME="$TEST_ROOT/home" \
        SPUR_JOB_ID=42 \
        SPUR_JOB_GPUS=3 \
        ROCR_VISIBLE_DEVICES=3 \
        AKA_SLURM_EXPECTED_GPUS=1 \
        bash "$RUNNER" _inside smoke 2>/dev/null
)
assert_has "ROCR_VISIBLE_DEVICES=3" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_ISOLATED_HOME=1" "${args[@]}"
assert_not_has "/dev/mem" "${args[@]}"

# The wrapper fails before Docker when the scheduler exposes a different number
# of GPUs than requested.
if env \
    PATH="$TEST_BIN:$PATH" \
    SPUR_JOB_ID=43 \
    SPUR_JOB_GPUS=0,1 \
    AKA_SLURM_EXPECTED_GPUS=8 \
    bash "$RUNNER" _inside parallel-smoke >"$TEST_ROOT/mismatch.out" 2>&1; then
    fail "GPU allocation mismatch unexpectedly succeeded"
fi
grep -q "Requested 8 GPU(s), but allocation exposes 2" "$TEST_ROOT/mismatch.out" \
    || fail "GPU allocation mismatch did not report the expected error"

echo "PASS: Slurm/Spur resource and Docker handoff argument tests"

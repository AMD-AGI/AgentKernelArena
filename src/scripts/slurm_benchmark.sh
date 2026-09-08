#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# sbatch copies the submitted script into its spool directory before execution.
# Preserve the real shared-filesystem checkout path through the job environment.
HOST_ROOT="${AKA_SLURM_HOST_ROOT:-$SCRIPT_ROOT}"
DOCKER_RUNNER="$HOST_ROOT/src/scripts/docker_benchmark.sh"

DEFAULT_PARTITION="amd-spur"
DEFAULT_GPU_TYPE="mi355x"
DEFAULT_GPU_ARCH="gfx950"
DEFAULT_DEV_CPUS=16
DEFAULT_DEV_MEM="64G"
DEFAULT_DEV_TIME="04:00:00"
DEFAULT_PARALLEL_GPUS=8
DEFAULT_PARALLEL_CPUS=64
DEFAULT_PARALLEL_MEM="512G"
DEFAULT_PARALLEL_TIME="1-00:00:00"

usage() {
    cat <<'EOF'
Usage:
  src/scripts/slurm_benchmark.sh shell
  src/scripts/slurm_benchmark.sh smoke
  src/scripts/slurm_benchmark.sh parallel-smoke
  src/scripts/slurm_benchmark.sh check-agents [docker check-agents args...]
  src/scripts/slurm_benchmark.sh run [main.py args...]
  src/scripts/slurm_benchmark.sh parallel-run [main.py args...]
  src/scripts/slurm_benchmark.sh submit [main.py args...]
  src/scripts/slurm_benchmark.sh parallel-submit [main.py args...]

The login node only submits or starts the allocation. On the allocated GPU
node, the runner reuses src/scripts/docker_benchmark.sh and maps the GPUs that
Spur/Slurm assigned to the job into the Docker worker containers.

Environment overrides:
  AKA_SLURM_PARTITION          Partition (default: amd-spur)
  AKA_SLURM_GPU_TYPE           Slurm GPU type (default: mi355x)
  AKA_SLURM_GPU_ARCH           Runtime gfx architecture (default: gfx950)
  AKA_SLURM_GPU_COUNT          GPU count for shell/smoke/run/submit (default: 1)
  AKA_SLURM_PARALLEL_GPU_COUNT GPU count for parallel modes (default: 8)
  AKA_SLURM_CPUS               CPUs for single-GPU modes (default: 16)
  AKA_SLURM_MEM                Memory for single-GPU modes (default: 64G)
  AKA_SLURM_TIME               Time for single-GPU modes (default: 04:00:00)
  AKA_SLURM_PARALLEL_CPUS      CPUs for parallel modes (default: 64)
  AKA_SLURM_PARALLEL_MEM       Memory for parallel modes (default: 512G)
  AKA_SLURM_PARALLEL_TIME      Time for parallel modes (default: 1-00:00:00)
  AKA_SLURM_ACCOUNT            Optional account
  AKA_SLURM_QOS                Optional QoS
  AKA_SLURM_NODELIST           Optional node restriction
  AKA_SLURM_EXCLUSIVE          Set to 1/true/yes for exclusive-node allocation
  AKA_SLURM_LOG_DIR            Batch log directory (default: logs/slurm)
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

in_allocation() {
    [[ -n "${SPUR_JOB_ID:-${SLURM_JOB_ID:-}}" ]]
}

allocation_job_id() {
    printf '%s\n' "${SPUR_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
}

allocated_gpu_ids() {
    local raw
    raw="${SPUR_JOB_GPUS:-${SLURM_JOB_GPUS:-${ROCR_VISIBLE_DEVICES:-}}}"
    [[ -n "$raw" ]] || die "No allocated GPU IDs found in SPUR_JOB_GPUS, SLURM_JOB_GPUS, or ROCR_VISIBLE_DEVICES"
    printf '%s\n' "${raw//,/ }" | tr ' ' '\n' | sed '/^$/d'
}

load_allocated_gpus() {
    ALLOCATED_GPUS=()
    local gpu_id
    while IFS= read -r gpu_id; do
        [[ -n "$gpu_id" ]] && ALLOCATED_GPUS+=("$gpu_id")
    done < <(allocated_gpu_ids)

    [[ "${#ALLOCATED_GPUS[@]}" -gt 0 ]] || die "The allocation contains no GPUs"
    if [[ -n "${AKA_SLURM_EXPECTED_GPUS:-}" && "${#ALLOCATED_GPUS[@]}" -ne "$AKA_SLURM_EXPECTED_GPUS" ]]; then
        die "Requested $AKA_SLURM_EXPECTED_GPUS GPU(s), but allocation exposes ${#ALLOCATED_GPUS[@]}: ${ALLOCATED_GPUS[*]}"
    fi
}

configure_allocated_runtime() {
    local mode="$1"
    local job_id
    job_id="$(allocation_job_id)"

    export AKA_GPU_ARCH="${AKA_SLURM_GPU_ARCH:-$DEFAULT_GPU_ARCH}"
    export AKA_SKIP_DEV_MEM=1
    export AGENT_HOME_ISOLATION=1
    export AKA_CONTAINER_HOME="/tmp/aka-home-slurm-${job_id}-${mode}"
    export AKA_CACHE_SUFFIX="slurm-${job_id}-${mode}"
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed on allocated node $(hostname)"
    docker info >/dev/null 2>&1 || die "docker daemon is not accessible on allocated node $(hostname)"
}

run_parallel_smoke() {
    local -a pids=()
    local worker_id gpu_id
    local failed=0

    for worker_id in "${!ALLOCATED_GPUS[@]}"; do
        gpu_id="${ALLOCATED_GPUS[$worker_id]}"
        (
            echo "Parallel smoke worker $worker_id: host_gpu=$gpu_id" >&2
            export AKA_VISIBLE_GPU="$gpu_id"
            export AKA_WORKER_ID="smoke-$worker_id"
            export AKA_CONTAINER_HOME="/tmp/aka-home-slurm-$(allocation_job_id)-smoke-$worker_id"
            export AKA_CACHE_SUFFIX="slurm-$(allocation_job_id)-smoke-$worker_id"
            bash "$DOCKER_RUNNER" smoke
        ) &
        pids+=("$!")
    done

    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

run_inside_allocation() {
    local mode="${1:-}"
    [[ -n "$mode" ]] || die "internal allocation mode is required"
    shift

    in_allocation || die "internal mode '$mode' must run inside a Spur/Slurm allocation"
    load_allocated_gpus
    configure_allocated_runtime "$mode"
    require_docker

    echo "Slurm allocation: job=$(allocation_job_id) node=$(hostname) gpus=${ALLOCATED_GPUS[*]} mode=$mode" >&2

    case "$mode" in
        shell)
            export AKA_VISIBLE_GPU="${ALLOCATED_GPUS[0]}"
            exec bash "$DOCKER_RUNNER" shell
            ;;
        smoke)
            export AKA_VISIBLE_GPU="${ALLOCATED_GPUS[0]}"
            exec bash "$DOCKER_RUNNER" smoke
            ;;
        parallel-smoke)
            run_parallel_smoke
            ;;
        check-agents)
            export AKA_VISIBLE_GPU="${ALLOCATED_GPUS[0]}"
            exec bash "$DOCKER_RUNNER" check-agents "$@"
            ;;
        run)
            export AKA_VISIBLE_GPU="${ALLOCATED_GPUS[0]}"
            exec bash "$DOCKER_RUNNER" run "$@"
            ;;
        parallel-run)
            export GPU_IDS
            GPU_IDS="$(IFS=,; echo "${ALLOCATED_GPUS[*]}")"
            exec bash "$DOCKER_RUNNER" parallel-run "$@"
            ;;
        *)
            die "unknown internal allocation mode: $mode"
            ;;
    esac
}

build_resource_args() {
    local gpu_count="$1"
    local profile="$2"
    local job_name="$3"
    is_positive_integer "$gpu_count" || die "GPU count must be a positive integer; got '$gpu_count'"

    local cpus mem time_limit
    if [[ "$profile" == "parallel" ]]; then
        cpus="${AKA_SLURM_PARALLEL_CPUS:-$DEFAULT_PARALLEL_CPUS}"
        mem="${AKA_SLURM_PARALLEL_MEM:-$DEFAULT_PARALLEL_MEM}"
        time_limit="${AKA_SLURM_PARALLEL_TIME:-$DEFAULT_PARALLEL_TIME}"
    else
        cpus="${AKA_SLURM_CPUS:-$DEFAULT_DEV_CPUS}"
        mem="${AKA_SLURM_MEM:-$DEFAULT_DEV_MEM}"
        time_limit="${AKA_SLURM_TIME:-$DEFAULT_DEV_TIME}"
    fi

    RESOURCE_ARGS=(
        -p "${AKA_SLURM_PARTITION:-$DEFAULT_PARTITION}"
        -N 1
        -n 1
        -c "$cpus"
        --mem "$mem"
        -t "$time_limit"
        -G "${AKA_SLURM_GPU_TYPE:-$DEFAULT_GPU_TYPE}:${gpu_count}"
        -D "$HOST_ROOT"
        -J "$job_name"
    )

    [[ -n "${AKA_SLURM_ACCOUNT:-}" ]] && RESOURCE_ARGS+=(-A "$AKA_SLURM_ACCOUNT")
    [[ -n "${AKA_SLURM_QOS:-}" ]] && RESOURCE_ARGS+=(-q "$AKA_SLURM_QOS")
    [[ -n "${AKA_SLURM_NODELIST:-}" ]] && RESOURCE_ARGS+=(-w "$AKA_SLURM_NODELIST")
    case "${AKA_SLURM_EXCLUSIVE:-0}" in
        1|true|yes) RESOURCE_ARGS+=(--exclusive) ;;
        0|false|no|'') ;;
        *) die "AKA_SLURM_EXCLUSIVE must be 0/1, false/true, or no/yes" ;;
    esac
}

run_with_srun() {
    local mode="$1"
    local gpu_count="$2"
    local profile="$3"
    shift 3

    if in_allocation; then
        AKA_SLURM_EXPECTED_GPUS="$gpu_count" run_inside_allocation "$mode" "$@"
        return
    fi

    command -v srun >/dev/null 2>&1 || die "srun is not installed on this host"
    build_resource_args "$gpu_count" "$profile" "aka-$mode"
    export AKA_SLURM_HOST_ROOT="$HOST_ROOT"
    export AKA_SLURM_EXPECTED_GPUS="$gpu_count"
    if [[ "$mode" == "shell" ]]; then
        exec srun "${RESOURCE_ARGS[@]}" --pty bash "$0" _inside "$mode" "$@"
    fi
    exec srun "${RESOURCE_ARGS[@]}" bash "$0" _inside "$mode" "$@"
}

submit_with_sbatch() {
    local mode="$1"
    local gpu_count="$2"
    local profile="$3"
    shift 3

    in_allocation && die "submit commands must be invoked from outside an existing allocation"
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not installed on this host"
    build_resource_args "$gpu_count" "$profile" "aka-$mode"

    local log_dir="${AKA_SLURM_LOG_DIR:-$HOST_ROOT/logs/slurm}"
    mkdir -p "$log_dir"
    RESOURCE_ARGS+=(
        --output "$log_dir/%x-%j.out"
        --error "$log_dir/%x-%j.err"
    )
    export AKA_SLURM_HOST_ROOT="$HOST_ROOT"
    export AKA_SLURM_EXPECTED_GPUS="$gpu_count"
    sbatch --parsable "${RESOURCE_ARGS[@]}" "$0" _inside "$mode" "$@"
}

case "${1:-}" in
    shell)
        shift
        run_with_srun shell "${AKA_SLURM_GPU_COUNT:-1}" dev "$@"
        ;;
    smoke)
        shift
        run_with_srun smoke "${AKA_SLURM_GPU_COUNT:-1}" dev "$@"
        ;;
    parallel-smoke)
        shift
        run_with_srun parallel-smoke "${AKA_SLURM_PARALLEL_GPU_COUNT:-$DEFAULT_PARALLEL_GPUS}" parallel "$@"
        ;;
    check-agents)
        shift
        run_with_srun check-agents "${AKA_SLURM_GPU_COUNT:-1}" dev "$@"
        ;;
    run)
        shift
        run_with_srun run "${AKA_SLURM_GPU_COUNT:-1}" dev "$@"
        ;;
    parallel-run)
        shift
        run_with_srun parallel-run "${AKA_SLURM_PARALLEL_GPU_COUNT:-$DEFAULT_PARALLEL_GPUS}" parallel "$@"
        ;;
    submit)
        shift
        submit_with_sbatch run "${AKA_SLURM_GPU_COUNT:-1}" dev "$@"
        ;;
    parallel-submit)
        shift
        submit_with_sbatch parallel-run "${AKA_SLURM_PARALLEL_GPU_COUNT:-$DEFAULT_PARALLEL_GPUS}" parallel "$@"
        ;;
    _inside)
        shift
        run_inside_allocation "$@"
        ;;
    -h|--help|help|'')
        usage
        ;;
    *)
        usage >&2
        die "unknown command: $1"
        ;;
esac

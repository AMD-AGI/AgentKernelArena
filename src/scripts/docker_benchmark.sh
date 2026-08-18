#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DOCKER_IMAGE_GFX942="${AKA_DOCKER_IMAGE_GFX942:-lmsysorg/sglang:v0.5.12-rocm720-mi30x}"
GFX950_V0514_DOCKER_IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705"
GFX950_V0514_MANIFEST_DIGEST="sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78"
GFX950_V0514_IMMUTABLE_IMAGE="lmsysorg/sglang-rocm@${GFX950_V0514_MANIFEST_DIGEST}"
DEFAULT_DOCKER_IMAGE_GFX950="${AKA_DOCKER_IMAGE_GFX950:-$GFX950_V0514_DOCKER_IMAGE}"
CONTAINER_WORKDIR="${AKA_DOCKER_WORKDIR:-/workspace}"
HOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_HOME="${HOME:?HOME must be set}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
SELECTED_GPU_ARCH=""
SELECTED_IMAGE=""
AGENT_STATE_MOUNT_ROOT="${AKA_AGENT_STATE_MOUNT_ROOT:-/opt/aka-agent-state}"
DEFAULT_RUN_CONFIG="example_configs/quickstart_claude_mi300.yaml"
# Set by host-side commands after reading the selected run config. Keep this
# separate from REQUIRED_AGENTS because geak_v4 is normalized to claude_code
# before Docker arguments are built.
GEAK_V4_RUNTIME=0
# quality_loop keeps the repository checkout read-only in the agent container.
# Only these host-validated, run-specific subdirectories are over-mounted rw.
QUALITY_LOOP_ARTIFACT_REL=""
QUALITY_LOOP_WORKTREE_REL=""
EVAL_TOOL_SOCKET_CONTAINER_DIR="/run/aka-eval-tools"
EVAL_TOOL_INPUT_CONTAINER_DIR="/input"
EVAL_TOOL_FRAMEWORK_CONTAINER_ROOT="/opt/aka-eval-tools"
EVAL_TOOL_SCRATCH_CONTAINER_DIR="/work"
EVAL_TOOL_ARTIFACT_CONTAINER_DIR="/artifacts"
EVAL_TOOL_RUNTIME_DIR=""
EVAL_TOOL_SOCKET_HOST_DIR=""
EVAL_TOOL_ARTIFACT_HOST_ROOT=""
EVAL_TOOL_ARTIFACT_SCORING_ROOT=""
EVAL_TOOL_SELECTED=""
eval_tool_docker_args=()
eval_tool_container_names=()
eval_tool_ids=()

# /opt/venv/bin is placed before /usr/local/bin and /usr/bin so that a bare
# `python3` / `pytest` resolves to the torch-enabled venv interpreter rather than
# the system python (which lacks torch). Without this, repository tasks whose
# commands call `python3 scripts/task_runner.py` fail with ModuleNotFoundError: torch.
container_path="/opt/claude-node/bin:/opt/node/bin:${HOST_HOME}/.local/bin:/opt/venv/bin:/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

usage() {
    cat <<'EOF'
Usage:
  src/scripts/docker_benchmark.sh run [main.py args...]
  src/scripts/docker_benchmark.sh parallel-run [main.py args...]
  src/scripts/docker_benchmark.sh preflight [--config_name <run-config.yaml>]
  src/scripts/docker_benchmark.sh shell
  src/scripts/docker_benchmark.sh check-agents [--config_name <run-config.yaml>]
  src/scripts/docker_benchmark.sh quality-loop [--config <quality-loop-config.yaml>] [quality_loop args...]
  src/scripts/docker_benchmark.sh smoke
  src/scripts/docker_benchmark.sh eval-tools-smoke
  src/scripts/docker_benchmark.sh build-eval-tool-images

Default run config:
  example_configs/quickstart_claude_mi300.yaml (MI300/MI300X).
  On another GPU, pass --config_name with a matching run configuration.

Environment overrides:
  GPU_IDS                 Comma/space separated GPU indices for parallel-run.
  AKA_LOGICAL_GPU         Logical GPU index inside a masked worker container (default: 0).
  AKA_DOCKER_IMAGE        Absolute Docker image override.
  AKA_GPU_ARCH            GPU arch override for shell/smoke, or run configs without target_gpu_model.
  AKA_DOCKER_IMAGE_<ARCH> Per-arch image override, e.g. AKA_DOCKER_IMAGE_GFX950=...
  AKA_DOCKER_IMAGE_GFX942 Default image for gfx942.
  AKA_DOCKER_IMAGE_GFX950 Default image for gfx950.
  AKA_NODE_PREFIX         Host Node prefix containing bin/node and npm-installed agent CLI(s).
  AKA_AGENTS              Agent CLI(s) to check, comma/space separated; use all for all three.
  AKA_EVAL_TOOLS          Override evaluation_tools.enabled (comma/space separated).
  AKA_EVAL_TOOL_IMAGE_<TOOL>
                           Per-tool sidecar image override, e.g. AKA_EVAL_TOOL_IMAGE_GPU_ASAN.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

warn() {
    echo "WARNING: $*" >&2
}

require_path() {
    local path="$1"
    local label="$2"
    [[ -e "$path" ]] || die "$label not found: $path"
}

normalize_gpu_arch() {
    local arch="$1"
    arch="${arch%%:*}"
    case "$arch" in
        gfx*) printf '%s\n' "$arch" ;;
        [0-9]*) printf 'gfx%s\n' "$arch" ;;
        *) printf '%s\n' "$arch" ;;
    esac
}

docker_image_for_arch() {
    local arch="$1"
    local arch_upper env_name env_image
    arch_upper="$(printf '%s' "$arch" | tr '[:lower:]' '[:upper:]')"
    env_name="AKA_DOCKER_IMAGE_${arch_upper}"
    env_image="${!env_name:-}"
    if [[ -n "$env_image" ]]; then
        printf '%s\n' "$env_image"
        return
    fi

    case "$arch" in
        gfx942) printf '%s\n' "$DEFAULT_DOCKER_IMAGE_GFX942" ;;
        gfx950) printf '%s\n' "$DEFAULT_DOCKER_IMAGE_GFX950" ;;
        *)
            die "No Docker image mapping for GPU arch '$arch'. Set AKA_DOCKER_IMAGE or ${env_name}."
            ;;
    esac
}

uses_gfx950_v0514_runtime() {
    [[ "$SELECTED_GPU_ARCH" == "gfx950" ]] || return 1
    [[ "$SELECTED_IMAGE" == "$GFX950_V0514_DOCKER_IMAGE" \
        || "$SELECTED_IMAGE" == "$GFX950_V0514_IMMUTABLE_IMAGE" \
        || ( -n "${AKA_SCORING_IMAGE_RUNTIME_REF:-}" \
            && "$SELECTED_IMAGE" == "$AKA_SCORING_IMAGE_RUNTIME_REF" ) ]]
}

read_target_gpu_model() {
    local config="$1"
    [[ -f "$config" ]] || die "config file not found: $config"
    sed -nE "s/^[[:space:]]*target_gpu_model[[:space:]]*:[[:space:]]*['\"]?([^'\"#[:space:]]+).*/\1/p" "$config" | head -n 1
}

resolve_gfx_arch_from_model() {
    local model="$1"
    local cheatsheet="$HOST_ROOT/src/prompts/cheatsheet/default_cheatsheet.yaml"
    [[ -f "$cheatsheet" ]] || die "default cheatsheet not found: $cheatsheet"
    awk -v key="$model" '
        function trim(s) {
            sub(/^[[:space:]]+/, "", s)
            sub(/[[:space:]]+$/, "", s)
            return s
        }
        /^[[:space:]]{2}[^[:space:]][^:]*:[[:space:]]*$/ {
            current = $0
            sub(/^[[:space:]]*/, "", current)
            sub(/:.*/, "", current)
            current = trim(current)
        }
        current != "" && toupper(current) == toupper(key) && /gfx_arch[[:space:]]*:/ {
            val = $0
            sub(/.*gfx_arch[[:space:]]*:[[:space:]]*/, "", val)
            sub(/[[:space:]#].*/, "", val)
            print val
            exit
        }
    ' "$cheatsheet"
}

resolve_config_gpu_arch() {
    local config="$1"
    local model
    model="$(read_target_gpu_model "$config")"
    if [[ -z "$model" ]]; then
        if [[ -n "${AKA_GPU_ARCH:-}" ]]; then
            normalize_gpu_arch "$AKA_GPU_ARCH"
            return
        fi
        die "target_gpu_model not found in $config; set AKA_GPU_ARCH or add target_gpu_model"
    fi

    local arch
    arch="$(resolve_gfx_arch_from_model "$model")"
    [[ -n "$arch" ]] || die "No gfx_arch mapping for target_gpu_model '$model' in default_cheatsheet.yaml"
    normalize_gpu_arch "$arch"
}

detect_host_gpu_arch() {
    if [[ -n "${AKA_GPU_ARCH:-}" ]]; then
        normalize_gpu_arch "$AKA_GPU_ARCH"
        return
    fi

    local enumerator=""
    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        enumerator="$(command -v rocm_agent_enumerator)"
    elif [[ -x /opt/rocm/bin/rocm_agent_enumerator ]]; then
        enumerator="/opt/rocm/bin/rocm_agent_enumerator"
    fi

    if [[ -n "$enumerator" ]]; then
        "$enumerator" 2>/dev/null | sed -nE 's/^(gfx[0-9a-zA-Z]+).*/\1/p' | head -n 1
        return
    fi

    local info=""
    if command -v rocminfo >/dev/null 2>&1; then
        info="$(command -v rocminfo)"
    elif [[ -x /opt/rocm/bin/rocminfo ]]; then
        info="/opt/rocm/bin/rocminfo"
    fi

    if [[ -n "$info" ]]; then
        "$info" 2>/dev/null | sed -nE 's/.*Name:[[:space:]]*(gfx[0-9a-zA-Z]+).*/\1/p' | head -n 1
    fi
}

select_runtime() {
    local arch="$1"
    [[ -n "$arch" ]] || die "Could not infer GPU arch; set AKA_GPU_ARCH=gfx942 or AKA_GPU_ARCH=gfx950"

    SELECTED_GPU_ARCH="$(normalize_gpu_arch "$arch")"
    if [[ -n "${AKA_DOCKER_IMAGE:-}" ]]; then
        SELECTED_IMAGE="$AKA_DOCKER_IMAGE"
    else
        SELECTED_IMAGE="$(docker_image_for_arch "$SELECTED_GPU_ARCH")"
    fi
    echo "Docker runtime: arch=${SELECTED_GPU_ARCH} image=${SELECTED_IMAGE}" >&2
}

select_runtime_for_config() {
    local config="$1"
    select_runtime "$(resolve_config_gpu_arch "$config")"
}

select_runtime_for_host() {
    select_runtime "$(detect_host_gpu_arch)"
}

detect_node_prefix() {
    if [[ -n "${AKA_NODE_PREFIX:-}" ]]; then
        printf '%s\n' "$AKA_NODE_PREFIX"
        return
    fi

    local node_bin
    node_bin="$(command -v node || true)"
    [[ -n "$node_bin" ]] || die "node not found on host PATH; needed for mounted Codex CLI"

    node_bin="$(readlink -f "$node_bin")"
    dirname "$(dirname "$node_bin")"
}

# Resolve the Node prefix that owns a CLI installed with `npm -g`. Prefer an
# explicit override, then the CLI found on PATH, then the active Node prefix.
# Return non-zero when no prefix contains both bin/node and bin/<cli>.
detect_node_cli_prefix() {
    local cli="$1"
    local prefix="" cli_bin=""

    if [[ -n "${AKA_NODE_PREFIX:-}" ]]; then
        prefix="$AKA_NODE_PREFIX"
        if [[ -e "$prefix/bin/node" && -e "$prefix/bin/$cli" ]]; then
            printf '%s\n' "$prefix"
            return 0
        fi
        return 1
    fi

    cli_bin="$(command -v "$cli" || true)"
    if [[ -n "$cli_bin" && "$(basename "$(dirname "$cli_bin")")" == "bin" ]]; then
        prefix="$(dirname "$(dirname "$cli_bin")")"
        if [[ -e "$prefix/bin/node" && -e "$prefix/bin/$cli" ]]; then
            printf '%s\n' "$prefix"
            return 0
        fi
    fi

    if command -v node >/dev/null 2>&1; then
        prefix="$(detect_node_prefix)"
        if [[ -e "$prefix/bin/node" && -e "$prefix/bin/$cli" ]]; then
            printf '%s\n' "$prefix"
            return 0
        fi
    fi
    return 1
}

docker_args=()
declare -A _MOUNTED_TARGETS=()

add_mount() {
    local source="$1"
    local target="$2"
    local mode="${3:-}"
    # Skip duplicate targets (e.g. ~/.local/bin is shared by claude + cursor).
    if [[ -n "${_MOUNTED_TARGETS[$target]:-}" ]]; then
        return 0
    fi
    _MOUNTED_TARGETS[$target]=1
    if [[ -n "$mode" ]]; then
        docker_args+=(-v "${source}:${target}:${mode}")
    else
        docker_args+=(-v "${source}:${target}")
    fi
}

# Require a path only when strict; otherwise return non-zero so the caller can
# skip an agent that is not installed (best-effort provisioning).
need_path() {
    local path="$1" label="$2" strict="${3:-1}"
    if [[ -e "$path" ]]; then
        return 0
    fi
    [[ "$strict" == "1" ]] && die "$label not found: $path"
    return 1
}

# Parse the configured agent template from a run config (best-effort).
read_agent_template() {
    local config="$1"
    [[ -f "$config" ]] || return 0
    sed -nE 's/^[[:space:]]+template:[[:space:]]*["'"'"']?([A-Za-z0-9_]+).*/\1/p' "$config" | head -n 1
}

configure_geak_v4_runtime() {
    local config="$1"
    GEAK_V4_RUNTIME=0
    if [[ "$(read_agent_template "$config")" == "geak_v4" ]]; then
        GEAK_V4_RUNTIME=1
    fi
}

agent_list_contains() {
    local agents="$1"
    local expected="$2"
    local agent
    for agent in $agents; do
        [[ "$agent" == "$expected" ]] && return 0
    done
    return 1
}

# task_validator delegates to a backend CLI; read which one.
read_validator_backend() {
    local cfg="$HOST_ROOT/agents/task_validator/agent_config.yaml"
    [[ -f "$cfg" ]] || { printf 'claude_code\n'; return; }
    sed -nE 's/^backend:[[:space:]]*["'"'"']?([A-Za-z0-9_]+).*/\1/p' "$cfg" | head -n 1
}

# Decide which agent CLIs to provision into the container.
# AKA_AGENTS env (comma/space list) overrides; else derive from config's
# agent.template (task_validator -> its backend); else all three.
resolve_required_agents() {
    local config="${1:-}"
    if [[ -n "${AKA_AGENTS:-}" ]]; then
        printf '%s\n' "${AKA_AGENTS//,/ }"
        return
    fi
    local tmpl=""
    [[ -n "$config" ]] && tmpl="$(read_agent_template "$config")"
    if [[ -z "$tmpl" ]]; then
        printf 'codex claude_code cursor\n'
        return
    fi
    [[ "$tmpl" == "task_validator" ]] && tmpl="$(read_validator_backend)"
    case "$tmpl" in
        claude|claude_code) printf 'claude_code\n' ;;
        cursor|cursor-agent) printf 'cursor\n' ;;
        codex) printf 'codex\n' ;;
        # GEAK v4 drives Claude Code; extra deps handled in build/preflight.
        geak_v4|geak-v4|geak) printf 'claude_code\n' ;;
        *) printf '%s\n' "$tmpl" ;;
    esac
}

# Normalize the user-facing check list and reject specialized integrations that
# do not use one of the three host CLI mount paths.
normalize_check_agents() {
    local raw="$*"
    raw="${raw//,/ }"
    local -a normalized=()
    local agent
    for agent in $raw; do
        case "$agent" in
            all)
                normalized+=(codex claude_code cursor)
                ;;
            claude|claude_code|geak_v4|geak-v4|geak)
                normalized+=(claude_code)
                ;;
            cursor|cursor-agent)
                normalized+=(cursor)
                ;;
            codex)
                normalized+=(codex)
                ;;
            *)
                die "docker-check-agents only supports codex, claude_code, cursor, or all; got '$agent'"
                ;;
        esac
    done
    [[ "${#normalized[@]}" -gt 0 ]] || die "No agent selected for docker-check-agents"
    printf '%s\n' "${normalized[*]}"
}

# Mount one agent's CLI install + auth dirs. $2=strict (1 require, 0 best-effort).
mount_agent() {
    local agent="$1" strict="${2:-1}"
    local isolate="${AGENT_HOME_ISOLATION:-0}"
    case "$agent" in
        codex)
            local node_prefix
            node_prefix="$(detect_node_cli_prefix codex || true)"
            if [[ -z "$node_prefix" ]]; then
                [[ "$strict" == "1" ]] && die "npm-installed Codex not found on host PATH or under AKA_NODE_PREFIX"
                warn "npm-installed Codex not found; skipping Codex agent mounts"
                return 0
            fi
            need_path "$node_prefix/bin/node" "host node" "$strict" || return 0
            need_path "$node_prefix/bin/codex" "host codex" "$strict" || return 0
            need_path "$HOST_HOME/.codex" "Codex auth/config directory" "$strict" || return 0
            add_mount "$node_prefix" /opt/node ro
            if [[ "$isolate" == "1" ]]; then
                add_mount "$HOST_HOME/.codex" "$AGENT_STATE_MOUNT_ROOT/.codex" ro
            else
                add_mount "$HOST_HOME/.codex" "$HOST_HOME/.codex"
            fi
            ;;
        claude_code)
            local native_claude_bin="$HOST_HOME/.local/bin/claude"
            local native_claude_root="$HOST_HOME/.local/share/claude"
            local claude_node_prefix=""
            if [[ -e "$native_claude_bin" && -d "$native_claude_root" ]]; then
                add_mount "$HOST_HOME/.local/bin" "$HOST_HOME/.local/bin" ro
                add_mount "$native_claude_root" "$native_claude_root" ro
            else
                claude_node_prefix="$(detect_node_cli_prefix claude || true)"
                if [[ -z "$claude_node_prefix" ]]; then
                    if [[ "$strict" == "1" ]]; then
                        die "Claude Code not found; install it with npm -g or the native installer, then ensure 'claude' is on PATH"
                    fi
                    warn "Claude Code not found; skipping Claude Code mounts"
                    return 0
                fi
                add_mount "$claude_node_prefix" /opt/claude-node ro
            fi
            need_path "$HOST_HOME/.claude" "Claude Code auth directory" "$strict" || return 0
            need_path "$HOST_HOME/.claude.json" "Claude Code auth/config file" "$strict" || return 0
            if [[ "$isolate" == "1" ]]; then
                add_mount "$HOST_HOME/.claude" "$AGENT_STATE_MOUNT_ROOT/.claude" ro
                add_mount "$HOST_HOME/.claude.json" "$AGENT_STATE_MOUNT_ROOT/.claude.json" ro
            else
                add_mount "$HOST_HOME/.claude" "$HOST_HOME/.claude"
                add_mount "$HOST_HOME/.claude.json" "$HOST_HOME/.claude.json"
            fi
            ;;
        cursor)
            need_path "$HOST_HOME/.local/bin" "host local bin directory" "$strict" || return 0
            need_path "$HOST_HOME/.local/share/cursor-agent" "Cursor Agent local install" "$strict" || return 0
            need_path "$HOST_HOME/.cursor" "Cursor Agent state directory" "$strict" || return 0
            need_path "$HOST_HOME/.config/cursor" "Cursor Agent config directory" "$strict" || return 0
            add_mount "$HOST_HOME/.local/bin" "$HOST_HOME/.local/bin" ro
            add_mount "$HOST_HOME/.local/share/cursor-agent" "$HOST_HOME/.local/share/cursor-agent" ro
            if [[ "$isolate" == "1" ]]; then
                add_mount "$HOST_HOME/.cursor" "$AGENT_STATE_MOUNT_ROOT/.cursor" ro
                add_mount "$HOST_HOME/.config/cursor" "$AGENT_STATE_MOUNT_ROOT/.config/cursor" ro
            else
                add_mount "$HOST_HOME/.cursor" "$HOST_HOME/.cursor"
                add_mount "$HOST_HOME/.config/cursor" "$HOST_HOME/.config/cursor"
            fi
            ;;
        *)
            warn "Unknown agent '$agent'; not provisioning any CLI for it"
            ;;
    esac
}

add_device_if_present() {
    local dev="$1"
    if [[ -e "$dev" ]]; then
        docker_args+=(--device="$dev")
    else
        warn "Skipping missing device $dev"
    fi
}

normalize_eval_tool_id() {
    local tool="${1//-/_}"
    tool="$(printf '%s' "$tool" | tr '[:upper:]' '[:lower:]')"
    case "$tool" in
        triton_fpsan|gpu_asan|rocjitsu|rocjitsu_waitcheck|rocjitsu_consan|hip_fpsan) printf '%s\n' "$tool" ;;
        *) die "Unsupported evaluation tool '$1'" ;;
    esac
}

resolve_eval_tools() {
    local config="${1:-}"
    local raw=""
    if [[ -n "${AKA_EVAL_TOOLS:-}" ]]; then
        raw="${AKA_EVAL_TOOLS//,/ }"
    elif [[ -n "$config" && -f "$config" ]]; then
        raw="$(python3 - "$config" <<'PY'
import sys
import yaml

value = (yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}).get("evaluation_tools")
if not value:
    raise SystemExit(0)
enabled = value.get("enabled", ()) if isinstance(value, dict) else ()
if enabled is True:
    enabled = (
        "triton_fpsan",
        "gpu_asan",
        "rocjitsu",
        "rocjitsu_waitcheck",
        "rocjitsu_consan",
        "hip_fpsan",
    )
elif isinstance(enabled, str):
    enabled = (enabled,)
print(" ".join(str(item) for item in enabled or ()))
PY
)"
    fi

    local seen=" " tool normalized
    for tool in $raw; do
        normalized="$(normalize_eval_tool_id "$tool")"
        [[ "$seen" != *" $normalized "* ]] \
            || die "Duplicate evaluation tool '$normalized'"
        seen+="$normalized "
        printf '%s\n' "$normalized"
    done
}

eval_tool_image() {
    local tool="$1" tool_upper env_name override
    [[ "$SELECTED_GPU_ARCH" == "gfx950" ]] \
        || die "Evaluation-tool sidecars are verified only for gfx950; selected ${SELECTED_GPU_ARCH:-unknown}"
    tool_upper="$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
    env_name="AKA_EVAL_TOOL_IMAGE_${tool_upper}"
    override="${!env_name:-}"
    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
    else
        printf 'agent-kernel-arena/eval-tool-%s:gfx950\n' "${tool//_/-}"
    fi
}

verify_eval_tool_scoring_image() {
    local selected_id pinned_id
    [[ "$SELECTED_GPU_ARCH" == "gfx950" ]] \
        || die "Evaluation-tool sidecars are verified only for gfx950"
    selected_id="$(docker image inspect --format '{{.Id}}' "$SELECTED_IMAGE" 2>/dev/null || true)"
    pinned_id="$(docker image inspect --format '{{.Id}}' "$GFX950_V0514_IMMUTABLE_IMAGE" 2>/dev/null || true)"
    [[ "$selected_id" == sha256:* ]] \
        || die "Could not resolve immutable scoring image ID for $SELECTED_IMAGE"
    [[ "$pinned_id" == sha256:* ]] \
        || die "Pinned evaluation scoring image is unavailable: $GFX950_V0514_IMMUTABLE_IMAGE"
    [[ "$selected_id" == "$pinned_id" ]] \
        || die "Evaluation tools are unverified with scoring image $SELECTED_IMAGE ($selected_id); expected $GFX950_V0514_DOCKER_IMAGE ($pinned_id)"
    export AKA_SCORING_IMAGE_RUNTIME_REF="$selected_id"
    export AKA_SCORING_IMAGE_REFERENCE="$SELECTED_IMAGE"
    # Launch by immutable local config ID after verification.  This closes the
    # gap in which a mutable tag could move between inspection and docker run.
    SELECTED_IMAGE="$selected_id"
}

path_has_symlink_component() {
    local path="$1" current="/" component
    local -a components=()
    [[ "$path" == /* ]] || return 0
    IFS='/' read -r -a components <<< "${path#/}"
    for component in "${components[@]}"; do
        [[ -z "$component" ]] && continue
        current="${current%/}/$component"
        [[ ! -L "$current" ]] || return 0
    done
    return 1
}

prepare_eval_tool_artifact_dir() {
    local artifact_host_parent="$1" label="$2"
    local artifact_label artifact_host_dir artifact_namespace_root
    local artifact_parent_real host_root_real
    artifact_label="$(safe_label "$label")"
    [[ -n "$artifact_label" && "$artifact_label" != "." && "$artifact_label" != ".." ]] \
        || die "invalid evaluation-tool artifact label: $label"
    [[ -d "$artifact_host_parent" ]] \
        || die "evaluation-tool artifact parent not found: $artifact_host_parent"
    path_has_symlink_component "$artifact_host_parent" \
        && die "evaluation-tool artifact parent contains a symlink: $artifact_host_parent"
    host_root_real="$(realpath -e "$HOST_ROOT")"
    artifact_parent_real="$(realpath -e "$artifact_host_parent")"
    case "$artifact_parent_real" in
        "$host_root_real"|"$host_root_real"/*) ;;
        *) die "evaluation-tool artifact parent must stay inside repository: $artifact_parent_real" ;;
    esac
    artifact_namespace_root="$artifact_parent_real/.eval-tool-artifacts"
    [[ ! -L "$artifact_namespace_root" ]] \
        || die "evaluation-tool artifact namespace must not be a symlink: $artifact_namespace_root"
    mkdir -p "$artifact_namespace_root"
    artifact_host_dir="$artifact_namespace_root/$artifact_label"
    [[ ! -L "$artifact_host_dir" ]] \
        || die "evaluation-tool worker artifact directory must not be a symlink: $artifact_host_dir"
    mkdir -p "$artifact_host_dir"
    artifact_host_dir="$(realpath -e "$artifact_host_dir")"
    case "$artifact_host_dir" in
        "$artifact_namespace_root"/*) ;;
        *) die "evaluation-tool artifact root escaped its namespace: $artifact_host_dir" ;;
    esac
    # Docker's daemon may be root-squashed on NFS homes. Execute-only access
    # lets it resolve the already-created bind source without exposing a
    # directory listing; the host/container worker UID remains the owner.
    chmod 0711 "$artifact_namespace_root" "$artifact_host_dir"
    printf '%s\n' "$artifact_host_dir"
}

prepare_rocjitsu_passwd_file() {
    local image="$1" target="$2"
    local base_passwd

    # ROCr currently crashes inside rocJITsu when getpwuid(3) cannot resolve
    # the caller. Keep the sidecar at the host UID (never root), but give this
    # one container an image-derived passwd file containing that UID. The file
    # lives in the mode-0700 per-run runtime directory and is mounted read-only.
    base_passwd="$(
        docker run --rm --network=none --entrypoint /bin/cat \
            "$image" /etc/passwd
    )"
    [[ -n "$base_passwd" ]] \
        || die "Could not read /etc/passwd from rocJITsu image $image"
    printf '%s\n' "$base_passwd" > "$target"

    if ! awk -F: -v uid="$HOST_UID" \
        '$3 == uid { found = 1 } END { exit(found ? 0 : 1) }' "$target"; then
        printf 'aka-eval:x:%s:%s:rocJITsu eval worker:%s:/usr/sbin/nologin\n' \
            "$HOST_UID" "$HOST_GID" "${EVAL_TOOL_SCRATCH_CONTAINER_DIR}/home" \
            >> "$target"
    fi
    chmod 0644 "$target"
}

build_eval_tool_docker_args() {
    local tool="$1" image="$2" container_name="$3"
    local socket_host_dir="$4" scratch_host_dir="$5" artifact_host_dir="$6"
    local runtime_ref="${7:-unverified}"
    local passwd_host_file="${8:-}"

    # Inspection, plan identity, and execution must refer to the same immutable
    # object. Refuse a tag here so future callers cannot reopen a tag-movement
    # race after start_eval_tool_sidecars has resolved the local config ID.
    [[ "$image" == sha256:* && "$image" == "$runtime_ref" ]] \
        || die "evaluation-tool sidecars must launch by their resolved immutable image ID"

    require_path "$HOST_ROOT" "evaluation-tool input root"
    require_path "$socket_host_dir" "evaluation-tool socket directory"
    require_path "$scratch_host_dir" "evaluation-tool scratch directory"
    require_path "$artifact_host_dir" "evaluation-tool artifact directory"
    if [[ "$tool" == "rocjitsu" ]]; then
        [[ -n "$passwd_host_file" ]] \
            || die "rocJITsu sidecar requires an image-derived passwd file"
        require_path "$passwd_host_file" "rocJITsu passwd file"
    elif [[ -n "$passwd_host_file" ]]; then
        die "passwd override is restricted to the rocJITsu sidecar"
    fi

    eval_tool_docker_args=(
        run -d --rm
        --name "$container_name"
        --entrypoint /opt/venv/bin/python
        --network=none
        --cap-drop=ALL
        --security-opt=no-new-privileges
        --read-only
        --user "${HOST_UID}:${HOST_GID}"
        --tmpfs "/tmp:rw,nosuid,nodev,uid=${HOST_UID},gid=${HOST_GID},mode=1777"
        -e "HOME=${EVAL_TOOL_SCRATCH_CONTAINER_DIR}/home"
        -e "TMPDIR=/tmp"
        -e "XDG_CACHE_HOME=${EVAL_TOOL_SCRATCH_CONTAINER_DIR}/cache"
        -e "TORCH_EXTENSIONS_DIR=${EVAL_TOOL_SCRATCH_CONTAINER_DIR}/torch-extensions"
        -e "TRITON_CACHE_DIR=${EVAL_TOOL_SCRATCH_CONTAINER_DIR}/triton-cache"
        # Keep trusted worker/helper imports exclusive to the image-owned tree.
        # Candidate files remain addressable by argv and cwd under /input:ro.
        -e "PYTHONPATH=${EVAL_TOOL_FRAMEWORK_CONTAINER_ROOT}"
        -e "PYTHONDONTWRITEBYTECODE=1"
        -e "AKA_EVAL_TOOL_FRAMEWORK_ROOT=${EVAL_TOOL_FRAMEWORK_CONTAINER_ROOT}"
        -e "AKA_EVAL_TOOL_RUNTIME_REF=${runtime_ref}"
        -e "AKA_EVAL_TOOL_INSTANCE=${container_name}"
        -v "${HOST_ROOT}:${EVAL_TOOL_INPUT_CONTAINER_DIR}:ro"
        -v "${scratch_host_dir}:${EVAL_TOOL_SCRATCH_CONTAINER_DIR}:rw"
        -v "${artifact_host_dir}:${EVAL_TOOL_ARTIFACT_CONTAINER_DIR}:rw"
        -v "${socket_host_dir}:${EVAL_TOOL_SOCKET_CONTAINER_DIR}:rw"
    )

    if [[ "$tool" == "rocjitsu" ]]; then
        eval_tool_docker_args+=(
            -v "${passwd_host_file}:/etc/passwd:ro"
        )
    fi

    if [[ "$tool" != "rocjitsu_waitcheck" ]]; then
        local gpu_grp gpu_gid
        for gpu_grp in render video; do
            gpu_gid="$(getent group "$gpu_grp" 2>/dev/null | cut -d: -f3 || true)"
            [[ -z "$gpu_gid" ]] || eval_tool_docker_args+=(--group-add "$gpu_gid")
        done
        [[ ! -e /dev/kfd ]] || eval_tool_docker_args+=(--device=/dev/kfd)
        [[ ! -e /dev/dri ]] || eval_tool_docker_args+=(--device=/dev/dri)

        if [[ -n "${AKA_VISIBLE_GPU:-}" ]]; then
            eval_tool_docker_args+=(
                -e "ROCR_VISIBLE_DEVICES=${AKA_VISIBLE_GPU}"
                -e "HIP_VISIBLE_DEVICES=${AKA_LOGICAL_GPU:-0}"
                -e "CUDA_VISIBLE_DEVICES=${AKA_LOGICAL_GPU:-0}"
                -e "GPU_DEVICE_ORDINAL=${AKA_LOGICAL_GPU:-0}"
            )
        fi
    fi

    eval_tool_docker_args+=(
        "$image"
        -m src.eval_tools.worker
        --tool "$tool"
        --socket "${EVAL_TOOL_SOCKET_CONTAINER_DIR}/${tool}.sock"
        --input-root "$EVAL_TOOL_INPUT_CONTAINER_DIR"
        --scratch-root "$EVAL_TOOL_SCRATCH_CONTAINER_DIR"
        --artifact-root "$EVAL_TOOL_ARTIFACT_CONTAINER_DIR"
    )
}

start_eval_tool_sidecars() {
    local config="$1" label="$2"
    local artifact_host_parent="${3:-$HOST_ROOT}"
    local artifact_scoring_parent="${4:-$CONTAINER_WORKDIR}"
    eval_tool_ids=()
    while IFS= read -r tool; do
        [[ -z "$tool" ]] || eval_tool_ids+=("$tool")
    done < <(resolve_eval_tools "$config")
    [[ "${#eval_tool_ids[@]}" -gt 0 ]] || return 0
    verify_eval_tool_scoring_image

    [[ "$artifact_scoring_parent" == /* ]] \
        || die "evaluation-tool scoring artifact parent must be absolute: $artifact_scoring_parent"
    local artifact_namespace artifact_host_dir artifact_scoring_root artifact_label
    artifact_label="$(safe_label "$label")"
    artifact_namespace=".eval-tool-artifacts/$artifact_label"
    artifact_host_dir="$(prepare_eval_tool_artifact_dir "$artifact_host_parent" "$label")"
    artifact_scoring_root="${artifact_scoring_parent%/}/$artifact_namespace"

    EVAL_TOOL_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aka-eval-tools-$(safe_label "$label").XXXXXX")"
    EVAL_TOOL_SOCKET_HOST_DIR="$EVAL_TOOL_RUNTIME_DIR/sockets"
    EVAL_TOOL_ARTIFACT_HOST_ROOT="$artifact_host_dir"
    EVAL_TOOL_ARTIFACT_SCORING_ROOT="$artifact_scoring_root"
    EVAL_TOOL_SELECTED="$(IFS=,; printf '%s' "${eval_tool_ids[*]}")"
    mkdir -p "$EVAL_TOOL_SOCKET_HOST_DIR" "$EVAL_TOOL_RUNTIME_DIR/scratch"
    chmod 0700 "$EVAL_TOOL_RUNTIME_DIR" "$EVAL_TOOL_SOCKET_HOST_DIR"
    export AKA_EVAL_TOOL_SOCKET_HOST_DIR="$EVAL_TOOL_SOCKET_HOST_DIR"
    export AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT="$EVAL_TOOL_ARTIFACT_HOST_ROOT"
    export AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT="$EVAL_TOOL_ARTIFACT_SCORING_ROOT"
    export AKA_EVAL_TOOLS_SELECTED="$EVAL_TOOL_SELECTED"
    eval_tool_container_names=()

    local tool image name scratch socket socket_host_dir attempt runtime_ref runtime_env passwd_host_file
    for tool in "${eval_tool_ids[@]}"; do
        image="$(eval_tool_image "$tool")"
        runtime_ref="$(docker image inspect --format '{{.Id}}' "$image")"
        [[ "$runtime_ref" == sha256:* ]] \
            || die "Could not resolve immutable image ID for $tool image $image"
        runtime_env="AKA_EVAL_TOOL_RUNTIME_REF_$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
        printf -v "$runtime_env" '%s' "$runtime_ref"
        export "$runtime_env"
        name="aka-eval-$(safe_label "$label")-${tool//_/-}-$$"
        scratch="$EVAL_TOOL_RUNTIME_DIR/scratch/$tool"
        # Each sidecar receives only its own writable socket directory. The
        # scoring container mounts their parent read-only and selects the
        # matching nested path, so one tool cannot replace another tool's UDS.
        socket_host_dir="$EVAL_TOOL_SOCKET_HOST_DIR/$tool"
        mkdir -p "$scratch/home" "$scratch/cache" "$socket_host_dir"
        chmod 0700 "$socket_host_dir"
        passwd_host_file=""
        if [[ "$tool" == "rocjitsu" ]]; then
            passwd_host_file="$EVAL_TOOL_RUNTIME_DIR/rocjitsu.passwd"
            prepare_rocjitsu_passwd_file "$runtime_ref" "$passwd_host_file"
        fi
        build_eval_tool_docker_args \
            "$tool" "$runtime_ref" "$name" "$socket_host_dir" \
            "$scratch" "$artifact_host_dir" "$runtime_ref" "$passwd_host_file"
        docker "${eval_tool_docker_args[@]}" >/dev/null
        eval_tool_container_names+=("$name")
        socket="$socket_host_dir/$tool.sock"
        # Workers run one synthetic known-bug positive control before exposing
        # the socket. HIP compilation and rocJITsu simulation can take a minute
        # on a cold cache, so readiness is bounded at five minutes.
        for attempt in $(seq 1 600); do
            [[ -S "$socket" ]] && break
            if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
                die "Evaluation-tool sidecar exited before readiness: tool=$tool container=$name"
            fi
            sleep 0.5
        done
        [[ -S "$socket" ]] || die "Timed out waiting for evaluation-tool sidecar: $tool"
        # Publish a stable flat client path only after readiness. The sidecar
        # cannot tamper with this symlink because its RW bind is the nested
        # directory, while the scoring container receives the parent read-only.
        ln -s -- "$tool/$tool.sock" "$EVAL_TOOL_SOCKET_HOST_DIR/$tool.sock"
    done
}

stop_eval_tool_sidecars() {
    local name
    for name in "${eval_tool_container_names[@]:-}"; do
        [[ -z "$name" ]] || docker stop --time 5 "$name" >/dev/null 2>&1 || true
    done
    eval_tool_container_names=()
    unset AKA_EVAL_TOOL_SOCKET_HOST_DIR
    unset AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT
    unset AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT
    unset AKA_EVAL_TOOLS_SELECTED
    unset AKA_SCORING_IMAGE_RUNTIME_REF
    unset AKA_SCORING_IMAGE_REFERENCE
    local tool runtime_env
    for tool in triton_fpsan gpu_asan rocjitsu rocjitsu_waitcheck rocjitsu_consan hip_fpsan; do
        runtime_env="AKA_EVAL_TOOL_RUNTIME_REF_$(printf '%s' "$tool" | tr '[:lower:]' '[:upper:]')"
        unset "$runtime_env"
    done
    if [[ -n "$EVAL_TOOL_RUNTIME_DIR" && -d "$EVAL_TOOL_RUNTIME_DIR" ]]; then
        case "$EVAL_TOOL_RUNTIME_DIR" in
            "${TMPDIR:-/tmp}"/aka-eval-tools-*) rm -rf -- "$EVAL_TOOL_RUNTIME_DIR" ;;
            *) warn "Refusing to remove unexpected eval-tool runtime path: $EVAL_TOOL_RUNTIME_DIR" ;;
        esac
    fi
    EVAL_TOOL_RUNTIME_DIR=""
    EVAL_TOOL_SOCKET_HOST_DIR=""
    EVAL_TOOL_ARTIFACT_HOST_ROOT=""
    EVAL_TOOL_ARTIFACT_SCORING_ROOT=""
    EVAL_TOOL_SELECTED=""
}

build_eval_tool_images() {
    select_runtime_for_host
    [[ "$SELECTED_GPU_ARCH" == "gfx950" ]] \
        || die "Tool images currently have verified build locks only for gfx950"
    local tool image dockerfile target
    for tool in triton_fpsan gpu_asan rocjitsu rocjitsu_waitcheck rocjitsu_consan hip_fpsan; do
        image="$(eval_tool_image "$tool")"
        target=""
        case "$tool" in
            rocjitsu_waitcheck|rocjitsu_consan)
                dockerfile="$HOST_ROOT/docker/eval-tools/rocjitsu-sanitizers/Dockerfile"
                target="${tool//_/-}-runtime"
                ;;
            *)
                dockerfile="$HOST_ROOT/docker/eval-tools/${tool//_/-}/Dockerfile"
                ;;
        esac
        if [[ -n "$target" ]]; then
            docker build --pull=false --target "$target" -f "$dockerfile" -t "$image" "$HOST_ROOT"
        else
            docker build --pull=false -f "$dockerfile" -t "$image" "$HOST_ROOT"
        fi
    done
}

build_docker_args() {
    local interactive="${1:-0}"
    # Which agent CLIs to provision, and whether their absence is fatal.
    # Defaults (no caller override) are best-effort over all three — used by
    # interactive `shell`/`smoke` so any installed agent works.
# Use `-` (not `:-`) so an explicitly-empty REQUIRED_AGENTS means "no agents"
# (e.g. setup-flydsl), while unset falls back to all three.
    local agents="${REQUIRED_AGENTS-codex claude_code cursor}"
    local strict="${AGENTS_STRICT:-0}"
    local container_home="${AKA_CONTAINER_HOME:-$HOST_HOME}"
    local codex_home="${AKA_CODEX_HOME:-$container_home/.codex}"
    local cache_suffix="${AKA_CACHE_SUFFIX:-}"
    local cache_postfix=""

    if [[ -n "$cache_suffix" ]]; then
        cache_suffix="${cache_suffix//[^A-Za-z0-9_.-]/_}"
        cache_postfix="-$cache_suffix"
    fi

    [[ -n "$SELECTED_IMAGE" ]] || select_runtime_for_host

    docker_args=(run --rm --entrypoint bash)
    unset _MOUNTED_TARGETS
    declare -gA _MOUNTED_TARGETS=()
    if [[ "$interactive" == "1" && -t 0 ]]; then
        docker_args+=(-it)
    fi

    docker_args+=(
        --ipc=host
        --network=host
        --privileged
        --cap-add=SYS_ADMIN
        --cap-add=SYS_PTRACE
        --security-opt=seccomp=unconfined
        --user "${HOST_UID}:${HOST_GID}"
        -e "HOME=${container_home}"
        -e "CODEX_HOME=${codex_home}"
        -e "XDG_CACHE_HOME=/tmp/agent-cache${cache_postfix}"
        -e "MPLCONFIGDIR=/tmp/matplotlib${cache_postfix}"
        -e "TORCH_EXTENSIONS_DIR=/tmp/torch-extensions${cache_postfix}"
        -e "TRITON_CACHE_DIR=/tmp/triton-cache${cache_postfix}"
        -e "PYTHONUSERBASE=${CONTAINER_WORKDIR}/.aka-pyuserbase"
        -e "MIOPEN_USER_DB_PATH=/tmp/miopen-cache${cache_postfix}"
        -e "MIOPEN_CACHE_DIR=/tmp/miopen-cache${cache_postfix}"
        -e "MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache${cache_postfix}"
        -e "AGENT_KERNEL_ARENA_DOCKER=1"
        -e "AGENT_KERNEL_ARENA_WORKDIR=${CONTAINER_WORKDIR}"
        -e "AGENT_KERNEL_ARENA_GPU_ARCH=${SELECTED_GPU_ARCH}"
        -e "PYTORCH_ROCM_ARCH=${SELECTED_GPU_ARCH}"
        -e "AGENT_STATE_MOUNT_ROOT=${AGENT_STATE_MOUNT_ROOT}"
        -e "PATH=${container_path}"
        -w "$CONTAINER_WORKDIR"
    )

    # geak_v4's claude-agent-sdk is installed with `pip install --target` into
    # this host-mounted dir (see container_setup_geak). Only put it on
    # PYTHONPATH for GEAK runs so its dependency closure cannot shadow the
    # runtime image's pinned packages for existing agents.
    if [[ "$GEAK_V4_RUNTIME" == "1" ]]; then
        docker_args+=(-e "PYTHONPATH=${CONTAINER_WORKDIR}/.aka-pyuserbase/geak-sdk")
    fi

    # The pinned gfx950 image ships root-owned AITER/FlyDSL caches, and its
    # /tmp/aiter_configs directory is not writable by the host UID used below.
    # Keep these overrides tied to that exact runtime so custom images and
    # other GPU architectures retain their existing cache behavior.
    if uses_gfx950_v0514_runtime; then
        docker_args+=(
            -e "AITER_JIT_DIR=/tmp/aiter-jit${cache_postfix}"
            -e "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-runtime-cache${cache_postfix}"
            --tmpfs "/tmp/aiter_configs:rw,uid=${HOST_UID},gid=${HOST_GID},mode=1777"
        )
    fi

    if [[ -n "${AKA_VISIBLE_GPU:-}" ]]; then
        local logical_gpu="${AKA_LOGICAL_GPU:-0}"
        docker_args+=(
            -e "AGENT_KERNEL_ARENA_HOST_GPU_ID=${AKA_VISIBLE_GPU}"
            -e "ROCR_VISIBLE_DEVICES=${AKA_VISIBLE_GPU}"
            -e "HIP_VISIBLE_DEVICES=${logical_gpu}"
            -e "CUDA_VISIBLE_DEVICES=${logical_gpu}"
            -e "GPU_DEVICE_ORDINAL=${logical_gpu}"
        )
    fi
    if [[ -n "${AKA_WORKER_ID:-}" ]]; then
        docker_args+=(-e "AGENT_KERNEL_ARENA_WORKER_ID=${AKA_WORKER_ID}")
    fi
    if [[ "${AGENT_HOME_ISOLATION:-0}" == "1" ]]; then
        docker_args+=(-e "AGENT_KERNEL_ARENA_ISOLATED_HOME=1")
    fi
    # Forward the host's Claude / Anthropic auth+config only for GEAK execution
    # containers that provision Claude Code. Requiring both conditions keeps
    # existing Claude/task-validator runs unchanged and prevents setup-geak
    # (which has no agent CLI) from receiving runtime credentials. Each var is
    # passed by name only (no "=value") so secrets stay out of argv / process
    # listings.
    if [[ "$GEAK_V4_RUNTIME" == "1" ]] && agent_list_contains "$agents" claude_code; then
        local claude_env_var
        for claude_env_var in \
            ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_BASE_URL \
            ANTHROPIC_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
            ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL \
            CLAUDE_CODE_SUBAGENT_MODEL API_TIMEOUT_MS \
            CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS \
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC \
            NODE_EXTRA_CA_CERTS SSL_CERT_FILE CURL_CA_BUNDLE \
            REQUESTS_CA_BUNDLE NODE_TLS_REJECT_UNAUTHORIZED; do
            if [[ -n "${!claude_env_var:-}" ]]; then
                docker_args+=(-e "$claude_env_var")
            fi
        done
    fi

    # GPU device nodes are group-owned (ROCm): /dev/dri/renderD* by `render` and
    # /dev/kfd by `render` or `video` depending on the host's udev rules. Add the
    # non-root container user to both supplementary groups so it can reach the ROCm
    # compute device (otherwise torch.cuda is unavailable).
    local gpu_grp gpu_gid
    for gpu_grp in render video; do
        gpu_gid="$(getent group "$gpu_grp" 2>/dev/null | cut -d: -f3 || true)"
        if [[ -n "$gpu_gid" ]]; then
            docker_args+=(--group-add "$gpu_gid")
        fi
    done

    add_device_if_present /dev/kfd
    add_device_if_present /dev/dri
    add_device_if_present /dev/mem

    if [[ -n "$QUALITY_LOOP_ARTIFACT_REL" || -n "$QUALITY_LOOP_WORKTREE_REL" ]]; then
        [[ -n "$QUALITY_LOOP_ARTIFACT_REL" && -n "$QUALITY_LOOP_WORKTREE_REL" ]] \
            || die "quality_loop requires both artifact and worktree mount paths"
        require_path "$HOST_ROOT/$QUALITY_LOOP_ARTIFACT_REL" "quality_loop artifact directory"
        require_path "$HOST_ROOT/$QUALITY_LOOP_WORKTREE_REL" "quality_loop worktree"
        add_mount "$HOST_ROOT" "$CONTAINER_WORKDIR" ro
        add_mount \
            "$HOST_ROOT/$QUALITY_LOOP_ARTIFACT_REL" \
            "$CONTAINER_WORKDIR/$QUALITY_LOOP_ARTIFACT_REL"
        add_mount \
            "$HOST_ROOT/$QUALITY_LOOP_WORKTREE_REL" \
            "$CONTAINER_WORKDIR/$QUALITY_LOOP_WORKTREE_REL"
    else
        add_mount "$HOST_ROOT" "$CONTAINER_WORKDIR"
    fi
    # A scoring container receives the per-worker Unix-socket directory and its
    # dedicated report tree. Tool images, credentials, Docker access, and the
    # rest of the experiments/workspace tree never enter a sidecar writable.
    if [[ -n "${AKA_EVAL_TOOL_SOCKET_HOST_DIR:-}" ]]; then
        local artifact_host_namespace artifact_scoring_namespace
        local artifact_host_label artifact_scoring_label
        [[ -d "$AKA_EVAL_TOOL_SOCKET_HOST_DIR" ]] \
            || die "evaluation-tool socket directory not found: $AKA_EVAL_TOOL_SOCKET_HOST_DIR"
        add_mount \
            "$AKA_EVAL_TOOL_SOCKET_HOST_DIR" \
            "$EVAL_TOOL_SOCKET_CONTAINER_DIR" ro
        [[ -d "${AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT:-}" ]] \
            || die "evaluation-tool artifact directory not found: ${AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT:-unset}"
        [[ "${AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT:-}" == /* ]] \
            || die "evaluation-tool scoring artifact root must be absolute"
        artifact_host_namespace="${AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT%/*}"
        artifact_scoring_namespace="${AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT%/*}"
        artifact_host_label="${AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT##*/}"
        artifact_scoring_label="${AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT##*/}"
        [[ "${artifact_host_namespace##*/}" == ".eval-tool-artifacts" \
            && "${artifact_scoring_namespace##*/}" == ".eval-tool-artifacts" ]] \
            || die "evaluation-tool artifact roots must use the .eval-tool-artifacts namespace"
        [[ -n "$artifact_host_label" && "$artifact_host_label" == "$artifact_scoring_label" ]] \
            || die "evaluation-tool host/scoring worker labels must match"
        [[ -d "$artifact_host_namespace" ]] \
            || die "evaluation-tool artifact namespace not found: $artifact_host_namespace"
        # The broad repository mount is writable in ordinary/parallel runs.
        # Hide its artifact namespace behind a read-only bind, then over-mount
        # only this worker's validated child as writable. A sibling worker can
        # neither alter reports nor swap another worker's bind source by symlink.
        add_mount \
            "$artifact_host_namespace" \
            "$artifact_scoring_namespace" ro
        add_mount \
            "$AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT" \
            "$AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT"
        docker_args+=(
            -e "AKA_EVAL_TOOL_SOCKET_DIR=$EVAL_TOOL_SOCKET_CONTAINER_DIR"
            -e "AKA_EVAL_TOOL_SCORING_ROOT=$CONTAINER_WORKDIR"
            -e "AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT=$AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT"
            -e "AKA_SCORING_IMAGE_RUNTIME_REF=${AKA_SCORING_IMAGE_RUNTIME_REF:?missing verified scoring image ID}"
            -e "AKA_SCORING_IMAGE_REFERENCE=${AKA_SCORING_IMAGE_REFERENCE:?missing scoring image reference}"
        )
        if [[ -n "${AKA_EVAL_TOOLS_SELECTED:-}" ]]; then
            docker_args+=(-e "AKA_EVAL_TOOLS_SELECTED=${AKA_EVAL_TOOLS_SELECTED}")
        fi
        local eval_tool runtime_env
        for eval_tool in triton_fpsan gpu_asan rocjitsu rocjitsu_waitcheck rocjitsu_consan hip_fpsan; do
            runtime_env="AKA_EVAL_TOOL_RUNTIME_REF_$(printf '%s' "$eval_tool" | tr '[:lower:]' '[:upper:]')"
            if [[ -n "${!runtime_env:-}" ]]; then
                docker_args+=(-e "$runtime_env=${!runtime_env}")
            fi
        done
    fi
    # Persistent pip user-base (PYTHONUSERBASE) so `make docker-setup-flydsl` survives
    # across runs. It lives INSIDE the repo dir, which is already bind-mounted above and
    # is owned by the host user — this avoids a separate mount whose source the docker
    # daemon would have to create (which fails on NFS/root-squashed homes).
    mkdir -p "$HOST_ROOT/.aka-pyuserbase" 2>/dev/null || true
    local _agent
    for _agent in $agents; do
        mount_agent "$_agent" "$strict"
    done

    # Mount the GEAK kernel_workflow checkout only for GEAK runs so an exported
    # host setting does not change the container surface for existing agents.
    if [[ "$GEAK_V4_RUNTIME" == "1" && -n "${GEAK_V4_WORKFLOW_DIR:-}" ]]; then
        local geak_dir
        geak_dir="$(cd "$GEAK_V4_WORKFLOW_DIR" 2>/dev/null && pwd || true)"
        if [[ -n "$geak_dir" && -d "$geak_dir" ]]; then
            add_mount "$geak_dir" "$geak_dir" ro
            docker_args+=(-e "GEAK_V4_WORKFLOW_DIR=$geak_dir")
        elif [[ "$strict" == "1" ]]; then
            die "GEAK_V4_WORKFLOW_DIR is set but is not a directory: $GEAK_V4_WORKFLOW_DIR"
        else
            warn "GEAK_V4_WORKFLOW_DIR is set but is not a directory: $GEAK_V4_WORKFLOW_DIR; skipping GEAK mount"
        fi
    fi

    # The base image lacks the GNU `time` binary and the container runs as a
    # non-root user (so it cannot apt-install it). Bind-mount the host binary
    # read-only so commands that invoke `/usr/bin/time` do not fail with 127.
    if [[ -x /usr/bin/time ]]; then
        add_mount /usr/bin/time /usr/bin/time ro
    fi

    if [[ -e "$HOST_HOME/.gitconfig" ]]; then
        add_mount "$HOST_HOME/.gitconfig" "$HOST_HOME/.gitconfig" ro
    fi

    docker_args+=("$SELECTED_IMAGE")
}

docker_exec() {
    local interactive="${1:-0}"
    shift
    build_docker_args "$interactive"
    docker "${docker_args[@]}" -lc 'cd "$AGENT_KERNEL_ARENA_WORKDIR" && if [[ "${AGENT_KERNEL_ARENA_ISOLATED_HOME:-0}" == "1" ]]; then bash src/scripts/docker_benchmark.sh _container_prepare_worker_home; fi && exec "$@"' _ "$@"
}

extract_config_name() {
    local config="$DEFAULT_RUN_CONFIG"
    local arg
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --config_name)
                shift
                [[ $# -gt 0 ]] || die "--config_name requires a value"
                config="$1"
                ;;
            --config_name=*)
                config="${arg#--config_name=}"
                ;;
        esac
        shift || true
    done
    printf '%s\n' "$config"
}

extract_quality_loop_config() {
    local config="agents/quality_loop/agent_config.yaml"
    local arg
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --config)
                shift
                [[ $# -gt 0 ]] || die "--config requires a value"
                config="$1"
                ;;
            --config=*)
                config="${arg#--config=}"
                ;;
        esac
        shift || true
    done
    printf '%s\n' "$config"
}

extract_quality_loop_resume() {
    local arg
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --resume)
                shift
                [[ $# -gt 0 ]] || die "--resume requires a run ID"
                printf '%s\n' "$1"
                return
                ;;
            --resume=*)
                printf '%s\n' "${arg#--resume=}"
                return
                ;;
        esac
        shift || true
    done
}

container_smoke() {
    python - <<'PY'
import importlib
import os
import shutil
import sys

print(f"python={sys.executable}")
print(f"version={sys.version.split()[0]}")

for cmd in ("hipcc", "rocprof-compute"):
    path = shutil.which(cmd)
    if not path:
        raise SystemExit(f"missing command: {cmd}")
    print(f"{cmd}={path}")

for mod_name in ("torch", "triton", "pytest", "yaml", "numpy"):
    mod = importlib.import_module(mod_name)
    print(f"{mod_name}=ok {getattr(mod, '__version__', '')}")

try:
    flydsl = importlib.import_module("flydsl")
    print(f"flydsl=ok {getattr(flydsl, '__version__', '')}")
except ModuleNotFoundError:
    print("flydsl=optional-missing (run `make docker-setup-flydsl` before FlyDSL tasks)")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False")
print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
selected_arch = os.environ.get("AGENT_KERNEL_ARENA_GPU_ARCH")
actual_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
if actual_arch:
    print(f"torch_cuda_arch={actual_arch}")
if selected_arch and actual_arch and not actual_arch.startswith(selected_arch):
    raise SystemExit(
        f"selected GPU arch {selected_arch} does not match visible device arch {actual_arch}; "
        "fix target_gpu_model for experiment runs, or use AKA_GPU_ARCH only for shell/smoke diagnostics"
    )
PY
}

container_check_agents() {
    # Verify only the requested agents (default: all three). Driven by the same
    # agent set as the mounts, so a single-agent run does not require the others.
    local agents="$*"
    [[ -n "$agents" ]] || agents="codex claude_code cursor"
    AKA_CHECK_AGENTS="$agents" python - <<'PY'
import json
import os
import shutil
import subprocess

agents = os.environ.get("AKA_CHECK_AGENTS", "").split()


def require_cmd(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"missing command: {name}")
    print(f"{name}={path}")
    return path


def run_checked(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed with exit {proc.returncode}:\n{output[:1000]}")
    return output.strip()


if "codex" in agents:
    require_cmd("codex")
    codex_status = run_checked(["codex", "login", "status"])
    codex_line = next((line for line in codex_status.splitlines() if "Logged in" in line), codex_status.splitlines()[-1])
    print(f"codex_status={codex_line}")

if "claude_code" in agents:
    require_cmd("claude")
    claude_version = run_checked(["claude", "--version"]).splitlines()[-1]
    claude_status_raw = run_checked(["claude", "auth", "status"])
    claude_status = json.loads(claude_status_raw)
    if not claude_status.get("loggedIn"):
        raise SystemExit("claude is not logged in")
    print(
        "claude_status=loggedIn "
        f"authMethod={claude_status.get('authMethod')} "
        f"subscriptionType={claude_status.get('subscriptionType')} "
        f"version={claude_version}"
    )

if "cursor" in agents:
    require_cmd("cursor-agent")
    cursor_version = run_checked(["cursor-agent", "--version"]).splitlines()[-1]
    cursor_status = json.loads(run_checked(["cursor-agent", "status", "--format", "json"]))
    if not cursor_status.get("isAuthenticated"):
        raise SystemExit("cursor-agent is not authenticated")
    print(
        "cursor_status=authenticated "
        f"hasAccessToken={cursor_status.get('hasAccessToken')} "
        f"hasRefreshToken={cursor_status.get('hasRefreshToken')} "
        f"version={cursor_version}"
    )
PY
}

container_preflight() {
    local config_name="${1:-$DEFAULT_RUN_CONFIG}"
    container_smoke
    # Only verify the agent(s) this config actually uses (mounts are scoped the same way).
    container_check_agents $(resolve_required_agents "$config_name")
    # GEAK v4 also needs the Claude Agent SDK and its kernel_workflow checkout.
    if [[ "$(read_agent_template "$config_name")" == geak_v4 ]]; then
        container_setup_geak
        container_check_geak
    fi
python - "$config_name" <<'PY'
import pathlib
import sys

import yaml

config_path = pathlib.Path(sys.argv[1])
if not config_path.exists():
    raise SystemExit(f"config file not found: {config_path}")

config = yaml.safe_load(config_path.read_text()) or {}
if not isinstance(config, dict):
    raise SystemExit(f"config file must contain a mapping: {config_path}")

print(f"config_ok={config_path}")
PY
}

container_setup_flydsl() {
    # If the image already provides FlyDSL, do nothing — installing a --user copy
    # could shadow the image version with an incompatible one.
    if python -c 'import flydsl' 2>/dev/null; then
        python -c 'import flydsl; print("flydsl already provided by image: " + str(getattr(flydsl, "__version__", "unknown")) + "; nothing to install")'
        return 0
    fi
    # Otherwise install into the persistent pip user-base (PYTHONUSERBASE), a
    # host-mounted dir, so it survives the --rm container and is importable in later runs.
    echo "flydsl not found in image; installing into persistent pip user-base..."
    python -m pip install --user --upgrade flydsl
    python -c 'import flydsl; print("flydsl=" + str(getattr(flydsl, "__version__", "unknown")) + " setup OK")'
}

container_setup_geak() {
    # Install claude-agent-sdk when the image does not ship it.
    if python -c 'import claude_agent_sdk' 2>/dev/null; then
        python -c 'import claude_agent_sdk; print("claude-agent-sdk already provided by image: " + str(getattr(claude_agent_sdk, "__version__", "unknown")) + "; nothing to install")'
        return 0
    fi
    # Install into a host-mounted target dir (survives the --rm container) and
    # rely on the forwarded PYTHONPATH (see build_docker_args) to import it.
    # `pip install --target` is the only option that works on the standard sglang
    # runtimes: their python is a virtualenv rooted at /opt/venv, which both
    # rejects `--user` (user-site disabled) and is unwritable by the non-root
    # container user (so a plain in-venv install fails with EACCES). --target
    # writes to a dir owned by the host UID and works on system-python images too.
    # Trade-off: --target cannot see the venv's already-installed deps, so it
    # pulls the SDK's full dependency closure (a few hundred MB, several minutes
    # on first run). This is a one-time provisioning cost — later runs import the
    # SDK via PYTHONPATH and short-circuit above.
    local target="${PYTHONUSERBASE:-$PWD/.aka-pyuserbase}/geak-sdk"
    echo "claude-agent-sdk not found in image; installing into $target ..."
    python -m pip install --target "$target" claude-agent-sdk
    PYTHONPATH="$target${PYTHONPATH:+:$PYTHONPATH}" python -c 'import claude_agent_sdk; print("claude-agent-sdk=" + str(getattr(claude_agent_sdk, "__version__", "unknown")) + " setup OK")'
}

container_check_geak() {
    # Confirm the kernel_workflow checkout is reachable inside the container.
    local dir="${GEAK_V4_WORKFLOW_DIR:-/opt/geak/kernel_workflow}"
    if [[ ! -f "$dir/kernel_workflow.js" ]]; then
        die "GEAK kernel workflow not found: $dir/kernel_workflow.js. Export GEAK_V4_WORKFLOW_DIR on the host (the runner mounts and forwards it) to your GEAK kernel_workflow directory."
    fi
    echo "geak_workflow=$dir/kernel_workflow.js"
}

container_prepare_worker_home() {
    local state_root="${AGENT_STATE_MOUNT_ROOT:-/opt/aka-agent-state}"
    mkdir -p "$HOME"

    if [[ -d "$state_root/.codex" && ! -e "$HOME/.codex" ]]; then
        cp -a "$state_root/.codex" "$HOME/.codex"
        chmod -R u+rwX "$HOME/.codex" 2>/dev/null || true
    fi

    if [[ -d "$state_root/.claude" && ! -e "$HOME/.claude" ]]; then
        cp -a "$state_root/.claude" "$HOME/.claude"
        chmod -R u+rwX "$HOME/.claude" 2>/dev/null || true
    fi
    if [[ -f "$state_root/.claude.json" && ! -e "$HOME/.claude.json" ]]; then
        cp -a "$state_root/.claude.json" "$HOME/.claude.json"
        chmod u+rw "$HOME/.claude.json" 2>/dev/null || true
    fi

    if [[ -d "$state_root/.cursor" && ! -e "$HOME/.cursor" ]]; then
        cp -a "$state_root/.cursor" "$HOME/.cursor"
        chmod -R u+rwX "$HOME/.cursor" 2>/dev/null || true
    fi
    if [[ -d "$state_root/.config/cursor" && ! -e "$HOME/.config/cursor" ]]; then
        mkdir -p "$HOME/.config"
        cp -a "$state_root/.config/cursor" "$HOME/.config/cursor"
        chmod -R u+rwX "$HOME/.config/cursor" 2>/dev/null || true
    fi
}

read_workspace_prefix() {
    local config="$1"
    sed -nE "s/^[[:space:]]*workspace_directory_prefix[[:space:]]*:[[:space:]]*['\"]?([^'\"#[:space:]]+).*/\1/p" "$config" | head -n 1
}

extract_run_suffix_arg() {
    local arg
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --run-suffix)
                shift
                [[ $# -gt 0 ]] || die "--run-suffix requires a value"
                printf '%s\n' "$1"
                return
                ;;
            --run-suffix=*)
                printf '%s\n' "${arg#--run-suffix=}"
                return
                ;;
        esac
        shift || true
    done
}

extract_resume_run_arg() {
    local arg
    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            --resume-run)
                shift
                [[ $# -gt 0 ]] || die "--resume-run requires a value"
                printf '%s\n' "$1"
                return
                ;;
            --resume-run=*)
                printf '%s\n' "${arg#--resume-run=}"
                return
                ;;
        esac
        shift || true
    done
}

has_arg() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        [[ "$arg" == "$needle" ]] && return 0
    done
    return 1
}

resolve_workspace_dir_for_config() {
    local config="$1"
    local prefix model agent
    prefix="$(read_workspace_prefix "$config")"
    model="$(read_target_gpu_model "$config")"
    agent="$(read_agent_template "$config")"
    [[ -n "$prefix" ]] || die "workspace_directory_prefix not found in $config"
    [[ -n "$model" ]] || die "target_gpu_model not found in $config"
    [[ -n "$agent" ]] || die "agent.template not found in $config"
    printf '%s/experiments/%s_%s_%s\n' "$HOST_ROOT" "$prefix" "$model" "$agent"
}

resolve_latest_run_name() {
    local config="$1"
    local workspace_dir
    workspace_dir="$(resolve_workspace_dir_for_config "$config")"
    [[ -d "$workspace_dir" ]] || die "No workspace directory found for resume-latest: $workspace_dir"
    find "$workspace_dir" -maxdepth 1 -mindepth 1 -type d -name 'run_*' ! -name '*_heldout' \
        -printf '%f\n' | sort -r | head -n 1
}

resolve_parallel_run_name() {
    local config="$1"
    shift
    local resume_run suffix latest
    resume_run="$(extract_resume_run_arg "$@" || true)"
    if [[ -n "$resume_run" ]]; then
        printf '%s\n' "$resume_run"
        return
    fi

    if has_arg --resume-latest "$@"; then
        latest="$(resolve_latest_run_name "$config")"
        [[ -n "$latest" ]] || die "No run directories found for --resume-latest"
        printf '%s\n' "$latest"
        return
    fi

    suffix="$(extract_run_suffix_arg "$@" || true)"
    if [[ -n "$suffix" ]]; then
        [[ "$suffix" =~ ^[A-Za-z0-9._-]+$ ]] || die "--run-suffix may only contain letters, numbers, dot, underscore, and dash"
        printf 'run_%s_%s\n' "$(date +%Y%m%d_%H%M%S)" "$suffix"
    else
        printf 'run_%s\n' "$(date +%Y%m%d_%H%M%S)"
    fi
}

resolve_gpu_ids() {
    if [[ -n "${GPU_IDS:-}" ]]; then
        printf '%s\n' "${GPU_IDS//,/ }" | tr ' ' '\n' | sed '/^$/d'
        return
    fi

    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showid 2>/dev/null \
            | sed -nE 's/.*GPU\[([0-9]+)\].*/\1/p' \
            | sort -n \
            | uniq
        return
    fi

    die "GPU_IDS is not set and rocm-smi is not available for GPU discovery"
}

safe_label() {
    local value="$1"
    value="${value//[^A-Za-z0-9_.-]/_}"
    printf '%s\n' "$value"
}

run_parallel() {
    local config_name
    config_name="$(extract_config_name "$@")"
    select_runtime_for_config "$config_name"
    configure_geak_v4_runtime "$config_name"

    REQUIRED_AGENTS="$(resolve_required_agents "$config_name")"
    AGENTS_STRICT=1

    local run_name safe_run_name
    run_name="$(resolve_parallel_run_name "$config_name" "$@")"
    safe_run_name="$(safe_label "$run_name")"

    local -a gpu_ids=()
    local gpu_id
    while IFS= read -r gpu_id; do
        [[ -n "$gpu_id" ]] && gpu_ids+=("$gpu_id")
    done < <(resolve_gpu_ids)
    [[ "${#gpu_ids[@]}" -gt 0 ]] || die "No GPU IDs available; set GPU_IDS=0,1,..."

    echo "Parallel run: run_name=${run_name} workers=${#gpu_ids[@]} gpu_ids=${gpu_ids[*]}" >&2

    (
        export AKA_VISIBLE_GPU="${gpu_ids[0]}"
        export AKA_WORKER_ID="preflight"
        export AKA_CONTAINER_HOME="/tmp/aka-home-${safe_run_name}-preflight"
        export AKA_CACHE_SUFFIX="${safe_run_name}-preflight"
        export AGENT_HOME_ISOLATION=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_preflight "$config_name"
    )

    (
        export AKA_VISIBLE_GPU="${gpu_ids[0]}"
        export AKA_WORKER_ID="init"
        export AKA_CONTAINER_HOME="/tmp/aka-home-${safe_run_name}-init"
        export AKA_CACHE_SUFFIX="${safe_run_name}-init"
        export AGENT_HOME_ISOLATION=1
        docker_exec 0 python main.py "$@" --parallel-init --run-name "$run_name"
    )

    local -a pids=()
    local worker_id
    for worker_id in "${!gpu_ids[@]}"; do
        gpu_id="${gpu_ids[$worker_id]}"
        (
            export AKA_VISIBLE_GPU="$gpu_id"
            export AKA_WORKER_ID="$worker_id"
            export AKA_CONTAINER_HOME="/tmp/aka-home-${safe_run_name}-worker-${worker_id}"
            export AKA_CACHE_SUFFIX="${safe_run_name}-worker-${worker_id}"
            export AGENT_HOME_ISOLATION=1
            trap stop_eval_tool_sidecars EXIT
            start_eval_tool_sidecars "$config_name" "${safe_run_name}-worker-${worker_id}"
            docker_exec 0 python main.py "$@" --parallel-worker --worker-id "$worker_id" --run-name "$run_name"
        ) &
        pids+=("$!")
    done

    local worker_failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            worker_failed=1
        fi
    done

    local postprocess_failed=0
    if ! (
        export AKA_VISIBLE_GPU="${gpu_ids[0]}"
        export AKA_WORKER_ID="postprocess"
        export AKA_CONTAINER_HOME="/tmp/aka-home-${safe_run_name}-postprocess"
        export AKA_CACHE_SUFFIX="${safe_run_name}-postprocess"
        export AGENT_HOME_ISOLATION=1
        docker_exec 0 python main.py "$@" --postprocess-only --run-name "$run_name"
    ); then
        postprocess_failed=1
    fi

    if [[ "$worker_failed" != "0" || "$postprocess_failed" != "0" ]]; then
        return 1
    fi
}

case "${1:-}" in
    run)
        shift
        config_name="$(extract_config_name "$@")"
        select_runtime_for_config "$config_name"
        configure_geak_v4_runtime "$config_name"
        # Only the configured agent's CLI/auth is required for a run.
        REQUIRED_AGENTS="$(resolve_required_agents "$config_name")"
        AGENTS_STRICT=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_preflight "$config_name"
        trap stop_eval_tool_sidecars EXIT
        start_eval_tool_sidecars "$config_name" "run-${BASHPID}"
        docker_exec 0 python main.py "$@"
        stop_eval_tool_sidecars
        trap - EXIT
        ;;
    parallel-run)
        shift
        run_parallel "$@"
        ;;
    quality-loop)
        shift
        quality_loop_config="$(extract_quality_loop_config "$@")"
        [[ -f "$quality_loop_config" ]] || die "quality_loop config file not found: $quality_loop_config"
        if has_arg --plan "$@"; then
            python3 -m agents.quality_loop "$@"
            exit
        fi
        quality_loop_resume="$(extract_quality_loop_resume "$@" || true)"
        if [[ -n "$quality_loop_resume" ]]; then
            quality_loop_run_id="$(python3 -m agents.quality_loop.host check "$@")"
        else
            quality_loop_run_id="$(python3 -m agents.quality_loop.host start "$@")"
        fi
        echo "quality_loop run ID: $quality_loop_run_id" >&2
        mapfile -t quality_loop_paths < <(
            python3 -m agents.quality_loop.host paths "$@" --run-id "$quality_loop_run_id"
        )
        [[ "${#quality_loop_paths[@]}" -eq 2 ]] \
            || die "quality_loop host returned invalid runtime paths"
        QUALITY_LOOP_ARTIFACT_REL="${quality_loop_paths[0]}"
        QUALITY_LOOP_WORKTREE_REL="${quality_loop_paths[1]}"
        select_runtime_for_config "$quality_loop_config"
        REQUIRED_AGENTS="codex"
        AGENTS_STRICT=1
        AGENT_HOME_ISOLATION=1
        AKA_CONTAINER_HOME="/tmp/aka-quality-loop-${quality_loop_run_id}"
        AKA_CACHE_SUFFIX="quality-loop-${quality_loop_run_id}"
        quality_loop_container_args=("$@")
        if [[ -z "$quality_loop_resume" ]]; then
            quality_loop_container_args+=(--resume "$quality_loop_run_id")
        fi
        quality_loop_container_args+=(--defer-github --skip-preflight)
        trap stop_eval_tool_sidecars EXIT
        start_eval_tool_sidecars \
            "$quality_loop_config" \
            "quality-loop-${quality_loop_run_id}"
        docker_exec 0 python3 -m agents.quality_loop "${quality_loop_container_args[@]}"
        stop_eval_tool_sidecars
        trap - EXIT
        python3 -m agents.quality_loop.host finalize "$@" --run-id "$quality_loop_run_id"
        ;;
    preflight)
        shift
        config_name="$(extract_config_name "$@")"
        select_runtime_for_config "$config_name"
        configure_geak_v4_runtime "$config_name"
        REQUIRED_AGENTS="$(resolve_required_agents "$config_name")"
        AGENTS_STRICT=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_preflight "$config_name"
        ;;
    shell)
        select_runtime_for_host
        # Interactive shell: provision whichever agents are installed (best-effort).
        REQUIRED_AGENTS="${AKA_AGENTS:-codex claude_code cursor}"
        REQUIRED_AGENTS="${REQUIRED_AGENTS//,/ }"
        AGENTS_STRICT=0
        build_docker_args 1
        docker "${docker_args[@]}"
        ;;
    check-agents)
        shift
        select_runtime_for_host
        config_name="$(extract_config_name "$@")"
        if [[ -z "${AKA_AGENTS:-}" ]]; then
            [[ -f "$config_name" ]] || die "config file not found: $config_name"
        fi
        configure_geak_v4_runtime "$config_name"
        # By default, check only the CLI selected by CONFIG. AKA_AGENTS can
        # request one, several, or `all` explicitly.
        REQUIRED_AGENTS="$(normalize_check_agents "$(resolve_required_agents "$config_name")")"
        AGENTS_STRICT=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_check_agents $REQUIRED_AGENTS
        ;;
    smoke)
        select_runtime_for_host
        REQUIRED_AGENTS="${AKA_AGENTS:-codex claude_code cursor}"
        REQUIRED_AGENTS="${REQUIRED_AGENTS//,/ }"
        AGENTS_STRICT=0
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_smoke
        ;;
    eval-tools-smoke)
        select_runtime_for_host
        export AKA_EVAL_TOOLS="${AKA_EVAL_TOOLS:-triton_fpsan,gpu_asan,rocjitsu,rocjitsu_waitcheck,rocjitsu_consan,hip_fpsan}"
        trap stop_eval_tool_sidecars EXIT
        start_eval_tool_sidecars "" "smoke-${BASHPID}"
        for eval_tool in "${eval_tool_ids[@]}"; do
            python3 -m src.eval_tools health \
                --socket "$EVAL_TOOL_SOCKET_HOST_DIR/$eval_tool.sock"
        done
        stop_eval_tool_sidecars
        trap - EXIT
        ;;
    build-eval-tool-images)
        build_eval_tool_images
        ;;
    setup-flydsl)
        select_runtime_for_host
        # FlyDSL install needs no agent CLIs.
        REQUIRED_AGENTS=""
        AGENTS_STRICT=0
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_setup_flydsl
        ;;
    setup-geak)
        select_runtime_for_host
        GEAK_V4_RUNTIME=1
        REQUIRED_AGENTS=""
        AGENTS_STRICT=0
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_setup_geak
        ;;
    _container_setup_flydsl)
        container_setup_flydsl
        ;;
    _container_setup_geak)
        container_setup_geak
        container_check_geak
        ;;
    _container_smoke)
        container_smoke
        ;;
    _container_check_agents)
        shift
        container_check_agents "$@"
        ;;
    _container_preflight)
        shift
        container_preflight "$@"
        ;;
    _container_prepare_worker_home)
        container_prepare_worker_home
        ;;
    _print_eval_tool_docker_args)
        shift
        [[ "$#" -eq 6 || "$#" -eq 7 || "$#" -eq 8 ]] \
            || die "_print_eval_tool_docker_args expects TOOL IMAGE NAME SOCKET SCRATCH ARTIFACT [RUNTIME_REF [PASSWD_FILE]]"
        build_eval_tool_docker_args "$@"
        printf '%s\n' "${eval_tool_docker_args[@]}"
        ;;
    _verify_eval_tool_scoring_image)
        shift
        [[ "$#" -eq 2 ]] \
            || die "_verify_eval_tool_scoring_image expects ARCH IMAGE"
        SELECTED_GPU_ARCH="$1"
        SELECTED_IMAGE="$2"
        verify_eval_tool_scoring_image
        printf '%s\n' "$SELECTED_IMAGE" "$AKA_SCORING_IMAGE_RUNTIME_REF" "$AKA_SCORING_IMAGE_REFERENCE"
        ;;
    _prepare_eval_tool_artifact_dir)
        shift
        [[ "$#" -eq 2 ]] \
            || die "_prepare_eval_tool_artifact_dir expects PARENT LABEL"
        prepare_eval_tool_artifact_dir "$1" "$2"
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

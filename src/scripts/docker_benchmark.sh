#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DOCKER_IMAGE_GFX942="${AKA_DOCKER_IMAGE_GFX942:-lmsysorg/sglang:v0.5.12-rocm720-mi30x}"
DEFAULT_DOCKER_IMAGE_GFX950="${AKA_DOCKER_IMAGE_GFX950:-lmsysorg/sglang:v0.5.12-rocm720-mi35x}"
DEFAULT_DOCKER_IMAGE_NAVI="${AKA_DOCKER_IMAGE_NAVI:-rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0}"
CONTAINER_WORKDIR="${AKA_DOCKER_WORKDIR:-/workspace}"
HOST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_HOME="${HOME:?HOME must be set}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
SELECTED_GPU_ARCH=""
SELECTED_IMAGE=""

# /opt/venv/bin is placed before /usr/local/bin and /usr/bin so that a bare
# `python3` / `pytest` resolves to the torch-enabled venv interpreter rather than
# the system python (which lacks torch). Without this, repository tasks whose
# commands call `python3 scripts/task_runner.py` fail with ModuleNotFoundError: torch.
container_path="/opt/node/bin:${HOST_HOME}/.local/bin:/opt/venv/bin:/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

usage() {
    cat <<'EOF'
Usage:
  src/scripts/docker_benchmark.sh run [main.py args...]
  src/scripts/docker_benchmark.sh preflight [--config_name config.yaml]
  src/scripts/docker_benchmark.sh shell
  src/scripts/docker_benchmark.sh check-agents
  src/scripts/docker_benchmark.sh smoke

Environment overrides:
  AKA_DOCKER_IMAGE        Absolute Docker image override.
  AKA_GPU_ARCH            GPU arch override for shell/smoke, or run configs without target_gpu_model.
  AKA_DOCKER_IMAGE_<ARCH> Per-arch image override, e.g. AKA_DOCKER_IMAGE_GFX950=...
  AKA_DOCKER_IMAGE_GFX942 Default image for gfx942.
  AKA_DOCKER_IMAGE_GFX950 Default image for gfx950.
  AKA_DOCKER_IMAGE_NAVI   Default image for Navi/RDNA/Radeon gfx10*/gfx11*/gfx12*.
  AKA_FLYDSL_SPEC         FlyDSL pip requirement for setup-flydsl (default: flydsl>=0.2,<0.3).
  AKA_FLYDSL_EXTRA_INDEX_URL
                          Extra pip index for FlyDSL wheels (default: ROCm nightly gfx942/gfx950).
  AKA_NODE_PREFIX         Host Node prefix containing bin/node and bin/codex.
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
        gfx10*|gfx11*|gfx12*) printf '%s\n' "$DEFAULT_DOCKER_IMAGE_NAVI" ;;
        *)
            die "No Docker image mapping for GPU arch '$arch'. Set AKA_DOCKER_IMAGE or ${env_name}."
            ;;
    esac
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
    [[ -n "$arch" ]] || die "Could not infer GPU arch; set AKA_GPU_ARCH=gfx942, gfx950, or a Navi/RDNA arch such as gfx1100"

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
        *) printf '%s\n' "$tmpl" ;;
    esac
}

# Mount one agent's CLI install + auth dirs. $2=strict (1 require, 0 best-effort).
mount_agent() {
    local agent="$1" strict="${2:-1}"
    case "$agent" in
        codex)
            if ! command -v node >/dev/null 2>&1; then
                [[ "$strict" == "1" ]] && die "Codex agent requires host node on PATH"
                warn "node not found on PATH; skipping Codex agent mounts"
                return 0
            fi
            local node_prefix
            node_prefix="$(detect_node_prefix)"
            need_path "$node_prefix/bin/node" "host node" "$strict" || return 0
            need_path "$node_prefix/bin/codex" "host codex" "$strict" || return 0
            need_path "$HOST_HOME/.codex" "Codex auth/config directory" "$strict" || return 0
            add_mount "$node_prefix" /opt/node ro
            add_mount "$HOST_HOME/.codex" "$HOST_HOME/.codex"
            ;;
        claude_code)
            need_path "$HOST_HOME/.local/bin" "host local bin directory" "$strict" || return 0
            need_path "$HOST_HOME/.local/share/claude" "Claude Code local install" "$strict" || return 0
            need_path "$HOST_HOME/.claude" "Claude Code auth directory" "$strict" || return 0
            need_path "$HOST_HOME/.claude.json" "Claude Code auth/config file" "$strict" || return 0
            add_mount "$HOST_HOME/.local/bin" "$HOST_HOME/.local/bin" ro
            add_mount "$HOST_HOME/.local/share/claude" "$HOST_HOME/.local/share/claude" ro
            add_mount "$HOST_HOME/.claude" "$HOST_HOME/.claude"
            add_mount "$HOST_HOME/.claude.json" "$HOST_HOME/.claude.json"
            ;;
        cursor)
            need_path "$HOST_HOME/.local/bin" "host local bin directory" "$strict" || return 0
            need_path "$HOST_HOME/.local/share/cursor-agent" "Cursor Agent local install" "$strict" || return 0
            need_path "$HOST_HOME/.cursor" "Cursor Agent state directory" "$strict" || return 0
            need_path "$HOST_HOME/.config/cursor" "Cursor Agent config directory" "$strict" || return 0
            add_mount "$HOST_HOME/.local/bin" "$HOST_HOME/.local/bin" ro
            add_mount "$HOST_HOME/.local/share/cursor-agent" "$HOST_HOME/.local/share/cursor-agent" ro
            add_mount "$HOST_HOME/.cursor" "$HOST_HOME/.cursor"
            add_mount "$HOST_HOME/.config/cursor" "$HOST_HOME/.config/cursor"
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

build_docker_args() {
    local interactive="${1:-0}"
    # Which agent CLIs to provision, and whether their absence is fatal.
    # Defaults (no caller override) are best-effort over all three — used by
    # interactive `shell`/`smoke` so any installed agent works.
# Use `-` (not `:-`) so an explicitly-empty REQUIRED_AGENTS means "no agents"
# (e.g. setup-flydsl), while unset falls back to all three.
    local agents="${REQUIRED_AGENTS-codex claude_code cursor}"
    local strict="${AGENTS_STRICT:-0}"

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
        -e "HOME=${HOST_HOME}"
        -e "CODEX_HOME=${HOST_HOME}/.codex"
        -e "XDG_CACHE_HOME=/tmp/agent-cache"
        -e "MPLCONFIGDIR=/tmp/matplotlib"
        -e "TORCH_EXTENSIONS_DIR=/tmp/torch-extensions"
        -e "TRITON_CACHE_DIR=/tmp/triton-cache"
        -e "PYTHONUSERBASE=${CONTAINER_WORKDIR}/.aka-pyuserbase"
        -e "PYTHONPATH=${CONTAINER_WORKDIR}/.aka-pyuserbase/site-packages"
        -e "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-cache"
        -e "FLYDSL_AUTOTUNE_CACHE_DIR=/tmp/flydsl-autotune"
        -e "MIOPEN_USER_DB_PATH=/tmp/miopen-cache"
        -e "MIOPEN_CACHE_DIR=/tmp/miopen-cache"
        -e "MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache"
        -e "AGENT_KERNEL_ARENA_DOCKER=1"
        -e "AGENT_KERNEL_ARENA_WORKDIR=${CONTAINER_WORKDIR}"
        -e "AGENT_KERNEL_ARENA_GPU_ARCH=${SELECTED_GPU_ARCH}"
        -e "PYTORCH_ROCM_ARCH=${SELECTED_GPU_ARCH}"
        -e "PATH=${container_path}"
        -w "$CONTAINER_WORKDIR"
    )

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

    add_mount "$HOST_ROOT" "$CONTAINER_WORKDIR"
    # Persistent Python package overlay so `make docker-setup-flydsl` survives
    # across runs. It lives INSIDE the repo dir, which is already bind-mounted above and
    # is owned by the host user — this avoids a separate mount whose source the docker
    # daemon would have to create (which fails on NFS/root-squashed homes).
    mkdir -p "$HOST_ROOT/.aka-pyuserbase/site-packages" 2>/dev/null || true
    local _agent
    for _agent in $agents; do
        mount_agent "$_agent" "$strict"
    done

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
    docker "${docker_args[@]}" -lc 'cd "$AGENT_KERNEL_ARENA_WORKDIR" && exec "$@"' _ "$@"
}

extract_config_name() {
    local config="config.yaml"
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

for mod_name in ("torch", "triton", "pytest", "yaml", "numpy", "flydsl"):
    mod = importlib.import_module(mod_name)
    print(f"{mod_name}=ok {getattr(mod, '__version__', '')}")

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
        "fix target_gpu_model for benchmark runs, or use AKA_GPU_ARCH only for shell/smoke diagnostics"
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
    local config_name="${1:-config.yaml}"
    container_smoke
    # Only verify the agent(s) this config actually uses (mounts are scoped the same way).
    container_check_agents $(resolve_required_agents "$config_name")
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
    local flydsl_spec="${AKA_FLYDSL_SPEC:-flydsl>=0.2,<0.3}"
    local extra_index="${AKA_FLYDSL_EXTRA_INDEX_URL:-https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/}"
    local expected_prefix="${AKA_FLYDSL_EXPECTED_PREFIX:-0.2}"
    local install_target="${PYTHONUSERBASE:-${CONTAINER_WORKDIR}/.aka-pyuserbase}/site-packages"

    python - <<'PY' || true
try:
    import flydsl
    print("existing flydsl=" + str(getattr(flydsl, "__version__", "unknown")) + " at " + str(getattr(flydsl, "__file__", "unknown")))
except Exception as exc:
    print("existing flydsl import failed: " + repr(exc))
PY

    # Always install into the persistent package overlay. The benchmark images
    # may already ship FlyDSL, but those baked-in versions can lag the task suite.
    # The image Python runs from a venv where --user installs are disabled, so use
    # --target plus PYTHONPATH instead.
    mkdir -p "$install_target"
    echo "Installing FlyDSL requirement '${flydsl_spec}' into ${install_target}..."
    python -m pip install --upgrade --force-reinstall --pre --target "$install_target" --extra-index-url "$extra_index" "$flydsl_spec"

    AKA_EXPECTED_FLYDSL_PREFIX="$expected_prefix" python - <<'PY'
import os
import sys

import flydsl

version = str(getattr(flydsl, "__version__", "unknown"))
path = str(getattr(flydsl, "__file__", "unknown"))
expected = os.environ.get("AKA_EXPECTED_FLYDSL_PREFIX", "0.2")
print(f"flydsl={version} path={path} setup OK")
if not version.startswith(expected):
    raise SystemExit(f"FlyDSL version {version!r} does not start with expected prefix {expected!r}")
PY
}

case "${1:-}" in
    run)
        shift
        config_name="$(extract_config_name "$@")"
        select_runtime_for_config "$config_name"
        # Only the configured agent's CLI/auth is required for a run.
        REQUIRED_AGENTS="$(resolve_required_agents "$config_name")"
        AGENTS_STRICT=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_preflight "$config_name"
        docker_exec 0 python main.py "$@"
        ;;
    preflight)
        shift
        config_name="$(extract_config_name "$@")"
        select_runtime_for_config "$config_name"
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
        select_runtime_for_host
        # check-agents is the strict, all-three verification path.
        REQUIRED_AGENTS="codex claude_code cursor"
        AGENTS_STRICT=1
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_check_agents
        ;;
    smoke)
        select_runtime_for_host
        REQUIRED_AGENTS="${AKA_AGENTS:-codex claude_code cursor}"
        REQUIRED_AGENTS="${REQUIRED_AGENTS//,/ }"
        AGENTS_STRICT=0
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_smoke
        ;;
    setup-flydsl)
        select_runtime_for_host
        # FlyDSL install needs no agent CLIs.
        REQUIRED_AGENTS=""
        AGENTS_STRICT=0
        docker_exec 0 bash src/scripts/docker_benchmark.sh _container_setup_flydsl
        ;;
    _container_setup_flydsl)
        container_setup_flydsl
        ;;
    _container_smoke)
        container_smoke
        ;;
    _container_check_agents)
        container_check_agents
        ;;
    _container_preflight)
        shift
        container_preflight "$@"
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

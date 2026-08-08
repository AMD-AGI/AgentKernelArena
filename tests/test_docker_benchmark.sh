#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/src/scripts/docker_benchmark.sh"
cd "$ROOT"
PINNED_GFX950_IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705"
GFX950_V0514_DOCKER_IMAGE_ID="sha256:0a78d51f2f1db80a1abfe23350fc2e5733ac5acb1528d6dc7ce3679bdb099aff"
export GFX950_V0514_DOCKER_IMAGE_ID
OLD_GFX950_IMAGE="lmsysorg/sglang:v0.5.12-rocm720-mi35x"

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
    fail "missing Docker argument: $expected"
}

assert_not_has() {
    local unexpected="$1"
    shift
    local value
    for value in "$@"; do
        [[ "$value" != "$unexpected" ]] || fail "unexpected Docker argument: $unexpected"
    done
}

find_arg_with_prefix() {
    local prefix="$1"
    shift
    local value
    for value in "$@"; do
        if [[ "$value" == "$prefix"* ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done
    return 1
}

find_arg_with_suffix() {
    local suffix="$1"
    shift
    local value
    for value in "$@"; do
        if [[ "$value" == *"$suffix" ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done
    return 1
}

# Capture the exact argv that the runner would pass to Docker without requiring
# a daemon, GPU devices, or the benchmark images on this host.
docker() {
    if [[ "${1:-}" == image && "${2:-}" == inspect ]]; then
        case "${4:-}" in
            '{{.Id}}') printf '%s\n' "$GFX950_V0514_DOCKER_IMAGE_ID" ;;
            '{{json .RepoDigests}}') printf '["example.invalid/runtime@sha256:%064d"]\n' 0 ;;
            *) return 2 ;;
        esac
        return 0
    fi
    local argument
    for argument in "$@"; do
        if [[ "$argument" == "com.amd.aka.apex-runtime-preflight=1" ]]; then
            printf '{"runtime_manifest_sha256":"%064d","schema":"aka.apex-runtime-snapshot/v1"}\n' 0
            return 0
        fi
    done
    printf '%s\n' "$@"
}
export -f docker

run_shell_args() {
    env \
        HOME="$TEST_HOME" \
        AKA_AGENTS=test-noop-agent \
        "$@" \
        bash "$RUNNER" shell 2>/dev/null
}

run_check_args() {
    local home="$1"
    local config="$2"
    shift 2
    env \
        HOME="$home" \
        AKA_GPU_ARCH=gfx950 \
        "$@" \
        bash "$RUNNER" check-agents --config_name "$config" 2>/dev/null
}

assert_cache_args_present() {
    local suffix="$1"
    shift
    assert_has "AITER_JIT_DIR=/tmp/aiter-jit${suffix}" "$@"
    assert_has "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-runtime-cache${suffix}" "$@"
    assert_has "/tmp/aiter_configs:rw,uid=$(id -u),gid=$(id -g),mode=1777" "$@"
}

assert_cache_args_absent() {
    assert_not_has "AITER_JIT_DIR=/tmp/aiter-jit" "$@"
    assert_not_has "FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl-runtime-cache" "$@"
    assert_not_has "/tmp/aiter_configs:rw,uid=$(id -u),gid=$(id -g),mode=1777" "$@"
}

TEST_HOME="$(mktemp -d)"
cleanup_test_home() {
    chmod -R u+w "$TEST_HOME" 2>/dev/null || true
    rm -rf "$TEST_HOME"
}
trap cleanup_test_home EXIT
UNRELATED_GEAK_WORKFLOW_DIR="$TEST_HOME/unrelated-geak-workflow"
GEAK_SDK_PYTHONPATH="PYTHONPATH=/workspace/.aka-pyuserbase/geak-sdk"
mkdir -p "$UNRELATED_GEAK_WORKFLOW_DIR"
touch "$UNRELATED_GEAK_WORKFLOW_DIR/kernel_workflow.js"

bash -n "$RUNNER"
grep -Fq 'json.dumps(dict(device_count=count' "$RUNNER" \
    || fail "runtime GPU observation must avoid nested single-quote corruption"
grep -Fq -- '--unshare-pid --unshare-ipc' "$RUNNER" \
    || fail "rootless bwrap preflight must create a private attempt PID namespace"
grep -Fq -- '--proc /proc' "$RUNNER" \
    || fail "rootless bwrap preflight must mount the attempt-private procfs"

# Formal HOME preparation is fail-closed: it overrides every caller-provided
# mutable path. The same helper must be a strict no-op outside a formal campaign.
(
    # shellcheck source=../src/scripts/docker_benchmark.sh
    source "$RUNNER"
    CAMPAIGN_PROVENANCE=0
    AKA_CONTAINER_HOME="/caller/home"
    AKA_CODEX_HOME="/caller/codex"
    AKA_CACHE_SUFFIX="caller-cache"
    AGENT_HOME_ISOLATION=0
    prepare_formal_container_home "ignored/label"
    [[ "$AKA_CONTAINER_HOME" == "/caller/home" ]]
    [[ "$AKA_CODEX_HOME" == "/caller/codex" ]]
    [[ "$AKA_CACHE_SUFFIX" == "caller-cache" ]]
    [[ "$AGENT_HOME_ISOLATION" == "0" ]]

    CAMPAIGN_PROVENANCE=1
    prepare_formal_container_home "formal/worker"
    [[ "$AKA_CONTAINER_HOME" == "/tmp/aka-home-formal_worker" ]]
    [[ "$AKA_CODEX_HOME" == "/tmp/aka-home-formal_worker/.codex" ]]
    [[ "$AKA_CACHE_SUFFIX" == "formal_worker" ]]
    [[ "$AGENT_HOME_ISOLATION" == "1" ]]
) || fail "formal container HOME preparation contract failed"

# Exercise the formal lease/inventory path itself with CPU-only command fakes.
# This catches shell-scope regressions such as passing an unset inventory path.
FORMAL_SHELL_ROOT="$TEST_HOME/formal-shell"
FORMAL_INVENTORY_MARKER="$FORMAL_SHELL_ROOT/inventory-path"
mkdir -p "$FORMAL_SHELL_ROOT/data"
printf '{}\n' > "$FORMAL_SHELL_ROOT/gpu-plan.json"
(
    # shellcheck source=../src/scripts/docker_benchmark.sh
    source "$RUNNER"
    CAMPAIGN_PROVENANCE=1
    CAMPAIGN_DATA_ROOT="$FORMAL_SHELL_ROOT/data"
    CAMPAIGN_GPU_PLAN_HOST="$FORMAL_SHELL_ROOT/gpu-plan.json"
    CAMPAIGN_GPU_PLAN_SHA256="$(printf '%064d' 0)"
    CAMPAIGN_GPU_LEASE_FDS=()
    AKA_GPU_LEASE_ROOT="$FORMAL_SHELL_ROOT/leases"

    python3() {
        local script="$1"
        shift
        case "$(basename "$script"):${1:-}" in
            gpu_exclusivity.py:lease-keys)
                printf '0x0000000000000001\n'
                ;;
            kfd_process_inventory.py:--output)
                printf '{"cpu_only_fake_inventory":true}\n' > "$2"
                ;;
            gpu_exclusivity.py:create-receipt)
                local inventory="" output=""
                while [[ "$#" -gt 0 ]]; do
                    case "$1" in
                        --kfd-process-inventory)
                            inventory="$2"
                            shift 2
                            ;;
                        --output)
                            output="$2"
                            shift 2
                            ;;
                        *)
                            shift
                            ;;
                    esac
                done
                [[ -n "$inventory" && -f "$inventory" && -s "$inventory" ]] || return 70
                printf '%s\n' "$inventory" > "$FORMAL_INVENTORY_MARKER"
                printf '{"cpu_only_fake_receipt":true}\n' > "$output"
                ;;
            gpu_exclusivity.py:verify-receipt)
                printf '%064d\n' 0
                ;;
            *)
                return 71
                ;;
        esac
    }

    acquire_campaign_gpu_exclusivity "cpu-only-formal-shell-test"
    [[ -s "$FORMAL_INVENTORY_MARKER" ]]
    inventory_path="$(<"$FORMAL_INVENTORY_MARKER")"
    [[ "$inventory_path" == "$FORMAL_SHELL_ROOT/data/"* ]]
    [[ -f "$inventory_path" && -s "$inventory_path" ]]
    [[ "$(stat -c '%a' "$inventory_path")" == "444" ]]
    [[ -f "$CAMPAIGN_GPU_EXCLUSIVITY_HOST" ]]
) || fail "formal GPU acquire path did not pass a published KFD inventory artifact"

# The internal container subcommand must forward the selected agent list instead
# of falling back to its all-three default. A fake Python records the environment
# without invoking any real agent CLI.
FAKE_BIN="$TEST_HOME/fake-bin"
mkdir -p "$FAKE_BIN"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$AKA_CHECK_AGENTS"\n' > "$FAKE_BIN/python"
chmod +x "$FAKE_BIN/python"
forwarded_agents="$(PATH="$FAKE_BIN:$PATH" bash "$RUNNER" _container_check_agents cursor)"
[[ "$forwarded_agents" == "cursor" ]] || fail "container check received '$forwarded_agents', expected cursor"

# The gfx950 default resolves to the pinned image and enables writable caches.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950)
assert_has "$PINNED_GFX950_IMAGE" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID=$GFX950_V0514_DOCKER_IMAGE_ID" "${args[@]}"
assert_has 'AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS=["example.invalid/runtime@sha256:0000000000000000000000000000000000000000000000000000000000000000"]' "${args[@]}"
assert_cache_args_present "" "${args[@]}"

# A caller-supplied expected image ID is a check, never an unchecked override.
if run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE_ID="sha256:$(printf '%064d' 9)" \
    >/dev/null 2>&1; then
    fail "mismatched AKA_DOCKER_IMAGE_ID was accepted"
fi

mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_GPU_POOL=0,1,2,3,4,5,6,7)
assert_has "AGENT_KERNEL_ARENA_GPU_POOL=0,1,2,3,4,5,6,7" "${args[@]}"

# A worker suffix must isolate both runtime cache directories.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_CACHE_SUFFIX=worker/3)
assert_cache_args_present "-worker_3" "${args[@]}"

# Explicitly selecting the same verified tag has the same behavior.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE="$PINNED_GFX950_IMAGE")
assert_cache_args_present "" "${args[@]}"

# Old and custom gfx950 images retain their existing Docker arguments.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE="$OLD_GFX950_IMAGE")
assert_cache_args_absent "${args[@]}"
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx950 AKA_DOCKER_IMAGE_GFX950=example.invalid/custom:latest)
assert_cache_args_absent "${args[@]}"

# The unchanged gfx942 default does not receive the gfx950-only configuration.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx942)
assert_has "lmsysorg/sglang:v0.5.12-rocm720-mi30x" "${args[@]}"
assert_cache_args_absent "${args[@]}"

# Image equality alone is insufficient: the selected architecture must be gfx950.
mapfile -t args < <(run_shell_args AKA_GPU_ARCH=gfx942 AKA_DOCKER_IMAGE="$PINNED_GFX950_IMAGE")
assert_cache_args_absent "${args[@]}"

# By default, check-agents provisions only the CLI selected by the config.
CURSOR_HOME="$TEST_HOME/cursor-home"
CURSOR_CONFIG="$TEST_HOME/cursor-config.yaml"
mkdir -p \
    "$CURSOR_HOME/.local/bin" \
    "$CURSOR_HOME/.local/share/cursor-agent" \
    "$CURSOR_HOME/.cursor" \
    "$CURSOR_HOME/.config/cursor"
touch "$CURSOR_HOME/.local/bin/cursor-agent"
printf 'agent:\n  template: cursor\n' > "$CURSOR_CONFIG"

mapfile -t args < <(run_check_args \
    "$CURSOR_HOME" \
    "$CURSOR_CONFIG" \
    ANTHROPIC_AUTH_TOKEN=cursor-must-not-receive-this \
    ANTHROPIC_BASE_URL=https://gateway.example/api \
    GEAK_V4_WORKFLOW_DIR="$UNRELATED_GEAK_WORKFLOW_DIR")
assert_has "$CURSOR_HOME/.local/share/cursor-agent:$CURSOR_HOME/.local/share/cursor-agent:ro" "${args[@]}"
assert_has "$CURSOR_HOME/.cursor:$CURSOR_HOME/.cursor" "${args[@]}"
assert_has "$CURSOR_HOME/.config/cursor:$CURSOR_HOME/.config/cursor" "${args[@]}"
assert_has "_container_check_agents" "${args[@]}"
assert_has "cursor" "${args[@]}"
assert_not_has "$CURSOR_HOME/.claude:$CURSOR_HOME/.claude" "${args[@]}"
assert_not_has "$CURSOR_HOME/.codex:$CURSOR_HOME/.codex" "${args[@]}"
assert_not_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"
assert_not_has "ANTHROPIC_BASE_URL" "${args[@]}"
assert_not_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"
assert_not_has "$UNRELATED_GEAK_WORKFLOW_DIR:$UNRELATED_GEAK_WORKFLOW_DIR:ro" "${args[@]}"
assert_not_has "GEAK_V4_WORKFLOW_DIR=$UNRELATED_GEAK_WORKFLOW_DIR" "${args[@]}"

# A Codex-only config likewise receives neither Claude credentials nor GEAK's
# dependency path/mount, even when both are configured on the host.
CODEX_HOME="$TEST_HOME/codex-home"
CODEX_PREFIX="$TEST_HOME/codex-node"
CODEX_CONFIG="$TEST_HOME/codex-config.yaml"
mkdir -p "$CODEX_HOME/.codex" "$CODEX_PREFIX/bin"
touch "$CODEX_PREFIX/bin/node" "$CODEX_PREFIX/bin/codex"
printf 'agent:\n  template: codex\n' > "$CODEX_CONFIG"

mapfile -t args < <(run_check_args \
    "$CODEX_HOME" \
    "$CODEX_CONFIG" \
    AKA_NODE_PREFIX="$CODEX_PREFIX" \
    ANTHROPIC_API_KEY=codex-must-not-receive-this \
    GEAK_V4_WORKFLOW_DIR="$UNRELATED_GEAK_WORKFLOW_DIR")
assert_has "$CODEX_PREFIX:/opt/node:ro" "${args[@]}"
assert_has "$CODEX_HOME/.codex:$CODEX_HOME/.codex" "${args[@]}"
assert_has "codex" "${args[@]}"
assert_not_has "ANTHROPIC_API_KEY" "${args[@]}"
assert_not_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"
assert_not_has "$UNRELATED_GEAK_WORKFLOW_DIR:$UNRELATED_GEAK_WORKFLOW_DIR:ro" "${args[@]}"
assert_not_has "GEAK_V4_WORKFLOW_DIR=$UNRELATED_GEAK_WORKFLOW_DIR" "${args[@]}"

# Apex is an orchestrating agent. The runner mounts its checkout read-only and
# provisions only the backend selected in agents/apex/agent_config.yaml (Codex
# by default), rather than exposing all three backend credentials.
APEX_ROOT="$TEST_HOME/apex-checkout"
APEX_CONFIG="$TEST_HOME/apex-config.yaml"
mkdir -p "$APEX_ROOT"
touch "$APEX_ROOT/main.py"
mkdir -p "$APEX_ROOT/.venv/bin"
touch "$APEX_ROOT/.venv/bin/python"
chmod +x "$APEX_ROOT/.venv/bin/python"
printf 'agent:\n  template: apex\n' > "$APEX_CONFIG"

mapfile -t args < <(run_check_args \
    "$CODEX_HOME" \
    "$APEX_CONFIG" \
    AKA_APEX_ROOT="$APEX_ROOT" \
    AKA_NODE_PREFIX="$CODEX_PREFIX" \
    ANTHROPIC_API_KEY=apex-codex-must-not-receive-this)
assert_has "$APEX_ROOT:$APEX_ROOT:ro" "${args[@]}"
assert_has "APEX_ROOT=$APEX_ROOT" "${args[@]}"
assert_has "APEX_PYTHON=$APEX_ROOT/.venv/bin/python" "${args[@]}"
assert_has "PYTHONDONTWRITEBYTECODE=1" "${args[@]}"
assert_not_has "PYTHONPATH=" "${args[@]}"
assert_has "$CODEX_PREFIX:/opt/node:ro" "${args[@]}"
assert_has "$CODEX_HOME/.codex:$CODEX_HOME/.codex" "${args[@]}"
assert_has "codex" "${args[@]}"
assert_not_has "$CODEX_HOME/.claude:$CODEX_HOME/.claude" "${args[@]}"
assert_not_has "$CODEX_HOME/.local/share/cursor-agent:$CODEX_HOME/.local/share/cursor-agent:ro" "${args[@]}"
assert_not_has "ANTHROPIC_API_KEY" "${args[@]}"

# A matched direct-Codex campaign receives only the Apex Git receipt, not the
# Apex source mount. This pins the treatment revision without letting the
# baseline agent inspect the treatment implementation.
CAMPAIGN_APEX_ROOT="$TEST_HOME/campaign-apex-checkout"
CAMPAIGN_CODEX_CONFIG="$TEST_HOME/campaign-codex-config.yaml"
mkdir -p "$CAMPAIGN_APEX_ROOT"
git -C "$CAMPAIGN_APEX_ROOT" init -q
touch "$CAMPAIGN_APEX_ROOT/main.py"
printf '.venv\n' > "$CAMPAIGN_APEX_ROOT/.gitignore"
mkdir -p "$CAMPAIGN_APEX_ROOT/.venv/bin"
mkdir -p "$CAMPAIGN_APEX_ROOT/.venv/lib/python3.10/site-packages"
ln -s /usr/bin/python3 "$CAMPAIGN_APEX_ROOT/.venv/bin/python"
printf 'include-system-site-packages = false\n' > "$CAMPAIGN_APEX_ROOT/.venv/pyvenv.cfg"
git -C "$CAMPAIGN_APEX_ROOT" add main.py .gitignore
git -C "$CAMPAIGN_APEX_ROOT" \
    -c user.name=AKA -c user.email=aka@example.invalid \
    commit -q -m initial
CAMPAIGN_APEX_COMMIT="$(git -C "$CAMPAIGN_APEX_ROOT" rev-parse HEAD)"
CAMPAIGN_DATA_ROOT="$TEST_HOME/campaign-data"
printf 'agent:\n  template: codex\ncampaign:\n  comparison: apex_vs_codex\nworkspace_directory_prefix: %s/workspace\n' \
    "$CAMPAIGN_DATA_ROOT" > "$CAMPAIGN_CODEX_CONFIG"

mapfile -t args < <(run_check_args \
    "$CODEX_HOME" \
    "$CAMPAIGN_CODEX_CONFIG" \
    AKA_APEX_ROOT="$CAMPAIGN_APEX_ROOT" \
    AKA_NODE_PREFIX="$CODEX_PREFIX")
assert_has "AGENT_KERNEL_ARENA_APEX_COMMIT=$CAMPAIGN_APEX_COMMIT" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_APEX_DIRTY=false" "${args[@]}"
apex_runtime_digest_arg="$(
    find_arg_with_prefix "AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256=" "${args[@]}"
)" || fail "formal campaign did not bind the Apex runtime manifest"
[[ "${apex_runtime_digest_arg#*=}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "formal Apex runtime manifest digest is invalid"
assert_has "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID=$GFX950_V0514_DOCKER_IMAGE_ID" "${args[@]}"
assert_not_has "$CAMPAIGN_APEX_ROOT:$CAMPAIGN_APEX_ROOT:ro" "${args[@]}"
assert_not_has "APEX_ROOT=$CAMPAIGN_APEX_ROOT" "${args[@]}"
assert_has "$CAMPAIGN_DATA_ROOT:$CAMPAIGN_DATA_ROOT" "${args[@]}"
aka_runtime_mount="$(find_arg_with_suffix ":/workspace:ro" "${args[@]}")" \
    || fail "formal campaign did not mount a sealed AKA runtime"
[[ "$aka_runtime_mount" == /tmp/agentkernelarena-formal-runtime.*:/workspace:ro ]] \
    || fail "formal AKA runtime mount is not an isolated immutable mount: $aka_runtime_mount"
assert_not_has "$ROOT:/workspace:ro" "${args[@]}"
assert_not_has "$ROOT:/workspace" "${args[@]}"
assert_has "$CODEX_HOME/.codex:/opt/aka-agent-state/.codex:ro" "${args[@]}"
assert_not_has "$CODEX_HOME/.codex:$CODEX_HOME/.codex" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_ISOLATED_HOME=1" "${args[@]}"
formal_home_arg="$(find_arg_with_prefix "HOME=/tmp/aka-home-check-agents-" "${args[@]}")" \
    || fail "formal check-agents did not receive an ephemeral HOME"
formal_home="${formal_home_arg#HOME=}"
assert_has "CODEX_HOME=$formal_home/.codex" "${args[@]}"
formal_label="${formal_home#/tmp/aka-home-}"
assert_has "XDG_CACHE_HOME=/tmp/agent-cache-$formal_label" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT=$CAMPAIGN_DATA_ROOT" "${args[@]}"
assert_has "AGENT_KERNEL_ARENA_AKA_RUNTIME_ROOT=/workspace" "${args[@]}"
aka_manifest_arg="$(
    find_arg_with_prefix "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST=" "${args[@]}"
)" || fail "formal campaign did not mount an AKA execution manifest"
[[ "${aka_manifest_arg#*=}" == "$CAMPAIGN_DATA_ROOT/aka-runtime-manifest-"*.json ]] \
    || fail "formal AKA manifest is not a persistent campaign artifact"
aka_image_arg="$(
    find_arg_with_prefix "AGENT_KERNEL_ARENA_AKA_RUNTIME_IMAGE_SHA256=" "${args[@]}"
)" || fail "formal campaign did not bind the AKA immutable image"
[[ "${aka_image_arg#*=}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "formal AKA image digest is invalid"
assert_has "/usr/bin/bwrap:/usr/bin/bwrap:ro" "${args[@]}"
formal_requirements_mount="$(
    find_arg_with_suffix "/agents/codex/formal_requirements.toml:/etc/codex/requirements.toml:ro" "${args[@]}"
)" || fail "formal Codex requirements were not sourced from the sealed AKA runtime"
[[ "$formal_requirements_mount" == /tmp/agentkernelarena-formal-runtime.* ]] \
    || fail "formal Codex requirements came from a live checkout"
assert_has "--security-opt=seccomp=unconfined" "${args[@]}"
assert_has "--security-opt=apparmor=unconfined" "${args[@]}"
assert_has "--security-opt=no-new-privileges:true" "${args[@]}"
assert_has "--security-opt=systempaths=unconfined" "${args[@]}"
assert_has "--cap-drop=ALL" "${args[@]}"
assert_not_has "--privileged" "${args[@]}"
assert_not_has "--cap-add=SYS_ADMIN" "${args[@]}"
assert_not_has "--cap-add=SYS_PTRACE" "${args[@]}"
assert_not_has "--pid=host" "${args[@]}"
assert_not_has "--device=/dev/mem" "${args[@]}"
assert_not_has "--device=/dev/dri" "${args[@]}"
assert_not_has "--device=/dev/kfd" "${args[@]}"

# The Apex arm executes only from its digest-addressed SquashFS snapshot. The
# live treatment checkout remains provenance text and is not visible in Docker.
CAMPAIGN_APEX_AGENT_CONFIG="$TEST_HOME/campaign-apex-agent-config.yaml"
printf 'agent:\n  template: apex\ncampaign:\n  comparison: apex_vs_codex\nworkspace_directory_prefix: %s/workspace-apex\n' \
    "$CAMPAIGN_DATA_ROOT" > "$CAMPAIGN_APEX_AGENT_CONFIG"
mapfile -t apex_args < <(run_check_args \
    "$CODEX_HOME" \
    "$CAMPAIGN_APEX_AGENT_CONFIG" \
    AKA_APEX_ROOT="$CAMPAIGN_APEX_ROOT" \
    AKA_NODE_PREFIX="$CODEX_PREFIX")
apex_snapshot_arg="$(
    find_arg_with_prefix "AGENT_KERNEL_ARENA_APEX_RUNTIME_SNAPSHOT_ROOT=" "${apex_args[@]}"
)" || fail "formal Apex arm did not receive its immutable runtime root"
apex_snapshot="${apex_snapshot_arg#*=}"
[[ "$apex_snapshot" == /tmp/agentkernelarena-formal-runtime.*/*/* \
    && "$(basename "$apex_snapshot")" =~ ^[0-9a-f]{64}$ ]] \
    || fail "formal Apex runtime root is not digest-addressed: $apex_snapshot"
assert_has "$apex_snapshot:$apex_snapshot:ro" "${apex_args[@]}"
assert_has "APEX_ROOT=$apex_snapshot/repo" "${apex_args[@]}"
assert_has "APEX_PYTHON=$apex_snapshot/sealed-bin/python" "${apex_args[@]}"
assert_has "AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT=$CAMPAIGN_APEX_ROOT" "${apex_args[@]}"
assert_not_has "$CAMPAIGN_APEX_ROOT:$CAMPAIGN_APEX_ROOT:ro" "${apex_args[@]}"
apex_image_arg="$(
    find_arg_with_prefix "AGENT_KERNEL_ARENA_APEX_RUNTIME_IMAGE_SHA256=" "${apex_args[@]}"
)" || fail "formal Apex arm did not bind its immutable image"
[[ "${apex_image_arg#*=}" =~ ^[0-9a-f]{64}$ ]] \
    || fail "formal Apex immutable image digest is invalid"

# A natively installed Claude CLI is a launcher in ~/.local/bin that resolves
# into ~/.local/share/claude/versions. Both sides of that symlink must be
# mounted at the same absolute paths inside the container.
NATIVE_CLAUDE_HOME="$TEST_HOME/native-claude-home"
NATIVE_CLAUDE_CONFIG="$TEST_HOME/native-claude-config.yaml"
mkdir -p \
    "$NATIVE_CLAUDE_HOME/.local/bin" \
    "$NATIVE_CLAUDE_HOME/.local/share/claude/versions" \
    "$NATIVE_CLAUDE_HOME/.claude"
touch \
    "$NATIVE_CLAUDE_HOME/.local/share/claude/versions/2.1.0" \
    "$NATIVE_CLAUDE_HOME/.claude.json"
ln -s ../share/claude/versions/2.1.0 "$NATIVE_CLAUDE_HOME/.local/bin/claude"
printf 'agent:\n  template: claude_code\n' > "$NATIVE_CLAUDE_CONFIG"

mapfile -t args < <(run_check_args \
    "$NATIVE_CLAUDE_HOME" \
    "$NATIVE_CLAUDE_CONFIG" \
    ANTHROPIC_AUTH_TOKEN=claude-must-not-receive-this \
    GEAK_V4_WORKFLOW_DIR="$UNRELATED_GEAK_WORKFLOW_DIR")
assert_has "$NATIVE_CLAUDE_HOME/.local/bin:$NATIVE_CLAUDE_HOME/.local/bin:ro" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.local/share/claude:$NATIVE_CLAUDE_HOME/.local/share/claude:ro" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.claude:$NATIVE_CLAUDE_HOME/.claude" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.claude.json:$NATIVE_CLAUDE_HOME/.claude.json" "${args[@]}"
assert_has "claude_code" "${args[@]}"
assert_not_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"
assert_not_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"
assert_not_has "$UNRELATED_GEAK_WORKFLOW_DIR:$UNRELATED_GEAK_WORKFLOW_DIR:ro" "${args[@]}"
assert_not_has "GEAK_V4_WORKFLOW_DIR=$UNRELATED_GEAK_WORKFLOW_DIR" "${args[@]}"

# Omitting --config_name uses the one-task MI300/MI300X Claude quickstart.
mapfile -t args < <(
    env \
        HOME="$NATIVE_CLAUDE_HOME" \
        AKA_GPU_ARCH=gfx942 \
        bash "$RUNNER" check-agents 2>/dev/null
)
assert_has "claude_code" "${args[@]}"
assert_not_has "cursor" "${args[@]}"

# An npm-installed Claude CLI is mounted from its Node prefix; the native
# ~/.local/share/claude layout is not required.
CLAUDE_HOME="$TEST_HOME/claude-home"
CLAUDE_PREFIX="$TEST_HOME/claude-node"
CLAUDE_CONFIG="$TEST_HOME/claude-config.yaml"
mkdir -p "$CLAUDE_HOME/.claude" "$CLAUDE_PREFIX/bin"
touch \
    "$CLAUDE_HOME/.claude.json" \
    "$CLAUDE_PREFIX/bin/node" \
    "$CLAUDE_PREFIX/bin/claude"
printf 'agent:\n  template: claude_code\n' > "$CLAUDE_CONFIG"

mapfile -t args < <(run_check_args \
    "$CLAUDE_HOME" \
    "$CLAUDE_CONFIG" \
    AKA_NODE_PREFIX="$CLAUDE_PREFIX")
assert_has "$CLAUDE_PREFIX:/opt/claude-node:ro" "${args[@]}"
assert_has "PATH=/opt/claude-node/bin:/opt/node/bin:$CLAUDE_HOME/.local/bin:/opt/venv/bin:/usr/local/bin:/opt/rocm/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin" "${args[@]}"
assert_has "$CLAUDE_HOME/.claude:$CLAUDE_HOME/.claude" "${args[@]}"
assert_has "$CLAUDE_HOME/.claude.json:$CLAUDE_HOME/.claude.json" "${args[@]}"
assert_has "_container_check_agents" "${args[@]}"
assert_has "claude_code" "${args[@]}"
assert_not_has "$CLAUDE_HOME/.local/share/claude:$CLAUDE_HOME/.local/share/claude:ro" "${args[@]}"
assert_not_has "$CLAUDE_HOME/.codex:$CLAUDE_HOME/.codex" "${args[@]}"

# AGENTS=all is an explicit override and expands to all three first-class CLIs.
ALL_HOME="$TEST_HOME/all-home"
ALL_NODE_PREFIX="$TEST_HOME/all-node"
mkdir -p \
    "$ALL_HOME/.codex" \
    "$ALL_HOME/.claude" \
    "$ALL_HOME/.local/bin" \
    "$ALL_HOME/.local/share/cursor-agent" \
    "$ALL_HOME/.cursor" \
    "$ALL_HOME/.config/cursor" \
    "$ALL_NODE_PREFIX/bin"
touch \
    "$ALL_HOME/.claude.json" \
    "$ALL_NODE_PREFIX/bin/node" \
    "$ALL_NODE_PREFIX/bin/codex" \
    "$ALL_NODE_PREFIX/bin/claude"

mapfile -t args < <(run_check_args \
    "$ALL_HOME" \
    "$TEST_HOME/not-needed.yaml" \
    AKA_NODE_PREFIX="$ALL_NODE_PREFIX" \
    AKA_AGENTS=all)
assert_has "$ALL_NODE_PREFIX:/opt/node:ro" "${args[@]}"
assert_has "$ALL_NODE_PREFIX:/opt/claude-node:ro" "${args[@]}"
assert_has "$ALL_HOME/.local/share/cursor-agent:$ALL_HOME/.local/share/cursor-agent:ro" "${args[@]}"
assert_has "codex" "${args[@]}"
assert_has "claude_code" "${args[@]}"
assert_has "cursor" "${args[@]}"
assert_not_has "all" "${args[@]}"

# A geak_v4 config provisions the Claude Code CLI/auth and, when
# GEAK_V4_WORKFLOW_DIR is exported, mounts that checkout and forwards the var.
GEAK_HOME="$TEST_HOME/geak-home"
GEAK_PREFIX="$TEST_HOME/geak-node"
GEAK_CONFIG="$TEST_HOME/geak-config.yaml"
GEAK_WORKFLOW_DIR="$TEST_HOME/geak-checkout/kernel_workflow"
mkdir -p "$GEAK_HOME/.claude" "$GEAK_PREFIX/bin" "$GEAK_WORKFLOW_DIR"
touch \
    "$GEAK_HOME/.claude.json" \
    "$GEAK_PREFIX/bin/node" \
    "$GEAK_PREFIX/bin/claude" \
    "$GEAK_WORKFLOW_DIR/kernel_workflow.js"
printf 'agent:\n  template: geak_v4\n' > "$GEAK_CONFIG"

mapfile -t args < <(run_check_args \
    "$GEAK_HOME" \
    "$GEAK_CONFIG" \
    AKA_NODE_PREFIX="$GEAK_PREFIX" \
    GEAK_V4_WORKFLOW_DIR="$GEAK_WORKFLOW_DIR")
assert_has "$GEAK_PREFIX:/opt/claude-node:ro" "${args[@]}"
assert_has "$GEAK_HOME/.claude:$GEAK_HOME/.claude" "${args[@]}"
assert_has "$GEAK_HOME/.claude.json:$GEAK_HOME/.claude.json" "${args[@]}"
assert_has "claude_code" "${args[@]}"
assert_has "$GEAK_WORKFLOW_DIR:$GEAK_WORKFLOW_DIR:ro" "${args[@]}"
assert_has "GEAK_V4_WORKFLOW_DIR=$GEAK_WORKFLOW_DIR" "${args[@]}"
# The Claude Agent SDK is installed with `pip install --target` into the mounted
# user-base (setup-geak); its dir must be forwarded on PYTHONPATH so the venv
# python in the standard sglang images can import it.
assert_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"

# The explicit setup command has no run config or required agent CLI, but still
# needs the GEAK-only dependency path and workflow mount for its container check.
mapfile -t args < <(
    env \
        HOME="$TEST_HOME" \
        AKA_GPU_ARCH=gfx950 \
        GEAK_V4_WORKFLOW_DIR="$GEAK_WORKFLOW_DIR" \
        ANTHROPIC_AUTH_TOKEN=setup-must-not-receive-this \
        bash "$RUNNER" setup-geak 2>/dev/null
)
assert_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"
assert_has "$GEAK_WORKFLOW_DIR:$GEAK_WORKFLOW_DIR:ro" "${args[@]}"
assert_has "GEAK_V4_WORKFLOW_DIR=$GEAK_WORKFLOW_DIR" "${args[@]}"
assert_has "_container_setup_geak" "${args[@]}"
assert_not_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"

# Without GEAK_V4_WORKFLOW_DIR the checkout mount/env are absent.
mapfile -t args < <(run_check_args \
    "$GEAK_HOME" \
    "$GEAK_CONFIG" \
    AKA_NODE_PREFIX="$GEAK_PREFIX")
assert_has "claude_code" "${args[@]}"
assert_not_has "GEAK_V4_WORKFLOW_DIR=$GEAK_WORKFLOW_DIR" "${args[@]}"
assert_has "$GEAK_SDK_PYTHONPATH" "${args[@]}"

# The host's Claude gateway credentials are forwarded by name (value stays out of
# argv). These hosts use the AMD Core42 / Primus-safe gateway, where the credential
# is an ANTHROPIC_AUTH_TOKEN paired with an ANTHROPIC_BASE_URL.
mapfile -t args < <(run_check_args \
    "$GEAK_HOME" \
    "$GEAK_CONFIG" \
    AKA_NODE_PREFIX="$GEAK_PREFIX" \
    ANTHROPIC_AUTH_TOKEN=dummy-token-value \
    ANTHROPIC_BASE_URL=https://gateway.example/api)
assert_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"
assert_not_has "ANTHROPIC_AUTH_TOKEN=dummy-token-value" "${args[@]}"
assert_has "ANTHROPIC_BASE_URL" "${args[@]}"

# A plain ANTHROPIC_API_KEY (e.g. api.anthropic.com auth) is likewise forwarded.
mapfile -t args < <(env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL \
    HOME="$GEAK_HOME" AKA_GPU_ARCH=gfx950 AKA_NODE_PREFIX="$GEAK_PREFIX" \
    ANTHROPIC_API_KEY=dummy-key-value \
    bash "$RUNNER" check-agents --config_name "$GEAK_CONFIG" 2>/dev/null)
assert_has "ANTHROPIC_API_KEY" "${args[@]}"
assert_not_has "ANTHROPIC_API_KEY=dummy-key-value" "${args[@]}"

# When no Claude credentials are present on the host, none are forwarded.
mapfile -t args < <(env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL \
    HOME="$GEAK_HOME" AKA_GPU_ARCH=gfx950 AKA_NODE_PREFIX="$GEAK_PREFIX" \
    bash "$RUNNER" check-agents --config_name "$GEAK_CONFIG" 2>/dev/null)
assert_not_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"
assert_not_has "ANTHROPIC_API_KEY" "${args[@]}"

echo "PASS: docker_benchmark runtime and agent-selection argument tests"

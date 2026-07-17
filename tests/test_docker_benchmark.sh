#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/src/scripts/docker_benchmark.sh"
cd "$ROOT"
PINNED_GFX950_IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705"
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

# Capture the exact argv that the runner would pass to Docker without requiring
# a daemon, GPU devices, or the benchmark images on this host.
docker() {
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
trap 'rm -rf "$TEST_HOME"' EXIT

bash -n "$RUNNER"

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
assert_cache_args_present "" "${args[@]}"

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

mapfile -t args < <(run_check_args "$CURSOR_HOME" "$CURSOR_CONFIG")
assert_has "$CURSOR_HOME/.local/share/cursor-agent:$CURSOR_HOME/.local/share/cursor-agent:ro" "${args[@]}"
assert_has "$CURSOR_HOME/.cursor:$CURSOR_HOME/.cursor" "${args[@]}"
assert_has "$CURSOR_HOME/.config/cursor:$CURSOR_HOME/.config/cursor" "${args[@]}"
assert_has "_container_check_agents" "${args[@]}"
assert_has "cursor" "${args[@]}"
assert_not_has "$CURSOR_HOME/.claude:$CURSOR_HOME/.claude" "${args[@]}"
assert_not_has "$CURSOR_HOME/.codex:$CURSOR_HOME/.codex" "${args[@]}"

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

mapfile -t args < <(run_check_args "$NATIVE_CLAUDE_HOME" "$NATIVE_CLAUDE_CONFIG")
assert_has "$NATIVE_CLAUDE_HOME/.local/bin:$NATIVE_CLAUDE_HOME/.local/bin:ro" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.local/share/claude:$NATIVE_CLAUDE_HOME/.local/share/claude:ro" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.claude:$NATIVE_CLAUDE_HOME/.claude" "${args[@]}"
assert_has "$NATIVE_CLAUDE_HOME/.claude.json:$NATIVE_CLAUDE_HOME/.claude.json" "${args[@]}"
assert_has "claude_code" "${args[@]}"

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

echo "PASS: docker_benchmark runtime and agent-selection argument tests"

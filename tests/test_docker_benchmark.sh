#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/src/scripts/docker_benchmark.sh"
cd "$ROOT"
PINNED_GFX950_IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705"
PINNED_GFX950_IMMUTABLE_IMAGE="lmsysorg/sglang-rocm@sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78"
export PINNED_GFX950_IMMUTABLE_IMAGE
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

assert_before() {
    local first="$1" second="$2"
    shift 2
    local value index=0 first_index="" second_index=""
    for value in "$@"; do
        [[ "$value" != "$first" || -n "$first_index" ]] || first_index="$index"
        [[ "$value" != "$second" || -n "$second_index" ]] || second_index="$index"
        index=$((index + 1))
    done
    [[ -n "$first_index" && -n "$second_index" && "$first_index" -lt "$second_index" ]] \
        || fail "Docker argument '$first' must precede '$second'"
}

# Capture the exact argv that the runner would pass to Docker without requiring
# a daemon, GPU devices, or the benchmark images on this host.
docker() {
    if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
        local reference="${!#}"
        if [[ "$reference" == "$PINNED_GFX950_IMMUTABLE_IMAGE" ]]; then
            printf '%s\n' "${FAKE_PINNED_IMAGE_ID:-sha256:pinned-image-id}"
        else
            printf '%s\n' "${FAKE_SELECTED_IMAGE_ID:-sha256:pinned-image-id}"
        fi
        return 0
    fi
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
PATH_TEST_PARENT="$ROOT/.eval-tool-runner-test-$$"
trap 'rm -rf "$TEST_HOME" "$PATH_TEST_PARENT"' EXIT
UNRELATED_GEAK_WORKFLOW_DIR="$TEST_HOME/unrelated-geak-workflow"
GEAK_SDK_PYTHONPATH="PYTHONPATH=/workspace/.aka-pyuserbase/geak-sdk"
mkdir -p "$UNRELATED_GEAK_WORKFLOW_DIR"
touch "$UNRELATED_GEAK_WORKFLOW_DIR/kernel_workflow.js"

bash -n "$RUNNER"

# Sanitizer sidecars accept the scoring runtime only when the selected reference
# and the known manifest reference resolve to the same immutable local config ID,
# and launch that ID rather than a mutable tag. An alias to identical content is
# valid; a retagged image is rejected.
mapfile -t verified_image < <(
    FAKE_SELECTED_IMAGE_ID=sha256:verified-config \
    FAKE_PINNED_IMAGE_ID=sha256:verified-config \
    bash "$RUNNER" _verify_eval_tool_scoring_image gfx950 "$PINNED_GFX950_IMAGE"
)
[[ "${verified_image[0]}" == "sha256:verified-config" ]] \
    || fail "verified scoring launch did not freeze the immutable image ID"
[[ "${verified_image[1]}" == "sha256:verified-config" ]] \
    || fail "verified scoring image ID was not exported"
[[ "${verified_image[2]}" == "$PINNED_GFX950_IMAGE" ]] \
    || fail "human-readable scoring image reference was not preserved"

mapfile -t verified_alias < <(
    FAKE_SELECTED_IMAGE_ID=sha256:verified-config \
    FAKE_PINNED_IMAGE_ID=sha256:verified-config \
    bash "$RUNNER" _verify_eval_tool_scoring_image gfx950 example.invalid/scoring:alias
)
[[ "${verified_alias[0]}" == "sha256:verified-config" ]] \
    || fail "byte-identical scoring image alias was not frozen by ID"

if FAKE_SELECTED_IMAGE_ID=sha256:retagged \
    FAKE_PINNED_IMAGE_ID=sha256:verified-config \
    bash "$RUNNER" _verify_eval_tool_scoring_image \
        gfx950 "$PINNED_GFX950_IMAGE" >/dev/null 2>&1; then
    fail "retagged scoring image unexpectedly passed immutable verification"
fi

# Artifact bind sources must be physical repository directories. Reject both a
# symlink in the parent and symlinks pre-created at namespace/worker boundaries.
mkdir -p "$PATH_TEST_PARENT/good"
prepared_path="$(
    bash "$RUNNER" _prepare_eval_tool_artifact_dir \
        "$PATH_TEST_PARENT/good" worker-safe
)"
[[ "$prepared_path" == "$PATH_TEST_PARENT/good/.eval-tool-artifacts/worker-safe" ]] \
    || fail "artifact directory was not physically canonicalized"

ln -s "$PATH_TEST_PARENT/good" "$PATH_TEST_PARENT/parent-link"
if bash "$RUNNER" _prepare_eval_tool_artifact_dir \
    "$PATH_TEST_PARENT/parent-link" worker >/dev/null 2>&1; then
    fail "symlinked artifact parent unexpectedly passed containment"
fi

mkdir -p "$PATH_TEST_PARENT/namespace-link" "$TEST_HOME/outside-namespace"
ln -s "$TEST_HOME/outside-namespace" \
    "$PATH_TEST_PARENT/namespace-link/.eval-tool-artifacts"
if bash "$RUNNER" _prepare_eval_tool_artifact_dir \
    "$PATH_TEST_PARENT/namespace-link" worker >/dev/null 2>&1; then
    fail "symlinked artifact namespace unexpectedly passed containment"
fi

mkdir -p "$PATH_TEST_PARENT/worker-link/.eval-tool-artifacts" "$TEST_HOME/outside-worker"
ln -s "$TEST_HOME/outside-worker" \
    "$PATH_TEST_PARENT/worker-link/.eval-tool-artifacts/worker"
if bash "$RUNNER" _prepare_eval_tool_artifact_dir \
    "$PATH_TEST_PARENT/worker-link" worker >/dev/null 2>&1; then
    fail "symlinked worker artifact directory unexpectedly passed containment"
fi

if bash "$RUNNER" _prepare_eval_tool_artifact_dir \
    "$PATH_TEST_PARENT/good" .. >/dev/null 2>&1; then
    fail "dot-dot artifact label unexpectedly passed validation"
fi

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

# task_validator may override its default backend in the run config. Provision
# that selected CLI rather than the backend from agent_config.yaml.
VALIDATOR_CODEX_CONFIG="$TEST_HOME/validator-codex-config.yaml"
printf 'agent:\n  template: task_validator\n  backend: codex\ntasks:\n  - backend: ignored\n' > "$VALIDATOR_CODEX_CONFIG"
mapfile -t args < <(run_check_args \
    "$CODEX_HOME" \
    "$VALIDATOR_CODEX_CONFIG" \
    AKA_NODE_PREFIX="$CODEX_PREFIX")
assert_has "codex" "${args[@]}"
assert_not_has "claude_code" "${args[@]}"

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

# Evaluation-tool sidecars have a deliberately narrower security boundary than
# the scoring container: no network, no credentials, no privileged mode, and
# only immutable input plus explicit scratch/artifact/socket writes.
EVAL_FRAMEWORK_ROOT="/opt/aka-eval-tools"
for eval_image in triton-fpsan gpu-asan rocjitsu hip-fpsan; do
    eval_dockerfile="$ROOT/docker/eval-tools/$eval_image/Dockerfile"
    grep -Fq "COPY src/eval_tools $EVAL_FRAMEWORK_ROOT/src/eval_tools" "$eval_dockerfile" \
        || fail "$eval_dockerfile does not bake the evaluation framework"
    grep -Fq "AKA_EVAL_TOOL_FRAMEWORK_ROOT=$EVAL_FRAMEWORK_ROOT" "$eval_dockerfile" \
        || fail "$eval_dockerfile does not declare its image-owned framework root"
    grep -Fq "PYTHONPATH=$EVAL_FRAMEWORK_ROOT" "$eval_dockerfile" \
        || fail "$eval_dockerfile does not default imports to its baked framework"
done
framework_record="$(
    env -u AKA_EVAL_TOOL_FRAMEWORK_ROOT PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path

from src.eval_tools.worker import _framework_provenance

root, source, module = _framework_provenance(Path.cwd())
print(f"{root}|{source}|{module}")
PY
)"
[[ "$framework_record" == "$ROOT|local_module_fallback|$ROOT/src/eval_tools/worker.py" ]] \
    || fail "unexpected local worker framework provenance: $framework_record"
AKA_EVAL_TOOL_FRAMEWORK_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
from pathlib import Path

from src.eval_tools.worker import _framework_provenance

try:
    _framework_provenance(Path.cwd())
except RuntimeError as error:
    assert "must resolve to image-owned" in str(error)
else:
    raise AssertionError("worker accepted a checkout as its configured framework root")
PY
EVAL_SOCKET_PARENT="$TEST_HOME/eval-sockets"
EVAL_SOCKET_DIR="$EVAL_SOCKET_PARENT/gpu_asan"
EVAL_SCRATCH_DIR="$TEST_HOME/eval-scratch"
EVAL_ARTIFACT_DIR="$TEST_HOME/eval-artifacts"
mkdir -p "$EVAL_SOCKET_DIR" "$EVAL_SCRATCH_DIR" "$EVAL_ARTIFACT_DIR"
mapfile -t args < <(
    HOME="$TEST_HOME" bash "$RUNNER" _print_eval_tool_docker_args \
        gpu_asan sha256:0123456789abcdef aka-eval-test \
        "$EVAL_SOCKET_DIR" "$EVAL_SCRATCH_DIR" "$EVAL_ARTIFACT_DIR" \
        sha256:0123456789abcdef
)
assert_has "--network=none" "${args[@]}"
assert_has "--cap-drop=ALL" "${args[@]}"
assert_has "--security-opt=no-new-privileges" "${args[@]}"
assert_has "--read-only" "${args[@]}"
assert_has "$ROOT:/input:ro" "${args[@]}"
assert_has "PYTHONPATH=$EVAL_FRAMEWORK_ROOT" "${args[@]}"
assert_not_has "PYTHONPATH=$EVAL_FRAMEWORK_ROOT:/input" "${args[@]}"
assert_not_has "PYTHONPATH=/input" "${args[@]}"
assert_has "AKA_EVAL_TOOL_FRAMEWORK_ROOT=$EVAL_FRAMEWORK_ROOT" "${args[@]}"
assert_has "$EVAL_SCRATCH_DIR:/work:rw" "${args[@]}"
assert_has "$EVAL_ARTIFACT_DIR:/artifacts:rw" "${args[@]}"
assert_has "$EVAL_SOCKET_DIR:/run/aka-eval-tools:rw" "${args[@]}"
assert_not_has "$EVAL_SOCKET_PARENT:/run/aka-eval-tools:rw" "${args[@]}"
assert_has "sha256:0123456789abcdef" "${args[@]}"
assert_not_has "example.invalid/gpu-asan:test" "${args[@]}"
assert_has "src.eval_tools.worker" "${args[@]}"
assert_has "AKA_EVAL_TOOL_RUNTIME_REF=sha256:0123456789abcdef" "${args[@]}"
assert_not_has "--privileged" "${args[@]}"
assert_not_has "--network=host" "${args[@]}"
assert_not_has "--cap-add=SYS_ADMIN" "${args[@]}"
assert_not_has "--cap-add=SYS_PTRACE" "${args[@]}"
assert_not_has "ANTHROPIC_AUTH_TOKEN" "${args[@]}"
assert_not_has "ANTHROPIC_API_KEY" "${args[@]}"
assert_not_has "$EVAL_SOCKET_DIR/rocjitsu.passwd:/etc/passwd:ro" "${args[@]}"

# rocJITsu remains non-root. ROCr crashes if the numeric container UID is not
# resolvable, so only this sidecar receives an image-derived passwd override.
EVAL_ROCJITSU_SOCKET_DIR="$EVAL_SOCKET_PARENT/rocjitsu"
EVAL_ROCJITSU_PASSWD="$EVAL_SOCKET_PARENT/rocjitsu.passwd"
mkdir -p "$EVAL_ROCJITSU_SOCKET_DIR"
printf 'root:x:0:0:root:/root:/bin/bash\naka-eval:x:%s:%s::/work/home:/usr/sbin/nologin\n' \
    "$(id -u)" "$(id -g)" > "$EVAL_ROCJITSU_PASSWD"
mapfile -t args < <(
    HOME="$TEST_HOME" bash "$RUNNER" _print_eval_tool_docker_args \
        rocjitsu sha256:fedcba9876543210 aka-eval-rocjitsu-test \
        "$EVAL_ROCJITSU_SOCKET_DIR" "$EVAL_SCRATCH_DIR" "$EVAL_ARTIFACT_DIR" \
        sha256:fedcba9876543210 "$EVAL_ROCJITSU_PASSWD"
)
assert_has "--user" "${args[@]}"
assert_has "$(id -u):$(id -g)" "${args[@]}"
assert_not_has "0:0" "${args[@]}"
assert_has "$EVAL_ROCJITSU_PASSWD:/etc/passwd:ro" "${args[@]}"
assert_has "$EVAL_ROCJITSU_SOCKET_DIR:/run/aka-eval-tools:rw" "${args[@]}"
assert_not_has "$EVAL_SOCKET_PARENT:/run/aka-eval-tools:rw" "${args[@]}"
assert_has "--cap-drop=ALL" "${args[@]}"
assert_has "--security-opt=no-new-privileges" "${args[@]}"

if HOME="$TEST_HOME" bash "$RUNNER" _print_eval_tool_docker_args \
    gpu_asan example.invalid/gpu-asan:mutable aka-eval-mutable \
    "$EVAL_SOCKET_DIR" "$EVAL_SCRATCH_DIR" "$EVAL_ARTIFACT_DIR" \
    sha256:0123456789abcdef >/dev/null 2>&1; then
    fail "mutable sidecar image reference unexpectedly passed launch validation"
fi

# The unchanged scoring container gets the read-only socket mount plus one
# dedicated per-worker artifact mount. The namespace parent shadows the broad
# repository mount read-only, preventing access to sibling worker reports.
EVAL_SCORING_ARTIFACT_NAMESPACE="$TEST_HOME/.eval-tool-artifacts"
EVAL_SCORING_ARTIFACT_DIR="$EVAL_SCORING_ARTIFACT_NAMESPACE/test-worker"
EVAL_QUALITY_ARTIFACT_DIR="$EVAL_SCORING_ARTIFACT_NAMESPACE/quality-test"
mkdir -p "$EVAL_SCORING_ARTIFACT_DIR" "$EVAL_QUALITY_ARTIFACT_DIR"
mapfile -t args < <(run_shell_args \
    AKA_GPU_ARCH=gfx950 \
    AKA_EVAL_TOOL_SOCKET_HOST_DIR="$EVAL_SOCKET_PARENT" \
    AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT="$EVAL_SCORING_ARTIFACT_DIR" \
    AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT=/workspace/.eval-tool-artifacts/test-worker \
    AKA_EVAL_TOOLS_SELECTED=gpu_asan \
    AKA_SCORING_IMAGE_RUNTIME_REF=sha256:scoring0123456789abcdef \
    AKA_SCORING_IMAGE_REFERENCE="$PINNED_GFX950_IMAGE" \
    AKA_EVAL_TOOL_RUNTIME_REF_GPU_ASAN=sha256:0123456789abcdef)
assert_has "$EVAL_SOCKET_PARENT:/run/aka-eval-tools:ro" "${args[@]}"
assert_has "$EVAL_SCORING_ARTIFACT_NAMESPACE:/workspace/.eval-tool-artifacts:ro" "${args[@]}"
assert_has "$EVAL_SCORING_ARTIFACT_DIR:/workspace/.eval-tool-artifacts/test-worker" "${args[@]}"
assert_not_has "$EVAL_SCORING_ARTIFACT_NAMESPACE:/workspace/.eval-tool-artifacts:rw" "${args[@]}"
assert_before \
    "$EVAL_SCORING_ARTIFACT_NAMESPACE:/workspace/.eval-tool-artifacts:ro" \
    "$EVAL_SCORING_ARTIFACT_DIR:/workspace/.eval-tool-artifacts/test-worker" \
    "${args[@]}"
assert_has "AKA_EVAL_TOOL_SOCKET_DIR=/run/aka-eval-tools" "${args[@]}"
assert_has "AKA_EVAL_TOOL_SCORING_ROOT=/workspace" "${args[@]}"
assert_has "AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT=/workspace/.eval-tool-artifacts/test-worker" "${args[@]}"
assert_has "AKA_SCORING_IMAGE_RUNTIME_REF=sha256:scoring0123456789abcdef" "${args[@]}"
assert_has "AKA_SCORING_IMAGE_REFERENCE=$PINNED_GFX950_IMAGE" "${args[@]}"
assert_has "AKA_EVAL_TOOLS_SELECTED=gpu_asan" "${args[@]}"
assert_has "AKA_EVAL_TOOL_RUNTIME_REF_GPU_ASAN=sha256:0123456789abcdef" "${args[@]}"

# The dedicated scoring path is overrideable without broadening sidecar mounts.
mapfile -t args < <(run_shell_args \
    AKA_GPU_ARCH=gfx950 \
    AKA_EVAL_TOOL_SOCKET_HOST_DIR="$EVAL_SOCKET_PARENT" \
    AKA_EVAL_TOOL_ARTIFACT_HOST_ROOT="$EVAL_QUALITY_ARTIFACT_DIR" \
    AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT=/workspace/.eval-tool-artifacts/quality-test \
    AKA_SCORING_IMAGE_RUNTIME_REF=sha256:scoring0123456789abcdef \
    AKA_SCORING_IMAGE_REFERENCE="$PINNED_GFX950_IMAGE")
assert_has \
    "AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT=/workspace/.eval-tool-artifacts/quality-test" \
    "${args[@]}"
assert_has "$EVAL_SCORING_ARTIFACT_NAMESPACE:/workspace/.eval-tool-artifacts:ro" "${args[@]}"
assert_has "$EVAL_QUALITY_ARTIFACT_DIR:/workspace/.eval-tool-artifacts/quality-test" "${args[@]}"

echo "PASS: docker_benchmark runtime, agent-selection, and eval-tool isolation tests"

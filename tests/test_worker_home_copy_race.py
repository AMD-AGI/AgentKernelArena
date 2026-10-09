"""The Docker worker must not publish a partially copied agent home."""

import os
from pathlib import Path
import subprocess


RUNNER = Path(__file__).resolve().parents[1] / "src/scripts/docker_benchmark.sh"


def _run_copy(tmp_path, mode):
    state = tmp_path / "mounted-state"
    codex = state / ".codex"
    codex.mkdir(parents=True)
    (codex / "auth.json").write_text("synthetic-auth")
    (codex / "cloud-config-bundle-cache.json").write_text("old-cache")
    (codex / "packages").mkdir()
    (codex / "packages" / "large-package").write_text("do-not-copy")
    (state / ".claude").mkdir()
    (state / ".claude" / "settings.json").write_text("synthetic-claude")
    (state / ".claude.json").write_text("synthetic-claude-root")
    (state / ".cursor").mkdir()
    (state / ".cursor" / "settings.json").write_text("synthetic-cursor")
    (state / ".config" / "cursor").mkdir(parents=True)
    (state / ".config" / "cursor" / "settings.json").write_text("synthetic-cursor-config")

    home = tmp_path / "worker-home"
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    attempts = tmp_path / "cloud-copy-attempts"
    attempts.write_text("0")
    shim = shim_dir / "cp"
    shim.write_text("""#!/usr/bin/env bash
set -euo pipefail
source_path="${2:?}"
destination="${3:?}"
if [[ "$source_path" == "$COPY_TEST_CODEX/cloud-config-bundle-cache.json" ]]; then
    count="$(cat "$COPY_TEST_ATTEMPTS")"
    count="$((count + 1))"
    printf '%s' "$count" > "$COPY_TEST_ATTEMPTS"
    [[ ! -e "$COPY_TEST_HOME/.codex" ]] || exit 91
    if [[ "$COPY_TEST_MODE" == always || "$count" == 1 ]]; then
        # This marker models files cp may already have put in the private
        # staging tree. A failed copy must never publish it as live state.
        printf partial > "$destination/partial-marker"
        if [[ "$COPY_TEST_MODE" == replace && "$count" == 1 ]]; then
            printf refreshed-cache > "$source_path.next"
            mv "$source_path.next" "$source_path"
        fi
        exit 42
    fi
fi
exec /bin/cp "$@"
""")
    shim.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "AKA_AGENT_STATE_MOUNT_ROOT": str(state),
           "PATH": str(shim_dir) + os.pathsep + os.environ["PATH"],
           "COPY_TEST_CODEX": str(codex), "COPY_TEST_ATTEMPTS": str(attempts),
           "COPY_TEST_HOME": str(home), "COPY_TEST_MODE": mode}
    result = subprocess.run(["bash", str(RUNNER), "_container_prepare_worker_home"],
                            env=env, capture_output=True, text=True, timeout=30)
    return result, home, attempts


def test_replaced_source_retries_before_publishing_complete_worker_state(tmp_path):
    result, home, attempts = _run_copy(tmp_path, "replace")
    assert result.returncode == 0, result.stderr
    assert attempts.read_text() == "2"
    assert (home / ".codex" / "auth.json").read_text() == "synthetic-auth"
    assert (home / ".codex" / "cloud-config-bundle-cache.json").read_text() == "refreshed-cache"
    assert not (home / ".codex" / "partial-marker").exists()
    assert not (home / ".codex" / "packages").exists()
    assert (home / ".claude" / "settings.json").read_text() == "synthetic-claude"
    assert (home / ".claude.json").read_text() == "synthetic-claude-root"
    assert (home / ".cursor" / "settings.json").read_text() == "synthetic-cursor"
    assert (home / ".config" / "cursor" / "settings.json").read_text() == "synthetic-cursor-config"
    assert not list(home.glob(".aka-agent-state-stage.*"))
    assert "retrying (1/4)" in result.stderr
    assert "synthetic-auth" not in result.stdout + result.stderr


def test_exhausted_copy_failure_leaves_no_live_or_staged_worker_state(tmp_path):
    result, home, attempts = _run_copy(tmp_path, "always")
    assert result.returncode != 0
    assert attempts.read_text() == "4"
    assert not (home / ".codex").exists()
    assert not (home / ".claude").exists()
    assert not list(home.glob(".aka-agent-state-stage.*"))
    assert "failed after four attempts" in result.stderr
    assert "synthetic-auth" not in result.stdout + result.stderr

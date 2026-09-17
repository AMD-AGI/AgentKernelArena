"""Exercise the role runner using local fake CLI processes; no paid inference."""
import json
import logging
import os
from pathlib import Path
import sys

import pytest

from agents.quality_loop.backend import CodexBackend
from agents.quality_loop.config import BackendConfig


def fake_cli(tmp_path, monkeypatch, body):
    binary = tmp_path / 'codex'
    binary.write_text(f'#!{sys.executable}\n{body}\n')
    binary.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    return workspace


def test_role_argv_and_environment_do_not_inherit_github_access(tmp_path, monkeypatch):
    workspace = fake_cli(tmp_path, monkeypatch, '''
import json, os, sys
from pathlib import Path
Path('call.json').write_text(json.dumps({'argv':sys.argv[1:], 'gh_present':'GH_TOKEN' in os.environ,
                                       'ssh_present':'SSH_AUTH_SOCK' in os.environ,
                                       'stdin':sys.stdin.read(),
                                       'gh_config':os.environ['GH_CONFIG_DIR']}))
print(json.dumps({'type':'turn.completed'}))
''')
    monkeypatch.setenv('GH_TOKEN', 'nonsecret-test-fixture')
    monkeypatch.setenv('SSH_AUTH_SOCK', 'nonsecret-test-fixture')
    backend = CodexBackend(BackendConfig(), logging.getLogger(__name__))
    backend.run('-literal prompt', workspace, role='optimizer')
    result = json.loads((workspace/'call.json').read_text())
    assert result['argv'][-2:] == ['--', '-']
    assert result['stdin'] == '-literal prompt'
    assert result['argv'][result['argv'].index('--model')+1] == 'gpt-5.6-terra'
    assert 'model_reasoning_effort="medium"' in result['argv']
    assert '--ephemeral' in result['argv']
    assert result['gh_present'] is False and result['ssh_present'] is False
    assert Path(result['gh_config']).parent == workspace


@pytest.mark.parametrize('body, message', [
    ('print("not a completed turn")', 'without a completed turn'),
    ('print(\'{"type":"turn.failed"}\')', 'failed turn'),
    ('print(\'{"type":"turn.completed"}\'); raise SystemExit(2)', 'failed \\(2\\)'),
])
def test_zero_exit_or_partial_stream_is_not_success(tmp_path, monkeypatch, body, message):
    workspace = fake_cli(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match=message):
        CodexBackend(BackendConfig(), logging.getLogger(__name__)).run('test', workspace, role='reviewer')


def test_timeout_stops_cli_and_child_process(tmp_path, monkeypatch):
    workspace = fake_cli(tmp_path, monkeypatch, '''
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
time.sleep(60)
''')
    backend = CodexBackend(BackendConfig(timeout_seconds=1), logging.getLogger(__name__))
    with pytest.raises(RuntimeError, match='timed out'):
        backend.run('test', workspace, role='repair')
    child = int((workspace/'child.pid').read_text())
    # A killed orphan may be briefly waiting for the init process to reap it.
    status = Path(f'/proc/{child}/stat')
    try:
        assert status.read_text().split()[2] == 'Z'
    except (FileNotFoundError, ProcessLookupError):
        pass

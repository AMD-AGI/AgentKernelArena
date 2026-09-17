"""Native authentication copies use synthetic tokens; never contact a provider."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["isolated", "mixed", "missing", "same_home"])
def test_native_codex_auth_is_explicit_and_isolated(tmp_path, mode):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Forge interpreter")
    script = r'''
import json, os, sys
from pathlib import Path
from agents.forge.codex_auth import install_cli_auth
from agents.forge.upstream import verify_release
from kernelforge.agent_backends.base import AgentRuntimeConfig, AgentRunSpec
from kernelforge.agent_backends.codex import CodexBackend
from kernelforge.agent_backends import codex
from kernelforge.llm import LlmGateway
root, mode = Path(sys.argv[1]), sys.argv[2]
source, isolated = root/'source', root/'sdk'
source.mkdir()
os.environ['CODEX_HOME'] = str(source)
for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL'):
    os.environ.pop(key, None)
payload = {'auth_mode':'chatgpt', 'tokens':{'access_token':'synthetic-test-token'}}
if mode != 'missing':
    (source/'auth.json').write_text(json.dumps(payload))
if mode == 'mixed':
    os.environ['OPENAI_API_KEY'] = 'synthetic-test-key'
verify_release()
original = codex._provider_overrides
if mode in ('missing', 'mixed'):
    try:
        install_cli_auth()
    except RuntimeError:
        assert codex._provider_overrides is original
    else:
        raise AssertionError('invalid configuration accepted')
else:
    install_cli_auth()
    backend = CodexBackend(runtime=AgentRuntimeConfig(provider='codex', model='test',
                          options={'home':str(source if mode == 'same_home' else isolated)}))
    if mode == 'same_home':
        try:
            backend._child_environment()
        except RuntimeError as error:
            assert 'isolated' in str(error)
        else:
            raise AssertionError('source HOME accepted')
    else:
        env = backend._child_environment()
        copied = Path(env['CODEX_HOME'])/'auth.json'
        assert json.loads(copied.read_text()) == payload
        assert copied.stat().st_mode & 0o777 == 0o600
        copied.write_text('refreshed by SDK')
        backend._child_environment()
        assert copied.read_text() == 'refreshed by SDK'
        assert json.loads((source/'auth.json').read_text()) == payload
        assert codex._provider_overrides(LlmGateway()) == ['model_provider="openai"']
        options = backend._thread_start_options(codex._load_codex_sdk(),
                    AgentRunSpec(cwd=str(isolated), model='test', user_prompt='test', system_prompt='test'))
        assert options['model_provider'] == 'openai'
        try:
            codex._provider_overrides(LlmGateway(base_url='https://example.invalid', key_env='OTHER'))
        except RuntimeError:
            pass
        else:
            raise AssertionError('gateway silently redirected')
'''
    run = subprocess.run([python, "-c", script, str(tmp_path), mode], cwd=ROOT,
                         capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr

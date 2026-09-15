"""Real local subprocess transport/capture checks; no provider or GPU calls."""
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import sys

import pytest

from agents.quality_loop import backend as module
from agents.quality_loop.config import BackendConfig


def make_backend(tmp_path, monkeypatch, body, timeout=3):
    cli = tmp_path / 'codex'
    cli.write_text(f'#!{sys.executable}\n{body}\n')
    cli.chmod(0o700)
    monkeypatch.setattr(module.shutil, 'which', lambda name: str(cli))
    workspace = tmp_path / 'task'
    workspace.mkdir()
    return module.CodexBackend(BackendConfig(timeout_seconds=timeout), logging.getLogger(__name__)), workspace


def receipts(workspace):
    found = sorted(workspace.parent.glob('.quality_loop-*/process.json'))
    assert found
    assert not list(workspace.rglob('process.json'))
    return [(path, json.loads(path.read_text())) for path in found]


def assert_streams(path, status):
    for stream in status['streams'].values():
        raw = (path.parent / stream['file']).read_bytes()
        assert len(raw) == stream['retained_bytes'] <= stream['limit_bytes']
        assert stream['retained_sha256'] == hashlib.sha256(raw).hexdigest()
        assert stat.S_IMODE((path.parent / stream['file']).stat().st_mode) == 0o600
        if not stream['truncated']:
            assert stream['observed_sha256'] == hashlib.sha256(raw).hexdigest()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_large_unicode_stdin_and_raw_activity_usage_are_retained(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import hashlib, json, os, sys
from pathlib import Path
payload = sys.stdin.buffer.read()
Path('transport.json').write_text(json.dumps({'argv': sys.argv[1:], 'hash': hashlib.sha256(payload).hexdigest(),
    'stdin_link': os.readlink('/proc/self/fd/0'), 'bytes': len(payload)}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution',
    'command': 'python local_check.py', 'exit_code': 0, 'aggregated_output': 'checked'}}))
print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '完成'}}))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 120, 'cached_input_tokens': 20, 'output_tokens': 31}}))
print('diagnostic only', file=sys.stderr)
''')
    prompt = '-literal\n汉字🙂 café\n' * 120000  # Well above exec argument limits.
    monkeypatch.setenv('GH_TOKEN', 'DO_NOT_RECORD_AUTH_FIXTURE')
    monkeypatch.setenv('PRIVATE_PROCESS_ENV', 'DO_NOT_RECORD_ENV_FIXTURE')
    result = backend.run(prompt, workspace, role='optimizer')
    assert '完成' in result
    transport = json.loads((workspace / 'transport.json').read_text())
    expected_hash = hashlib.sha256(prompt.encode()).hexdigest()
    assert transport['hash'] == expected_hash
    assert transport['bytes'] == len(prompt.encode())
    assert transport['argv'][-2:] == ['--', '-']
    assert len(json.dumps(transport['argv'])) < 2000
    assert '(deleted)' in transport['stdin_link']
    assert not transport['stdin_link'].startswith(str(workspace))
    path, status = receipts(workspace)[0]
    assert status['status'] == 'succeeded' and status['returncode'] == 0
    assert status['raw_receipts_synced'] is True
    assert status['prompt'] == {'transport': 'anonymous_stdin', 'bytes': len(prompt.encode()), 'sha256': expected_hash}
    assert status['model'] == backend.config.model and status['effort'] == backend.config.effort
    assert status['events']['last_terminal_events'] == [{'type': 'turn.completed', 'usage':
        {'input_tokens': 120, 'cached_input_tokens': 20, 'output_tokens': 31}}]
    raw = (path.parent / 'stdout.log').read_text()
    assert 'command_execution' in raw and 'aggregated_output' in raw
    assert (path.parent / 'stderr.log').read_text() == 'diagnostic only\n'
    for file in path.parent.iterdir():
        assert b'DO_NOT_RECORD_' not in file.read_bytes()
    assert_streams(path, status)


def test_role_and_repeated_invocation_receipts_do_not_overwrite(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, 'print(\'{"type":"turn.completed"}\')')
    for role in ['reviewer', 'repair', 'reviewer']:
        backend.run(role, workspace, role=role)
    records = receipts(workspace)
    assert len(records) == 3
    assert sorted(s['role'] for _, s in records) == ['repair', 'reviewer', 'reviewer']
    assert all(p.parent.name.startswith(f'.quality_loop-{s["role"]}-') for p, s in records)
    assert all(s['workspace'] == str(workspace) for _, s in records)
    with pytest.raises(ValueError, match='identifier'):
        backend.run('test', workspace, role='../escape')


@pytest.mark.parametrize('events,code,match', [
    (['turn.completed'], 3, r'failed \(3\)'),
    (['turn.failed', 'turn.completed'], 0, 'failed turn'),
    (['thread.failed', 'turn.completed'], 0, 'failed turn'),
    (['error', 'turn.completed'], 0, 'failed turn'),
    (['item.completed'], 0, 'without a completed turn'),
])
def test_error_evidence_survives_all_rejected_terminal_paths(tmp_path, monkeypatch, events, code, match):
    body = f'''import json, sys
for kind in {events!r}: print(json.dumps({{'type': kind}}))
print('retained failure diagnostic', file=sys.stderr)
sys.exit({code})'''
    backend, workspace = make_backend(tmp_path, monkeypatch, body)
    with pytest.raises(RuntimeError, match=match):
        backend.run('test', workspace, role='reviewer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['returncode'] == code
    assert 'retained failure diagnostic' in (path.parent / 'stderr.log').read_text()
    assert status['streams']['stdout']['eof'] and status['streams']['stderr']['eof']
    assert_streams(path, status)


def test_timeout_keeps_partial_stdout_stderr_and_kills_descendant(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
print('{"type":"item.started"}', flush=True)
print('partial stderr', file=sys.stderr, flush=True)
time.sleep(60)
''', timeout=1)
    with pytest.raises(RuntimeError, match='timed out'):
        backend.run('test', workspace, role='repair')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'timed_out' and status['timed_out'] is True
    assert status['timeout_phase'] == 'role'
    assert status['returncode'] < 0
    assert status['events']['completed_turn'] is False
    assert b'item.started' in (path.parent / 'stdout.log').read_bytes()
    assert b'partial stderr' in (path.parent / 'stderr.log').read_bytes()
    child = int((workspace / 'child.pid').read_text())
    try:
        assert Path(f'/proc/{child}/stat').read_text().split()[2] == 'Z'
    except (FileNotFoundError, ProcessLookupError):
        pass
    assert_streams(path, status)


@pytest.mark.parametrize('terminal,accepted', [('turn.completed', True), ('error', False)])
def test_truncation_still_hashes_and_parses_late_terminal_events(tmp_path, monkeypatch, terminal, accepted):
    monkeypatch.setattr(module, 'STDOUT_LIMIT', 64)
    monkeypatch.setattr(module, 'STDERR_LIMIT', 32)
    output = b'x\n' * 90000 + json.dumps({'type': terminal, 'usage': {'input_tokens': 4}}).encode() + b'\n'
    backend, workspace = make_backend(tmp_path, monkeypatch, f'''
import json, sys
sys.stdout.buffer.write(b'x\\n' * 90000)
print(json.dumps({{'type': {terminal!r}, 'usage': {{'input_tokens': 4}}}}))
sys.stderr.write('e' * 90000)
''')
    if accepted:
        backend.run('test', workspace, role='optimizer')
    else:
        with pytest.raises(RuntimeError, match='failed turn'):
            backend.run('test', workspace, role='optimizer')
    path, status = receipts(workspace)[0]
    assert status['status'] == ('succeeded' if accepted else 'failed')
    assert status['streams']['stdout']['truncated'] is True
    assert status['streams']['stderr']['truncated'] is True
    assert status['streams']['stdout']['observed_sha256'] == hashlib.sha256(output).hexdigest()
    assert status['events']['last_terminal_events'] == [{'type': terminal, 'usage': {'input_tokens': 4}}]
    assert_streams(path, status)


def test_oversized_event_cannot_hide_error_and_then_claim_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'EVENT_LINE_LIMIT', 128)
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import json
print(json.dumps({'type': 'error', 'message': 'large' * 100}))
print('{"type":"turn.completed"}')
''')
    with pytest.raises(RuntimeError, match='event line limit'):
        backend.run('test', workspace, role='reviewer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['events']['oversized_line']
    assert_streams(path, status)


def test_missing_cli_still_has_failed_receipt(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '')
    monkeypatch.setattr(module.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='CLI is required'):
        backend.run('test', workspace, role='case_enhancer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and 'pid' not in status
    assert_streams(path, status)


def test_evidence_write_failure_is_not_role_success(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, 'print(\'{"type":"turn.completed"}\')')
    real_write = module._write_status
    def quota_failure(directory, status):
        if status['status'] == 'succeeded':
            raise OSError(122, 'Disk quota exceeded (fixture)')
        return real_write(directory, status)
    monkeypatch.setattr(module, '_write_status', quota_failure)
    with pytest.raises(OSError, match='quota'):
        backend.run('test', workspace, role='optimizer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['evidence_error_errno'] == 122
    assert (path.parent / 'stdout.log').read_bytes() == b'{"type":"turn.completed"}\n'


def test_spawn_failure_is_recorded_without_environment_dump(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '')
    monkeypatch.setattr(module.shutil, 'which', lambda name: str(tmp_path / 'missing-executable'))
    with pytest.raises(FileNotFoundError):
        backend.run('not in argv', workspace, role='repair')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['exception_type'] == 'FileNotFoundError'
    assert status['argv'][-2:] == ['--', '-'] and 'pid' not in status
    assert_streams(path, status)


def test_split_unicode_event_and_terminal_without_newline(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import json, os, time
message = json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '汉字🙂'}}, ensure_ascii=False).encode()
pos = message.index('汉'.encode()) + 1
os.write(1, message[:pos])
time.sleep(.03)
os.write(1, message[pos:] + b'\\n')
os.write(1, b'{"type":"turn.completed"}')
''')
    assert backend.run('test', workspace, role='reviewer') == '汉字🙂\n{"type":"turn.completed"}'
    path, status = receipts(workspace)[0]
    assert status['events']['completed_turn'] is True
    assert_streams(path, status)


def test_stream_write_error_kills_process_and_preserves_failed_status(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import time
print('{"type":"item.started"}', flush=True)
time.sleep(60)
''')
    real_feed = module._StreamCapture.feed
    def failed_write(self, data):
        real_feed(self, data)
        raise OSError(122, 'fixture stream quota error')
    monkeypatch.setattr(module._StreamCapture, 'feed', failed_write)
    with pytest.raises(OSError, match='quota'):
        backend.run('test', workspace, role='optimizer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['returncode'] < 0
    assert status['events']['completed_turn'] is False
    assert_streams(path, status)


def test_short_disk_write_is_failed_with_actual_retained_byte_count(tmp_path):
    class ShortWriter:
        def __init__(self, stream):
            self.stream, self.name = stream, stream.name

        def write(self, data):
            return self.stream.write(data[:2])

    path = tmp_path / 'stdout.log'
    with path.open('wb', buffering=0) as stream:
        capture = module._StreamCapture(ShortWriter(stream), 100)
        with pytest.raises(OSError, match='Incomplete'):
            capture.feed(b'abcdef')
        recorded = capture.summary()
    assert path.read_bytes() == b'ab'
    assert recorded['retained_bytes'] == 2 and recorded['observed_bytes'] == 6
    assert recorded['retained_sha256'] == hashlib.sha256(b'ab').hexdigest()
    assert recorded['observed_sha256'] == hashlib.sha256(b'abcdef').hexdigest()
    assert recorded['truncated'] is True and recorded['eof'] is False


@pytest.mark.parametrize('timeout,cleanup,phase', [(30, 0.2, 'exit_drain'), (1, 5, 'role')])
def test_exited_cli_with_descendant_pipe_reports_actual_timeout(tmp_path, monkeypatch, timeout, cleanup, phase):
    monkeypatch.setattr(module, 'CLEANUP_SECONDS', cleanup)
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
print('{"type":"turn.completed"}', flush=True)
''', timeout=timeout)
    with pytest.raises(RuntimeError) as caught:
        backend.run('test', workspace, role='reviewer')
    path, status = receipts(workspace)[0]
    assert status['status'] == 'timed_out' and status['timeout_phase'] == phase
    if phase == 'exit_drain':
        assert 'CLI exited but output pipes remained open' in str(caught.value)
        assert 'timed out after 30s' not in str(caught.value)
        assert status['exit_drain_limit_seconds'] == cleanup
    else:
        assert f'timed out after {timeout}s' in str(caught.value)
    assert status['timeout_seconds'] == timeout and status['returncode'] == 0
    assert status['events']['completed_turn'] is True  # Still a rejected call.
    assert status['raw_receipts_synced'] is True
    child = int((workspace / 'child.pid').read_text())
    try:
        assert Path(f'/proc/{child}/stat').read_text().split()[2] == 'Z'
    except (FileNotFoundError, ProcessLookupError):
        pass
    assert_streams(path, status)


def test_raw_and_status_fsync_precede_success_publication(tmp_path, monkeypatch):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import sys
print('{"type":"turn.completed"}')
print('stderr evidence', file=sys.stderr)
''')
    operations = []
    real_fsync, real_replace = module.os.fsync, Path.replace
    def fsync(descriptor):
        path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
        state = json.loads(path.read_text())['status'] if path.name == 'process.json.tmp' else None
        operations.append(('fsync', path.name, state))
        return real_fsync(descriptor)
    def replace(path, destination):
        operations.append(('replace', path.name, json.loads(path.read_text())['status']))
        return real_replace(path, destination)
    monkeypatch.setattr(module.os, 'fsync', fsync)
    monkeypatch.setattr(Path, 'replace', replace)
    backend.run('test', workspace, role='optimizer')
    success_sync = operations.index(('fsync', 'process.json.tmp', 'succeeded'))
    success_replace = operations.index(('replace', 'process.json.tmp', 'succeeded'))
    assert operations.index(('fsync', 'stdout.log', None)) < success_sync < success_replace
    assert operations.index(('fsync', 'stderr.log', None)) < success_sync
    for state in ('starting', 'running'):
        assert operations.index(('fsync', 'process.json.tmp', state)) < operations.index(('replace', 'process.json.tmp', state))
    path, status = receipts(workspace)[0]
    assert status['status'] == 'succeeded' and status['raw_receipts_synced'] is True
    assert_streams(path, status)


@pytest.mark.parametrize('failing_file', ['stdout.log', 'stderr.log', 'process.json.tmp'])
def test_fsync_quota_error_cannot_publish_success(tmp_path, monkeypatch, failing_file):
    backend, workspace = make_backend(tmp_path, monkeypatch, '''
import sys
print('{"type":"turn.completed"}')
print('retained stderr', file=sys.stderr)
''')
    real_fsync = module.os.fsync
    injected = []
    def fsync(descriptor):
        path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
        if path.name == failing_file:
            # Permit initial state and the best-effort failed receipt, but reject
            # every attempt to sync successful metadata. Raw sync fails always.
            if path.name != 'process.json.tmp' or json.loads(path.read_text())['status'] == 'succeeded':
                injected.append(path.name)
                raise OSError(122, 'fsync quota fixture')
        return real_fsync(descriptor)
    monkeypatch.setattr(module.os, 'fsync', fsync)
    with pytest.raises(OSError, match='fsync quota'):
        backend.run('test', workspace, role='optimizer')
    assert injected
    path, status = receipts(workspace)[0]
    assert status['status'] == 'failed' and status['evidence_error_errno'] == 122
    assert status['returncode'] == 0 and status['events']['completed_turn'] is True
    assert status['raw_receipts_synced'] is (failing_file == 'process.json.tmp')
    assert (path.parent / 'stdout.log').read_bytes() == b'{"type":"turn.completed"}\n'
    assert_streams(path, status)

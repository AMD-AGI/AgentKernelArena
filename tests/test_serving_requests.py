"""Real socket lifecycle; synthetic child actions are not GPU qualification."""
import json
from pathlib import Path
import socket
import socketserver
import threading
import tempfile
import time
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.serving_runtime import CleanExecutor, request_handler


def test_disconnect_releases_serial_executor_for_final_action(tmp_path):
    entered, cleaned = threading.Event(), threading.Event()
    calls = []
    class Executor:
        def execute(self, request, *, cancelled):
            calls.append(request['action'])
            if request['action'] == 'performance':
                entered.set()
                limit = time.monotonic() + 5
                while not cancelled():
                    assert time.monotonic() < limit
                    time.sleep(.01)
                cleaned.set()
            else:
                assert cleaned.is_set()
            return {'returncode': 0}
    # Short path also fits macOS's Unix socket path limit.
    with tempfile.TemporaryDirectory(prefix='aka-sock-', dir='/tmp') as sockets:
        _exercise_socket_lifecycle(str(Path(sockets) / 's'), Executor(), entered, calls)


def _exercise_socket_lifecycle(endpoint, executor, entered, calls):
    with socketserver.UnixStreamServer(endpoint, request_handler(executor)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.socket(socket.AF_UNIX) as first:
                first.connect(endpoint)
                first.sendall(b'{"action":"performance"}\n')
                assert entered.wait(2)
            with socket.socket(socket.AF_UNIX) as final:
                final.settimeout(2)
                final.connect(endpoint)
                final.sendall(b'{"action":"compile"}\n')
                assert json.loads(final.recv(1024))['returncode'] == 0
            assert calls == ['performance', 'compile']
        finally:
            server.shutdown()
            thread.join(2)


@pytest.mark.parametrize('disconnected,deadline', [(False, 1), (True, None)])
def test_stale_requests_never_read_sources_or_launch_docker(tmp_path, disconnected, deadline):
    executor = CleanExecutor({}, tmp_path / 'checkout', tmp_path / 'trusted', '0', tmp_path / 'cache')
    spec = SimpleNamespace(action=lambda *_: SimpleNamespace(commands=[['runner']], timeout_s=60))
    executor.tasks['test'] = (tmp_path, spec, {})
    request = dict(task_id='test', role='candidate', action='compile', phase='candidate_evaluation',
                   workspace='/workspace/missing', timeout=60)
    if deadline is not None:
        request['deadline_epoch'] = deadline
    with pytest.raises((TimeoutError, ConnectionAbortedError)):
        executor.execute(request, cancelled=lambda: disconnected)


def test_cleanup_failure_disables_further_gpu_requests(tmp_path):
    executor = CleanExecutor({}, tmp_path / 'checkout', tmp_path / 'trusted', '0', tmp_path / 'cache')
    executor.cleanup_error = 'Serving cleanup failed; refusing further GPU actions'
    with pytest.raises(RuntimeError, match='cleanup failed'):
        executor.execute({})


@pytest.mark.parametrize('lingering', [False, True])
def test_cancel_kills_child_and_proves_container_absent_before_next_action(tmp_path, monkeypatch, lingering):
    import src.serving_runtime as runtime
    checkout = tmp_path / 'checkout'
    (checkout / 'run').mkdir(parents=True)
    template = tmp_path / 'template'
    template.mkdir()
    executor = CleanExecutor({}, checkout, tmp_path / 'trusted', '0', tmp_path / 'cache')
    spec = SimpleNamespace(action=lambda *_: SimpleNamespace(commands=[['runner']], timeout_s=60))
    executor.tasks['test'] = (template, spec, dict(model={}, dependencies={}, gpu_count=1, image='fixture'))
    monkeypatch.setattr(executor.runtime_verifier, 'verify', lambda *args: None)
    monkeypatch.setattr(runtime, 'runtime_mounts', lambda *args: [])
    real_popen = subprocess.Popen
    children, calls = [], []
    def launch(argv, **kwargs):
        calls.append('launch')
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    def cleanup(argv, **kwargs):
        calls.append(argv[1])
        assert children[-1].poll() is not None, 'Docker client must be reaped first'
        return SimpleNamespace(returncode=0, stdout='container-id\n' if lingering and argv[1] == 'ps' else '')
    monkeypatch.setattr(runtime.subprocess, 'Popen', launch)
    monkeypatch.setattr(runtime.subprocess, 'run', cleanup)
    request = dict(task_id='test', workspace='/workspace/run', role='baseline',
                   action='performance', phase='task_validation', timeout=60)
    if lingering:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            executor.execute(request, cancelled=lambda: bool(children))
        with pytest.raises(RuntimeError, match='cleanup failed'):
            executor.execute(request)
        assert calls == ['launch', 'rm', 'ps']
    else:
        for _ in range(2):
            count = len(children)
            result = executor.execute(request, cancelled=lambda: len(children) > count)
            assert result['returncode'] < 0
            assert 'caller disconnected' in result['stderr']
            assert executor.active_container is None
        assert calls == ['launch', 'rm', 'ps', 'launch', 'rm', 'ps']
    assert all(child.poll() is not None for child in children)
    assert len(list((tmp_path / 'trusted').glob('*/command_evidence.json'))) == len(children)

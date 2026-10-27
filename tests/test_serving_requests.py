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
    monkeypatch.setattr(runtime, '_CLEANUP_TIMEOUT_S', .01)
    checkout = tmp_path / 'checkout'
    (checkout / 'run').mkdir(parents=True)
    template = tmp_path / 'template'
    template.mkdir()
    executor = CleanExecutor({}, checkout, tmp_path / 'trusted', '0', tmp_path / 'cache')
    spec = SimpleNamespace(action=lambda *_: SimpleNamespace(commands=[['runner', '--value', 'two words']], timeout_s=60))
    executor.tasks['test'] = (template, spec, dict(model={}, dependencies={}, gpu_count=1, image='fixture'))
    monkeypatch.setattr(executor.runtime_verifier, 'verify', lambda *args: None)
    monkeypatch.setattr(runtime, 'runtime_mounts', lambda *args: [])
    real_popen = subprocess.Popen
    children, calls = [], []
    def launch(argv, **kwargs):
        # An image's serving CLI must not consume trusted task-action arguments.
        assert argv[argv.index('--entrypoint')+1] == 'runner'
        assert argv[argv.index('fixture')+1:] == ['--value', 'two words']
        calls.append('launch')
        child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    def cleanup(argv, **kwargs):
        calls.append(argv[1])
        assert children[-1].poll() is not None, 'Docker client must be reaped first'
        return SimpleNamespace(returncode=0, stdout='container-id\n' if lingering and argv[1] == 'ps' else '', stderr='')
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


def test_timed_out_removal_still_checks_actual_container_absence(tmp_path, monkeypatch):
    import src.serving_runtime as runtime
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == 'rm':
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    proof = runtime._remove_action_container('aka-eval-owned', tmp_path / 'cleanup.json')
    assert proof['status'] == 'PASS'
    assert 'TimeoutExpired' in proof['attempts'][0]['remove']['error']
    assert calls[-1] == ['docker', 'ps', '-aq', '--filter', 'name=^/aka-eval-owned$']
    assert json.loads((tmp_path / 'cleanup.json').read_text()) == proof


def test_cleanup_evidence_replaces_untrusted_link_without_writing_its_target(tmp_path, monkeypatch):
    import src.serving_runtime as runtime
    outside = tmp_path / 'outside.txt'
    outside.write_text('unchanged')
    evidence = tmp_path / 'cleanup.json'
    evidence.symlink_to(outside)
    monkeypatch.setattr(runtime.subprocess, 'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='', stderr=''))
    proof = runtime._remove_action_container('aka-eval-owned', evidence)
    assert proof['status'] == 'PASS'
    assert outside.read_text() == 'unchanged'
    assert not evidence.is_symlink()
    assert json.loads(evidence.read_text()) == proof


@pytest.mark.parametrize('initial_state', ['present', 'unknown'])
def test_cleanup_waits_for_confirmed_absence_after_transient_state(tmp_path, monkeypatch, initial_state):
    import src.serving_runtime as runtime
    monkeypatch.setattr(runtime, '_CLEANUP_TIMEOUT_S', 1)
    monkeypatch.setattr(runtime, '_CLEANUP_POLL_S', .001)
    inspections = 0
    def run(argv, **kwargs):
        nonlocal inspections
        if argv[1] == 'ps':
            inspections += 1
            if inspections == 1:
                return SimpleNamespace(returncode=1 if initial_state == 'unknown' else 0,
                    stdout='still-removing\n' if initial_state == 'present' else '',
                    stderr='daemon temporarily unavailable' if initial_state == 'unknown' else '')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    proof = runtime._remove_action_container('aka-eval-owned', tmp_path / 'cleanup.json')
    assert proof['status'] == 'PASS' and inspections == 2
    assert proof['attempts'][0]['inspect']['returncode'] == (1 if initial_state == 'unknown' else 0)
    assert proof['attempts'][1]['inspect']['returncode'] == 0


@pytest.mark.parametrize('unknown', [False, True])
def test_cleanup_never_accepts_lingering_or_unknown_container_state(tmp_path, monkeypatch, unknown):
    import src.serving_runtime as runtime
    monkeypatch.setattr(runtime, '_CLEANUP_TIMEOUT_S', .01)
    def run(argv, **kwargs):
        if argv[1] == 'ps':
            return SimpleNamespace(returncode=1 if unknown else 0,
                                   stdout='' if unknown else 'still-present\n', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='absence could not be established'):
        runtime._remove_action_container('aka-eval-owned', tmp_path / 'cleanup.json')
    assert json.loads((tmp_path / 'cleanup.json').read_text())['status'] == 'FAIL'


def native_client_review_fixture(tmp_path, members=None):
    import hashlib
    import io
    import tarfile
    root=tmp_path/'task';(root/'assets').mkdir(parents=True)
    archive=root/'assets/client.tar.gz'
    members=members or [('python/site/client/phase.py',b'def phase():\n    return "profiling"\n')]
    with tarfile.open(archive,'w:gz') as package:
        for name,data in members:
            info=tarfile.TarInfo(name);info.size=len(data)
            package.addfile(info,io.BytesIO(data))
    lock=dict(client_runtime=dict(kind='archive',asset='client.tar.gz',
        review_sources={n:hashlib.sha256(data).hexdigest() for n,data in members}),
        assets={'client.tar.gz':{'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}})
    return root,lock


def test_native_client_timer_source_evidence_is_read_from_digest_bound_archive(tmp_path):
    from src.tools.perf.serving_scenarios import client_review_sources
    import hashlib
    root,lock=native_client_review_fixture(tmp_path)
    records=client_review_sources(root,lock)
    assert len(records)==1
    row=records[0]
    assert row['path']=='client-runtime:python/site/client/phase.py'
    assert row['sha256']==hashlib.sha256(row['source'].encode()).hexdigest()
    assert row['archive_sha256']==lock['assets']['client.tar.gz']['sha256']
    assert not (tmp_path/'python').exists()
    lock['client_runtime']['review_sources']['python/site/client/phase.py']='0'*64
    with pytest.raises(ValueError,match='locked digest'):client_review_sources(root,lock)
    lock['assets']['client.tar.gz']['sha256']='1'*64
    with pytest.raises(ValueError,match='archive differs'):client_review_sources(root,lock)
    assert client_review_sources(root,{})==[]
    assert client_review_sources(root,dict(client_runtime=None))==[]


@pytest.mark.parametrize('name',['/etc/example.py','../example.py','python/../example.py',
    'python//example.py','python/./example.py','python/auth.json',''])
def test_native_client_review_source_paths_cannot_escape_or_read_arbitrary_data(name):
    from src.tools.perf.serving_scenarios import validate_client_review_sources
    with pytest.raises(ValueError):
        validate_client_review_sources(dict(review_sources={name:'a'*64}))


def test_native_client_review_source_evidence_rejects_ambiguous_and_large_sources(tmp_path):
    from src.tools.perf.serving_scenarios import client_review_sources
    root,lock=native_client_review_fixture(tmp_path/'duplicate', [('phase.py',b'a=1\n'),('phase.py',b'a=1\n')])
    with pytest.raises(ValueError,match='ambiguous'):client_review_sources(root,lock)
    root,lock=native_client_review_fixture(tmp_path/'large',[('phase.py',b' '*(256*1024+1))])
    with pytest.raises(ValueError,match='budget'):client_review_sources(root,lock)
    root,lock=native_client_review_fixture(tmp_path/'total',[(str(i)+'.py',b' '*(200*1024)) for i in range(3)])
    with pytest.raises(ValueError,match='combined budget'):client_review_sources(root,lock)

"""High-level checks for model-wide tasks; no enumerated single-kernel suite."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def source_policy():
    from _aka_serving_bundle import Bundle
    return Bundle(ROOT, json.loads((ROOT/'runtime.lock.json').read_text())).check_sources()


def prepare_runtime(lock):
    """Fail on missing/offline client prerequisites before loading GPU weights."""
    from _aka_serving import prepare_client
    from _aka_serving_scenarios import client_python
    client = prepare_client(lock)
    if lock['scenario'] != 'text':
        return {}
    executable = client_python(ROOT, lock)
    env = {**os.environ, 'PYTHONPATH': str(client), 'PYTHONDONTWRITEBYTECODE': '1',
           'PIP_NO_INDEX': '1', 'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never',
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
    command = ['bash', '-c', 'source "$1"; magpie_infx_client_ready "$2" "$3"',
               'client-preflight', str(client/'benchmarks/magpie_bench_remote_compat.sh'), executable, str(client)]
    completed = subprocess.run(command, cwd=client, env=env, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise ValueError('Prepared native client runtime is incomplete; no bootstrap is permitted: ' + completed.stderr[-2000:])
    return dict(prepared_client_runtime='PASS', client_runtime=lock['client_runtime'])


def launch_command(config, port):
    from _aka_serving import prepare_client
    from _aka_serving_backends import launch_command as launch
    lock = json.loads((ROOT/'runtime.lock.json').read_text())
    prepare_client(lock)
    return launch(ROOT, config, port, lock)


def server_environment(config, port, install_kernel):
    from _aka_serving_backends import server_environment as environment
    return environment(ROOT, config, port, install_kernel, json.loads((ROOT/'runtime.lock.json').read_text()))


def performance(config, lock, limits, client, port, row):
    from _aka_serving_scenarios import performance as measure
    return measure(ROOT, config, lock, limits, client, port, row)


def workload_identity(config, lock, limits, row):
    from _aka_serving_scenarios import text_identity
    return text_identity(ROOT, config, lock, limits, row)


def operator_checks(lock):
    """No per-kernel test list: shared model checks and separate review are required.

    This empty list is never evidence that individual operators were tested.
    """
    source_policy()
    return []

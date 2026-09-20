"""Real legacy helper preload and worker completion controls on CPU."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import yaml

from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = [
    'glm-5.3-flash__gemm_a16w16_bf16_cijk',
    'glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle',
    'qwen3.8-2.4t__gemma_fused_add_rmsnorm',
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=OPERATIONS)
def actual_legacy_worker(request, tmp_path, monkeypatch):
    original_task = task_directory(request.param)
    task = tmp_path / 'task'
    for folder in ('scripts', 'ut', 'source', 'build'):
        (task / folder).mkdir(parents=True)
    for filename in ('_trusted_worker.py', 'runtime_integrity.py', '_bench.py', 'task_runner.py'):
        shutil.copyfile(original_task / 'scripts' / filename, task / 'scripts' / filename)
    for filename in ('unittest.py', 'cases.py', 'harness_lib.py', 'meta.json'):
        shutil.copyfile(original_task / 'ut' / filename, task / 'ut' / filename)
    shutil.copyfile(ROOT / 'src/tools/perf/aka_benchmark.py', task / 'scripts/_aka_benchmark.py')
    # Only GPU/image availability is substituted. Real task modules, aliases,
    # source loading, monitoring and the common completion nonce are exercised.
    (task / 'scripts/runtime_preflight.py').write_text('''import sys
def require_runtime(cfg, *, phase="complete"):
    assert "aiter" not in sys.modules and "sglang" not in sys.modules
    return {"phase": phase}
''')
    cfg = yaml.safe_load((original_task / 'config.yaml').read_text())
    declared = cfg['headkernel']['trusted_worker_modules']
    assert declared['_legacy_correctness'] == 'ut/unittest.py'
    assert declared['_bench'] == 'scripts/_bench.py'
    declared['legacy_probe'] = 'scripts/probe.py'
    (task / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (task / 'scripts/probe.py').write_text('''import importlib.util,json,sys
from pathlib import Path
import torch

def main():
    cases = sys.modules['_headkernel_cases']
    checks = sys.modules['_legacy_correctness']
    assert sys.modules['legacy_probe'].main is main
    assert checks.h is sys.modules['harness_lib']
    alias = 'fused_add_rmsnorm_cases' if 'fused_add_rmsnorm_cases' in sys.modules else '_glm_cases'
    loader = checks._load if hasattr(checks, '_load') else checks.load
    assert loader(alias, cases.__file__) is cases
    assert sys.modules['_bench'].load_module('_headkernel_cases', cases.__file__) is cases
    if alias == '_glm_cases':
        assert checks.cases is cases
        assert len(cases.correctness_cases(cases.META)) in (21, 27)
    path = Path(__file__).resolve().parents[1] / 'source/kernel.py'
    spec = importlib.util.spec_from_file_location('legacy_candidate', path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules['legacy_candidate'] = candidate
    spec.loader.exec_module(candidate)
    assert torch.equal(candidate.kernel(torch.tensor([1., 2., 3.])), torch.tensor([2., 4., 6.]))
    print(json.dumps({'status': 'ok', 'same_helper_objects': True}))
    return 0
''')
    runner = load('legacy_runner', task / 'scripts/task_runner.py')
    runner.TASK_DIR, runner.UT_DIR, runner.BUILD_DIR = task, task / 'ut', task / 'build'
    original_command = runner.worker_command
    dependencies = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    bootstrap = ('import json,runpy,sys\nsys.path[:0]=json.loads(sys.argv.pop(1))\n'
                 'sys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n')
    def command(*args, **kwargs):
        actual = original_command(*args, **kwargs)
        return [sys.executable, '-B', '-c', bootstrap, json.dumps(dependencies), *actual[2:]]
    monkeypatch.setattr(runner, 'worker_command', command)
    def execute(source):
        (task / 'source/kernel.py').write_text(source)
        return runner.run_worker(task / 'scripts/probe.py', [], None, 30, True)
    return task, execute


def test_legacy_worker_accepts_unchanged_helpers(actual_legacy_worker):
    _, execute = actual_legacy_worker
    result = execute('def kernel(x): return x * 2\n')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['same_helper_objects'] is True


@pytest.mark.parametrize('attack', ['function', 'alias'])
def test_legacy_worker_rejects_actual_case_helper_replacement(actual_legacy_worker, attack):
    task, execute = actual_legacy_worker
    change = ("name = 'correctness_cases' if hasattr(cases, 'correctness_cases') else '_validate_outputs'\n"
              "    setattr(cases, name, lambda *args, **kwargs: [])" if attack == 'function' else
              "sys.modules['_headkernel_cases'] = types.ModuleType('replacement')")
    result = execute('import sys,types\ndef kernel(x):\n    cases = sys.modules["_headkernel_cases"]\n    '
                     + change + '\n    return x * 2\n')
    assert result.returncode != 0
    assert 'IntegrityError' in result.stderr
    assert not list((task / 'build').glob('_worker_completion_*'))

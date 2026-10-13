import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / 'src/tools/perf/aka_benchmark.py'
TASKS = ('points_in_boxes', 'roiaware_pool3d', 'roipoint_pool3d')


class ImportBoundaryTests(unittest.TestCase):
    def _stage(self, destination, task):
        rel = Path('tasks/hip2hip') / task
        shutil.copytree(REPO / rel, destination, ignore=shutil.ignore_patterns('__pycache__', 'build', '*.pyc'))
        shutil.copy2(BENCHMARK, destination / 'scripts/_aka_benchmark.py')
        for name in ('reference_checks', 'replay_validation', '_aka_benchmark', 'task_runner'):
            (destination / f'{name}.py').write_text("raise RuntimeError('ROOT SHADOW EXECUTED')\n")
        (destination / 'torch.py').write_text("raise RuntimeError('ROOT TORCH SHADOW EXECUTED')\n")
        (destination / 'scripts/torch.py').write_text("raise RuntimeError('SCRIPTS TORCH SHADOW EXECUTED')\n")
        for name in ('kernel_loader', f'{task}_wrapper'):
            (destination / 'scripts' / f'{name}.py').write_text("raise RuntimeError('SCRIPTS SHADOW EXECUTED')\n")
        return destination

    def _run(self, task_dir):
        code = '''
import importlib.util, pathlib, sys, types
root = pathlib.Path(sys.argv[1]).resolve()
for name in ('task_runner', '_aka_benchmark', 'reference_checks', 'replay_validation'):
    bogus = types.ModuleType(name)
    bogus.__file__ = str(root / (name + '.py'))
    sys.modules[name] = bogus
spec = importlib.util.spec_from_file_location('trusted_evaluate', root / 'scripts/evaluate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
harness = module._protected_runner()
for name in ('_aka_benchmark', 'reference_checks', 'replay_validation'):
    found = getattr(harness, '_PROTECTED_' + {'_aka_benchmark':'BENCHMARK', 'reference_checks':'REFERENCE_CHECKS', 'replay_validation':'REPLAY_VALIDATION'}[name])
    assert pathlib.Path(found.__file__).resolve() == root / 'scripts' / (name + '.py'), (name, found.__file__)
for name in ('kernel_loader', root.name + '_wrapper'):
    found = importlib.util.find_spec(name)
    assert pathlib.Path(found.origin).resolve() == root / (name + '.py'), (name, found.origin)
assert module.run(['validate-task']) == 0
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-c', code, str(task_dir)], cwd=task_dir, env=env, text=True, capture_output=True, timeout=60)

    def test_root_policy_shadows_and_inverse_scripts_shadows(self):
        with tempfile.TemporaryDirectory(prefix='roi-shadow-') as temp:
            for task in TASKS:
                with self.subTest(task=task):
                    task_dir = self._stage(Path(temp) / task, task)
                    result = self._run(task_dir)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    self.assertIn('"status": "PASS"', result.stdout)

    def test_bare_import_fixture_is_shadowable(self):
        # This is the former bare-import lookup pattern, isolated as a fixture.
        # The production test above verifies that the repaired runners do not use it.
        with tempfile.TemporaryDirectory(prefix='roi-bare-import-') as temp:
            task_dir = Path(temp)
            (task_dir / 'reference_checks.py').write_text("raise RuntimeError('ROOT SHADOW EXECUTED')\n")
            result = subprocess.run(
                [sys.executable, '-c',
                 'import sys; sys.path.insert(0, sys.argv[1]); import reference_checks',
                 str(task_dir)],
                cwd=task_dir, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                text=True, capture_output=True, timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('ROOT SHADOW EXECUTED', result.stderr + result.stdout)

    def test_preloaded_task_local_torch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='roi-preloaded-torch-') as temp:
            task_dir = self._stage(Path(temp) / 'points_in_boxes', 'points_in_boxes')
            code = '''
import importlib.util, pathlib, sys, types
root = pathlib.Path(sys.argv[1]).resolve()
bad = types.ModuleType('torch')
bad.__file__ = str(root / 'torch.py')
sys.modules['torch'] = bad
spec = importlib.util.spec_from_file_location('trusted_evaluate', root / 'scripts/evaluate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.exit(module.run(['validate-task']))
'''
            result = subprocess.run(
                [sys.executable, '-c', code, str(task_dir)],
                cwd=task_dir, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                text=True, capture_output=True, timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Task-local torch module', result.stdout + result.stderr)

if __name__ == '__main__':
    unittest.main()

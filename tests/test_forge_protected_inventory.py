"""Exercise inventory compatibility with the real pinned engine, CPU only."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_upstream(tmp_path, script):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run([python, "-c", script, str(tmp_path)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_inventory_matches_upstream_and_refreshes_symlinks(tmp_path):
    run_upstream(tmp_path, r'''
import os, sys
from pathlib import Path
from unittest.mock import patch
from agents.forge.protected_inventory import protected_path_inventory as fast
from kernelforge.llm.workspace_policy import protected_path_inventory as original
root = Path(sys.argv[1])
for name in ['plain.data', 'other.data', 'src/kernel.py', 'nested/tests/input.bin',
             'nested/test_oracle.py', 'deep/golden.data', '.git/tests/ignored.py']:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(name)
(root / 'link').symlink_to('plain.data')
(root / 'broken').symlink_to('missing.data')
(root / 'dir-link').symlink_to('nested', target_is_directory=True)
os.mkfifo(root / 'tests-fifo')
exact = ['link', 'broken', 'dir-link', 'future/missing.data', '', None]
kwargs = dict(exact_paths=exact, extra_globs=['*/golden.data'])
before = fast(root, **kwargs)
assert before == original(root, **kwargs)
assert root/'plain.data' in before and root/'other.data' not in before
assert root/'future/missing.data' in before and root/'link' in before
assert root/'nested/tests/input.bin' in before
assert root/'.git/tests/ignored.py' not in before
assert root/'src/kernel.py' not in before
assert root/'tests-fifo' not in before
(root/'link').unlink()
(root/'link').symlink_to('other.data')
after = fast(root, **kwargs)
assert after == original(root, **kwargs)
assert root/'other.data' in after and root/'plain.data' not in after
with patch('os.walk', side_effect=PermissionError('unreadable source')):
    try:
        fast(root, **kwargs)
    except PermissionError:
        pass
    else:
        raise AssertionError('inventory swallowed filesystem failure')
''')


def test_resolve_work_is_linear_for_large_exact_inventory(tmp_path):
    run_upstream(tmp_path, r'''
import sys
from pathlib import Path
from unittest.mock import patch
from agents.forge.protected_inventory import protected_path_inventory as fast
from kernelforge.llm.workspace_policy import protected_path_inventory as original
root = Path(sys.argv[1])
paths = [root/f'file_{i}.data' for i in range(128)]
for p in paths:
    p.write_text('reference material')
resolve = Path.resolve
counts = {}
outputs = {}
for name, scan in [('original', original), ('adapter', fast)]:
    count = [0]
    def counted(self, *args, **kwargs):
        count[0] += 1
        return resolve(self, *args, **kwargs)
    with patch.object(Path, 'resolve', counted):
        outputs[name] = scan(root, exact_paths=paths)
    counts[name] = count[0]
assert outputs['original'] == outputs['adapter'] == tuple(sorted(paths, key=str))
assert counts['original'] > len(paths)**2
assert counts['adapter'] <= 3*len(paths)+1, counts
''')


def test_real_guard_keeps_rollback_and_missing_file_detection(tmp_path):
    run_upstream(tmp_path, r'''
import subprocess, sys
from pathlib import Path
from agents.forge import upstream, protected_inventory
from kernelforge.agent_backends import workspace_guard
from kernelforge.agent_backends.base import AgentRunSpec
from kernelforge.loop import insession_gate
from kernelforge.llm import workspace_policy
upstream.probe()
protected_inventory.install()
assert workspace_guard.protected_path_inventory is protected_inventory.protected_path_inventory
assert insession_gate.protected_path_inventory is protected_inventory.protected_path_inventory
assert workspace_policy.protected_path_inventory is protected_inventory.protected_path_inventory
root = Path(sys.argv[1])
def git(*args):
    subprocess.run(['git', *args], cwd=root, capture_output=True, check=True)
git('init', '-q')
git('config', 'user.name', 'test')
git('config', 'user.email', 'test@example.invalid')
for name, content in [('kernel.py','initial'), ('reference.data','trusted'), ('.gitignore','ignored/\n')]:
    (root/name).write_text(content)
git('add', '.')
git('commit', '-qm', 'fixture')
missing = root/'ignored/future.data'
spec = AgentRunSpec(system_prompt='', user_prompt='', cwd=str(root),
    target_files=[str(root/'kernel.py')],
    protected_paths=[str(root/'reference.data'), str(missing)])
guard = workspace_guard.WorkspaceGuard(spec)
guard.prepare()
(root/'kernel.py').write_text('legal implementation')
assert guard.verify() == ['kernel.py']
(root/'reference.data').write_text('tampered')
try:
    guard.verify()
except workspace_guard.WorkspaceSafetyError as error:
    assert 'protected' in str(error)
else:
    raise AssertionError('protected reference accepted')
assert (root/'reference.data').read_text() == 'trusted'
assert (root/'kernel.py').read_text() == 'initial'
guard = workspace_guard.WorkspaceGuard(spec)
guard.prepare()
missing.parent.mkdir()
missing.write_text('injected ignored reference')
try:
    guard.verify()
except workspace_guard.WorkspaceSafetyError as error:
    assert 'protected' in str(error)
else:
    raise AssertionError('creation of missing protected path accepted')
assert not missing.exists()
''')


def test_unreviewed_policy_cannot_install_any_hooks(tmp_path):
    run_upstream(tmp_path, r'''
import sys
from pathlib import Path
from agents.forge import upstream
from kernelforge.llm import workspace_policy
original = workspace_policy.protected_path_inventory
changed = Path(sys.argv[1])/'changed_policy.py'
changed.write_text('# unreviewed upstream\n')
workspace_policy.__file__ = str(changed)
try:
    upstream.install_hooks({})
except RuntimeError as error:
    assert 'Unreviewed KernelForge source: kernelforge.llm.workspace_policy' in str(error)
else:
    raise AssertionError('unreviewed policy accepted')
assert workspace_policy.protected_path_inventory is original
''')

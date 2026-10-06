"""Retain evidence for transient package changes without accepting them."""
import ast
import hashlib
import json
from pathlib import Path
import secrets

import pytest


@pytest.fixture
def package(tmp_path):
    runner=Path(__file__).resolve().parents[1]/'tasks/headkernel/glm-5.3-flash__fused_moe_kernel/scripts/task_runner.py'
    names={'package_hash','package_snapshot','check_package_unchanged'}
    functions=[node for node in ast.parse(runner.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in names]
    def file_sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    namespace={'ROOT':tmp_path,'hashlib':hashlib,'file_sha':file_sha,'json':json,'secrets':secrets}
    # Exercise the real hash and diagnostic, without importing GPU dependencies.
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(runner),'exec'),namespace)
    for name,data in [('source/kernels.py',b'GPU body'),('ut/oracle.py',b'protected oracle'),('cases.json',b'cases')]:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return tmp_path,namespace


def test_unchanged_inputs_and_runtime_outputs_do_not_create_a_diagnostic(package):
    root,runner=package;before=runner['package_snapshot']()
    for name in ('build/compile_report.json','.validator_audit/compile.stderr.log','.validator_torch_extensions/extension.so'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('runtime output')
    runner['check_package_unchanged'](before,'compile')
    assert runner['package_hash']()==before['sha256']
    assert not list((root/'build').glob('package_hash_mismatch_*.json'))


@pytest.mark.parametrize('mutation',['added','removed','changed'])
def test_transient_input_change_fails_and_survives_later_cleanup(package,mutation):
    root,runner=package;path=root/'source/kernels.py';original=path.read_bytes()
    if mutation=='added':path.unlink()
    before=runner['package_snapshot']()
    if mutation=='removed':path.unlink()
    else:path.write_bytes(b'bad body')  # Same length as original; content must identify it.
    with pytest.raises(ValueError,match='Task package changed during evaluation; diagnostic: build/'):
        runner['check_package_unchanged'](before,'compile')
    after=runner['package_snapshot']()
    # A final workspace inventory can miss the transient mutation.
    if mutation=='added':path.unlink()
    else:path.write_bytes(original)
    assert runner['package_hash']()==before['sha256']
    files=list((root/'build').glob('package_hash_mismatch_compile_*.json'));assert len(files)==1
    diagnostic=json.loads(files[0].read_text())
    assert diagnostic['phase']=='compile'
    assert diagnostic['before']==before
    assert diagnostic['after']==after
    for kind in ('added','removed','changed'):
        assert diagnostic[kind]==(['source/kernels.py'] if kind==mutation else [])
    if mutation!='removed':
        assert diagnostic['after']['files']['source/kernels.py']=={'sha256':hashlib.sha256(b'bad body').hexdigest(),'size_bytes':8}


@pytest.mark.parametrize('name',['validation_report.yaml','.validation_complete','ut/oracle.py','cases.json'])
def test_report_marker_and_immutable_input_changes_still_fail(package,name):
    root,runner=package;before=runner['package_snapshot']();path=root/name
    path.write_text('changed input')
    with pytest.raises(ValueError,match='Task package changed during evaluation'):
        runner['check_package_unchanged'](before,'correctness')
    diagnostic=json.loads(next((root/'build').glob('package_hash_mismatch_*.json')).read_text())
    assert name in diagnostic['added']+diagnostic['changed']


def test_repeated_mismatches_keep_separate_evidence_and_do_not_hash_diagnostics(package):
    root,runner=package;before=runner['package_snapshot']();(root/'source/kernels.py').write_bytes(b'bad body')
    changed=runner['package_hash']()
    for _ in range(2):
        with pytest.raises(ValueError,match='Task package changed during evaluation'):
            runner['check_package_unchanged'](before,'performance')
    paths=list((root/'build').glob('package_hash_mismatch_performance_*.json'))
    assert len(paths)==2
    assert runner['package_hash']()==changed
    assert all(json.loads(path.read_text())['after']['sha256']==changed for path in paths)


def test_unwritable_diagnostic_does_not_suppress_the_package_failure(package):
    root,runner=package;before=runner['package_snapshot']();(root/'source/kernels.py').write_bytes(b'bad body')
    (root/'build').write_text('blocks diagnostic directory creation')
    with pytest.raises(ValueError,match='Task package changed during evaluation; could not write hash diagnostic:'):
        runner['check_package_unchanged'](before,'compile')

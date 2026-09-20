"""Integration gates for generated Kimi tasks; no native GPU claim."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
from head_kernel_generated_test_utils import generated_helper

def kimi_helper(filename):
    return generated_helper('kimi', filename)
TASKS = {p.parent.name: p.parent for p in (ROOT/'tasks/head_kernels/kimi-k3').rglob('config.yaml')}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_shared_workers_and_public_runtime_are_preserved():
    for task in TASKS.values():
        cfg = yaml.safe_load((task/'config.yaml').read_text())
        prior = yaml.safe_load(subprocess.check_output(['git','show','908879dc:'+str((task/'config.yaml').relative_to(ROOT))], cwd=ROOT))
        assert cfg['headkernel']['docker'] == prior['headkernel']['docker']
        assert cfg['headkernel']['runtime'] == prior['headkernel']['runtime']
        for name in ('runtime_integrity.py','_trusted_worker.py','runtime_preflight.py'):
            assert (task/'scripts'/name).read_bytes() == (ROOT/'tasks/head_kernels/_support'/name).read_bytes()
        assert cfg['headkernel']['trusted_worker_modules']['generated_worker'] == 'scripts/generated_worker.py'
        assert cfg['harness_path'] == 'scripts/generated_worker.py'
        assert 'def attest(' not in (task/'scripts/generated_worker.py').read_text()


@pytest.mark.parametrize('name', list(TASKS))
def test_actual_preloader_imports_helpers_without_candidate_access(name):
    task = TASKS[name]
    code = f'''import importlib.util,pathlib,yaml,sys
path=pathlib.Path({str(ROOT/'tasks/head_kernels/_support/_trusted_worker.py')!r})
spec=importlib.util.spec_from_file_location('bootstrap',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
task=pathlib.Path({str(task)!r});cfg=yaml.safe_load((task/'config.yaml').read_text())
modules,files=m.load_declared_modules(task,cfg)
assert files[(task/'scripts/generated_worker.py').resolve()] is modules['generated_worker']
assert modules['generated_contract'] is sys.modules['generated_contract']
assert '_kimi_generated_candidate' not in sys.modules
'''
    result = subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr


def test_unobserved_and_boundary_workloads_fail_before_gpu_scoring(tmp_path):
    controller = load('controller',kimi_helper('generated_controller.py'))
    for task in TASKS.values():
        meta=json.loads((task/'ut/meta.json').read_text())
        assert meta['workload_scoring']['enabled'] is False
        reports=[]
        runner=SimpleNamespace(UT_DIR=task/'ut',write_report=lambda name,data:reports.append(data))
        assert controller.run_performance(runner,{},1)==[]
        assert reports[-1]['status']=='fail'
        assert 'Workload scoring disabled' in reports[-1]['error']
        assert reports[-1]['test_cases']==[]


def test_single_launch_gate_cannot_be_satisfied_by_a_median():
    controller=load('controller',kimi_helper('generated_controller.py'))
    task=TASKS['moe_gemm1_stage1'];meta=json.loads((task/'ut/meta.json').read_text())
    compact=json.loads((task/'ut/generated_cases.json').read_text())
    expected=controller.expected_profiles(meta,compact)['single_launch']
    assert len(expected)==len(meta['case_specs'])*meta['random_draws']==9
    response={'schema_version':1,'profile':'single_launch','seed':1,'reference':False,
              'correctness_policy':'elementwise_median_21','rows':[{'id':name,'output':0} for name in expected]}
    proc=SimpleNamespace(returncode=0,stderr='',stdout=controller.PREFIX+json.dumps(response))
    with pytest.raises(RuntimeError,match='replaced by a median'):
        controller.parse_worker(proc,'single_launch',1,False,expected,True)
    reference=meta['single_launch_reference']
    assert hashlib.sha256((task/reference['source']).read_bytes()).hexdigest()==reference['source_sha256']
    source=(task/'ut/generated_single_launch.py').read_text()
    assert 'torch_moe_stage1(' in source and '.median(' not in source
    assert 'activation=ActivationType.Situv2' in source and 'doweight=False' in source


def test_one_bad_individual_launch_fails_even_if_other_trials_match():
    controller=load('controller',kimi_helper('generated_controller.py'))
    contract=load('contract',kimi_helper('generated_contract.py'))
    expected=[{'id':f'case:single:{i}','output':contract.encode_output(torch.ones(3),torch)} for i in range(3)]
    actual=json.loads(json.dumps(expected));actual[1]['output']=contract.encode_output(torch.full((3,),9.),torch)
    with pytest.raises(RuntimeError,match='reference mismatch'):
        controller.compare_rows(expected,actual,contract,.02,torch)

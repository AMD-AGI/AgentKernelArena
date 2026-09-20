"""Exercise the finalized common worker/nonce boundary with GLM generated helpers."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]


def source(name):
    support=ROOT/'tasks/head_kernels/_support'
    return (support/name).read_bytes()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


@pytest.fixture
def actual_worker(tmp_path,monkeypatch):
    task=tmp_path/'task'
    for name in ['scripts','ut','source','build']:(task/name).mkdir(parents=True)
    for name in ['_trusted_worker.py','runtime_integrity.py','_bench.py','task_runner.py']:
        (task/'scripts'/name).write_bytes(source(name))
    shutil.copyfile(ROOT/'src/tools/perf/aka_benchmark.py',task/'scripts/_aka_benchmark.py')
    shutil.copyfile(generated_helper("glm", "generated_worker.py"),task/'scripts/generated_worker.py')
    shutil.copyfile(generated_helper("glm", "generated_contract.py"),task/'ut/generated_contract.py')
    (task/'scripts/runtime_preflight.py').write_text('def require_runtime(cfg, **kwargs): pass\n')
    (task/'ut/harness_lib.py').write_text('import torch\ndef correct(a,b,tol): return bool(torch.equal(a,b)), 0.0\n')
    (task/'scripts/probe.py').write_text('''import importlib.util,json,sys
from pathlib import Path
import torch
import generated_contract
import generated_worker

def main():
    if sys.modules['probe_worker'].main is not main:
        raise RuntimeError('unattested entrypoint copy')
    if sys.modules['contract_alias'] is not generated_contract:
        raise RuntimeError('alias did not reuse the attested helper')
    task=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('candidate',task/'source/kernel.py')
    candidate=importlib.util.module_from_spec(spec);sys.modules['candidate']=candidate
    spec.loader.exec_module(candidate)
    values={'scale':torch.tensor([1.,2.,3.])}
    result=generated_worker.invoke_checked(lambda args:candidate.kernel(args['scale']),values,generated_contract,torch,[])
    print(generated_worker.PREFIX+json.dumps({'schema_version':1,'profile':'recorded','seed':731,
        'reference':False,'rows':[{'id':'case','inputs':[],**result}]}))
    return 0
''')
    cfg={'headkernel':{'generated_input_revision':'glm-generated-test','trusted_worker_modules':{
        'generated_contract':'ut/generated_contract.py','contract_alias':'ut/generated_contract.py',
        'generated_worker':'scripts/generated_worker.py','_bench':'scripts/_bench.py','probe_worker':'scripts/probe.py'}}}
    (task/'config.yaml').write_text(yaml.safe_dump(cfg))
    runner=load(task/'scripts/task_runner.py','glm_common_runner')
    runner.TASK_DIR,runner.UT_DIR,runner.BUILD_DIR=task,task/'ut',task/'build'
    original=runner.worker_command
    dependencies=[x for x in os.environ.get('PYTHONPATH','').split(os.pathsep) if x]
    bootstrap='import json,runpy,sys\nsys.path[:0]=json.loads(sys.argv.pop(1))\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n'
    def command(*args,**kwargs):
        actual=original(*args,**kwargs)
        return [sys.executable,'-B','-c',bootstrap,json.dumps(dependencies),*actual[2:]]
    monkeypatch.setattr(runner,'worker_command',command)
    controller=load(generated_helper("glm", "generated_correctness.py"),'glm_parent_controller')
    contract=load(generated_helper("glm", "generated_contract.py"),'glm_parent_contract')
    expected=[{'id':'case','inputs':[],'aliases':[],
               'output':contract.encode_output(torch.tensor([2.,4.,6.]),torch)}]
    def execute(text):
        (task/'source/kernel.py').write_text(text)
        proc=runner.run_worker(task/'scripts/probe.py',[],None,30,True)
        accepted=False
        if proc.returncode==0:
            rows=controller.parse_worker(proc,'recorded',731,False,['case'])
            try:
                controller.compare_rows(expected,rows,contract,0.02,torch)
                accepted=True
            except RuntimeError:
                pass
        return accepted,proc
    return task,execute


def test_common_worker_accepts_real_unchanged_source(actual_worker):
    _,execute=actual_worker
    accepted,proc=execute('def kernel(scale): return scale*2\n')
    assert accepted,(proc.stdout,proc.stderr)


@pytest.mark.parametrize('attack',[
    "sys.modules['generated_contract'].require_inputs_unchanged=lambda *a:None\n    scale.add_(10)",
    "sys.modules['generated_contract'].encode_output=lambda *a: {'stride':[1]}",
    "sys.modules['contract_alias']=types.ModuleType('fake')",
    "torch.Tensor.numpy=lambda self: old_numpy(torch.tensor([2.,4.,6.]))",
])
def test_common_finalization_rejects_checker_encoder_alias_and_numpy_tampering(actual_worker,attack):
    task,execute=actual_worker
    text='import sys,types,torch\nold_numpy=torch.Tensor.numpy\ndef kernel(scale):\n    out=scale*2\n    '+attack+'\n    return out\n'
    accepted,proc=execute(text)
    assert not accepted and proc.returncode!=0
    assert 'IntegrityError' in proc.stderr
    assert not list((task/'build').glob('_worker_completion_*'))

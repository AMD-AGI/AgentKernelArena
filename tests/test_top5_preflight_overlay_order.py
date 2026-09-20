"""Real import/cache regressions through the shared worker; no GPU execution."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
SUPPORT=ROOT/'tasks/head_kernels/_support'
GLM=ROOT/'tasks/head_kernels/glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/fused_moe_kernel'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def test_environment_phase_does_not_import_native_dependencies_or_target(tmp_path,monkeypatch):
    runtime=load('phase_runtime_test',SUPPORT/'runtime_preflight.py')
    image='registry/sglang:v0.5.18-rocm720-mi35x-profilerfix'
    cfg={'headkernel':{'docker':image,'target_callable':'native_fixture:run',
                       'runtime':{'required_modules':['native_fixture']}}}
    requirements=runtime.runtime_requirements(cfg,tmp_path)
    for name,value in requirements['environment'].items():monkeypatch.setenv(name,value)
    for name in requirements['cache_environment']:monkeypatch.setenv(name,str(tmp_path/name))
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DOCKER','1')
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DOCKER_IMAGE',image)
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID','sha256:'+'a'*64)
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST','sha256:'+'a'*64)
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID_ROLE','config_digest')
    modules={name:SimpleNamespace(__version__='1.0') for name in requirements['required_modules']}
    modules['sglang'].__version__='0.5.18'
    modules['native_fixture'].run=lambda value:value
    modules['torch']=SimpleNamespace(__version__='2.9.1+rocm7.2.0',version=SimpleNamespace(hip='7.2.26015'),
        cuda=SimpleNamespace(is_available=lambda:True,current_device=lambda:0,
            get_device_properties=lambda _:SimpleNamespace(gcnArchName='gfx950')))
    calls=[]
    def imported(name):
        calls.append(name)
        return modules[name]
    monkeypatch.setattr(runtime.importlib,'import_module',imported)
    before=runtime.require_runtime(cfg,tmp_path,phase='environment')
    assert calls==['torch']
    assert before['status']=='ok' and before['native_resolution_complete'] is False
    assert before['target_resolution']=='deferred_until_protected_overlay'
    assert (tmp_path/'build/runtime_preflight_environment.json').is_file()
    after=runtime.require_runtime(cfg,tmp_path)
    assert after['phase']=='complete' and after['native_resolution_complete'] is True
    assert 'native_fixture' in calls and after['target_resolution']['target']=='native_fixture:run'
    assert (tmp_path/'build/runtime_preflight.json').is_file()
    with pytest.raises(ValueError,match='phase'):
        runtime.preflight(cfg,tmp_path,phase='skip')


@pytest.fixture
def import_task(tmp_path,monkeypatch):
    task=tmp_path/'task'
    for name in ('ut','scripts','source','build'):(task/name).mkdir(parents=True)
    for name in ('_trusted_worker.py','runtime_integrity.py','task_runner.py'):
        shutil.copyfile(SUPPORT/name,task/'scripts'/name)
    shutil.copyfile(ROOT/'src/tools/perf/aka_benchmark.py',task/'scripts/_aka_benchmark.py')
    (task/'ut/harness_lib.py').write_text('import torch\ndef correct(a,b,tol): return bool(torch.equal(a,b)),0.0\n')
    # Only hardware/image availability is substituted. Native target imports,
    # aliases, overlays, attestation and the completion nonce are real.
    (task/'scripts/runtime_preflight.py').write_text('''import importlib,sys
PHASES=[]
def require_runtime(cfg, *, phase="complete"):
    PHASES.append(phase)
    if phase=="environment":
        if "native_pkg" in sys.modules:
            raise RuntimeError("native package was already imported")
        return {"phase":phase,"native_resolution_complete":False}
    owner=importlib.import_module("native_pkg.dispatcher")
    if not callable(owner.run):raise RuntimeError("target is not callable")
    return {"phase":phase,"native_resolution_complete":True}
''')
    (task/'ut/declared.py').write_text('import sys\nNATIVE_WAS_CACHED="native_pkg" in sys.modules\n')
    (task/'scripts/probe.py').write_text('''import json,sys
import declared_helper

def main():
    import native_pkg
    from native_pkg import dispatcher,device
    if sys.modules["probe_worker"].main is not main:
        raise RuntimeError("entrypoint was not the attested helper")
    if declared_helper.NATIVE_WAS_CACHED:
        raise RuntimeError("native package imported before helper preload")
    if not native_pkg.import_had_guard:
        raise RuntimeError("native package imported before attestation")
    if native_pkg.exported is not dispatcher.run:
        raise RuntimeError("package cached a stale dispatcher alias")
    if sys.modules["runtime_preflight"].PHASES != ["environment","complete"]:
        raise RuntimeError("preflight did not execute both phases")
    print(json.dumps({"value":dispatcher.run(7),"device_file":device.__file__,
                      "dispatcher_file":dispatcher.__file__}))
    return 0
''')
    cfg={'headkernel':{'generated_input_revision':'import-order-test','trusted_worker_modules':{
        'declared_helper':'ut/declared.py','probe_worker':'scripts/probe.py'}}}
    (task/'config.yaml').write_text(yaml.safe_dump(cfg))
    installed=tmp_path/'installed/native_pkg';installed.mkdir(parents=True)
    (installed/'__init__.py').write_text('''import sys
monitor=sys.modules.get("runtime_integrity")
import_had_guard=monitor is not None and monitor.ACTIVE_GUARD is not None
from .dispatcher import run as exported
''')
    (installed/'dispatcher.py').write_text('from .device import kernel\ndef run(x): return kernel(x)\n')
    (installed/'device.py').write_text('def kernel(x): return -999\n')
    overlays=[]
    for name,multiplier in [('baseline_overlay',2),('_cand_overlay',3)]:
        overlay=task/'ut'/name;(overlay/'_patched').mkdir(parents=True)
        shutil.copyfile(GLM/'ut/baseline_overlay/sitecustomize.py',overlay/'sitecustomize.py')
        (overlay/'_patched/device.py').write_text(f'def kernel(x): return x*{multiplier}\n')
        (overlay/'_patched/dispatcher.py').write_text('from native_pkg.device import kernel\ndef run(x): return kernel(x)\n')
        (overlay/'_overlay_manifest.json').write_text(json.dumps({'modules':[
            {'module':'native_pkg.device','file':'_patched/device.py'},
            {'module':'native_pkg.dispatcher','file':'_patched/dispatcher.py'}],
            'rebinds':[],'captures':[],'markers':[]}))
        overlays.append(overlay)
    runner=load('import_order_runner',task/'scripts/task_runner.py')
    runner.TASK_DIR,runner.UT_DIR,runner.BUILD_DIR=task,task/'ut',task/'build'
    original=runner.worker_command
    paths=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p]+[str(installed.parent)]
    boot='import json,runpy,sys\nsys.path[:0]=json.loads(sys.argv.pop(1))\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n'
    def command(*args,**kwargs):
        actual=original(*args,**kwargs)
        return [sys.executable,'-B','-c',boot,json.dumps(paths),*actual[2:]]
    monkeypatch.setattr(runner,'worker_command',command)
    return task,installed.parent,overlays,runner


def test_shared_worker_resolves_frozen_and_candidate_after_attestation(import_task):
    task,_,overlays,runner=import_task
    outputs=[]
    for i,overlay in enumerate(overlays):
        proc=runner.run_worker(task/'scripts/probe.py',[],str(overlay),30,bool(i))
        assert proc.returncode==0,proc.stderr
        value=json.loads(proc.stdout)
        assert Path(value['device_file']).resolve()==overlay/'_patched/device.py'
        assert Path(value['dispatcher_file']).resolve()==overlay/'_patched/dispatcher.py'
        outputs.append(value['value'])
    assert outputs==[14,21]
    assert not list((task/'build').glob('_worker_completion_*'))


def test_old_import_order_reproduces_stale_native_alias(import_task):
    _,installed,overlays,_=import_task
    program='''import importlib,json,runpy,sys
sys.path.insert(0,sys.argv[1])
import native_pkg
before=importlib.import_module("native_pkg.dispatcher")
runpy.run_path(sys.argv[2]+"/sitecustomize.py")
after=importlib.import_module("native_pkg.dispatcher")
print(json.dumps({"cached":before is after,"value":after.run(7),"guard":native_pkg.import_had_guard}))
'''
    proc=subprocess.run([sys.executable,'-B','-c',program,str(installed),str(overlays[0])],
                        capture_output=True,text=True,check=True)
    assert json.loads(proc.stdout)=={'cached':True,'value':-999,'guard':False}


def test_shared_worker_rejects_premature_native_import_from_a_helper(import_task):
    task,_,overlays,runner=import_task
    (task/'ut/declared.py').write_text('import native_pkg\nNATIVE_WAS_CACHED=True\n')
    proc=runner.run_worker(task/'scripts/probe.py',[],str(overlays[1]),30,True)
    assert proc.returncode!=0
    assert 'source module loaded before protected overlay' in proc.stderr
    assert not list((task/'build').glob('_worker_completion_*'))


@pytest.mark.parametrize("attack", [
    'sys.modules["runtime_preflight"].require_runtime = lambda *args, **kwargs: {}',
    'importlib.import_module = lambda *args, **kwargs: None',
    'importlib.metadata.version = lambda *args, **kwargs: "spoofed"',
])
def test_native_source_cannot_disable_complete_preflight(import_task, attack):
    task,_,overlays,runner=import_task
    (overlays[1]/'_patched/device.py').write_text(
        'import sys,importlib,importlib.metadata\n' + attack + '\ndef kernel(x): return x*3\n')
    proc=runner.run_worker(task/'scripts/probe.py',[],str(overlays[1]),30,True)
    assert proc.returncode!=0
    assert 'IntegrityError' in proc.stderr
    assert not list((task/'build').glob('_worker_completion_*'))

"""Real CPU tensor failure artifacts, with no verifier retry or GPU readback."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from src.tools import trusted_task_eval as trusted

UT=Path(__file__).resolve().parents[1]/'tasks/headkernel/glm-5.3-flash__fused_moe_kernel/ut'
sys.path.insert(0,str(UT))
spec=importlib.util.spec_from_file_location('tested_moe_receipts',UT/'replay_receipts.py')
receipts_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(receipts_module)
from failure_bundle import file_sha, json_sha, save_cpu_failure


@pytest.fixture
def capture(tmp_path):
    base=torch.arange(16,dtype=torch.float32).reshape(4,4)
    truth={'seed':512174569,'inputs':{'x':base[1:3,::2].t(),'alias':base[0]}}
    snapshots={};calls=[];failure=AssertionError('original native mismatch')
    def reset(seed):truth['seed']=seed;calls.append('reset');return truth
    def verify(value):
        calls.append('verify')
        actual=torch.tensor([[1.0,2.5]],dtype=torch.bfloat16)
        expected=torch.tensor([[1.0,2.0]],dtype=torch.bfloat16)
        observed_inputs={name:tensor.clone() for name,tensor in value['inputs'].items()}
        snapshots.update(actual=actual,expected=expected,observed_inputs=observed_inputs)
        raise failure
    request={'request_id':'original-request','phase':'performance','challenge_seed':688901543,
             'source_sha256':{'source/kernels.py':'a'*64},'manifest_sha256':'b'*64}
    provenance={'runtime_image':'image@sha256:'+'c'*64,'native_source_manifest_sha256':'d'*64,
                'native_sources':{'native_codeobject_arithmetic_evidence':{'path':'hsa/native.co','sha256':'e'*64}}}
    case={'case_id':'prefill','fixture':{'path':'fixtures/real.json','sha256':'f'*64}}
    receipts=receipts_module.ReplayReceipts(tmp_path/'build',512174497,'a'*64,request=request,provenance=provenance)
    checked_reset,checked_verify=receipts.leg('prefill','native_production',reset,verify,case=case)
    return {'root':tmp_path,'truth':truth,'snapshots':snapshots,'calls':calls,'failure':failure,'verify':verify,
            'request':request,'provenance':provenance,'case':case,'receipts':receipts,'reset':checked_reset,'check':checked_verify}


def run_failure(capture):
    capture['reset'](512174569)
    with pytest.raises(AssertionError) as caught:capture['check'](capture['truth'])
    assert caught.value is capture['failure']
    events=[json.loads(line) for line in capture['receipts'].path.read_text().splitlines()]
    failure=next(row for row in reversed(events) if row['event']=='verify_failure')
    assert events[-1]['event']=='verify_failure_artifact'
    return {**failure,'failure_artifact':events[-1]['failure_artifact']}


def test_exact_cpu_tensors_seed_context_and_hashes_survive_capture(capture,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('GPU/readback method called by persistence')
    monkeypatch.setattr(torch.Tensor,'cpu',forbidden)
    monkeypatch.setattr(torch.Tensor,'cuda',forbidden)
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    event=run_failure(capture);artifact=event['failure_artifact'];build=capture['receipts'].path.parent
    assert capture['calls']==['reset','verify']
    assert artifact['status']=='complete'
    manifest_path=build/artifact['manifest_file'];manifest=json.loads(manifest_path.read_text())
    assert file_sha(manifest_path)==artifact['manifest_sha256']
    assert manifest['context']['native_comparison_challenge_seed']==512174497
    assert manifest['context']['seed']==512174569 and manifest['context']['iteration']==72
    assert manifest['enclosing_request']==capture['request']
    assert manifest['enclosing_request']['challenge_seed']==688901543
    assert manifest['enclosing_request_sha256']==json_sha(capture['request'])
    assert manifest['provenance']==capture['provenance'] and manifest['case']==capture['case']
    assert manifest['tensor_manifest_sha256']==json_sha(manifest['tensor_manifest'])
    bundle=build/manifest['bundle']['file'];assert file_sha(bundle)==manifest['bundle']['sha256']
    saved=torch.load(bundle,map_location='cpu',weights_only=True)
    for name in ['actual','expected']:assert torch.equal(saved[name],capture['snapshots'][name])
    for group,original in [('truth_inputs',capture['truth']['inputs']),('observed_inputs',capture['snapshots']['observed_inputs'])]:
        for name,value in original.items():
            assert torch.equal(saved[group][name],value)
            assert saved[group][name].stride()==value.stride()
            assert saved[group][name].storage_offset()==value.storage_offset()
    assert saved['truth_inputs']['x'].untyped_storage().data_ptr()==saved['truth_inputs']['alias'].untyped_storage().data_ptr()


def test_only_one_bundle_is_attempted_per_comparison(capture):
    first=run_failure(capture);artifact=first['failure_artifact'];path=capture['receipts'].path.parent/artifact['bundle']['file']
    digest=file_sha(path);second=run_failure(capture)
    assert second['failure_artifact']['status']=='already_attempted'
    assert file_sha(path)==digest and len(list(path.parent.glob('*.tensor_failure.pt')))==1
    assert capture['calls']==['reset','verify','reset','verify']


def test_original_failure_is_durable_before_tensor_serialization(capture,monkeypatch):
    original=receipts_module.save_cpu_failure
    def check_order(*args,**kwargs):
        last=json.loads(capture['receipts'].path.read_text().splitlines()[-1])
        assert last['event']=='verify_failure' and last['seed']==512174569
        return original(*args,**kwargs)
    monkeypatch.setattr(receipts_module,'save_cpu_failure',check_order)
    run_failure(capture)


@pytest.mark.parametrize('limit',[1,256])
def test_storage_and_serialized_byte_limits_do_not_mask_verifier_failure(capture,monkeypatch,limit):
    def bounded(*args,**kwargs):return save_cpu_failure(*args,**kwargs,max_bytes=limit)
    monkeypatch.setattr(receipts_module,'save_cpu_failure',bounded)
    event=run_failure(capture);artifact=event['failure_artifact'];manifest=json.loads((capture['receipts'].path.parent/artifact['manifest_file']).read_text())
    assert artifact['status']=='capture_error'
    if manifest['bundle'] is not None:
        assert manifest['bundle']['bytes']<=limit
        assert manifest['bundle']['complete'] is False


@pytest.mark.parametrize('kind,limit',[('expanded',256),('overlapping',256),('aggregate',128)])
def test_logical_byte_limit_rejects_views_before_any_materialization(capture,monkeypatch,kind,limit):
    if kind=='expanded':view=torch.ones(1).expand(1024)
    elif kind=='overlapping':view=torch.as_strided(torch.ones(16),(8,8),(1,1))
    else:
        view=torch.ones(1).expand(16)
        capture['truth']['inputs']={'x':view}
    def verify(truth):
        actual=view;expected=view;observed_inputs=truth['inputs']
        raise capture['failure']
    _,capture['check']=capture['receipts'].leg('prefill','native_production',lambda seed:capture['truth'],verify,case=capture['case'])
    def bounded(*args,**kwargs):return save_cpu_failure(*args,**kwargs,max_bytes=limit)
    monkeypatch.setattr(receipts_module,'save_cpu_failure',bounded)
    materializations=[]
    def forbidden(*args,**kwargs):
        materializations.append(True)
        raise AssertionError('logical view was materialized')
    monkeypatch.setattr(torch.Tensor,'contiguous',forbidden)
    monkeypatch.setattr(torch,'save',forbidden)
    artifact=run_failure(capture)['failure_artifact']
    manifest=json.loads((capture['receipts'].path.parent/artifact['manifest_file']).read_text())
    assert artifact['status']=='capture_error' and manifest['bundle'] is None
    assert manifest['unique_storage_bytes']<=limit<manifest['logical_tensor_bytes']
    assert 'logical bytes' in manifest['capture_error']['message']
    assert materializations==[]


@pytest.mark.parametrize('kind',['expanded','overlapping'])
def test_bounded_logical_views_keep_their_data_and_strides(capture,kind):
    view=torch.ones(1).expand(8) if kind=='expanded' else torch.as_strided(torch.arange(5.),(3,3),(1,1))
    def verify(truth):
        actual=view;expected=view;observed_inputs=truth['inputs']
        raise capture['failure']
    _,capture['check']=capture['receipts'].leg('prefill','native_production',lambda seed:capture['truth'],verify,case=capture['case'])
    artifact=run_failure(capture)['failure_artifact'];build=capture['receipts'].path.parent
    manifest=json.loads((build/artifact['manifest_file']).read_text())
    assert artifact['status']=='complete'
    saved=torch.load(build/manifest['bundle']['file'],weights_only=True)
    assert torch.equal(saved['actual'],view) and saved['actual'].stride()==view.stride()


def test_non_cpu_snapshots_are_rejected_without_readback(capture,monkeypatch):
    def verify(truth):
        actual=torch.empty(2,device='meta');expected=torch.ones(2)
        observed_inputs=truth['inputs']
        raise capture['failure']
    _,capture['check']=capture['receipts'].leg('prefill','native_production',lambda seed:capture['truth'],verify,case=capture['case'])
    def forbidden(*args,**kwargs):raise AssertionError('readback attempted')
    monkeypatch.setattr(torch.Tensor,'cpu',forbidden)
    event=run_failure(capture);artifact=event['failure_artifact'];manifest=json.loads((capture['receipts'].path.parent/artifact['manifest_file']).read_text())
    assert artifact['status']=='capture_error' and manifest['bundle'] is None
    assert 'CPU tensors' in manifest['capture_error']['message']


def test_tensor_attributes_cannot_smuggle_device_tensors_into_serialization(capture,monkeypatch):
    capture['truth']['inputs']['x'].device_reference=torch.empty(1,device='meta')
    def forbidden(*args,**kwargs):raise AssertionError('unsafe payload reached torch.save')
    monkeypatch.setattr(torch,'save',forbidden)
    artifact=run_failure(capture)['failure_artifact']
    assert artifact['status']=='capture_error' and artifact['bundle'] is None


def test_json_tensor_attributes_are_metadata_not_pickle_objects(capture):
    capture['truth']['inputs']['x'].is_shuffled=True
    artifact=run_failure(capture)['failure_artifact'];build=capture['receipts'].path.parent
    manifest=json.loads((build/artifact['manifest_file']).read_text())
    metadata=next(row for row in manifest['tensor_manifest'] if row['path']==['truth_inputs','x'])
    assert metadata['python_attributes']=={'is_shuffled':True}
    saved=torch.load(build/manifest['bundle']['file'],weights_only=True)
    assert saved['truth_inputs']['x'].__dict__=={}


def test_missing_expected_snapshot_is_explicitly_partial(capture):
    def verify(truth):
        actual=torch.ones(2);observed_inputs=truth['inputs']
        raise capture['failure']
    _,capture['check']=capture['receipts'].leg('prefill','native_production',lambda seed:capture['truth'],verify,case=capture['case'])
    event=run_failure(capture);artifact=event['failure_artifact'];manifest=json.loads((capture['receipts'].path.parent/artifact['manifest_file']).read_text())
    assert artifact['status']=='partial' and manifest['missing_snapshots']==['expected']
    saved=torch.load(capture['receipts'].path.parent/manifest['bundle']['file'],weights_only=True)
    assert saved['expected'] is None


@pytest.mark.parametrize('failure_site',['bundle','receipt'])
def test_persistence_io_failure_preserves_original_exception(capture,monkeypatch,failure_site):
    def broken(*args,**kwargs):raise OSError('disk is full')
    if failure_site=='bundle':monkeypatch.setattr(receipts_module,'save_cpu_failure',broken)
    else:
        original=capture['receipts'].emit
        def emit(event,**fields):
            if event=='verify_failure':raise OSError('receipt is unwritable')
            original(event,**fields)
        monkeypatch.setattr(capture['receipts'],'emit',emit)
    capture['reset'](512174569)
    with pytest.raises(AssertionError) as caught:capture['check'](capture['truth'])
    assert caught.value is capture['failure'] and caught.value.__notes__
    assert capture['calls']==['reset','verify']


def test_interrupt_does_not_start_large_tensor_capture(capture,monkeypatch):
    failure=TimeoutError('phase deadline')
    def verify(truth):raise failure
    _,checked=capture['receipts'].leg('prefill','native_production',lambda seed:capture['truth'],verify)
    def forbidden(*args,**kwargs):raise AssertionError('capture on timeout')
    monkeypatch.setattr(receipts_module,'save_cpu_failure',forbidden)
    with pytest.raises(TimeoutError) as caught:checked(capture['truth'])
    assert caught.value is failure


def test_declared_tensor_bundle_survives_real_copy_and_build_removal(capture):
    if shutil.which('rclone') is None:pytest.skip('rclone required for real diagnostic preservation')
    artifact=run_failure(capture)['failure_artifact'];build=capture['receipts'].path.parent
    (build/'unrelated.pt').write_bytes(b'not declared')
    preserved=capture['root']/'preserved';trusted.preserve_diagnostics(build,preserved);shutil.rmtree(build)
    manifest=json.loads((preserved/artifact['manifest_file']).read_text());bundle=preserved/manifest['bundle']['file']
    hashes=json.loads((preserved/'hashes.json').read_text())
    assert file_sha(bundle)==hashes[bundle.name]==manifest['bundle']['sha256']
    assert file_sha(preserved/artifact['manifest_file'])==artifact['manifest_sha256']
    assert torch.equal(torch.load(bundle,weights_only=True)['actual'],capture['snapshots']['actual'])
    assert not (preserved/'unrelated.pt').exists()


@pytest.mark.parametrize('mutation',['missing','content','path','oversize'])
def test_copier_rejects_invalid_declared_bundle(capture,mutation):
    if shutil.which('rclone') is None:pytest.skip('rclone required for real diagnostic preservation')
    artifact=run_failure(capture)['failure_artifact'];build=capture['receipts'].path.parent
    manifest_path=build/artifact['manifest_file'];manifest=json.loads(manifest_path.read_text());bundle=build/manifest['bundle']['file']
    if mutation=='missing':bundle.unlink()
    elif mutation=='content':
        raw=bytearray(bundle.read_bytes());raw[-1]^=1;bundle.write_bytes(raw)
    elif mutation=='path':manifest['bundle']['file']='../outside.pt'
    else:manifest['bundle']['bytes']=(2<<30)+1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):trusted.preserve_diagnostics(build,capture['root']/'preserved')
    assert (capture['root']/'preserved'/capture['receipts'].path.name).is_file()
    assert (capture['root']/'preserved'/artifact['manifest_file']).is_file()

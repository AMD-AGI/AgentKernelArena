"""CPU-only regressions for the observed MLA decode control distribution."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/deepseek-v4-pro__unified_paged_attention_decode"
_spec = importlib.util.spec_from_file_location("mla_decode_distribution", TASK / "ut/mla_decode_distribution.py")
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)
POLICY = json.loads((TASK / "ut/mla_control_distribution.json").read_text())
MANIFEST = json.loads((TASK / "cases.json").read_text())


def test_original_cases_policy_and_parent_anchor_are_preserved():
    policy = recipe.load_policy(TASK, MANIFEST)
    assert policy == POLICY
    assert len(MANIFEST['cases']) == 4
    assert recipe.fingerprint(MANIFEST['cases'][:3]) == POLICY['base_cases_sha256']
    assert MANIFEST['measurement']['warmup_iterations'] == 10
    assert MANIFEST['measurement']['benchmark_iterations'] == 100
    assert MANIFEST['measurement']['correctness_seeds'] == [42,43]
    assert MANIFEST['measurement']['negative_controls'] == ['no_op','wrong_output']
    assert POLICY['missing_actual_lengths'] == list(range(193,200))


@pytest.mark.parametrize('length', range(192,201))
def test_csr_repack_preserves_each_parent_row_and_full_padding(length):
    parent = list(range(12800)) + [-1]*(16384-12800)
    actual,ptr = recipe.controls_for_length(parent,[200*i for i in range(65)],length)
    assert len(actual)==16384 and ptr==[length*i for i in range(65)]
    for row in range(64):
        assert actual[ptr[row]:ptr[row+1]]==parent[200*row:200*row+length]
    assert actual[ptr[-1]:]==[-1]*(16384-ptr[-1])
    assert parent==list(range(12800))+[-1]*(16384-12800)


@pytest.mark.parametrize('bad', [191,201,0,True])
def test_unobserved_length_is_rejected(bad):
    with pytest.raises(ValueError):recipe.controls_for_length(list(range(12800))+[-1]*3584,[200*i for i in range(65)],bad)


def test_histogram_boundaries_use_exact_observed_weights():
    position = 0
    for row in POLICY['histogram']:
        for ticket in (position, position+row['weight']-1):
            fake=SimpleNamespace(digest=lambda ticket=ticket:ticket.to_bytes(32,'big'))
            with patch.object(recipe.hashlib,'sha256',return_value=fake):
                assert recipe.choose_length(123,POLICY['histogram'],'case')==row['length']
        position += row['weight']
    assert position == 253952


def test_same_private_challenge_matches_both_legs_without_forcing_domain():
    def draw(seed):return [recipe.choose_length(seed+i,POLICY['histogram'],'case') for i in range(110)]
    assert draw(725934)==draw(725934)
    assert draw(725934)!=draw(725935)
    assert set(draw(725934)) <= set(range(192,201))


def test_native_launch_geometry_and_scalar_controls_are_fixed_for_every_state():
    assert POLICY['max_length_scalar_argument'] is None
    assert set(POLICY['captured_public_bindings']) == {'q','unified_kv','kv_indices','kv_indptr','attn_sink','kv_scales','softmax_scale'}
    split,reduce=POLICY['source_supported_launches']
    assert split['grid']==[64,1,4] and reduce['grid']==[64,16,1]
    assert split['kwargs']['BLOCK_K']==16 and split['scalar_arguments']['KV_SPLITS']==4
    assert split['kwargs']['num_warps']==4 and split['kwargs']['num_stages']==2
    assert split['scalar_arguments']['q_stride_t']==32768
    assert all(row['native_launch_controls_identical'] for row in POLICY['per_state_source_work'])
    assert POLICY['per_state_source_work'][0]['live_tokens_by_split']==[48]*4
    assert [row['live_tokens_by_split'] for row in POLICY['per_state_source_work'][1:]]==[[64,64,64,n] for n in range(1,9)]


def test_dispatch_probe_rejects_changed_grid_kwargs_or_positional_stride():
    invoked=[]
    class Kernel:
        def __getitem__(self,grid):return lambda *args,**kwargs:invoked.append((args,kwargs))
    owner=SimpleNamespace(current_length=193,dispatch_counts=Counter())
    expected={'grid':[64,1,4],'kwargs':{'H':16},'argument_names':['pointer','stride'],'scalar_arguments':{'stride':512,'H':16}}
    probe=recipe.KernelProbe(Kernel(),'split',expected,owner,'reference')
    probe[(64,1,4)](object(),512,H=16)
    assert len(invoked)==1
    for grid,stride,h in [((64,1,1),512,16),((64,1,4),256,16),((64,1,4),512,32)]:
        with pytest.raises(ValueError):probe[grid](object(),stride,H=h)
    assert len(invoked)==1


def test_exhaustive_and_weighted_replay_proofs_reject_missing_work():
    obj=object.__new__(recipe.RuntimeRecipe);obj.policy=POLICY
    kernels=[x['kernel'] for x in POLICY['source_supported_launches']]
    obj.dispatch_counts=Counter({(leg,length,kernel):1 for leg in ('candidate','reference') for length in range(192,201) for kernel in kernels})
    obj.draws=[{'length':length,'forced':True} for length in range(192,201) for _ in range(4)]
    assert obj.proof('correctness',MANIFEST['measurement'])['exhaustive_lengths']==list(range(192,201))
    obj.draws.pop()
    with pytest.raises(ValueError):obj.proof('correctness',MANIFEST['measurement'])
    obj.draws=[{'length':recipe.choose_length(12345+i,POLICY['histogram'],'case'),'forced':False} for i in range(110)]
    proof=obj.proof('performance',MANIFEST['measurement'])
    assert proof['warmup_draws']==10 and proof['measured_draws']==100
    assert sum(proof['measured_histogram'].values())==100
    obj.draws.pop()
    with pytest.raises(ValueError):obj.proof('performance',MANIFEST['measurement'])


def test_canonical_checked_replays_keeps_110_resets_and_oracles_outside_100_measurements():
    path=TASK/'ut/evaluation_contract.py';spec=importlib.util.spec_from_file_location('distribution_contract',path)
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    events=[];case=MANIFEST['cases'][-1];state={}
    def reset(seed):
        state['length']=recipe.choose_length(seed,POLICY['histogram'],case['case_id']);events.append('reset');return state['length']
    def initialize():events.append('initialize')
    def replay():events.append('replay')
    def verify(truth):assert truth==state['length'];events.append('verify')
    def measure(call):events.append('measure_start');call();events.append('measure_stop');return 0.1
    result=contract.checked_replays(case,MANIFEST['measurement'],reset_inputs=reset,initialize_outputs=initialize,replay=replay,verify=verify,measure=measure,observe=lambda:case,seed=725934)
    counts=Counter(events)
    assert counts['reset']==counts['initialize']==counts['verify']==counts['replay']==110
    assert counts['measure_start']==counts['measure_stop']==len(result['samples_ms'])==100
    for i,event in enumerate(events):
        if event=='measure_start':assert events[i:i+4]==['measure_start','replay','measure_stop','verify']


def test_runtime_recipe_keeps_storage_aliases_and_checks_control_padding():
    torch=pytest.importorskip('torch')
    class Storage:
        def __init__(self,nbytes,pointer):self.size,self.pointer=nbytes,pointer
        def nbytes(self):return self.size
        def data_ptr(self):return self.pointer
    class MetadataTensor:
        def __init__(self,meta,pointer):
            self.shape=tuple(meta['shape']);self.dtype=meta['dtype'];self.meta=meta
            self.storage=Storage(meta['storage_nbytes'],pointer)
        def stride(self):return tuple(self.meta['stride'])
        def storage_offset(self):return self.meta['storage_offset']
        def untyped_storage(self):return self.storage
    class Kernel:
        def __getitem__(self,grid):return lambda *args,**kwargs:None
    policy=deepcopy(POLICY)
    inputs={name:MetadataTensor(meta,index+1) for index,(name,meta) in enumerate(policy['native_inputs'].items())}
    values={'kv_indices':list(range(12800))+[-1]*3584,'kv_indptr':[200*i for i in range(65)]}
    for name,data in values.items():
        inputs[name]=torch.tensor(data,dtype=torch.int32)
        policy['parent_controls_sha256'][name]=hashlib.sha256(struct.pack('<'+'i'*len(data),*data)).hexdigest()
    inputs.update(kv_scales=None,softmax_scale=policy['softmax_scale'])
    def module():return SimpleNamespace(**{row['kernel']:Kernel() for row in policy['source_supported_launches']})
    candidate,reference=module(),module();before={k:v.untyped_storage().data_ptr() for k,v in inputs.items() if hasattr(v,'untyped_storage')}
    runtime=recipe.RuntimeRecipe(MANIFEST['cases'][-1],policy,inputs,candidate,reference)
    try:
        runtime.apply(inputs,42,193)
        runtime.verify_controls(inputs)
        assert before=={k:v.untyped_storage().data_ptr() for k,v in inputs.items() if hasattr(v,'untyped_storage')}
        assert inputs['kv_indptr'].tolist()==[193*i for i in range(65)]
        inputs['kv_indices'][-1]=0
        with pytest.raises(ValueError,match='selected control values'):runtime.verify_controls(inputs)
        inputs['kv_indices'][-1]=-1
        inputs['q'].storage.size+=2
        with pytest.raises(ValueError,match='storage changed'):runtime.verify_controls(inputs)
        inputs['q'].storage.size-=2
        inputs['q'].storage.pointer=inputs['unified_kv'].storage.pointer
        with pytest.raises(ValueError,match='alias relationship'):runtime.verify_controls(inputs)
    finally:runtime.close()
    assert not isinstance(candidate._paged_decode_split_kernel,recipe.KernelProbe)


@pytest.mark.parametrize('change', ['base_case','measurement','fixture','status','recipe_hash'])
def test_distribution_admission_rejects_scope_or_identity_changes(change):
    manifest=deepcopy(MANIFEST)
    if change=='base_case':manifest['cases'][0]['occurrences']+=1
    if change=='measurement':manifest['measurement']['benchmark_iterations']=99
    if change=='fixture':manifest['cases'][-1]['fixture']['sha256']='0'*64
    if change=='status':manifest['status']='FROZEN_CURRENT_CAPTURE'
    if change=='recipe_hash':manifest['cases'][-1]['distribution_recipe']['sha256']='0'*64
    with pytest.raises(ValueError):recipe.load_policy(TASK,manifest)


def test_generated_package_passes_actual_trusted_admission():
    from src.tools.trusted_task_eval import package_contract
    from src.task_contract import fingerprint
    contract = package_contract(TASK)
    assert len(contract['manifest']['cases']) == 4
    assert contract['fixtures']['manifest']['case_manifest_fingerprint'] == fingerprint(contract['manifest'])
    assert len(contract['fixtures']['manifest']['assets']) == 19180

"""Materialize whole-MoE cases only from completed native served captures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import strict_json,validate_manifest
from fixture_codec import file_sha
from native_enums import enum_number

IMAGE='docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96'
REQUIRED={('prefill',8192),('decode',64)}
DECODE_BUCKETS={1,2,4,8,12,16,24,32,40,48,56,64}
SHAPES={'w1':[288,512,4096],'w2':[288,4096,256],'w1_scale':[288,4,32],'w2_scale':[288,32,2]}
DTYPES={'hidden_states':'bfloat16','w1':'float8_e4m3fn','w2':'float8_e4m3fn','topk_weight':'float32','topk_ids':'int32','w1_scale':'float32','w2_scale':'float32'}


def enum_value(value,kind=None):
    if isinstance(value,dict) and value.get('kind')=='enum':return enum_number(value,kind)
    return value


def validate_controls(controls):
    expected={'expert_mask':None,'activation':0,'quant_type':5,'doweight_stage1':False,'a1_scale':None,'a2_scale':None,
        'block_size_M':None,'num_local_tokens':None,'moe_sorting_dispatch_policy':0,'dtype':None,
        'hidden_pad':0,'intermediate_pad':0,'bias1':None,'bias2':None,'splitk':0,'swiglu_limit':10.0,
        'beta':None,'linear_beta':None,'gate_mode':'separated','shared_w1':None,'shared_w2':None,
        'shared_w1_scale':None,'shared_w2_scale':None,'shared_expert_id':-1,'stage2_scatter':None}
    if set(controls)!=set(expected)|{'tensor_attributes'}:raise ValueError('Whole-MoE control set differs from pinned public signature')
    for key,value in expected.items():
        actual=enum_value(controls[key],{'activation':'ActivationType','quant_type':'QuantType'}.get(key))
        if type(actual) is not type(value) or actual!=value:raise ValueError('Unsupported actual MoE control '+key+': '+repr(actual))
    for name in ('w1','w2'):
        attrs=controls['tensor_attributes'].get(name,{})
        if attrs.get('is_shuffled') is not True or attrs.get('is_guinterleave',False) or attrs.get('shuffle_layout',[16,16])!=[16,16]:raise ValueError('Actual packed weight layout does not match current separated (16,16) port')
    for name in ('w1_scale','w2_scale'):
        attrs=controls['tensor_attributes'].get(name,{})
        if attrs.get('is_shuffled',False) or attrs.get('is_guinterleave',False) or attrs.get('shuffle_layout') is not None:raise ValueError('Current FP8 block scales must remain plain row-major128x128 scales')


def validate_fixture(f,rank,native_sources):
    if f.get('schema')!='served-tensor-fixture-v1' or f['family']!='fmoe':raise ValueError('Expected served whole-MoE fixture')
    if f['source_sha256']!=native_sources['files']['aiter/fused_moe.py'] or f['provenance']['image']!=IMAGE or f['provenance']['run_id']!=rank['provenance']['run_id'] or f['served']['tp_rank']!=0:raise ValueError('MoE fixture source/run/rank differs from pinned capture')
    if f.get('startup_values') is not False or f['origin'] not in ('served_eager','served_graph'):raise ValueError('Only native served fixtures qualify')
    if set(f['inputs'])!=set(DTYPES) or set(f['outputs'])!={'result'}:raise ValueError('Unsupported tensor binding set')
    h=f['inputs']['hidden_states'];m=h['shape'][0];stage=f['served']['stage'];served=f['served']
    if stage=='prefill':
        if not 1<=m<=8192 or served['active_tokens']!=m:raise ValueError('Invalid actual prefill token geometry')
    elif stage=='decode':
        if not 1<=served['active_requests']<=served['active_tokens']<=m<=64:raise ValueError('Invalid actual decode token geometry')
    else:raise ValueError('Not a served inference stage')
    if not 1<=served['active_requests']<=64:raise ValueError('Outside the64-request workload')
    if f['origin']=='served_graph' and (stage!='decode' or m not in DECODE_BUCKETS or f['graph_bucket']!='decode-bs'+str(m)):raise ValueError('Wrong physical graph bucket')
    validate_controls(f['controls'])
    for name,dtype in DTYPES.items():
        meta=f['inputs'][name];expected_shape=SHAPES.get(name,[m,8 if name in ('topk_ids','topk_weight') else 4096])
        if meta['shape']!=expected_shape or meta['dtype'].removeprefix('torch.')!=dtype:raise ValueError('Actual '+name+' tensor differs from current operator contract')
        stride=1;expected_strides=[]
        for size in reversed(expected_shape):expected_strides.insert(0,stride);stride*=size
        if meta['stride']!=expected_strides:raise ValueError('Port requires captured contiguous inputs: '+name)
        # Tensor data_ptr already includes its storage offset. Preserve the actual
        # nonzero offset in the raw-storage reconstruction and observed contract.
        if type(meta['storage_offset']) is not int or meta['storage_offset']<0:raise ValueError('Invalid original input storage offset')
    result=f['outputs']['result']
    if result['shape']!=[m,4096] or result['stride']!=[4096,1] or result['storage_offset']!=0 or result['dtype'].removeprefix('torch.')!='bfloat16':raise ValueError('Unexpected native output ABI')
    return stage,m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture_manifest');args=ap.parse_args()
    manifest_path=Path(args.capture_manifest).resolve();capture_root=manifest_path.parent;rank=strict_json(manifest_path.read_text())
    if rank.get('schema')!='served-capture-rank-manifest-v1' or not rank.get('complete') or not rank.get('sealed') or rank.get('failures') or rank.get('additional_failure_count',0):raise ValueError('Native capture manifest is incomplete')
    if rank.get('checkpoint_reason') not in ('profile_stop','native_stop_profile') or not rank.get('required_cases_supplied'):raise ValueError('Native explicit stop receipt required')
    if rank['provenance']['image']!=IMAGE or rank['provenance']['tp_rank']!=0:raise ValueError('Wrong image or capture rank')
    if set(rank['required_cases'])!=set(rank['case_schemas']) or set(rank['runtime_case_counts'])!=set(rank['case_schemas']):raise ValueError('Capture case inventory is incomplete')
    native_sources=strict_json((ROOT/'provenance/NATIVE-SOURCES.json').read_text())
    selected={};observed=set()
    for key,ref in sorted(rank['cases'].items()):
        p=(capture_root/ref['path']).resolve()
        if not p.is_relative_to(capture_root) or file_sha(p)!=ref['sha256']:raise ValueError('Fixture reference mismatch')
        f=strict_json(p.read_text())
        if f['family']!='fmoe':continue
        if key!=f['case_key'] or rank['case_schemas'][key]['family']!='fmoe':raise ValueError('Fixture identity mismatch')
        bucket=validate_fixture(f,rank,native_sources)
        count=rank['runtime_case_counts'][key]
        if type(count) is not int or count<=0:raise ValueError('Missing actual MoE occurrence count')
        selected[key]=(p,f,bucket,count);observed.add(bucket)
    expected_keys={key for key,schema in rank['case_schemas'].items() if schema['family']=='fmoe'}
    if set(selected)!=expected_keys:raise ValueError('A notified MoE case has no selected real fixture')
    if not REQUIRED<=observed:raise ValueError('Missing actual core whole-MoE stages: '+str(REQUIRED-observed))
    destination=ROOT/'fixtures';destination.mkdir(exist_ok=True);cases=[]
    for key,(path,f,bucket,count) in sorted(selected.items()):
        case_id='whole-moe-'+bucket[0]+'-m'+str(bucket[1])+'-'+key.removeprefix('fmoe-');fixture_path=destination/(case_id+'.json')
        subprocess.run(['rclone','copyto','--transfers','64000','--progress',str(path),str(fixture_path)],check=True)
        for phase in f['payload'].values():
            for group in phase.values():
                for segment in group['segments']:
                    src=(capture_root/segment['blob']).resolve();dst=destination/segment['blob']
                    if not src.is_relative_to(capture_root) or file_sha(src)!=segment['sha256']:raise ValueError('Blob escapes capture or changed')
                    if not dst.exists():subprocess.run(['rclone','copyto','--transfers','64000','--progress',str(src),str(dst)],check=True)
                    if file_sha(dst)!=segment['sha256']:raise ValueError('Staged blob checksum mismatch')
        tensors={}
        for phase,role in [('inputs','input'),('outputs','output')]:
            for name,meta in f[phase].items():
                tensors[name]={'role':role,'shape':meta['shape'],'strides':meta['stride'],'storage_offset':meta['storage_offset'],'dtype':meta['dtype'].removeprefix('torch.'),'device_type':'cuda'}
        controls=dict(f['controls']);controls['port_launch']={'BM':32,'BN':64,'BK':128,'input_quant_block':128,'intermediate_quant_block':128,'output_reduce_block':256}
        cases.append({'case_id':case_id,'occurrences':count,'calls_per_sample':1,'scalars':controls,'tensors':tensors,
            'fixture':{'path':'fixtures/'+fixture_path.name,'sha256':file_sha(fixture_path),'native_capture_run':f['provenance']['run_id']},
            'served_geometry':f['served'],
            'frequency_evidence':{'run_id':rank['provenance']['run_id'],'case_key':key,'scope':'actual per-rank notifications across the complete root-marked64-request workload; includes after Torch profiler auto-stop','operator':'aiter.fused_moe:fused_moe'},
            'original_profile_evidence':{'run_id':'glm-flash-fp8-native-jit-194292','observed_core_counts_per_rank':{'prefill_m8192':336,'decode_m64_one_linked_graph':42},'timing_from_instrumented_capture_used':False}})
    manifest=validate_manifest({'schema_version':1,'runtime_image':IMAGE,'source_run':rank['provenance']['run_id'],'cases':cases,
        'measurement':{'method':'cuda_graph','warmup_iterations':10,'benchmark_iterations':100,'correctness_seeds':[0,1],
            'refresh_inputs':'each_replay','initialize_outputs':'each_replay','validate_outputs':'each_replay','negative_controls':['no_op','wrong_output']},
        'refresh_semantics':'Every replay permutes real routing rows and creates new numerical hidden activations using fresh per-row multipliers and per-element noise around the captured distribution. CPU input truth is fixed before candidate execution; a native reference is computed only after candidate output/input snapshots are on CPU.',
        'coverage':'Every notified whole-MoE case has its own fixture-backed mandatory case; core prefill8192/decode64 required, tail and alternate valid structural cases retained.'})
    (ROOT/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'NOT_BUILT').unlink(missing_ok=True)
    print(str(len(cases))+' actual whole-MoE cases materialized; GPU correctness and framework qualification still required')
if __name__=='__main__':main()

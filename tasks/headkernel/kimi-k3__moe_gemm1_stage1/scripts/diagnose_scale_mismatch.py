"""Targeted, non-scoreable eager replay diagnostic; preserves exact task checks."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import canonical,strict_json
from runtime_adapter import Prepared,create_runtime,file_sha,raw_storage,scale_offset


def output_statistics(prepared,actual,expected):
    t=prepared.torch
    a=actual['return_scale'].view(t.uint8).reshape(-1)[prepared.live_scale]
    b=expected['return_scale'].view(t.uint8).reshape(-1)[prepared.live_scale]
    bad=(a!=b).nonzero().flatten()
    aq=actual['return'].view(t.uint8).reshape(-1,384).index_select(0,prepared.live_payload)
    bq=expected['return'].view(t.uint8).reshape(-1,384).index_select(0,prepared.live_payload)
    samples=[]
    ids=prepared.inputs['sorted_token_ids'].cpu().tolist()
    experts=prepared.inputs['sorted_expert_ids'].cpu().tolist()
    for index in bad[:64].tolist():
        route,column=divmod(index,12);row=int(prepared.live_sorted[route]);fused=ids[row]
        samples.append({'sorted_row':row,'payload_row':int(prepared.live_payload[route]),
            'scale_column':column,'physical_byte_offset':scale_offset(row,column,16),
            'actual_byte':int(a[index]),'reference_byte':int(b[index]),
            'token':fused&0xffffff,'slot':fused>>24,'expert':experts[row//prepared.controls['tile_m']]})
    independently_indexed=[scale_offset(int(row),column,16) for row in prepared.live_sorted for column in range(12)]
    if independently_indexed!=prepared.live_scale.tolist():raise AssertionError('Vector and scalar physical scale indices disagree')
    dequant_max=None
    if not bool((a==255).any()) and not bool((b==255).any()):
        av=aq.view(t.float8_e4m3fn).float().reshape(-1,12,32)
        bv=bq.view(t.float8_e4m3fn).float().reshape(-1,12,32)
        av=av*t.exp2(a.float().reshape(-1,12,1)-127)
        bv=bv*t.exp2(b.float().reshape(-1,12,1)-127)
        difference=(av-bv).abs()
        if bool(t.isfinite(difference).all()):dequant_max=float(difference.max())
    return {'live_scale_mismatches':bad.numel(),'live_payload_byte_mismatches':int((aq!=bq).sum()),
            'live_scale_count':a.numel(),'live_payload_byte_count':aq.numel(),
            'scalar_scale_index_check':True,'dequantized_payload_max_abs_difference_diagnostic_only':dequant_max,
            'first_scale_mismatches':samples}


def tensor_attributes(values,torch):
    result={}
    for name,value in values.items():
        if value is None:continue
        attributes={}
        for key,item in getattr(value,'__dict__',{}).items():
            if torch.is_tensor(item):
                attributes[key]={'tensor_shape':list(item.shape),'dtype':str(item.dtype),'device':str(item.device)}
            elif item is None or type(item) in (str,int,float,bool):attributes[key]=item
            elif isinstance(item,(tuple,list)) and all(type(v) in (str,int,float,bool) for v in item):attributes[key]=list(item)
            else:attributes[key]={'type':type(item).__module__+'.'+type(item).__qualname__}
        result[name]=attributes
    return result


class Writer:
    def __init__(self,directory,torch):
        self.directory=directory;self.torch=torch;(directory/'blobs').mkdir()
    def tensor(self,value):
        if value.device.type!='cpu':raise AssertionError('Only CPU snapshots may be written as diagnostic truth')
        raw=value.contiguous().view(self.torch.uint8).reshape(-1)
        data=memoryview(raw.numpy());digest=hashlib.sha256(data).hexdigest()
        path=self.directory/'blobs'/(digest+'.bin')
        if not path.exists():
            with path.open('xb') as stream:
                for start in range(0,len(data),8<<20):stream.write(data[start:start+(8<<20)])
        return {'blob':'blobs/'+path.name,'sha256':digest,'bytes':len(data),
                'dtype':str(value.dtype),'shape':list(value.shape),'stride':list(value.stride())}
    def tree(self,value):
        if self.torch.is_tensor(value):return self.tensor(value)
        if isinstance(value,dict):return {key:self.tree(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [self.tree(item) for item in value]
        return value


@contextmanager
def allocation_poison(prepared,value):
    """Optional diagnostic only: identify unwritten output bytes without changing ABI."""
    if value is None:
        yield
        return
    t=prepared.torch;original=t.empty
    qshape=tuple(prepared.record['outputs']['return']['shape'])
    scale_bytes=prepared.record['outputs']['return_scale']['storage_nbytes']
    def empty(*args,**kwargs):
        tensor=original(*args,**kwargs)
        if tensor.device.type=='cuda' and (
            tuple(tensor.shape)==qshape and tensor.dtype==t.float8_e4m3fn
            or tensor.dtype==t.uint8 and tensor.ndim==1 and tensor.numel()==scale_bytes):
            raw_storage(tensor,t).fill_(value)
        return tensor
    t.empty=empty
    try:yield
    finally:t.empty=original


def run(args):
    task=Path(args.task_root).resolve()
    if (task/'validation_report.yaml').exists():
        raise ValueError('Refusing a validator result snapshot; prepare a new materialized task copy')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    (task/'build').mkdir(exist_ok=True)
    manifest=strict_json((task/'cases.json').read_text())
    case=next(copy.deepcopy(case) for case in manifest['cases'] if case['tensors']['inputs.a']['shape'][0]==args.tokens)
    original_histogram=case['work_distribution']['valid_rows_histogram']
    if args.valid_rows is not None:
        if args.valid_rows not in {row for row,_ in original_histogram}:raise ValueError('Diagnostic work amount was not observed in this case')
        case['work_distribution']['valid_rows_histogram']=[[args.valid_rows,1]]
    policy=strict_json((task/'ut/source_guard_policy.json').read_text())
    request={'request_id':'scale-diagnostic-'+secrets.token_hex(12),
             'source_sha256':{name:file_sha(task/name) for name in policy['sources']},'challenge_seed':args.seed}
    report={'schema':'kimi-stage1-scale-diagnostic-v1','scoreable':False,'seed':args.seed,
            'forced_valid_rows':args.valid_rows,'case_id':case['case_id'],'request':request,
            'case_manifest_sha256':file_sha(task/'cases.json'),'trials':[],
            'allocation_poison':args.poison_allocations,'graph_capture':False,
            'math_and_comparison_policy':'unchanged; exact live E8M0 bytes and original0.02 mixed RMS payload tolerance'}
    (output/'case.json').write_text(canonical(case)+'\n')
    runtime=create_runtime(task,manifest,request);t=runtime.torch;writer=Writer(output,t)
    initial_compare=Prepared.compare
    def capture_initial_failure(self,actual,expected):
        try:return initial_compare(self,actual,expected)
        except AssertionError as error:
            report.update(status='initialization_failure',error=str(error),
                actual=writer.tree(actual),reference=writer.tree(expected),input_truth=writer.tree(self.pristine))
            (output/'DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
            raise
    Prepared.compare=capture_initial_failure
    try:prepared=runtime.prepare_case(case)
    finally:Prepared.compare=initial_compare
    prepared.callbacks._compare=initial_compare.__get__(prepared,Prepared)
    report['captured_tensor_attribute_codec_id']=prepared.record.get('tensor_attribute_codec_id')
    report['captured_tensor_attributes']={phase+'.'+name:meta.get('attributes',{})
        for phase in ('inputs','outputs') for name,meta in prepared.record[phase].items() if meta is not None}
    report['native_controls']=prepared.controls
    report['source_closure_equal']={name:(task/name).read_bytes()==(task/policy['sources'][name]['reference']).read_bytes()
        for name in policy['sources']}
    original_reference=prepared.callbacks._reference;original_compare=prepared.callbacks._compare
    original_immutable=prepared.callbacks._immutable;original_replay=prepared.callbacks.replay
    state={};initial=None;first_input_truth=None
    def immutable(after,before):
        result=original_immutable(after,before)
        state['last_post_input']=after
        return result
    def reference(truth):
        state['reference_truth']=truth
        with allocation_poison(prepared,0x5a if args.poison_allocations else None):
            result=original_reference(truth)
        state['post_reference_inputs']={name:raw_storage(value,t).to(device='cpu',copy=True)
            for name,value in prepared.reference_inputs.items() if value is not None}
        return result
    def replay():
        with allocation_poison(prepared,0xa5 if args.poison_allocations else None):return original_replay()
    def compare(actual,expected):
        state['actual']=actual;state['reference']=expected
        state['input_truth']=prepared.callbacks._truth.value
        try:state['comparison']=output_statistics(prepared,actual,expected)
        except Exception as error:state['comparison']={'statistics_error':str(error)}
        return original_compare(actual,expected)
    prepared.callbacks._immutable=immutable;prepared.callbacks._reference=reference
    prepared.callbacks._compare=compare;prepared.callbacks.replay=replay
    for trial in range(args.repeats):
        state.clear();item={'trial':trial,'seed':args.seed}
        try:
            prepared.callbacks.check_once(args.seed)
            item['task_check']='pass'
        except AssertionError as error:
            item['task_check']='fail';item['error']=str(error)
        item['work_rows']=prepared.work_rows
        item['candidate_tensor_attributes']=tensor_attributes(prepared.inputs,t)
        item['reference_tensor_attributes']=tensor_attributes(prepared.reference_inputs,t)
        item['tensor_attributes_equal']=item['candidate_tensor_attributes']==item['reference_tensor_attributes']
        item.update(state.get('comparison',{}))
        for name in ('actual','reference','input_truth','last_post_input','reference_truth','post_reference_inputs'):
            if name in state:item[name]=writer.tree(state[name])
        if 'input_truth' in item:
            if first_input_truth is None:first_input_truth=item['input_truth']
            item['identical_input_truth_to_first_trial']=item['input_truth']==first_input_truth
            item['candidate_input_bytes_match_truth_after_call']=item.get('last_post_input')==item['input_truth']
            item['reference_input_bytes_match_truth_after_call']=item.get('post_reference_inputs')==item['input_truth']
        item['live_sorted_rows']=writer.tensor(prepared.live_sorted)
        item['live_payload_rows']=writer.tensor(prepared.live_payload)
        item['live_scale_offsets']=writer.tensor(prepared.live_scale)
        item['candidate_input_pointers']={name:value.data_ptr() for name,value in prepared.inputs.items() if value is not None}
        item['reference_input_pointers']={name:value.data_ptr() for name,value in prepared.reference_inputs.items() if value is not None}
        if 'actual' in state:
            if initial is None:initial={'actual':state['actual'],'reference':state['reference']}
            else:
                item['candidate_vs_first_candidate']=output_statistics(prepared,state['actual'],initial['actual'])
                item['reference_vs_first_reference']=output_statistics(prepared,state['reference'],initial['reference'])
        report['trials'].append(item)
        (output/'DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='mismatch_reproduced' if any(row['task_check']=='fail' for row in report['trials']) else 'no_mismatch_observed'
    report['native_engagement']=prepared.native_engagement()
    (output/'DIAGNOSTIC.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'trials':len(report['trials']),
                      'scale_mismatches':[row.get('live_scale_mismatches') for row in report['trials']],
                      'output':str(output)},indent=2))
    return 1 if report['status']=='mismatch_reproduced' else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root',default=str(ROOT));parser.add_argument('--output',required=True)
    parser.add_argument('--tokens',type=int,choices=(64,8192,16384),default=64)
    parser.add_argument('--seed',type=int,default=734170854)
    parser.add_argument('--valid-rows',type=int,default=13824)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--poison-allocations',action='store_true')
    args=parser.parse_args()
    if not 1<=args.repeats<=16:parser.error('--repeats must be between 1 and 16')
    raise SystemExit(run(args))

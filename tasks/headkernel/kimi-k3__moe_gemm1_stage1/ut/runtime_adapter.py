"""Frozen current-capture A8W4 stage-1 adapter; no GPU golden precedes a candidate."""
from array import array
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys

from evaluation_contract import canonical, observe_case, strict_json
from fresh_runner import FreshCallbacks, clear_device_reference, cpu_copy

IMAGE = 'docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96'
NATIVE_SOURCE = 'd08339bc94dfabd3529d415857e49d0383f9efe2ef62be9a84916c55c5857e68'
FAMILY = 'flydsl_moe_stage1'
DTYPES = {'a':'float8_e4m3fn','a1_scale':'float8_e8m0fnu','num_valid_ids':'int32',
          'sorted_expert_ids':'int32','sorted_token_ids':'int32','topk_ids':'int32',
          'w1':'float4_e2m1fn_x2','w1_scale':'float8_e8m0fnu'}
NONE_INPUTS = ['bias','out','sorted_weights']
FIXED_CONTROLS = {'a_dtype':'fp8','a_scale_one':False,'act':'situv2','b_dtype':'fp4',
    'gate_mode':'interleave','inter_dim_pad':0,'k_batch':1,'k_batch_intra_block':None,
    'k_wave':1,'model_dim_pad':0,'out_dtype':'fp8','out_supplied':False,'persist_m':0,
    'situ_beta':4.0,'situ_linear_beta':25.0,'swiglu_limit':None,'tile_k':256,'tile_n':128,
    'topk':16,'use_async_copy':True}


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):result.update(part)
    return result.hexdigest()


def checked_path(root, relative, expected=None):
    path=Path(root)/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Fixture/source must be a regular task-local file: '+str(relative))
    if expected is not None and file_sha(path)!=expected:raise ValueError('Fixture/source hash differs: '+str(relative))
    return path


def validate_tensor_attributes(record):
    if record.get('tensor_attribute_codec_id') is not None:
        raise ValueError('Stage-1 captured ABI has no tensor-attribute codec')
    for phase in ('inputs','outputs'):
        for name,meta in record[phase].items():
            if meta is not None and meta.get('attributes'):
                raise ValueError('Unexpected captured tensor attributes: '+phase+'.'+name)


def validate_fixture(record):
    if (record.get('schema')!='served-tensor-fixture-v1' or record.get('family')!=FAMILY
            or record.get('source_sha256')!=NATIVE_SOURCE or record.get('startup_values') is not False
            or record['provenance']['image']!=IMAGE or record['provenance']['run_id']!='kimi-served-194550'):
        raise ValueError('Not an exact current served stage-1 fixture')
    validate_tensor_attributes(record)
    if set(record['inputs'])!=set(DTYPES)|set(NONE_INPUTS) or any(record['inputs'][name] is not None for name in NONE_INPUTS):
        raise ValueError('Native input binding/nullness differs')
    controls=record['controls']
    for key,value in FIXED_CONTROLS.items():
        if type(controls.get(key)) is not type(value) or controls[key]!=value:raise ValueError('Native control differs: '+key)
    m=record['inputs']['a']['shape'][0]
    if m not in (64,8192,16384):raise ValueError('Unobserved stage-1 token extent')
    expected={'tile_m':32 if m==64 else 64,'b_nt':2 if m==64 else 0,
              'v2_output_layout':m!=64,'waves_per_eu':4 if m==64 else 2,'xcd_swizzle':0 if m==64 else 4}
    if set(controls)!=set(FIXED_CONTROLS)|set(expected):raise ValueError('Unexpected native launch control')
    for key,value in expected.items():
        if type(controls[key]) is not type(value) or controls[key]!=value:raise ValueError('Launch geometry differs: '+key)
    if record['served']['active_tokens']!=m or record['served']['stage']!=('decode' if m==64 else 'prefill'):
        raise ValueError('Served token/stage binding differs')
    if record['origin']!=('served_graph' if m==64 else 'served_eager'):raise ValueError('Wrong served invocation origin')
    if record['inputs']['w1']['shape']!=[896,768,1792] or record['inputs']['w1_scale']['shape']!=[688128,112]:
        raise ValueError('The native packed FP4 weight/scales layout differs')
    aliases=[]
    for phase in ('inputs','outputs'):
        for name,meta in record[phase].items():
            if meta is None:continue
            if meta['storage_offset']!=0:raise ValueError('Unobserved nonzero storage offset')
            step=1;strides=[]
            for n in reversed(meta['shape']):strides.insert(0,step);step*=n
            if meta['stride']!=strides or meta['storage_nbytes']!=step*meta['element_size']:
                raise ValueError('Unexpected captured physical storage: '+name)
            if phase=='inputs' and meta['dtype']!='torch.'+DTYPES[name]:raise ValueError('Native dtype differs: '+name)
            aliases.append(meta['alias'])
            group=record['payload'][phase][meta['alias']]
            if (group['storage_nbytes']!=meta['storage_nbytes'] or len(group['segments'])!=1
                    or group['segments'][0]['offset_bytes']!=0 or group['segments'][0]['bytes']!=meta['storage_nbytes']):
                raise ValueError('Stage-1 fixture must cover every physical storage byte')
    if len(set(aliases))!=len(aliases):raise ValueError('Unobserved input/output alias')
    if set(record['outputs'])!={'out','return','return_scale'} or record['outputs']['out'] is not None:
        raise ValueError('Native return binding differs')
    if record['outputs']['return']['dtype']!='torch.float8_e4m3fn' or record['outputs']['return_scale']['dtype']!='torch.float8_e8m0fnu':
        raise ValueError('Native quantized return dtypes differ')
    rows=record['inputs']['sorted_expert_ids']['shape'][0]*controls['tile_m']
    if record['outputs']['return']['shape']!=([m,16,384] if m==64 else [rows,384]):raise ValueError('Output row layout differs')
    if record['outputs']['return_scale']['shape']!=[((rows+255)//256)*256,16]:raise ValueError('Output scale allocation differs')
    if record['inputs']['a1_scale']['shape']!=[((rows+255)//256)*256,112]:raise ValueError('Activation scale allocation differs')
    return m


def scale_offset(row, column, padded_columns):
    """AITER's native 32-row/8-column E8M0 physical byte permutation."""
    return ((row//32)*(padded_columns*32)+(column//8)*256+(column%4)*64
            +(row%16)*4+((column//4)%2)*2+((row//16)%2))


def weighted_work(histogram, rng):
    value=rng.randrange(sum(count for _,count in histogram))
    for rows,count in histogram:
        if value<count:return rows
        value-=count
    raise AssertionError('Invalid observed work histogram')


def make_routes(tokens, tile_m, valid_rows, capacity_rows, expert_slots, seed):
    """Fresh legal top-16 assignments with exactly an observed padded-work count.

    Every expert receives ceil(count/tile_m) sorted blocks, with no artificial
    empty blocks. Cyclic assignment makes each token appear exactly 16 times
    and prevents duplicate experts within a token.
    """
    if valid_rows%tile_m or valid_rows>capacity_rows:raise ValueError('Invalid observed sorted-row extent')
    rng=random.Random(seed); routes=tokens*16; blocks=valid_rows//tile_m
    active=min(896,blocks,routes)
    block_counts=[blocks//active+(i<blocks%active) for i in range(active)]
    counts=[(n-1)*tile_m+1 for n in block_counts]
    remaining=routes-sum(counts)
    if remaining<0 or remaining>active*(tile_m-1):raise ValueError('Observed work count cannot represent the frozen routing')
    order=list(range(active));rng.shuffle(order)
    for position,index in enumerate(order):
        following=len(order)-position-1
        low=max(0,remaining-following*(tile_m-1));high=min(tile_m-1,remaining)
        addition=rng.randint(low,high)
        counts[index]+=addition;remaining-=addition
    if remaining or max(counts)>tokens:raise ValueError('Route degrees do not fit distinct per-token experts')
    expert_ids=list(range(896));rng.shuffle(expert_ids)
    token_order=list(range(tokens));rng.shuffle(token_order)
    topk=array('i',[-1])*(tokens*16);sorted_ids=array('i',[16<<24])*capacity_rows
    sorted_experts=array('i',[-1])*expert_slots;slots=[0]*tokens
    row=0;cursor=0
    for expert,count,block_count in zip(expert_ids,counts,block_counts):
        for j in range(count):
            token=token_order[(cursor+j)%tokens];slot=slots[token];slots[token]+=1
            topk[token*16+slot]=expert
            sorted_ids[row+j]=(slot<<24)|token
        for block in range(block_count):sorted_experts[row//tile_m+block]=expert
        cursor+=count;row+=block_count*tile_m
    if row!=valid_rows or any(n!=16 for n in slots):raise AssertionError('Fresh routing did not preserve exact native work')
    return {'sorted_token_ids':sorted_ids,'sorted_expert_ids':sorted_experts,'topk_ids':topk,
            'num_valid_ids':array('i',[valid_rows,tokens])}


def raw_storage(tensor, torch):
    storage=tensor.untyped_storage()
    return torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(storage,0,(storage.nbytes(),),(1,))


def restore_phase(root, record, phase, torch):
    groups={}
    for alias,group in record['payload'][phase].items():
        data=torch.empty(group['storage_nbytes'],dtype=torch.uint8,device='cpu')
        for segment in group['segments']:
            path=checked_path(root,segment['blob'],segment['sha256'])
            offset=segment['offset_bytes'];written=0
            with path.open('rb') as stream:
                for part in iter(lambda:stream.read(8<<20),b''):
                    buffer=torch.frombuffer(bytearray(part),dtype=torch.uint8)
                    data[offset+written:offset+written+len(part)].copy_(buffer);written+=len(part)
            if written!=segment['bytes']:raise ValueError('Fixture byte size differs')
        groups[alias]=data
    result={}
    for name,meta in record[phase].items():
        if meta is None:result[name]=None;continue
        dtype=getattr(torch,meta['dtype'].removeprefix('torch.'))
        result[name]=torch.empty(0,dtype=dtype).set_(groups[meta['alias']].untyped_storage(),
            meta['storage_offset'],tuple(meta['shape']),tuple(meta['stride']))
    return result


def load_package(root, tag):
    spec=importlib.util.spec_from_file_location(tag,root/'__init__.py',submodule_search_locations=[str(root)])
    package=importlib.util.module_from_spec(spec);sys.modules[tag]=package;spec.loader.exec_module(package)
    function=package.flydsl_moe_stage1
    if not Path(inspect.getsourcefile(function)).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Native callable escaped its private source closure')
    return package,function


def resolve_reference_bindings(config,policy,provenance):
    """Use one frozen closure for the native runtime and generic evaluator."""
    reference_root=policy.get('native_reference_root')
    if reference_root!='ut/baseline_src/flydsl' or provenance.get('native_reference_root')!=reference_root:
        raise ValueError('The native reference root differs from the frozen source policy')
    sources=config.get('source_file_path',[])
    if len(sources)!=len(set(sources)) or set(sources)!=set(policy['sources']):
        raise ValueError('Configured candidate sources differ from the frozen source policy')
    expected={name:(Path(reference_root)/Path(name).relative_to('source/flydsl')).as_posix() for name in sources}
    configured=config.get('trusted_evaluation',{}).get('reference_sources')
    guarded={name:entry['reference'] for name,entry in policy['sources'].items()}
    if configured!=expected or guarded!=expected:
        raise ValueError('Runtime, guard and generic evaluator must share the same frozen reference files')
    return reference_root,expected


def validate_sync_repairs(root,provenance,references):
    """Check the exact local patch while retaining original image-source pins."""
    entries={entry['file']:entry for entry in provenance['files']}
    if len(entries)!=len(provenance['files']):raise ValueError('Duplicate source provenance entries')
    repairs=provenance.get('synchronization_repairs',[])
    if not repairs or provenance.get('baseline_kind')!='synchronization_repaired_supplied_reference':
        raise ValueError('The synchronized reference requires an explicit source-pinned repair')
    seen=set()
    for repair in repairs:
        name=repair['file'];entry=entries[name]
        if repair['id'] in seen or repair.get('occurrences')!=1:raise ValueError('Duplicate or ambiguous synchronization repair')
        seen.add(repair['id'])
        if (repair['reference']!=references.get(name)
                or repair['image_source_sha256']!=entry['source_sha256']
                or repair['projected_sha256_after']!=entry['sha256']
                or entry.get('synchronization_repair_id')!=repair['id']):
            raise ValueError('Synchronization repair does not match the image/reference source mapping')
        path=checked_path(root,repair['reference'],repair['projected_sha256_after'])
        patched=path.read_text()
        if patched.count(repair['after_context'])!=1:raise ValueError('Synchronization repair context changed')
        restored=patched.replace(repair['after_context'],repair['before_context'],1)
        if hashlib.sha256(restored.encode()).hexdigest()!=repair['projected_sha256_before']:
            raise ValueError('Frozen reference contains changes beyond the declared synchronization repair')
        evidence=strict_json(checked_path(root,repair['evidence']).read_text())
        if (evidence.get('repair_id')!=repair['id']
                or evidence.get('projected_sha256_before')!=repair['projected_sha256_before']
                or evidence.get('projected_sha256_after')!=repair['projected_sha256_after']):
            raise ValueError('Synchronization repair evidence identifies different source bytes')
    return repairs


def reference_contract(root):
    import yaml
    config=yaml.safe_load((root/'config.yaml').read_text())
    policy=strict_json((root/'ut/source_guard_policy.json').read_text())
    provenance=strict_json((root/'SOURCE-PROVENANCE.json').read_text())
    relative,references=resolve_reference_bindings(config,policy,provenance)
    validate_sync_repairs(root,provenance,references)
    reference_root=root/relative
    if reference_root.is_symlink() or reference_root.resolve()!=reference_root.absolute():
        raise ValueError('Frozen reference closure must be a regular task-local directory')
    return reference_root,provenance


class Runtime:
    def __init__(self,root,manifest,request):
        reference_root,provenance=reference_contract(root)
        # Resolve Python's standard unittest package independently of task
        # module search paths while Torch/FlyDSL import their dependencies.
        previous_path=list(sys.path)
        sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=(root/'ut').resolve()]
        try:
            import unittest.mock
            import torch
            import aiter.ops.flydsl.moe_kernels as installed
        finally:
            sys.path[:]=previous_path
        self.torch=torch;self.root=root;self.manifest=manifest;self.request=request
        if file_sha(Path(installed.__file__))!=NATIVE_SOURCE:raise RuntimeError('Installed AITER source differs from the served capture')
        editable=set(request['source_sha256']); installed_root=Path(installed.__file__).parent
        for entry in provenance['files']:
            relative=Path(entry['file']).relative_to('source/flydsl')
            if file_sha(installed_root/relative)!=entry['source_sha256']:raise RuntimeError('Installed FlyDSL dependency differs: '+str(relative))
            checked_path(root,str(reference_root.relative_to(root)/relative),entry['sha256'])
            if entry['file'] not in editable:checked_path(root,entry['file'],entry['sha256'])
        identity=hashlib.sha256(canonical(request['source_sha256']).encode()).hexdigest()[:16]
        self.candidate_package,self.candidate=load_package(root/'source/flydsl','_kimi_candidate_'+identity)
        self.reference_package,self.reference=load_package(reference_root,'_kimi_reference_'+identity)
        self.identity=identity;self.reference_root=reference_root
        self.baseline_kind=provenance['baseline_kind']
    def prepare_case(self,case):return Prepared(self,case)


class Prepared:
    def __init__(self,runtime,case):
        self.runtime=runtime;self.root=runtime.root;self.torch=runtime.torch;self.case=case
        path=checked_path(self.root,case['fixture']['path'],case['fixture']['sha256'])
        self.record=strict_json(path.read_text());self.tokens=validate_fixture(self.record)
        self.controls=dict(self.record['controls']);self.controls.pop('out_supplied')
        self.pristine=restore_phase(path.parent,self.record,'inputs',self.torch)
        self.golden=restore_phase(path.parent,self.record,'outputs',self.torch)
        self.inputs=self._allocate_inputs();self.reference_inputs=self._allocate_inputs()
        self.output=None;self.warmed={'candidate':False,'reference':False};self.calls=Counter()
        self.draws=[];self.poisoned=set();self.current_seed=None
        self.base_a=self.pristine['a'].view(self.torch.uint8).to('cuda')
        self.base_w=self.pristine['w1'].view(self.torch.uint8).to('cuda')
        self._recover_token_scales()
        self.callbacks=FreshCallbacks(refresh_inputs=self.refresh,initialize_outputs=self.initialize_outputs,
            snapshot_inputs=self.snapshot_inputs,snapshot_outputs=lambda:self.output,
            validate_metadata=self.validate_metadata,assert_immutable=self.assert_immutable,
            reference=self.reference,compare=self.compare,replay=self.invoke_candidate)
        # Bind both private implementations to actual captured values before any
        # generated trials. Golden fixture tensors remain exclusively on CPU.
        self._restore_original(self.inputs)
        before=cpu_copy(self.snapshot_inputs(),self.torch)
        self.invoke_candidate();self.torch.cuda.synchronize();self.validate_metadata()
        actual=cpu_copy(self.output,self.torch);self.assert_immutable(cpu_copy(self.snapshot_inputs(),self.torch),before)
        self.compare(actual,self.golden)
        expected=self.reference(before);expected_cpu=cpu_copy(expected,self.torch)
        clear_device_reference(expected,self.torch);self.torch.cuda.synchronize()
        self.compare(expected_cpu,self.golden);self.compare(actual,expected_cpu)
        self.captured_parity=True

    def _allocate_inputs(self):
        t=self.torch;result={}
        for name,meta in self.record['inputs'].items():
            if meta is None:result[name]=None;continue
            storage=t.empty(meta['storage_nbytes'],dtype=t.uint8,device='cuda')
            result[name]=t.empty(0,dtype=getattr(t,meta['dtype'].removeprefix('torch.')),device='cuda').set_(
                storage.untyped_storage(),meta['storage_offset'],tuple(meta['shape']),tuple(meta['stride']))
        return result

    def _restore_original(self, inputs):
        for name,value in self.pristine.items():
            if value is not None:raw_storage(inputs[name],self.torch).copy_(raw_storage(value,self.torch))
        self._set_work(self.pristine['sorted_token_ids'],self.pristine['num_valid_ids'])

    def _invoke(self,leg,inputs):
        fn=self.runtime.candidate if leg=='candidate' else self.runtime.reference
        if not self.warmed[leg]:
            fn(**inputs,**self.controls);self.torch.cuda.synchronize();self.warmed[leg]=True
        result=fn(**inputs,**self.controls);self.calls[leg]+=1
        if not isinstance(result,tuple) or len(result)!=2:raise AssertionError('Native quantized stage-1 must return payload and scales')
        return {'out':None,'return':result[0],'return_scale':result[1]}

    def invoke_candidate(self):
        self.output=self._invoke('candidate',self.inputs)
        return self.output

    def _set_work(self, sorted_ids, valid_ids):
        t=self.torch;ids=sorted_ids.view(-1).to(device='cpu',copy=True).to(t.int64)
        count=int(valid_ids.view(-1)[0]);active=ids[:count]
        token=active&0xffffff;slot=active>>24;mask=(token<self.tokens)&(slot<16)&(slot>=0)
        self.live_sorted=t.arange(count,dtype=t.int64)[mask]
        self.live_payload=self.live_sorted if self.controls['v2_output_layout'] else (token[mask]*16+slot[mask])
        if self.live_payload.numel()!=self.tokens*16 or self.live_payload.unique().numel()!=self.tokens*16:
            raise AssertionError('Routing does not define exactly one result for every token/expert slot')
        self.work_rows=count
        columns=t.arange(12,dtype=t.int64)[None,:];rows=self.live_sorted[:,None]
        self.live_scale=((rows//32)*512+(columns//8)*256+(columns%4)*64+(rows%16)*4+((columns//4)%2)*2+((rows//16)%2)).reshape(-1)

    def _recover_token_scales(self):
        t=self.torch;ids=self.pristine['sorted_token_ids'].tolist();count=int(self.pristine['num_valid_ids'][0])
        first=[None]*self.tokens
        for row,fused in enumerate(ids[:count]):
            token=fused&0xffffff;slot=fused>>24
            if token<self.tokens and 0<=slot<16 and first[token] is None:first[token]=row
        if any(row is None for row in first):raise ValueError('Captured routing does not cover every activation row')
        rows=t.tensor(first,dtype=t.int64)[:,None];columns=t.arange(112,dtype=t.int64)[None,:]
        offsets=(rows//32)*3584+(columns//8)*256+(columns%4)*64+(rows%16)*4+((columns//4)%2)*2+((rows//16)%2)
        self.token_scales=self.pristine['a1_scale'].view(t.uint8).reshape(-1)[offsets].contiguous().to('cuda')
        if bool((self.token_scales==255).any()):raise ValueError('Captured live activation scale contains NaN')

    def refresh(self,seed):
        t=self.torch;rng=random.Random(seed);rows=weighted_work(self.case['work_distribution']['valid_rows_histogram'],rng)
        routes=make_routes(self.tokens,self.controls['tile_m'],rows,self.inputs['sorted_token_ids'].numel(),
                           self.inputs['sorted_expert_ids'].numel(),seed)
        for name,values in routes.items():
            cpu=t.frombuffer(values,dtype=t.int32).reshape(self.inputs[name].shape)
            self.inputs[name].copy_(cpu)
        self._set_work(t.frombuffer(routes['sorted_token_ids'],dtype=t.int32),t.frombuffer(routes['num_valid_ids'],dtype=t.int32))
        generator=t.Generator(device='cuda').manual_seed(seed)
        permutation=t.randperm(self.tokens,device='cuda',generator=generator)
        a=self.inputs['a'].view(t.uint8)
        a.copy_(self.base_a.index_select(0,permutation))
        a.bitwise_xor_(t.randint(0,2,a.shape,dtype=t.uint8,device='cuda',generator=generator)*128)
        # Sign changes preserve every FP4 group's magnitudes and original E8M0
        # scale, while changing weight values at every native output channel.
        self.inputs['w1'].view(t.uint8).copy_(self.base_w)
        signs=t.randint(0,4,(896,768,1),dtype=t.uint8,device='cuda',generator=generator)
        signs=(signs&1)*8+((signs>>1)&1)*128
        self.inputs['w1'].view(t.uint8).bitwise_xor_(signs)
        self.inputs['w1_scale'].view(t.uint8).copy_(self.pristine['w1_scale'].view(t.uint8))
        # Build the exact sorted/tiled A-scale bytes for the freshly routed
        # token permutation; no layout conversion is placed inside timing.
        scale=self.inputs['a1_scale'].view(t.uint8).reshape(-1);scale.zero_()
        live=self.live_sorted.to('cuda');encoded=self.inputs['sorted_token_ids'][live].to(t.int64)
        token=encoded&0xffffff;columns=t.arange(112,device='cuda',dtype=t.int64)[None,:];r=live[:,None]
        offsets=(r//32)*3584+(columns//8)*256+(columns%4)*64+(r%16)*4+((columns//4)%2)*2+((r//16)%2)
        values=self.token_scales.index_select(0,permutation.index_select(0,token))
        scale[offsets]=values
        self.current_seed=seed;self.draws.append({'seed':seed,'valid_rows':rows,'active_m_tiles':rows//self.controls['tile_m']})
        record={'case_id':self.case['case_id'],'request_id':self.runtime.request['request_id'],
                'observed_occurrences':self.case['occurrences'],'benchmark_draws_are_not_workload_counts':True,
                'draws':self.draws}
        path=self.root/'build'/(self.case['case_id']+'_work.json');path.write_text(canonical(record)+'\n')

    def initialize_outputs(self):
        self.poisoned=set()
        if self.output is not None:
            for value in self.output.values():
                if value is not None:
                    raw_storage(value,self.torch).fill_(0xff);self.poisoned.add(value.data_ptr())

    def snapshot_inputs(self):
        # FreshCallbacks owns the single CPU copy; expose complete storage views.
        return {name:raw_storage(value,self.torch)
                for name,value in self.inputs.items() if value is not None}

    def assert_immutable(self,after,before):
        if set(after)!=set(before):raise AssertionError('Input storage set changed')
        for name in before:
            if before[name].device.type!='cpu' or after[name].device.type!='cpu':raise AssertionError('Input truth must be CPU-owned')
            if not self.torch.equal(after[name],before[name]):raise AssertionError('Native stage-1 modified input storage: '+name)

    def reference(self,truth):
        for name,value in self.reference_inputs.items():
            if value is not None:raw_storage(value,self.torch).copy_(truth[name])
        expected=self._invoke('reference',self.reference_inputs);self.torch.cuda.synchronize()
        after={name:raw_storage(value,self.torch).to(device='cpu',copy=True) for name,value in self.reference_inputs.items() if value is not None}
        self.assert_immutable(after,truth)
        return expected

    def validate_metadata(self):
        if self.output is None:raise AssertionError('Candidate produced no native outputs')
        seen={}
        for phase,values in (('inputs',self.inputs),('outputs',self.output)):
            for name,meta in self.record[phase].items():
                value=values[name]
                if meta is None:
                    if value is not None:raise AssertionError('Native null output/input changed')
                    continue
                if (str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape'] or list(value.stride())!=meta['stride']
                        or value.storage_offset()!=meta['storage_offset'] or value.untyped_storage().nbytes()!=meta['storage_nbytes']
                        or value.device.type!='cuda'):
                    raise AssertionError('Native physical ABI differs: '+phase+'.'+name)
                address=value.untyped_storage().data_ptr()
                if address in seen and seen[address]!=meta['alias']:raise AssertionError('Native buffers unexpectedly alias')
                seen[address]=meta['alias']
        return True

    def compare(self,actual,expected):
        t=self.torch
        for values in (actual,expected):
            if set(values)!=set(self.record['outputs']) or values['out'] is not None:raise AssertionError('Output structure differs')
            for name in ('return','return_scale'):
                value=values[name];meta=self.record['outputs'][name]
                if value.device.type!='cpu' or str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape']:
                    raise AssertionError('Oracle comparisons require exact-ABI CPU output snapshots')
        aq=actual['return'].view(t.uint8).reshape(-1,384).index_select(0,self.live_payload)
        eq=expected['return'].view(t.uint8).reshape(-1,384).index_select(0,self.live_payload)
        asc=actual['return_scale'].view(t.uint8).reshape(-1)[self.live_scale]
        esc=expected['return_scale'].view(t.uint8).reshape(-1)[self.live_scale]
        if not t.equal(asc,esc):raise AssertionError('Native tiled E8M0 output-scale bytes differ on live routes')
        if bool((asc==255).any()):raise AssertionError('Nonfinite output scale on a live route')
        a=aq.view(t.float8_e4m3fn).float();e=eq.view(t.float8_e4m3fn).float()
        if not bool(t.isfinite(a).all()) or not bool(t.isfinite(e).all()):raise AssertionError('Nonfinite FP8 output on a live route')
        tolerance=0.02;atol=tolerance*e.square().mean().sqrt().clamp_min(1e-6)
        if not bool(((a-e).abs()<=atol+tolerance*e.abs()).all()):raise AssertionError('Native FP8 output differs from independent reference')
        return True

    def observe(self):
        tensors={phase+'.'+name:value for phase,values in (('inputs',self.inputs),('outputs',self.output))
                 for name,value in values.items() if value is not None}
        return observe_case(self.case,tensors,self.case['scalars'])

    def corrupt_outputs(self):
        for value in self.output.values():
            if value is not None:raw_storage(value,self.torch).fill_(0xff)

    def native_engagement(self):
        return {'source_sha256':self.runtime.request['source_sha256'],'candidate_invoked':self.calls['candidate']>0,
                'independent_reference_invoked':self.calls['reference']>0,'captured_cpu_golden_parity':self.captured_parity,
                'candidate_callable':self.runtime.candidate.__module__+':flydsl_moe_stage1',
                'reference_callable':self.runtime.reference.__module__+':flydsl_moe_stage1',
                'native_source_sha256':NATIVE_SOURCE,'baseline_kind':self.runtime.baseline_kind,
                'frozen_reference_root':str(self.runtime.reference_root.relative_to(self.root)),
                'work_log':self.case['case_id']+'_work.json'}


def create_runtime(root,manifest,request):return Runtime(Path(root),manifest,request)

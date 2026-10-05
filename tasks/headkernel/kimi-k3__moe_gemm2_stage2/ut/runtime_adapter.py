"""Fixture-backed native replay with CPU observations before independent GPU oracle."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from evaluation_contract import canonical, observe_case, strict_json
from fresh_runner import FreshCallbacks, clear_device_reference, cpu_copy
from fixture_helpers import checked_path, file_sha, make_routes, raw_storage, weighted_work
from native import NativeBindings

IMAGE='docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96'
RUN_ID='kimi-actual-v4-no-stack-194550'
LEAN='_decode_lean_attention_fwd'
PREFILL='_flydsl_v2_stage2_wrapper'
DECODE='opus_moe_stage2_a8w4_decode_fwd'


def restore_phase(root,record,phase,torch):
    groups={}
    for alias,group in record['payload'][phase].items():
        # Paged KV gaps are never read. Defined zero bytes in the gaps make
        # full-storage immutable snapshots deterministic without inventing KV.
        data=torch.zeros(group['storage_nbytes'],dtype=torch.uint8,device='cpu')
        covered=[]
        for segment in group['segments']:
            path=checked_path(root,segment['blob'],segment['sha256'])
            offset=segment['offset_bytes'];size=segment['bytes'];written=0
            if offset<0 or offset+size>data.numel():raise ValueError('Fixture segment exceeds storage')
            if any(offset<b and a<offset+size for a,b in covered):raise ValueError('Overlapping fixture segments')
            covered.append((offset,offset+size))
            with path.open('rb') as stream:
                for part in iter(lambda:stream.read(8<<20),b''):
                    if written+len(part)>size:raise ValueError('Fixture exceeds recorded segment')
                    data[offset+written:offset+written+len(part)].copy_(torch.frombuffer(bytearray(part),dtype=torch.uint8))
                    written+=len(part)
            if written!=size:raise ValueError('Fixture byte size differs')
        if sum(b-a for a,b in covered)!=data.numel():
            names={name for name,meta in record[phase].items() if meta and meta['alias']==alias}
            if phase!='inputs' or not names or not names<={'k_buffer','v_buffer'}:
                raise ValueError('Only explicitly paged KV may contain unrecorded storage')
        groups[alias]=data
    return groups,views(record[phase],groups,torch,'cpu')


def views(metadata,groups,torch,device):
    result={}
    for name,meta in metadata.items():
        if meta is None:result[name]=None;continue
        result[name]=torch.empty(0,dtype=getattr(torch,meta['dtype'].removeprefix('torch.')),device=device).set_(
            groups[meta['alias']].untyped_storage(),meta['storage_offset'],tuple(meta['shape']),tuple(meta['stride']))
    return result


class Runtime:
    def __init__(self,root,manifest,request):
        self.root=Path(root);self.manifest=manifest;self.request=request
        old_path=list(sys.path)
        sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=(self.root/'ut').resolve()]
        try:
            import unittest.mock
            import torch
            import aiter
        finally:sys.path[:]=old_path
        self.torch=torch;self.native=NativeBindings(self.root,request)
        self.expected=strict_json((self.root/'ut/expected_abi.json').read_text())
    def prepare_case(self,case):return Prepared(self,case)


class Prepared:
    def __init__(self,runtime,case):
        self.runtime=runtime;self.root=runtime.root;self.torch=runtime.torch;self.case=case
        self.expected=runtime.expected[case['case_id']];self.family=self.expected['family']
        path=checked_path(self.root,case['fixture']['path'],case['fixture']['sha256'])
        self.record=strict_json(path.read_text())
        if (self.record.get('schema')!='served-tensor-fixture-v1' or self.record['family']!=self.family
            or self.record.get('startup_values') is not False or self.record['provenance']['image']!=IMAGE
            or self.record['provenance']['run_id']!=RUN_ID or self.record['source_sha256']!=self.expected['source']
            or canonical(self.record['structural_case_schema'])!=canonical(self.expected)):
            raise ValueError('Fixture is not the exact sealed current native ABI')
        self.controls={k:v for k,v in self.expected['controls'].items() if not k.endswith('_supplied')}
        self.controls.update(self.controls.pop('_kwargs',{}))
        self.pristine_groups,self.pristine=restore_phase(path.parent,self.record,'inputs',self.torch)
        self.golden_groups,self.golden=restore_phase(path.parent,self.record,'outputs',self.torch)
        self.groups,self.inputs=self.allocate();self.ref_groups,self.ref_inputs=self.allocate()
        self.output=None;self.calls=Counter();self.draws=[];self.warmed=set()
        self.mutable={meta['alias'] for meta in self.record['outputs'].values() if meta}
        self.candidate=runtime.native.get(self.family,'candidate')
        self.reference_fn=runtime.native.get(self.family,'reference')
        self.restore_original(self.groups)
        if self.family==LEAN:self.prepare_attention()
        else:self.prepare_moe()
        self.callbacks=FreshCallbacks(refresh_inputs=self.refresh,initialize_outputs=self.initialize_outputs,
            snapshot_inputs=self.snapshot_inputs,snapshot_outputs=lambda:self.output,
            validate_metadata=self.validate_metadata,assert_immutable=self.assert_immutable,
            reference=self.reference,compare=self.compare,replay=self.invoke_candidate)
        # Compilation happens through this actual native invocation. The only
        # golden present before the candidate is immutable CPU fixture storage.
        truth=self.snapshot_inputs();self.invoke_candidate();self.torch.cuda.synchronize();self.validate_metadata()
        actual=cpu_copy(self.output,self.torch);self.assert_immutable(self.snapshot_inputs(),truth)
        self.compare(actual,self.golden)
        golden=self.reference(truth);expected_cpu=cpu_copy(golden,self.torch)
        clear_device_reference(golden,self.torch);self.torch.cuda.synchronize()
        self.compare(expected_cpu,self.golden);self.compare(actual,expected_cpu)
        self.captured_parity=True

    def allocate(self):
        t=self.torch
        groups={alias:t.empty(data.numel(),dtype=t.uint8,device='cuda') for alias,data in self.pristine_groups.items()}
        return groups,views(self.record['inputs'],groups,t,'cuda')

    def restore_original(self,groups):
        for alias,data in self.pristine_groups.items():groups[alias].copy_(data)

    def outputs_for(self,inputs,result):
        if self.family==LEAN:
            if result is not None:raise AssertionError('Lean launcher return changed')
            return {name:inputs[name] for name in ('o','Mp','Lp','Op','locks')}
        if not self.torch.is_tensor(result):raise AssertionError('Native stage-2 must return its output tensor')
        return {'out':inputs['out'],'return':result,'return_scale':None}

    def invoke(self,leg,inputs):
        arguments={name:inputs[name] for name in self.expected['inputs']}
        fn=self.candidate if leg=='candidate' else self.reference_fn
        if leg not in self.warmed:
            # Exercise the actual specialization once outside validation and
            # timing, then restore captured inputs, including atomic outputs.
            # Some native compiler adapters execute on their first call.
            fn(**arguments,**self.controls);self.torch.cuda.synchronize()
            self.restore_original(self.groups if leg=='candidate' else self.ref_groups)
            self.warmed.add(leg)
        result=fn(**arguments,**self.controls);self.calls[leg]+=1
        return self.outputs_for(inputs,result)

    def invoke_candidate(self):
        self.output=self.invoke('candidate',self.inputs)
        return self.output

    def snapshot_inputs(self):
        return {alias:data.to(device='cpu',copy=True) for alias,data in self.groups.items()}

    def assert_immutable(self,after,before):
        if set(after)!=set(before):raise AssertionError('Input storage set changed')
        for alias in before:
            if before[alias].device.type!='cpu' or after[alias].device.type!='cpu':raise AssertionError('Truth must reside on CPU')
            if alias not in self.mutable and not self.torch.equal(after[alias],before[alias]):
                raise AssertionError('Native kernel modified readonly storage: '+alias)

    def reference(self,truth):
        for alias,data in truth.items():self.ref_groups[alias].copy_(data)
        result=self.invoke('reference',self.ref_inputs);self.torch.cuda.synchronize()
        after={alias:data.to(device='cpu',copy=True) for alias,data in self.ref_groups.items()}
        self.assert_immutable(after,truth)
        return result

    def prepare_attention(self):
        t=self.torch;self.tokens=64
        if self.controls['page_size']!=1 or self.inputs['q'].shape!=t.Size([64,12,576]):raise ValueError('Unreviewed Lean geometry')
        alias=self.record['inputs']['k_buffer']['alias'];group=self.record['payload']['inputs'][alias]
        row_bytes=self.inputs['k_buffer'].stride(0)*self.inputs['k_buffer'].element_size()
        spans=[]
        for segment in group['segments']:
            if segment['offset_bytes']%row_bytes or segment['bytes']%row_bytes:raise ValueError('KV segment must contain complete physical rows')
            spans.append(t.arange(segment['offset_bytes']//row_bytes,(segment['offset_bytes']+segment['bytes'])//row_bytes,dtype=t.int64))
        self.valid_kv=t.cat(spans)
        live=self.pristine['kv_indices'][:int(self.pristine['kv_indptr'][-1])]
        if not bool(t.isin(live,self.valid_kv).all()):raise ValueError('Captured KV access exceeds its read footprint')
        self.base_q=self.pristine['q'].to('cuda')

    def prepare_moe(self):
        t=self.torch;self.tokens=self.inputs['out'].shape[0];self.tile=self.controls['block_m']
        if self.tokens not in (64,8192,16384) or self.inputs['out'].shape[1]!=3584:raise ValueError('Unreviewed stage-2 shape')
        rows=int(self.pristine['num_valid_ids'][0]);encoded=self.pristine['sorted_token_ids'][:rows].to(t.int64)
        token=encoded&0xffffff;slot=encoded>>24;mask=(token<self.tokens)&(slot<16)&(slot>=0)
        live=t.arange(rows,dtype=t.int64)[mask];payload=live if self.family==PREFILL else token[mask]*16+slot[mask]
        route_ids=token[mask]*16+slot[mask]
        if payload.numel()!=self.tokens*16 or route_ids.unique().numel()!=self.tokens*16:raise ValueError('Captured routes are incomplete')
        self.base_inter=self.pristine['inter_states'].view(t.uint8).reshape(-1,384).index_select(0,payload).to('cuda')
        columns=t.arange(12,dtype=t.int64)[None,:];r=live[:,None]
        offsets=(r//32)*512+(columns//8)*256+(columns%4)*64+(r%16)*4+((columns//4)%2)*2+((r//16)%2)
        self.base_scales=self.pristine['a2_scale'].view(t.uint8).reshape(-1)[offsets].to('cuda')
        if bool((self.base_scales==255).any()):raise ValueError('Nonfinite live activation scale')

    def refresh_attention(self,seed):
        t=self.torch;rng=random.Random(seed)
        length=weighted_work(self.case['work_distribution']['sequence_length_histogram'],rng)
        if length<8193 or length>9216:raise ValueError('Unobserved KV length')
        generator=t.Generator(device='cuda').manual_seed(seed)
        self.inputs['q'].copy_(self.base_q)
        signs=t.randint(0,2,self.inputs['q'].shape,device='cuda',generator=generator,dtype=t.int8)*2-1
        self.inputs['q'].mul_(signs)
        # Each sequence receives unique valid physical rows. Sharing rows
        # between requests is legal; no uncaptured physical byte is read.
        starts=[rng.randrange(self.valid_kv.numel()) for _ in range(64)]
        order=(t.arange(length,dtype=t.int64)[None,:]+t.tensor(starts,dtype=t.int64)[:,None])%self.valid_kv.numel()
        ids=self.valid_kv[order].reshape(-1)
        self.inputs['kv_indices'].fill_(int(self.valid_kv[0]));self.inputs['kv_indices'][:ids.numel()].copy_(ids)
        self.inputs['kv_indptr'].copy_(t.arange(65,dtype=t.int32)*length)
        return {'sequence_length':length,'total_kv_tokens':64*length,'total_attention_tiles':64*((length+15)//16)}

    def refresh_moe(self,seed):
        t=self.torch;rng=random.Random(seed)
        rows=weighted_work(self.case['work_distribution']['valid_rows_histogram'],rng)
        routes=make_routes(self.tokens,self.tile,rows,self.inputs['sorted_token_ids'].numel(),self.inputs['sorted_expert_ids'].numel(),seed)
        for name in ('sorted_token_ids','sorted_expert_ids','num_valid_ids'):
            self.inputs[name].copy_(t.frombuffer(routes[name],dtype=t.int32).reshape(self.inputs[name].shape))
        encoded=t.frombuffer(routes['sorted_token_ids'],dtype=t.int32)[:rows].to(t.int64)
        token=encoded&0xffffff;slot=encoded>>24;mask=(token<self.tokens)&(slot<16)&(slot>=0)
        live=t.arange(rows,dtype=t.int64)[mask];payload=live if self.family==PREFILL else token[mask]*16+slot[mask]
        generator=t.Generator(device='cuda').manual_seed(seed)
        permutation=t.randperm(self.tokens*16,device='cuda',generator=generator)
        value=self.base_inter.index_select(0,permutation)
        value.bitwise_xor_(t.randint(0,2,value.shape,dtype=t.uint8,device='cuda',generator=generator)*128)
        inter=self.inputs['inter_states'].view(t.uint8).reshape(-1,384);inter.zero_();inter[payload.to('cuda')]=value
        scale=self.inputs['a2_scale'].view(t.uint8).reshape(-1);scale.fill_(127)
        r=live.to('cuda')[:,None];columns=t.arange(12,dtype=t.int64,device='cuda')[None,:]
        offsets=(r//32)*512+(columns//8)*256+(columns%4)*64+(r%16)*4+((columns//4)%2)*2+((r//16)%2)
        scale[offsets]=self.base_scales.index_select(0,permutation)
        weight=t.rand((self.tokens,16),device='cuda',generator=generator,dtype=t.float32)+0.01
        weight/=weight.sum(dim=1,keepdim=True)
        self.inputs['sorted_weights'].zero_()
        self.inputs['sorted_weights'][live.to('cuda')]=weight[token[mask].to('cuda'),slot[mask].to('cuda')]
        return {'valid_rows':rows,'active_m_tiles':rows//self.tile}

    def refresh(self,seed):
        self.restore_original(self.groups)
        work=self.refresh_attention(seed) if self.family==LEAN else self.refresh_moe(seed)
        self.draws.append({'seed':seed,**work})
        (self.root/'build'/(self.case['case_id']+'_work.json')).write_text(canonical({
            'case_id':self.case['case_id'],'request_id':self.runtime.request['request_id'],
            'observed_occurrences':self.case['occurrences'],'benchmark_draws_are_not_workload_counts':True,'draws':self.draws})+'\n')

    def initialize_outputs(self):
        if self.family==LEAN:
            self.inputs['o'].fill_(float('nan'))
            # Native writes only some scratch slots; retain exact initialized
            # values in untouched slots and validate all final scratch bytes.
            for name in ('Mp','Lp','Op','locks'):
                alias=self.record['inputs'][name]['alias'];self.groups[alias].copy_(self.pristine_groups[alias])
        elif self.family==DECODE:self.inputs['out'].zero_()
        else:self.inputs['out'].fill_(float('nan'))

    def validate_metadata(self):
        if self.output is None:raise AssertionError('Candidate did not bind native outputs')
        alias_to_address={};address_to_alias={}
        for phase,values in (('inputs',self.inputs),('outputs',self.output)):
            for name,meta in self.record[phase].items():
                value=values[name]
                if meta is None:
                    if value is not None:raise AssertionError('Native nullness changed')
                    continue
                if (str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape'] or list(value.stride())!=meta['stride']
                    or value.storage_offset()!=meta['storage_offset'] or value.untyped_storage().nbytes()!=meta['storage_nbytes']
                    or value.device.type!='cuda'):raise AssertionError('Native storage ABI differs: '+phase+'.'+name)
                address=value.untyped_storage().data_ptr();alias=meta['alias']
                if alias in alias_to_address and alias_to_address[alias]!=address:raise AssertionError('Required storage alias was lost')
                if address in address_to_alias and address_to_alias[address]!=alias:raise AssertionError('Unexpected storage alias')
                alias_to_address[alias]=address;address_to_alias[address]=alias
        return True

    def compare(self,actual,expected):
        t=self.torch;tolerance=0.02 if self.family==LEAN else 0.05
        if set(actual)!=set(expected) or set(actual)!=set(self.record['outputs']):raise AssertionError('Output structure differs')
        for name,meta in self.record['outputs'].items():
            a,e=actual[name],expected[name]
            if meta is None:
                if a is not None or e is not None:raise AssertionError('Unexpected output')
                continue
            for value in (a,e):
                if value.device.type!='cpu' or str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape']:
                    raise AssertionError('Oracle requires CPU output snapshots with exact dtype and shape')
            if not a.is_floating_point():
                if not t.equal(a,e):raise AssertionError('Integer output differs: '+name)
                continue
            if self.family==LEAN and name in ('Mp','Lp','Op'):
                untouched=expected['locks']==0
                if not t.equal(a[untouched].contiguous().view(t.uint8),e[untouched].contiguous().view(t.uint8)):
                    raise AssertionError('Untouched persistent scratch bytes differ: '+name)
            a=a.float();e=e.float();finite=t.isfinite(e)
            if name in ('o','out','return') and not bool(finite.all()):raise AssertionError('Nonfinite native result')
            if not t.equal(t.isnan(a),t.isnan(e)) or not t.equal(t.isposinf(a),t.isposinf(e)) or not t.equal(t.isneginf(a),t.isneginf(e)):
                raise AssertionError('Nonfinite output/scratch state differs: '+name)
            av,ev=a[finite],e[finite]
            if ev.numel():
                atol=tolerance*ev.double().square().mean().sqrt().clamp_min(1e-6)
                if not bool(((av-ev).abs()<=atol+tolerance*ev.abs()).all()):raise AssertionError('Independent native reference differs: '+name)
        return True

    def observe(self):
        tensors={phase+'.'+name:values[name] for phase,values in (('inputs',self.inputs),('outputs',self.output))
                 for name in self.expected[phase] if values[name] is not None}
        return observe_case(self.case,tensors,self.case['scalars'])

    def corrupt_outputs(self):
        self.inputs['o' if self.family==LEAN else 'out'].fill_(float('nan'))

    def native_engagement(self):
        return {'source_sha256':self.runtime.request['source_sha256'],'candidate_invoked':self.calls['candidate']>0,
            'independent_reference_invoked':self.calls['reference']>0,'captured_cpu_golden_parity':self.captured_parity,
            'candidate_callable':self.candidate.__module__+':'+self.candidate.__name__,
            'reference_callable':self.reference_fn.__module__+':'+self.reference_fn.__name__,
            'native_source_sha256':self.expected['source'],'work_log':self.case['case_id']+'_work.json',
            'candidate_binding':self.runtime.native.proofs[(self.family,'candidate')],
            'reference_binding':self.runtime.native.proofs[(self.family,'reference')]}


def create_runtime(root,manifest,request):return Runtime(root,manifest,request)

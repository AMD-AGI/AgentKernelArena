"""Full-shape aggregation trials plus explicitly sampled current native parity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from evaluation_contract import canonical, observe_case, strict_json
from fresh_runner import FreshCallbacks, OwnedCPUOutputs, clear_device_reference, cpu_copy
from native import bindings
from reference_math import reference as mathematical_reference, write_intervals
from cpu_policy import configure_cpu_threads

FAMILY='attn_res_hip'
NATIVE_SOURCE='95ebe877ec7176f869e5d19997c92b0c57079b32430da8e264ea13809aacc51b'


def checked(root,relative,sha=None):
    path=Path(root)/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Task path is not regular and local')
    data=path.read_bytes()
    if sha is not None and hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Fixture SHA256 differs')
    return data


def raw(value,torch):
    storage=value.untyped_storage()
    return torch.empty(0,dtype=torch.uint8,device=value.device).set_(storage,0,(storage.nbytes(),),(1,))


def tensor_views(metadata,groups,torch,device):
    return {name:None if meta is None else torch.empty(0,dtype=getattr(torch,meta['dtype'].removeprefix('torch.')),device=device).set_(
        groups[meta['alias']].untyped_storage(),meta['storage_offset'],tuple(meta['shape']),tuple(meta['stride'])) for name,meta in metadata.items()}


class Runtime:
    def __init__(self,root,manifest,request):
        self.root=Path(root);self.manifest=manifest;self.request=request
        previous=list(sys.path);sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=(self.root/'ut').resolve()]
        try:
            import unittest.mock
            import torch
        finally:sys.path[:]=previous
        self.torch=torch;self.cpu_thread_policy=configure_cpu_threads(torch,self.root/'build')
        self.native=bindings(self.root)
        self.expected=strict_json((self.root/'ut/expected_abi.json').read_text())
    def prepare_case(self,case):return Prepared(self,case)


class Prepared:
    def __init__(self,runtime,case):
        self.runtime=runtime;self.root=runtime.root;self.case=case;self.torch=runtime.torch
        self.expected=runtime.expected[case['case_id']];self.metadata=self.expected['native_abi']['inputs']
        ref=case['fixture'];record=strict_json(checked(self.root,ref['path'],ref['sha256']))
        if (record['schema']!='served-tensor-fixture-v1' or record['family']!=FAMILY or record['startup_values'] is not False
            or record['source_sha256']!=NATIVE_SOURCE or record['provenance']['run_id']!=runtime.manifest['run_id']
            or record['provenance']['image']!=runtime.manifest['runtime_image']
            or canonical(record['controls'])!=canonical(self.expected)):
            raise ValueError('Not the exact current native aggregation sample')
        self.record=record;self.rows=record['controls']['live_parity_sampling']['rows']
        if record['controls']['live_parity_sampling']['complete_native_tensor_dump'] is not False:
            raise ValueError('Sample evidence must not be presented as a full native tensor dump')
        self.controls={name:self.expected[name] for name in ('nvb','score_eps','out_eps','write_prefix')}
        self.segments={};self.payload={'inputs':{},'outputs':{}}
        base=(self.root/ref['path']).parent
        for phase,groups in record['payload'].items():
            for alias,group in groups.items():
                pieces=[]
                for segment in group['segments']:
                    key=(segment['blob'],segment['sha256'])
                    if key not in self.segments:
                        data=checked(base,*key)
                        if len(data)!=segment['bytes']:raise ValueError('Sample blob size differs')
                        self.segments[key]=self.torch.frombuffer(bytearray(data),dtype=self.torch.uint8)
                    pieces.append((segment['offset_bytes'],self.segments[key]))
                self.payload[phase][alias]=pieces
        self.groups,self.inputs=self.allocate();self.ref_groups,self.ref_inputs=self.allocate()
        self.output={name:self.inputs[name] for name in ('out','prefix_out','bank')}
        self.writes=write_intervals(self.metadata,self.controls);self.calls=Counter()
        self.callbacks=FreshCallbacks(refresh_inputs=self.refresh,initialize_outputs=self.initialize_outputs,
            snapshot_inputs=lambda:dict(self.groups),snapshot_outputs=lambda:self.output,
            validate_metadata=self.validate_metadata,assert_immutable=self.assert_immutable,
            reference=self.reference,compare=self.compare,replay=self.invoke_candidate,
            snapshot_candidate=lambda:self.snapshot_storage(self.groups))
        self.refresh(runtime.request['challenge_seed']);self.apply_samples();self.initialize_outputs()
        truth=cpu_copy(self.groups,self.torch);self.invoke_candidate();self.torch.cuda.synchronize();self.validate_metadata()
        actual,after=self.snapshot_storage(self.groups);self.assert_immutable(after,truth)
        self.compare_sample(actual)
        for alias,data in truth.items():self.ref_groups[alias].copy_(data)
        runtime.native['reference'](**self.ref_inputs,**self.controls);self.calls['native_reference']+=1
        native_result={name:self.ref_inputs[name] for name in self.output}
        native_cpu,native_after=self.snapshot_storage(self.ref_groups);self.assert_immutable(native_after,truth)
        self.compare(actual,native_cpu);self.compare_sample(native_cpu)
        clear_device_reference(native_result,self.torch);self.torch.cuda.synchronize()
        expected=self.reference(truth);expected_cpu=expected.value
        self.compare(actual,expected_cpu);self.compare_sample(expected_cpu);self.sample_parity=True

    def allocate(self):
        t=self.torch;groups={}
        for meta in self.metadata.values():
            if meta is not None and meta['alias'] not in groups:groups[meta['alias']]=t.empty(meta['storage_nbytes'],dtype=t.uint8,device='cuda')
        return groups,tensor_views(self.metadata,groups,t,'cuda')

    def refresh(self,seed):
        t=self.torch;generator=t.Generator(device='cuda').manual_seed(seed)
        for data in self.groups.values():data.zero_()
        seen=set()
        for name in ('prefix_sum','addend','bank','cw','ow','out','prefix_out'):
            value=self.inputs[name]
            if value is None:continue
            alias=self.metadata[name]['alias']
            if alias in seen:continue
            seen.add(alias);value.normal_(mean=0.0,std=0.3,generator=generator)
        # Score and norm weights are current captured model values. The
        # candidate receives only independent device storage, never the cache.
        for name in ('cw','ow'):
            if self.inputs[name] is None:continue
            alias=self.record['inputs'][name]['alias']
            for offset,data in self.payload['inputs'][alias]:self.groups[self.metadata[name]['alias']][offset:offset+data.numel()].copy_(data)

    def apply_samples(self):
        for name,meta in self.record['inputs'].items():
            if meta is None:continue
            for offset,data in self.payload['inputs'][meta['alias']]:self.groups[self.metadata[name]['alias']][offset:offset+data.numel()].copy_(data)

    def initialize_outputs(self):
        reads={self.metadata[name]['alias'] for name in ('prefix_sum','addend','bank','cw','ow') if self.metadata[name] is not None}
        for name in ('out','prefix_out'):
            if self.inputs[name] is None or (name=='prefix_out' and self.inputs['addend'] is None):continue
            if self.metadata[name]['alias'] not in reads:self.inputs[name].fill_(float('nan'))

    def invoke_candidate(self):
        result=self.runtime.native['candidate'](**self.inputs,**self.controls)
        if result is not None:raise AssertionError('Aggregation return ABI changed')
        self.calls['candidate']+=1

    def assert_immutable(self,after,before):
        if set(after)!=set(before):raise AssertionError('Storage identity changed')
        for alias,original in before.items():
            if original.device.type!='cpu' or after[alias].device.type!='cpu':raise AssertionError('Truth must be CPU-owned')
            position=0
            for start,end in self.writes.get(alias,[]):
                if not self.torch.equal(after[alias][position:start],original[position:start]):raise AssertionError('Readonly storage/padding changed')
                position=end
            if not self.torch.equal(after[alias][position:],original[position:]):raise AssertionError('Readonly storage/padding changed')

    def reference(self,truth):
        for alias,data in truth.items():self.ref_groups[alias].copy_(data)
        expected=mathematical_reference({**self.ref_inputs,**self.controls},self.torch);self.calls['mathematical_reference']+=1
        self.torch.cuda.synchronize();outputs,after=self.snapshot_storage(self.ref_groups);self.assert_immutable(after,truth)
        clear_device_reference(expected,self.torch);self.torch.cuda.synchronize()
        return OwnedCPUOutputs(outputs)

    def snapshot_storage(self,groups):
        after=cpu_copy(groups,self.torch)
        outputs=tensor_views(self.expected['native_abi']['outputs'],after,self.torch,'cpu')
        return outputs,after

    def validate_metadata(self):
        aliases={};addresses={}
        for phase,values in (('inputs',self.inputs),('outputs',self.output)):
            for name,meta in self.expected['native_abi'][phase].items():
                value=values[name]
                if meta is None:
                    if value is not None:raise AssertionError('Native nullness changed')
                    continue
                if (str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape'] or list(value.stride())!=meta['stride']
                    or value.storage_offset()!=meta['storage_offset'] or value.untyped_storage().nbytes()!=meta['storage_nbytes']
                    or value.device.type!='cuda'):raise AssertionError('Native ABI changed: '+name)
                address=value.untyped_storage().data_ptr();alias=meta['alias']
                if aliases.get(alias,address)!=address or addresses.get(address,alias)!=alias:raise AssertionError('Native alias relationships changed')
                aliases[alias]=address;addresses[address]=alias

    def compare_values(self,name,actual,expected):
        t=self.torch
        if name!='out':
            if not t.equal(actual.contiguous().view(t.uint8),expected.contiguous().view(t.uint8)):raise AssertionError('Prefix/bank state differs: '+name)
            return
        a,e=actual.float(),expected.float()
        if not bool(t.isfinite(a).all()) or not bool(t.isfinite(e).all()):raise AssertionError('Unwritten/nonfinite aggregation output')
        tolerance=0.02;atol=tolerance*e.double().square().mean().sqrt().clamp_min(1e-6)
        if not bool(((a-e).abs()<=atol+tolerance*e.abs()).all()):raise AssertionError('Independent aggregation reference differs')

    def compare(self,actual,expected):
        if set(actual)!=set(self.output) or set(expected)!=set(self.output):raise AssertionError('Output bindings differ')
        for name,meta in self.expected['native_abi']['outputs'].items():
            if meta is None:
                if actual[name] is not None or expected[name] is not None:raise AssertionError('Output nullness changed')
                continue
            for value in (actual[name],expected[name]):
                if value.device.type!='cpu' or str(value.dtype)!=meta['dtype'] or list(value.shape)!=meta['shape']:raise AssertionError('Comparison requires full-shape CPU observations')
            self.compare_values(name,actual[name],expected[name])
        return True

    def compare_sample(self,actual):
        t=self.torch
        for name,meta in self.record['outputs'].items():
            if meta is None:continue
            start=meta['storage_offset']*meta['element_size']
            extent=(1+sum((n-1)*s for n,s in zip(meta['shape'],meta['stride'])))*meta['element_size']
            data=t.zeros(extent,dtype=t.uint8);covered=0
            for offset,piece in self.payload['outputs'][meta['alias']]:
                a=max(start,offset);b=min(start+extent,offset+piece.numel())
                if b>a:data[a-start:b-start].copy_(piece[a-offset:b-offset]);covered+=b-a
            if covered<extent:raise ValueError('Live token sample is incomplete')
            value=t.empty(0,dtype=getattr(t,meta['dtype'].removeprefix('torch.'))).set_(data.untyped_storage(),0,tuple(meta['shape']),tuple(meta['stride']))
            self.compare_values(name,actual[name][:self.rows],value)

    def observe(self):
        tensors={phase+'.'+name:value for phase,values in (('inputs',self.inputs),('outputs',self.output)) for name,value in values.items() if value is not None}
        return observe_case(self.case,tensors,self.case['scalars'])

    def corrupt_outputs(self):self.inputs['out'].fill_(float('nan'))

    def native_engagement(self):
        return {'source_sha256':self.runtime.request['source_sha256'],'candidate_invoked':self.calls['candidate']>0,
            'cpu_thread_policy':self.runtime.cpu_thread_policy,
            'independent_reference_invoked':self.calls['mathematical_reference']>0,'independent_native_reference_invoked':self.calls['native_reference']>0,
            'current_native_token_sample_parity':self.sample_parity,'captured_sample_rows':self.rows,
            'full_shape_outputs_checked':True,'fixture_is_full_tensor_dump':False,
            'candidate_callable':self.runtime.native['candidate'].__module__+':attn_res_hip',
            'reference_callable':self.runtime.native['reference'].__module__+':attn_res_hip'}


def create_runtime(root,manifest,request):return Runtime(root,manifest,request)

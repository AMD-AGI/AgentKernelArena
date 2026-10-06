"""Protected complete-backing-storage reset and observation for dense operands."""
from functools import lru_cache
import json
from pathlib import Path
from evaluation_contract import fingerprint
ROOT=Path(__file__).resolve().parents[1]


def raw_storage(tensor):
    import torch
    return torch.empty(0,dtype=torch.uint8,device=tensor.device).set_(tensor.untyped_storage(),0,(tensor.untyped_storage().nbytes(),),(1,))


@lru_cache(maxsize=1)
def load_contract():return json.loads((ROOT/'ut/storage_contract.json').read_text())


def extents(case):
    row=load_contract()['cases'][case['case_id']]
    if row['case_sha256']!=fingerprint(case):raise ValueError('Backing-storage contract differs from frozen case')
    return row['storage_nbytes']


def allocate_view(spec,nbytes,device,fill=None):
    import torch
    dtype=getattr(torch,spec['dtype']);itemsize=torch.empty((),dtype=dtype).element_size()
    minimum=(spec['storage_offset']+1+sum((dim-1)*stride for dim,stride in zip(spec['shape'],spec['strides'])))*itemsize
    if type(nbytes) is not int or nbytes<minimum or nbytes%itemsize:raise ValueError('Invalid backing storage extent')
    raw=torch.empty((nbytes,),dtype=torch.uint8,device=device)
    if fill is not None:raw.fill_(fill)
    return torch.empty(0,dtype=dtype,device=device).set_(raw.untyped_storage(),spec['storage_offset'],spec['shape'],spec['strides'])


def fresh_input_storage(values,case,seed):
    sizes=extents(case);result={}
    for ordinal,(name,value) in enumerate(values.items()):
        spec=case['tensors'][name]
        logical=value.numel()*value.element_size()
        same=(value.untyped_storage().nbytes()==sizes[name] and value.storage_offset()==spec['storage_offset'] and list(value.stride())==spec['strides'])
        # Dense storage has no hidden bytes; its current logical values already
        # define every byte. Gapped/prefixed/tail storage gets fresh canaries.
        if same and logical==sizes[name]:result[name]=value
        else:
            fill=1+(int(seed)*17+ordinal*41)%255
            restored=allocate_view(spec,sizes[name],'cpu',fill=fill)
            restored.copy_(value);result[name]=restored
    return result


def copy_complete(destination,source):
    a=raw_storage(destination);b=raw_storage(source)
    if a.numel()!=b.numel():raise ValueError('Backing storage changed during reset')
    a.copy_(b)


def snapshot_complete(tensor):return raw_storage(tensor).detach().cpu().clone()


def assert_inputs_unchanged(tensors,inputs):
    import torch
    snapshots={name:snapshot_complete(tensors[name]) for name in inputs}
    for name,before in inputs.items():
        if not torch.equal(snapshots[name],raw_storage(before)):raise AssertionError('Input backing storage mutation: '+name)
    return snapshots


def output_guard_spans(spec,nbytes):
    import torch
    itemsize=torch.empty((),dtype=getattr(torch,spec['dtype'])).element_size()
    m,n=spec['shape'];row,col=spec['strides']
    if col!=1 or row<n:raise ValueError('Unsupported output storage geometry')
    start=spec['storage_offset']*itemsize;spans=[]
    if start:spans.append((0,start))
    for r in range(m-1):
        lo=start+(r*row+n)*itemsize;hi=start+(r+1)*row*itemsize
        if lo<hi:spans.append((lo,hi))
    end=start+((m-1)*row+n)*itemsize
    if end<nbytes:spans.append((end,nbytes))
    return spans


def initialize_output(tensor):
    raw_storage(tensor).fill_(167)
    tensor.fill_(float('nan'))


def assert_output_guards(snapshot,spec):
    for lo,hi in output_guard_spans(spec,snapshot.numel()):
        if not bool((snapshot[lo:hi]==167).all()):raise AssertionError('Output modified bytes outside its declared view')


def view_snapshot(snapshot,spec):
    import torch
    return torch.empty(0,dtype=getattr(torch,spec['dtype']),device='cpu').set_(snapshot.untyped_storage(),spec['storage_offset'],spec['shape'],spec['strides'])

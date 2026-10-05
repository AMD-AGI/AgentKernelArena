"""Protected raw-blob loader for served-tensor-fixture-v1; no pickle or inference."""
import hashlib
from pathlib import Path


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def restore_phase(root,fixture,phase,device='cpu'):
    import torch
    root=Path(root).resolve();groups=fixture['payload'][phase]
    if sum(g['storage_nbytes'] for g in groups.values())>2<<30:raise ValueError('Physical fixture extent exceeds 2GiB')
    if any(type(g['storage_nbytes']) is not int or g['storage_nbytes']<0 for g in groups.values()):raise ValueError('Invalid physical storage extent')
    raw={alias:torch.zeros(g['storage_nbytes'],dtype=torch.uint8,device=device) for alias,g in groups.items()}
    for alias,group in groups.items():
        for segment in group['segments']:
            original=root/segment['blob'];path=original.resolve()
            if not path.is_relative_to(root) or original.is_symlink() or file_sha(path)!=segment['sha256'] or path.stat().st_size!=segment['bytes']:raise ValueError('Fixture blob path/hash/size mismatch')
            offset=segment['offset_bytes'];size=segment['bytes']
            if not 0<=offset<=offset+size<=raw[alias].numel():raise ValueError('Fixture segment escapes storage')
            copied=0;actual_hash=hashlib.sha256()
            with path.open('rb') as stream:
                for data in iter(lambda:stream.read(8<<20),b''):
                    actual_hash.update(data)
                    block=torch.frombuffer(bytearray(data),dtype=torch.uint8)
                    raw[alias][offset+copied:offset+copied+len(data)].copy_(block.to(device));copied+=len(data)
            if copied!=size or actual_hash.hexdigest()!=segment['sha256']:raise ValueError('Fixture changed during restoration')
    out={}
    for name,meta in fixture[phase].items():
        if meta is None:out[name]=None;continue
        dtype=getattr(torch,meta['dtype'].removeprefix('torch.'),None)
        if not isinstance(dtype,torch.dtype):raise ValueError('Unknown tensor dtype')
        item_size=torch.empty((),dtype=dtype).element_size()
        if any(type(x) is not int or x<0 for x in [meta['storage_offset'],*meta['shape'],*meta['stride']]):raise ValueError('Invalid tensor geometry')
        start=meta['storage_offset']*item_size
        end=start if any(x==0 for x in meta['shape']) else start+(1+sum((n-1)*step for n,step in zip(meta['shape'],meta['stride'])))*item_size
        covered=start
        for segment in sorted(groups[meta['alias']]['segments'],key=lambda item:item['offset_bytes']):
            lo=segment['offset_bytes'];hi=lo+segment['bytes']
            if lo<=covered:covered=max(covered,hi)
        if not 0<=start<=end<=groups[meta['alias']]['storage_nbytes'] or covered<end:raise ValueError('Tensor reads uncaptured bytes or escapes original storage')
        out[name]=torch.empty(0,dtype=dtype,device=device).set_(raw[meta['alias']].untyped_storage(),meta['storage_offset'],meta['shape'],meta['stride'])
        for key,value in fixture['controls'].get('tensor_attributes',{}).get(name,{}).items():
            if key not in ('is_shuffled','shuffle_layout','is_guinterleave'):raise ValueError('Undeclared tensor attribute')
            setattr(out[name],key,value)
    return out




def fresh_numeric_fixture(inputs,seed):
    """Generate fresh activation values around real captured rows/routing."""
    import torch
    if any(value.device.type!='cpu' for value in inputs.values()):raise ValueError('Protected input truth must remain on CPU')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    m=inputs['hidden_states'].shape[0];order=torch.randperm(m,generator=generator)
    hidden=inputs['hidden_states'].index_select(0,order).float()
    rms=hidden.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1e-3)
    multiplier=0.75+0.5*torch.rand((m,1),generator=generator)
    noise=torch.randn(hidden.shape,generator=generator)*rms*0.05
    changed=(hidden*multiplier+noise).to(inputs['hidden_states'].dtype)
    fresh={name:(changed if name=='hidden_states' else value.index_select(0,order) if name in ('topk_ids','topk_weight') else value) for name,value in inputs.items()}
    return {'inputs':fresh,'seed':seed,'reference_policy':'native_after_candidate_CPU_snapshot'}

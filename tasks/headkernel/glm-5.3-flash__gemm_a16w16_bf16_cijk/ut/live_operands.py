"""Protected real operands and fresh numerical challenges with CPU-only truth."""
from functools import lru_cache
import json
from pathlib import Path
from fixture_codec import restore_phase,file_sha
ROOT=Path(__file__).resolve().parents[1]


def unpack(weight):
    n,k=weight.shape
    return weight.reshape(n//16,k//32,2,16,16).permute(0,3,1,2,4).contiguous().reshape(n,k)


def cpu_reference(values):
    import torch
    if 'SA' in values:
        ad=values['A'].float()*values['SA'].repeat_interleave(128,1)
        bd=unpack(values['B']).float()*values['SB'].repeat_interleave(128,0).repeat_interleave(128,1)
        return (ad@bd.t()).to(torch.bfloat16)
    return (values['A'].float()@values['B'].float()).to(values['A'].dtype)


@lru_cache(maxsize=2)
def load(path,checksum):
    import torch
    file=(ROOT/path).resolve()
    if not file.is_relative_to(ROOT) or file_sha(file)!=checksum:raise ValueError('Live operand fixture changed')
    fixture=json.loads(file.read_text());inputs=restore_phase(file.parent,fixture,'inputs');native=restore_phase(file.parent,fixture,'outputs')['result']
    if fixture['family']=='fp8_gemm':values={'A':inputs['XQ'],'B':inputs['WQ'],'SA':inputs['x_scale'],'SB':inputs['w_scale']}
    elif fixture['family'] in ('bf16_gemm','aten_bf16_mm'):values={'A':inputs['A'],'B':inputs['B'].t() if fixture['family']=='bf16_gemm' else inputs['B']}
    else:raise ValueError('Unexpected native operand family')
    expected=cpu_reference(values)
    from native_precision import calibrate
    calibrate({'case_id':file.stem,'live_fixture':{'capture_family':fixture['family']}},values,native,expected)
    return values,expected


def generate_live(case,seed):
    import torch
    metadata=case['live_fixture'];values,_=load(metadata['path'],metadata['sha256'])
    g=torch.Generator(device='cpu').manual_seed(seed);order=torch.randperm(values['A'].shape[0],generator=g)
    base=values['A'].float().index_select(0,order)
    rms=base.square().mean(dim=1,keepdim=True).sqrt().clamp_min(1e-6)
    factor=0.75+torch.rand((base.shape[0],1),generator=g)*0.5
    activation=base*factor+torch.randn(base.shape,generator=g)*rms*0.05
    if values['A'].element_size()==1:activation=activation.clamp(-448,448)
    fresh={'A':activation.to(values['A'].dtype),'B':values['B']}
    if 'SA' in values:
        shape=values['SA'].shape;scale=torch.empty_strided(shape,(1,shape[0]),dtype=values['SA'].dtype,device='cpu')
        scale.copy_(values['SA'].index_select(0,order));fresh.update(SA=scale,SB=values['SB'])
    return fresh,cpu_reference(fresh)

"""Protected native operand representatives and CPU-only independent reference."""
from functools import lru_cache
import json
from pathlib import Path
from fixture_codec import restore_phase,file_sha
ROOT=Path(__file__).resolve().parents[1]


def unpack(weight):
    n,k=weight.shape
    return weight.reshape(n//16,k//32,2,16,16).permute(0,3,1,2,4).contiguous().reshape(n,k)


@lru_cache(maxsize=2)
def load(path,checksum):
    import torch
    file=(ROOT/path).resolve()
    if not file.is_relative_to(ROOT) or file_sha(file)!=checksum:raise ValueError('Live operand fixture changed')
    fixture=json.loads(file.read_text());inputs=restore_phase(file.parent,fixture,'inputs');native=restore_phase(file.parent,fixture,'outputs')['result']
    if fixture['family']=='fp8_gemm':
        values={'A':inputs['XQ'],'B':inputs['WQ'],'SA':inputs['x_scale'],'SB':inputs['w_scale']}
        ad=values['A'].float()*values['SA'].repeat_interleave(128,1)
        bd=unpack(values['B']).float()*values['SB'].repeat_interleave(128,0).repeat_interleave(128,1)
        expected=(ad@bd.t()).to(torch.bfloat16)
    elif fixture['family'] in ('bf16_gemm','aten_bf16_mm'):
        values={'A':inputs['A'],'B':inputs['B'].t() if fixture['family']=='bf16_gemm' else inputs['B']}
        expected=(values['A'].float()@values['B'].float()).to(torch.bfloat16)
    else:raise ValueError('Unexpected native operand family')
    torch.testing.assert_close(expected,native,rtol=0.01,atol=0.02)
    return values,expected


def generate_live(case,seed):
    import torch
    metadata=case['live_fixture'];values,expected=load(metadata['path'],metadata['sha256'])
    order=torch.randperm(values['A'].shape[0],generator=torch.Generator(device='cpu').manual_seed(seed))
    # Raw-byte row gathering supports CPU FP8 builds without arithmetic/index
    # kernels for Float8. It preserves every bit of the captured activation.
    if values['A'].element_size()==1:
        activation=values['A'].view(torch.uint8).index_select(0,order).view(values['A'].dtype)
    else:activation=values['A'].index_select(0,order)
    fresh={'A':activation,'B':values['B']}
    if 'SA' in values:
        shape=values['SA'].shape
        scale=torch.empty_strided(shape,(1,shape[0]),dtype=values['SA'].dtype,device='cpu')
        scale.copy_(values['SA'].index_select(0,order))
        fresh.update(SA=scale,SB=values['SB'])
    return fresh,expected.index_select(0,order)

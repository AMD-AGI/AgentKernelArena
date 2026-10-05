"""Complete native input/output ABI shared by fixture intake and actual observation."""
import dataclasses
import enum


def runtime_abi(inputs, output):
    import torch
    tensors={}; scalars={}
    def walk(value,path,role):
        if torch.is_tensor(value):
            tensors[path]=value
            attrs={name:walk(v,path+'.attr.'+name,'input') for name,v in vars(value).items()}
            return {'tensor':path,'attributes':attrs}
        if dataclasses.is_dataclass(value) and not isinstance(value,type):
            # The alias module differs for each leg; the finite semantic class name does not.
            return {'dataclass':type(value).__qualname__,'fields':{f.name:walk(getattr(value,f.name),path+'.'+f.name,role) for f in dataclasses.fields(value)}}
        if type(value).__name__=='ActivationType' and hasattr(value,'value'): return {'aiter_activation':int(value.value)}
        if isinstance(value,enum.Enum): return {'enum':type(value).__qualname__,'name':value.name}
        if isinstance(value,(list,tuple)): return {'container':'tuple' if isinstance(value,tuple) else 'list','items':[walk(v,path+'.'+str(i),role) for i,v in enumerate(value)]}
        if isinstance(value,dict): return {k:walk(v,path+'.'+k,role) for k,v in value.items()}
        if value is None or type(value) in (str,int,float,bool): return value
        if type(value).__module__=='torch' and type(value).__name__=='dtype': return {'dtype':str(value)}
        raise ValueError('Unsupported semantic control type: '+str(type(value)))
    scalars['arguments']={key:walk(value,'arg.'+key,'inout' if key=='out' else 'input') for key,value in inputs.items()}
    scalars['return']=walk(output,'result','output')
    return tensors,scalars


def roles(tensors):
    return {name:('output' if name.startswith('result') else 'inout' if name=='arg.out' else 'input') for name in tensors}

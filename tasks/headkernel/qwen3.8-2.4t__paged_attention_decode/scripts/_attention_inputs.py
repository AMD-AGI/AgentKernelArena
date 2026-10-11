"""CPU mathematical attention over the exact original compact page/routing data."""
from __future__ import annotations

CHALLENGES = ('zero_query', 'negated_query', 'small_query', 'scaled_query')
IMMUTABLE = {'query':2, 'key_cache':3, 'value_cache':4, 'kv_indptr':6,
             'kv_page_indices':7, 'kv_last_page_lens':8, 'k_scale':15, 'v_scale':16}
TENSORS = {'out':0, 'workspace_buffer':1, **IMMUTABLE}


def tensor_contract(value):
    return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype)}


def case_params(args, reference_sha256):
    if len(args)!=19:
        raise ValueError('The captured paged-attention ABI has exactly 19 positional arguments')
    return {'regime':'decode','m':int(args[2].shape[0]),'positional_argument_count':19,
            'tensors':{name:tensor_contract(args[index]) for name,index in TENSORS.items()},
            'scale':float(args[5]),'block_size':int(args[9]),'max_num_partitions':int(args[10]),
            'alibi_slopes':args[11], 'kv_cache_dtype':args[12], 'kv_cache_layout':args[13],
            'logits_soft_cap':float(args[14]),'fp8_out_scale':args[17],
            'partition_size':int(args[18]),'default_mtp':1,
            'captured_reference_io_sha256':reference_sha256}


class CPUInputs:
    """Keep fixed captured KV/routing on CPU and regenerate queries continuously."""
    def __init__(self, torch, args, seed):
        self.torch, self.args = torch, tuple(args)
        if len(args)!=19 or any(v.device.type!='cpu' for v in args if isinstance(v,torch.Tensor)):
            raise ValueError('Attention oracle accepts the captured CPU argument tuple only')
        if (args[9]!=1 or args[11] is not None or args[12]!='auto' or args[13]!='NHD'
                or args[14]!=0.0 or args[17] is not None or args[18]!=256):
            raise ValueError('CPU oracle is restricted to the recorded page1 BF16 attention contract')
        q,k,v = args[2],args[3],args[4]
        if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
            raise ValueError('Recorded query and KV types must remain BF16')
        if k.ndim!=4 or tuple(k.shape[1:3])!=(1,1) or v.shape!=k.shape or k.shape[-1]!=q.shape[-1]:
            raise ValueError('Recorded compact NHD KV geometry differs')
        indptr,indices=args[6].long(),args[7].long()
        if (indptr.numel()!=q.shape[0]+1 or int(indptr[0])!=0 or int(indptr[-1])!=indices.numel()
                or not (indptr[1:]>=indptr[:-1]).all() or not ((indices>=0)&(indices<k.shape[0])).all()):
            raise ValueError('Invalid captured ragged/page metadata')
        self.pages=[]
        for start,end in zip(indptr[:-1].tolist(),indptr[1:].tolist()):
            if end<=start or end-start>int(args[10])*int(args[18]):
                raise ValueError('Captured context is empty or exceeds the native workspace capacity')
            selected=indices[start:end]
            self.pages.append((k[selected,0,0,:].float(),v[selected,0,0,:].float()))
        self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.generated=0

    def reference(self, query):
        t=self.torch
        if query.device.type!='cpu' or query.shape!=self.args[2].shape or query.dtype!=self.args[2].dtype:
            raise ValueError('CPU attention query differs from the recorded ABI')
        result=t.empty_like(query)
        for sequence,(key,value) in enumerate(self.pages):
            scores=t.matmul(query[sequence].float(),key.T)*float(self.args[5])
            probabilities=t.softmax(scores,dim=-1)
            result[sequence].copy_(t.matmul(probabilities,value).to(query.dtype))
        return result

    def next(self, kind='ordinary'):
        if kind not in ('ordinary',*CHALLENGES):
            raise ValueError('Unknown attention challenge')
        template=self.args[2]
        query=self.torch.empty_strided(template.shape,template.stride(),dtype=template.dtype,device='cpu')
        query.uniform_(-0.75,0.875,generator=self.rng)
        query.reshape(-1)[0]=0.5
        if kind=='zero_query':query.zero_()
        elif kind=='negated_query':query.neg_()
        elif kind=='small_query':query.mul_(1e-3)
        elif kind=='scaled_query':query.mul_(4.0)
        self.generated+=1
        return query,self.reference(query)

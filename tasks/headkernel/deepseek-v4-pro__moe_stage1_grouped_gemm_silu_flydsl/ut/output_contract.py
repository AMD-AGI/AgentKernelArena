"""Native stage1 output domain: dequantized FP8 payload and exact live E8M0 bytes.

The image allocates 256-row/8-column-padded scale storage with torch.empty.
Only sorted rows with real token/slot IDs and inter_dim//32 columns are output.
Offsets match the frozen mixed_moe_gemm_2stage_common.py scale-store formula.
The allocation is preserved; downstream vector loads may touch non-output padding.
"""
import hashlib
import json


def require(condition, message):
    if not condition: raise ValueError('Native output contract: '+message)


def comparison_inputs_from_snapshot(inputs, snapshots, tensor_bindings):
    """Reconstruct routing from owned CPU storage truth, never live GPU values."""
    import torch
    result=dict(inputs)
    for name in ('sorted_token_ids','sorted_expert_ids','num_valid_ids'):
        value=inputs[name];pointer=value.untyped_storage().data_ptr()
        aliases=[key for key,tensor in tensor_bindings.items()
                 if tensor.untyped_storage().data_ptr()==pointer and key in snapshots]
        require(len(aliases)==1,'routing storage is absent or ambiguous in immutable truth')
        raw=snapshots[aliases[0]]
        require(raw.device.type=='cpu' and raw.dtype==torch.uint8 and raw.is_contiguous()
                and raw.storage_offset()==0 and raw.untyped_storage().nbytes()==raw.numel(),
                'routing truth must be a complete owned CPU byte snapshot')
        shape=tuple(value.shape);strides=tuple(value.stride());offset=value.storage_offset();item=value.element_size()
        require(all(n>=0 for n in shape) and all(s>=0 for s in strides) and offset>=0,'invalid routing view geometry')
        end=offset*item if any(n==0 for n in shape) else (offset+1+sum((n-1)*s for n,s in zip(shape,strides)))*item
        require(end<=raw.numel(),'routing view escapes immutable CPU storage')
        result[name]=torch.empty(0,dtype=value.dtype,device='cpu').set_(raw.untyped_storage(),offset,shape,strides).clone()
    return result


def live_scale_domain(inputs, payload, scale):
    import torch
    require(payload.device.type=='cpu' and scale.device.type=='cpu','CPU snapshots are required')
    require(payload.ndim==3,'expected dense token/slot/intermediate FP8 payload')
    require(str(payload.dtype) in ('torch.float8_e4m3fn','torch.float8_e4m3fnuz'),'unsupported stage1 payload dtype')
    require(str(scale.dtype)=='torch.float8_e8m0fnu','expected native E8M0 scale dtype')
    require(scale.ndim==2 and scale.is_contiguous(),'expected native contiguous tiled scale allocation')
    m=int(inputs['a'].shape[0]);topk=inputs['topk'];inter_dim=int(inputs['w1'].shape[1])//2
    require(type(topk) is int and 0<topk<128 and 0<m<(1<<24),'unsupported token/slot domain')
    require(tuple(payload.shape)==(m,topk,inter_dim) and inter_dim>0 and inter_dim%32==0,'payload extent differs from inputs')
    ids=inputs['sorted_token_ids'];valid=inputs['num_valid_ids'];experts=inputs['sorted_expert_ids']
    require(ids.device.type=='cpu' and valid.device.type=='cpu' and experts.device.type=='cpu',
            'routing must come from immutable CPU input truth')
    require(ids.dtype==torch.int32 and ids.ndim==1 and valid.dtype==torch.int32 and valid.numel()>=1,'unexpected native routing dtype/shape')
    tile=inputs['tile_m'];require(type(tile) is int and tile>0,'invalid sort tile')
    scale_rows=((max(ids.numel(),experts.numel()*tile)+255)//256)*256
    columns=inter_dim//32;scale_columns=((columns+7)//8)*8
    require(tuple(scale.shape)==(scale_rows,scale_columns),'scale allocation differs from native padding rule')
    count=int(valid.reshape(-1)[:1].item())
    require(0<=count<=ids.numel(),'num_valid_ids escapes sorted routing storage')
    routes=ids[:count].to(torch.int64)
    tokens=routes&0xffffff;slots=routes>>24
    live=(tokens<m)&(slots>=0)&(slots<topk)
    rows=torch.nonzero(live,as_tuple=False).reshape(-1)
    pairs=tokens[live]*topk+slots[live]
    require(pairs.numel()==m*topk and torch.equal(torch.sort(pairs).values,torch.arange(m*topk)),
            'every token/slot must have exactly one live sorted row')
    require(experts.dtype==torch.int32 and experts.ndim==1 and int((rows//tile).max())<experts.numel(),
            'live rows escape native expert blocks')
    live_experts=experts.to(torch.int64)[rows//tile]
    require(bool(((live_experts>=0)&(live_experts<inputs['w1'].shape[0])).all()),'live expert IDs are invalid')
    row=rows[:,None];col=torch.arange(columns,dtype=torch.int64)[None,:]
    offsets=((row//32)*(scale_columns*32)+(col//8)*256+(col%4)*64+(row%16)*4
             +((col//4)%2)*2+((row//16)%2)).reshape(-1)
    require(offsets.numel()==m*topk*columns and int(offsets.min())>=0 and int(offsets.max())<scale.numel(),
            'live scale offsets escape native storage')
    require(torch.unique(offsets).numel()==offsets.numel(),'live scale offsets alias')
    proof={'contract':'native-stage1-dense-fp8-live-e8m0-v1','payload_shape':list(payload.shape),
        'scale_allocation_shape':list(scale.shape),'live_token_slot_rows':m*topk,'scale_columns':columns,
        'meaningful_scale_bytes':offsets.numel(),'allocated_scale_bytes':scale.numel(),
        'unwritten_scale_padding_bytes':scale.numel()-offsets.numel(),
        'routing_complete_and_unique':True,
        'live_expert_ids_valid':True,
        'routing_source':'immutable_cpu_input_storage_snapshot',
        'live_scale_offsets_sha256':hashlib.sha256(json.dumps(offsets.tolist(),separators=(',',':')).encode()).hexdigest()}
    return offsets,pairs,proof


def compare_stage1_outputs(actual,golden,inputs,tolerance,compare_dense,path='output'):
    import torch
    require(float(tolerance)==0.02,'dequantized semantic tolerance must be exactly 0.02')
    if type(actual)!=type(golden) or not isinstance(golden,(tuple,list)) or len(actual)!=2 or len(golden)!=2:
        raise AssertionError(path+': stage1 output structure differs')
    for index,(a,g) in enumerate(zip(actual,golden)):
        if not torch.is_tensor(a) or not torch.is_tensor(g) or a.shape!=g.shape or a.dtype!=g.dtype:
            raise AssertionError(path+'.'+str(index)+': shape/dtype differs')
        require(a.device.type=='cpu' and g.device.type=='cpu','candidate/reference comparisons require CPU snapshots')
    offsets,pairs,proof=live_scale_domain(inputs,golden[0],golden[1])
    expected=golden[1].view(torch.uint8).reshape(-1).index_select(0,offsets)
    # A corrupt reference is not a valid rejection oracle for a negative source.
    require(not bool((expected==255).any()),'reference has nonfinite live E8M0 scales')
    actual_bytes=actual[1].view(torch.uint8).reshape(-1).index_select(0,offsets)
    if bool((actual_bytes==255).any()):raise AssertionError(path+'.1: nonfinite live E8M0 scale')
    if not torch.equal(actual_bytes,expected):
        mismatch=torch.nonzero(actual_bytes!=expected,as_tuple=False).reshape(-1)
        first=int(mismatch[0])
        raise AssertionError(path+'.1: live E8M0 bytes differ at physical byte '+str(int(offsets[first])))
    # Scales are stored by sorted route row; payload is stored by token/slot.
    # Exact scale equality above lets both dequantizations use this one mapping.
    columns=proof['scale_columns'];dense_rows=proof['live_token_slot_rows']
    scales_by_route=torch.ldexp(torch.ones((dense_rows,columns),dtype=torch.float64),
                               expected.reshape(dense_rows,columns).to(torch.int32)-127)
    scales_by_dense_row=torch.empty_like(scales_by_route)
    scales_by_dense_row[pairs]=scales_by_route
    expanded=scales_by_dense_row.repeat_interleave(32,dim=1).reshape(golden[0].shape)
    expected_values=golden[0].to(torch.float64)*expanded
    actual_values=actual[0].to(torch.float64)*expanded
    require(bool(torch.isfinite(expected_values).all()),'reference has nonfinite dequantized payload')
    if not bool(torch.isfinite(actual_values).all()):raise AssertionError(path+'.0: nonfinite dequantized payload')
    atol=0.02*expected_values.square().mean().sqrt().clamp_min(1e-6)
    if not bool(((actual_values-expected_values).abs()<=atol+0.02*expected_values.abs()).all()):
        raise AssertionError(path+'.0: dequantized values differ beyond 0.02 semantic tolerance')
    return {**proof,'dense_payload_comparison':'all dense dequantized values',
            'semantic_tolerance':0.02,'dequantization_accumulator':'float64',
            'live_scale_comparison':'exact bytes','comparison_passed':True}

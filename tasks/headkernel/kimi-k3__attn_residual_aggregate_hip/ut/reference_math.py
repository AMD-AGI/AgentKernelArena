"""Independent mathematical attention-residual reference, including native side effects."""


def reference(arguments, torch):
    prefix=arguments['prefix_sum'];addend=arguments['addend'];bank=arguments['bank']
    nvb=arguments['nvb'];cw=arguments['cw'].float();ow=arguments['ow']
    # The materialized prefix is rounded before scores are evaluated, exactly
    # as downstream readers of prefix_out and the bank snapshot observe it.
    row=prefix.clone() if addend is None else (prefix.float()+addend.float()).to(prefix.dtype)
    if addend is not None:arguments['prefix_out'].copy_(row)
    if arguments['write_prefix']:bank[:,nvb,:].copy_(row)
    pv=row.float();tile=bank[:,:nvb,:].float()
    p_score=(pv*cw).sum(dim=-1)/(pv.square().mean(dim=-1)+arguments['score_eps']).sqrt()
    b_score=(tile*cw[None,None,:]).sum(dim=-1)/(tile.square().mean(dim=-1)+arguments['score_eps']).sqrt()
    scores=torch.cat((b_score,p_score[:,None]),dim=1)
    probabilities=torch.softmax(scores,dim=1)
    mixed=(tile*probabilities[:,:nvb,None]).sum(dim=1)+pv*probabilities[:,nvb,None]
    if ow is not None:
        mixed=mixed*torch.rsqrt(mixed.square().mean(dim=-1,keepdim=True)+arguments['out_eps'])*ow.float()
    arguments['out'].copy_(mixed.to(arguments['out'].dtype))
    return {name:arguments[name] for name in ('out','prefix_out','bank')}


def write_intervals(metadata,controls):
    """Every physical byte the native kernel is allowed to modify, by alias."""
    result={}
    def matrix(name):
        meta=metadata[name]
        if meta is None:return
        rows,width=meta['shape'];unit=meta['element_size']
        for row in range(rows):
            start=(meta['storage_offset']+row*meta['stride'][0])*unit
            result.setdefault(meta['alias'],[]).append((start,start+width*unit))
    matrix('out')
    if metadata['addend'] is not None:matrix('prefix_out')
    if controls['write_prefix']:
        meta=metadata['bank'];unit=meta['element_size']
        for row in range(meta['shape'][0]):
            start=(meta['storage_offset']+row*meta['stride'][0]+controls['nvb']*meta['stride'][1])*unit
            result.setdefault(meta['alias'],[]).append((start,start+meta['shape'][2]*unit))
    for alias,intervals in result.items():
        merged=[]
        for start,end in sorted(intervals):
            if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
            else:merged.append((start,end))
        result[alias]=merged
    return result

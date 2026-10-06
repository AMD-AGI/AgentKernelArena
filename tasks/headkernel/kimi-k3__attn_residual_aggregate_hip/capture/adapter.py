"""Native attn-residual capture extension for a NEW Kimi served-hook bundle.

The native call runs at its complete original shape. Four actual token rows
are retained as live parity evidence; the complete ABI and every occurrence
are recorded. This is explicitly a token sample, not a complete tensor dump.
"""
import functools

MODULE='sglang.kernels.ops.kimi_k3.attn_res_hip'
FAMILY='attn_res_hip'
SOURCE_SHA256='95ebe877ec7176f869e5d19997c92b0c57079b32430da8e264ea13809aacc51b'
SAMPLE_ROWS=4
TENSOR_NAMES=('prefix_sum','bank','cw','ow','out','addend','prefix_out')


def validate_arguments(arguments):
    prefix=arguments['prefix_sum'];tokens,hidden=prefix.shape
    nvb=arguments['nvb'];bank=arguments['bank']
    if hidden!=7168 or not 1<=nvb<=8 or tokens<=0:raise ValueError('Unreviewed aggregation shape')
    if bank.ndim!=3 or bank.shape[0]!=tokens or bank.shape[2]!=hidden or not nvb+int(arguments['write_prefix'])<=bank.shape[1]<=8:
        raise ValueError('Bank capacity does not cover native reads/write')
    if str(prefix.dtype)!='torch.bfloat16' or str(bank.dtype)!=str(prefix.dtype):raise ValueError('Current aggregation storage must be BF16')
    if bank.stride(2)!=1 or bank.stride(1)<hidden or bank.stride(0)<bank.shape[1]*bank.stride(1):
        raise ValueError('Token sampling requires disjoint contiguous rows per token')
    for name in ('prefix_sum','out','addend','prefix_out'):
        value=arguments[name]
        if value is not None and (tuple(value.shape)!=(tokens,hidden) or value.stride(1)!=1 or value.stride(0)<hidden):
            raise ValueError('Token sampling requires disjoint native tensor rows: '+name)
        if value is not None and str(value.dtype)!=str(prefix.dtype):raise ValueError('Native prefix/output storage dtype changed: '+name)
    if arguments['addend'] is not None and arguments['prefix_out'] is None:raise ValueError('Addend requires materialized prefix output')
    for name in ('cw','ow'):
        value=arguments[name]
        if value is not None and (tuple(value.shape)!=(hidden,) or value.stride(0)!=1):raise ValueError('Unreviewed score/norm weight layout')
    if str(arguments['cw'].dtype)!='torch.float32':raise ValueError('Native combined score weight must be FP32')
    return tokens,hidden


def sampled(value,rows):
    return value if value is None or value.ndim==1 else value[:rows]


def make_bindings(rc,arguments,source_hash):
    rc.require(source_hash==SOURCE_SHA256,'Aggregation runtime source differs from the pinned image')
    tokens,hidden=validate_arguments(arguments);rows=min(SAMPLE_ROWS,tokens)
    aliases={};full={}
    for name in sorted(TENSOR_NAMES):
        value=arguments[name]
        if value is None:full[name]=None;continue
        alias=aliases.setdefault(rc.storage_key(value),'s'+str(len(aliases)))
        full[name]=rc.tensor_metadata(value,alias)
        # Device ordinals differ across TP ranks and are not logical ABI.
        # The normal recorder metadata retains the actual device for audit.
        full[name].pop('device',None)
    token_aliases={}
    for name,meta in full.items():
        if meta is None or len(meta['shape'])<2:continue
        mapping=(meta['stride'][0],meta['storage_offset']//meta['stride'][0])
        previous=token_aliases.setdefault(meta['alias'],mapping)
        if previous!=mapping:raise ValueError('Cross-token aliases require a different full capture contract')
    bindings={name:sampled(arguments[name],rows) for name in TENSOR_NAMES}
    roles={name:rc.Role('readonly' if name in ('cw','ow') else 'mutable',optional=arguments[name] is None)
           for name in TENSOR_NAMES}
    output_roles={name:rc.Role('mutable',optional=arguments[name] is None) for name in ('out','prefix_out','bank')}
    controls={name:rc.controls_json(arguments[name]) for name in ('nvb','score_eps','out_eps','write_prefix')}
    controls.update(native_abi={'inputs':full,'outputs':{name:full[name] for name in ('out','prefix_out','bank')}},
        native_launch={'tokens':tokens,'hidden':hidden,'nvb':arguments['nvb'],'block_h':8192,
            'r_pad':1<<(arguments['nvb']-1).bit_length(),'num_warps':4,
            'has_add':arguments['addend'] is not None,'apply_out_norm':arguments['ow'] is not None,
            'write_bank':arguments['write_prefix']},
        live_parity_sampling={'schema':'native-independent-token-rows-v1','rows':rows,'full_native_tokens':tokens,
            'selection':'first actual token rows in every recorded structural class',
            'complete_native_tensor_dump':False,
            'justification':'one CTA per token; all bank/prefix accesses use the matching token index; no cross-token reduction'})
    family=rc.Family(FAMILY,source_hash,roles,output_roles)
    return family,bindings,controls


def outputs_for(arguments,result):
    if result is not None:raise ValueError('Native aggregation return ABI changed')
    rows=min(SAMPLE_ROWS,arguments['prefix_sum'].shape[0])
    return {name:sampled(arguments[name],rows) for name in ('out','prefix_out','bank')}


@functools.lru_cache(maxsize=4)
def _source_identity(path):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner_bindings(runtime,module,arguments):
    """Typed-owner API matching the shared dense capture adapter interface."""
    return make_bindings(runtime,arguments,_source_identity(module.__file__))


def configure(owner):
    """Call before owner.install() in a new combined supplemental bundle."""
    import work_classes
    if getattr(owner,'_attn_res_aggregation_capture_configured',False):raise ValueError('Aggregation capture adapter is already configured')
    if FAMILY in owner.TARGETS.get(MODULE,[]):raise ValueError('Aggregation native target is already registered')
    original_bindings=owner.make_bindings;original_outputs=owner.outputs_for
    original_classify=work_classes.classify
    owner.TARGETS.setdefault(MODULE,[]).append(FAMILY)
    def bindings(name,arguments,source_hash):
        return make_bindings(owner.rc,arguments,source_hash) if name==FAMILY else original_bindings(name,arguments,source_hash)
    def outputs(name,arguments,result):
        return outputs_for(arguments,result) if name==FAMILY else original_outputs(name,arguments,result)
    def classify(family,controls,values,input_metadata):
        if family!=FAMILY:return original_classify(family,controls,values,input_metadata)
        launch=controls['native_launch']
        return ({'policy':'kimi-attn-res-native-independent-tokens-v1','family':family,'native_launch':launch,
                 'full_native_abi_preserved_in_controls':True},
                {'native_tokens':launch['tokens'],'valid_bank_rows':launch['nvb'],
                 'actual_parity_sample_rows':controls['live_parity_sampling']['rows']})
    owner.make_bindings=bindings;owner.outputs_for=outputs;work_classes.classify=classify
    owner.SOURCE_HASHES['capture_adapter:attn_res_hip']=owner.rc.file_sha(__file__)
    owner._attn_res_aggregation_capture_configured=True

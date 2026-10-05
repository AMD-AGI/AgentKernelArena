"""Protected private bindings for the exact served native GPU implementations."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def load_file(path,name,tree=None):
    parent=name.rpartition('.')[0]
    if parent:importlib.import_module(parent)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:
        if tree is None:spec.loader.exec_module(module)
        else:exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def runtime_closure(root):
    import importlib.metadata
    if importlib.metadata.version('sglang')!='0.5.20':raise RuntimeError('SGLang 0.5.20 is required')
    cfg=json.loads((root/'provenance/SOURCE.json').read_text())
    for row in cfg['dependencies']:
        module=importlib.import_module(row['package'])
        if file_sha(Path(module.__file__).resolve().parent/row['path'])!=row['sha256']:
            raise RuntimeError('Installed native source differs: '+row['package']+'/'+row['path'])
    return cfg


def private_flydsl(root,leg,cfg,identity):
    selected=root/('source/flydsl' if leg=='candidate' else 'ut/reference/flydsl')
    name='aiter.ops.flydsl.kernels._aka_kimi_'+leg+'_'+identity
    package=types.ModuleType(name);package.__path__=[str(selected)];package.__package__=name
    sys.modules[name]=package
    dispatcher_path=selected/'mxmoe_dispatcher.py'
    tree=ast.parse(dispatcher_path.read_text(),filename=str(dispatcher_path))
    factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_gemm2_a4w4_port')
    matches=[n for n in factory.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='name' for t in n.targets)]
    if len(matches)!=1:raise RuntimeError('Frozen FlyDSL factory kernel symbol changed')
    matches[0].value=ast.BinOp(left=matches[0].value,op=ast.Add(),right=ast.Constant('_aka_'+leg+'_'+identity))
    dispatcher=load_file(dispatcher_path,name+'.mxmoe_dispatcher',tree)
    # Rebind only the frozen outer function's deferred dispatcher import. Each
    # leg resolves its own private module; no installed module is overwritten.
    host=root/'ut/host/fused_moe.py'
    host_tree=ast.parse(host.read_text(),filename=str(host))
    wrapper=next(n for n in host_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_flydsl_v2_stage2_wrapper')
    imports=[n for n in ast.walk(wrapper) if isinstance(n,ast.ImportFrom) and n.module=='aiter.ops.flydsl.kernels.mxmoe_dispatcher']
    if len(imports)!=1:raise RuntimeError('Frozen stage-2 dispatcher import changed')
    imports[0].module=name+'.mxmoe_dispatcher'
    module=load_file(host,'aiter._aka_kimi_fused_'+leg+'_'+identity,host_tree)
    return module._flydsl_v2_stage2_wrapper,{'private_dispatcher':dispatcher.__name__,
        'device_emitter':name+'.mxmoe_gemm_v2:gemm2_body_v2','kernel_symbol_suffix':'_aka_'+leg+'_'+identity}


class NativeBindings:
    def __init__(self,root,request):
        self.root=Path(root);self.request=request;self.cfg=runtime_closure(self.root)
        self.callables={};self.proofs={}
    def get(self,family,leg):
        key=(family,leg)
        if key in self.callables:return self.callables[key]
        source_hashes={relative:file_sha(self.root/(relative if leg=='candidate' else entry['reference']))
                       for relative,entry in self.cfg['editable_sources'].items()}
        identity=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()[:20]
        if family=='_decode_lean_attention_fwd':
            path=self.root/('source/decode_attention.py' if leg=='candidate' else 'ut/reference/decode_attention.py')
            module=load_file(path,'sglang.kernels.ops.attention._aka_kimi_lean_'+leg+'_'+identity)
            fn=module._decode_lean_attention_fwd
            proof={'binding':'private_triton_module','module':module.__name__}
        elif family=='_flydsl_v2_stage2_wrapper':
            fn,proof=private_flydsl(self.root,leg,self.cfg,identity)
        elif family=='opus_moe_stage2_a8w4_decode_fwd':
            from native_build import build_hip
            module=load_file(self.root/'ut/host/opus.py','aiter.ops.opus._aka_kimi_'+leg+'_'+identity)
            binding,proof=build_hip(self.root,self.cfg,leg)
            module._opus_moe_stage2_a8w4_decode_fwd_raw=binding('opus_moe_stage2_a8w4_decode_fwd')
            fn=module.opus_moe_stage2_a8w4_decode_fwd
        else:raise RuntimeError('Uncaptured native family')
        self.callables[key]=fn;self.proofs[key]={'source_sha256':source_hashes,**proof}
        return fn

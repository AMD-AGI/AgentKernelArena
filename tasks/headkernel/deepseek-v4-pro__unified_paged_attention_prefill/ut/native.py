"""Load complete native Python modules with the original package context."""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def private_torch_op_tree(source, targets, identity):
    """Give only selected frozen wrappers a private registration name.

    AITER's guard keys torch.ops.aiter by function name, not module identity.
    Re-exporting the decorated wrapper under its public module attribute keeps
    the native body/signature/decorator unchanged while isolating its globals.
    """
    tree=ast.parse(source);found={};body=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in targets:
            public=node.name
            decorators=[item for item in node.decorator_list if isinstance(item,ast.Call)
                        and isinstance(item.func,ast.Name) and item.func.id=='torch_compile_guard']
            if len(decorators)!=1:raise RuntimeError('Expected one frozen torch_compile_guard decorator')
            private='_aka_'+identity+'_'+public
            node.name=private;found[public]='aiter::'+private
            body.append(node)
            body.append(ast.Assign(targets=[ast.Name(id=public,ctx=ast.Store())],value=ast.Name(id=private,ctx=ast.Load())))
        else:body.append(node)
    if set(found)!=set(targets):raise RuntimeError('Missing frozen private torch-op wrapper')
    tree.body=body
    return ast.fix_missing_locations(tree),found


def load_file(path, qualified_name, *, private_torch_ops=()):
    parent = qualified_name.rpartition('.')[0]
    if parent: importlib.import_module(parent)
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        if private_torch_ops:
            identity=qualified_name.rpartition('.')[2]+'_'+uuid.uuid4().hex
            tree,operators=private_torch_op_tree(path.read_text(),private_torch_ops,identity)
            exec(compile(tree,str(path),'exec'),module.__dict__)
            module._aka_private_torch_ops=operators
            module._aka_private_backend_sha256=sha256(path)
        else:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified_name, None)
        raise
    return module


def runtime_closure(root):
    import importlib.metadata
    if importlib.metadata.version('sglang') != '0.5.20': raise RuntimeError('Exact SGLang 0.5.20 required')
    cfg=json.loads((root/'provenance/SOURCE.json').read_text())
    for row in cfg['dependencies']:
        pkg = importlib.import_module(row['package'])
        base = Path(pkg.__file__).resolve().parent
        path = base / row['path']
        if sha256(path) != row['sha256']: raise RuntimeError('Runtime source closure differs: ' + str(path))
    return cfg


def load_native(root, leg):
    cfg=json.loads((root/'provenance/SOURCE.json').read_text())
    if cfg.get('gpu_source_status')!='READY_FOR_GPU_VALIDATION':
        raise RuntimeError('No approved editable GPU implementation: '+cfg.get('gpu_source_status','unmapped'))
    cfg=runtime_closure(root)
    from native_build import source_hashes
    hashes=source_hashes(root,cfg,leg)
    parent=cfg['module'].rpartition('.')[0]
    alias=parent+'._aka_'+cfg['seam']+'_'+leg
    kind=cfg['gpu_binding']['kind'];proof={}
    if kind=='triton_module':
        relative=next(iter(cfg['editable_sources']))
        path=root/(relative if leg=='candidate' else cfg['editable_sources'][relative]['reference'])
        module=load_file(path,alias)
    else:
        path=root/cfg['host_source']
        if sha256(path)!=cfg['native_source_sha256']:raise RuntimeError('Frozen production wrapper changed')
        module=load_file(path,alias)
        if kind=='hip':
            from native_build import build_hip
            binding,proof=build_hip(root,cfg,leg)
            target=module
            if cfg['gpu_binding'].get('backend_source'):
                backend=cfg['gpu_binding']['backend_source']
                target=load_file(root/backend,'aiter.ops._aka_ds_prefill_'+leg,
                                 private_torch_ops=('pa_sparse_prefill_opus',))
                proof['private_torch_ops']=target._aka_private_torch_ops
                proof['private_backend_sha256']=target._aka_private_backend_sha256
                if not module._HAS_OPUS:raise RuntimeError('Current prefill contract requires the Opus path')
                module.pa_sparse_prefill_opus=target.pa_sparse_prefill_opus
            for python_name,native_name in cfg['gpu_binding']['exports'].items():
                setattr(target,python_name,binding(native_name))
        elif kind=='flydsl_stage1':
            from native_build import load_flydsl_emitter
            relative=next(iter(cfg['editable_sources']));entry=cfg['editable_sources'][relative]
            selected=root/(relative if leg=='candidate' else entry['reference'])
            emitter=load_flydsl_emitter(selected,'aiter.ops.flydsl.kernels._aka_ds_moe1_'+leg,hashes[relative])
            compiler=load_file(root/cfg['gpu_binding']['compiler_source'],'aiter.ops.flydsl.kernels._aka_ds_moe1_dispatch_'+leg)
            compiler.compile_mixed_moe_gemm1_common=emitter.compile_mixed_moe_gemm1_common
            import inspect
            signature=inspect.signature(module.compile_flydsl_moe_stage1)
            def compile_stage1(**kwargs):
                bound=signature.bind(**kwargs);bound.apply_defaults();values=dict(bound.arguments)
                if values['a_dtype']=='bf16' or values['b_dtype'] not in ('fp4','fp8'):
                    raise RuntimeError('Captured case does not use the mapped mixed-MoE GPU emitter')
                values['gate_mode']=compiler.GateMode(values['gate_mode'])
                return compiler.compile_mixed_moe_gemm1(**values)
            module._flydsl_moe_stage1_impl.__kwdefaults__['_compile_kernel']=compile_stage1
            proof={'emitter_source_sha256':hashes[relative],'kernel_symbol_suffix':'_aka_'+hashes[relative][:20],
                   'host_wrapper_frozen':True}
        else:raise RuntimeError('Unsupported protected GPU binding')
    fn=getattr(module,cfg['function'])
    import inspect
    expected=cfg['signature_parameters']
    names=[k for k,v in inspect.signature(fn).parameters.items() if v.kind != v.VAR_KEYWORD]
    if names != expected: raise RuntimeError('Native API changed')
    if kind=='flydsl_stage1':
        from native_build import guarded_flydsl_call
        fn=guarded_flydsl_call(fn)
    return module, fn, {'leg':leg,'source_sha256':hashes,'module':alias,'host_file':str(path.relative_to(root)),
                        'gpu_binding':kind,**proof}

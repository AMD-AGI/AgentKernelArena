"""Protected source-to-module adapters; neither launch code nor cache policy is editable."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import functools
import os
from pathlib import Path
import re
import sys
import uuid


def source_hashes(root,cfg,leg):
    result={}
    for relative,entry in cfg['editable_sources'].items():
        path=root/(relative if leg=='candidate' else entry['reference'])
        result[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def guarded_flydsl_call(function):
    """Reject dispatch branches outside the staged mixed-MoE emitter."""
    signature=inspect.signature(function)
    @functools.wraps(function)
    def invoke(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        if bound.arguments['a_dtype'] not in ('fp8','fp4','fp16') or bound.arguments['b_dtype'] not in ('fp4','fp8'):
            raise RuntimeError('Captured case selects an unstaged FlyDSL GPU implementation')
        return function(*args,**kwargs)
    return invoke


@contextmanager
def jit_scope(core,destination,csrc):
    old_env=os.environ.get('AITER_JIT_DIR');old_build=core.bd_dir;old_csrc=core.AITER_CSRC_DIR;old_path=list(sys.path)
    try:
        os.environ['AITER_JIT_DIR']=str(destination)
        core.get_user_jit_dir.cache_clear();core.get_user_jit_dir()
        core.bd_dir=str(destination/'build');core.AITER_CSRC_DIR=str(csrc)
        yield
    finally:
        if old_env is None:os.environ.pop('AITER_JIT_DIR',None)
        else:os.environ['AITER_JIT_DIR']=old_env
        core.bd_dir=old_build;core.AITER_CSRC_DIR=old_csrc
        core.get_user_jit_dir.cache_clear();core.get_user_jit_dir();sys.path[:]=old_path


def stage_csrc(root,cfg,leg):
    original=root/'ut/native/csrc'
    sources={path.relative_to(original).as_posix():path.read_text() for path in original.rglob('*') if path.is_file()}
    for relative,entry in cfg['editable_sources'].items():
        if 'native_relative' in entry:
            sources[entry['native_relative']]=(root/(relative if leg=='candidate' else entry['reference'])).read_text()
    digest=hashlib.sha256()
    for name,data in sorted(sources.items()):digest.update(name.encode()+b'\0'+data.encode())
    identity=digest.hexdigest();stage=root/'build/native_source'/f'{leg}_{identity[:20]}_{uuid.uuid4().hex}'
    stage.mkdir(parents=True)
    pattern=re.compile(r'#include\s*[<"]([^>"\n]+)[>"]')
    for name,data in sources.items():
        source=Path(name)
        def replace(match):
            for candidate in (source.parent/match[1],Path('include')/match[1],
                              Path('opus_moe/include')/match[1],Path('opus_gemm/include')/match[1]):
                normalized=os.path.normpath(candidate)
                if normalized in sources:return '#include "'+str(stage/normalized)+'"'
            return match[0]
        destination=stage/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(pattern.sub(replace,data))
    return stage,identity


def build_hip(root,cfg,leg):
    from aiter.jit import core
    stage,identity=stage_csrc(root,cfg,leg)
    recipe=cfg['native_build'];options=core.get_args_of_build(recipe['module'])
    if options.get('third_party'):raise RuntimeError('Native task build may not fetch dependencies')
    original_csrc=str(core.AITER_CSRC_DIR)
    def mapped(value):
        if isinstance(value,str):return value.replace(original_csrc,str(stage))
        if isinstance(value,list):return [mapped(item) for item in value]
        if isinstance(value,dict):return {key:mapped(item) for key,item in value.items()}
        return value
    options={key:mapped(value) for key,value in options.items()}
    module_name='module_aka_ds_'+leg+'_'+identity[:16]+'_'+uuid.uuid4().hex[:12]
    destination=root/'build/aiter_jit'/module_name;destination.mkdir(parents=True)
    options.update(md_name=module_name,srcs=[str(stage/name) for name in recipe['translation_units']])
    options['extra_include']=[str(stage/'include'),str(stage/'opus_moe/include'),str(stage/'opus_gemm/include')]+options.get('extra_include',[])
    with jit_scope(core,destination,stage):
        accepted=inspect.signature(core.build_module).parameters
        core.build_module(**{key:value for key,value in options.items() if key in accepted})
        native=core.get_module(module_name)
    extension=Path(native.__file__).resolve()
    if not extension.is_relative_to(destination.resolve()):raise RuntimeError('Native module did not load from its fresh build')
    convert,tensor_type,stream,device=core._pybind_develop_hooks()
    def binding(name):
        def invoke(*args,**kwargs):
            args=tuple(convert(value) if isinstance(value,tensor_type) else value for value in args)
            kwargs={key:convert(value) if isinstance(value,tensor_type) else value for key,value in kwargs.items()}
            native._set_current_hip_stream(stream(device()))
            return getattr(native,name)(*args,**kwargs)
        return invoke
    proof={'fresh_compilation':True,'source_tree_sha256':identity,'module_name':module_name,
           'extension_sha256':hashlib.sha256(extension.read_bytes()).hexdigest(),'extension_path':str(extension.relative_to(root))}
    return binding,proof


def load_flydsl_emitter(path,qualified_name,source_identity):
    """Add a source-keyed kernel symbol in the frozen factory, then compile its AST."""
    import importlib
    import importlib.util
    importlib.import_module(qualified_name.rpartition('.')[0])
    tree=ast.parse(path.read_text(),filename=str(path));factory=next(node for node in tree.body
        if isinstance(node,ast.FunctionDef) and node.name=='compile_mixed_moe_gemm1_common')
    matches=[]
    for index,node in enumerate(factory.body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='module_name' for t in node.targets):matches.append(index)
    if len(matches)!=1:raise RuntimeError('Frozen FlyDSL factory module-name assignment changed')
    index=matches[0]+1
    factory.body.insert(index,ast.AugAssign(target=ast.Name(id='module_name',ctx=ast.Store()),op=ast.Add(),
                                         value=ast.Constant(value='_aka_'+source_identity[:20])))
    spec=importlib.util.spec_from_file_location(qualified_name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[qualified_name]=module
    try:exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(qualified_name,None)
        raise
    return module

"""Closed GPU/IR-body edits with immutable host wrappers and launch scaffolding."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from cpp_body_guard import validate_cpp

SYNTAX={ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.AnnAssign,ast.AugAssign,
 ast.If,ast.IfExp,ast.For,ast.While,ast.With,ast.withitem,ast.Return,ast.Break,ast.Continue,
 ast.Pass,ast.Expr,ast.Raise,ast.Assert,ast.Name,ast.Attribute,ast.Call,ast.keyword,
 ast.Constant,ast.Subscript,ast.Slice,ast.Tuple,ast.List,ast.Dict,ast.Set,ast.ListComp,
 ast.comprehension,ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.JoinedStr,
 ast.FormattedValue,ast.Load,ast.Store,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,
 ast.Mod,ast.Pow,ast.MatMult,ast.USub,ast.UAdd,ast.Not,ast.Invert,ast.And,ast.Or,
 ast.BitAnd,ast.BitOr,ast.BitXor,ast.LShift,ast.RShift,ast.Eq,ast.NotEq,ast.Lt,
 ast.LtE,ast.Gt,ast.GtE,ast.Is,ast.IsNot,ast.In,ast.NotIn,ast.Starred}
PURE={'range','min','max','int','float','bool','len','list','tuple','enumerate','zip','abs','sum','RuntimeError','ValueError'}
FORBIDDEN={'torch','os','sys','inspect','gc','ctypes','threading','multiprocessing','asyncio','subprocess',
 'importlib','pickle','marshal','builtins','globals','locals','vars','dir','eval','exec','compile','open',
 'getattr','setattr','delattr','type','object','super','print','input','breakpoint'}
TL_OPS={'program_id','arange','load','store','zeros','full','dot','sum','max','min','exp','exp2','log','log2',
 'maximum','minimum','where','sqrt','rsqrt','abs','cdiv','multiple_of','max_contiguous','static_range',
 'range','reshape','trans','broadcast_to','gather','cast','fdiv','div_rn','sigmoid','ceil','floor','is_nan',
 'int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','bfloat16','float32','float64',
 'float8e4nv','float8e4b8','float8e5','constexpr'}


def dump(node):return ast.dump(node,include_attributes=False)


def signature(node):
    node=copy.deepcopy(node);node.body=[ast.Pass()];return dump(node)


def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        root=dotted(node.value)
        return None if root is None else root+'.'+node.attr
    return None


def bindings(node):
    return ({n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
            |{n.arg for n in ast.walk(node) if isinstance(n,ast.arg)}
            |{n.name for n in ast.walk(node) if isinstance(n,ast.FunctionDef)})


def functions(tree):
    result={}
    def walk(node,prefix):
        for child in ast.iter_child_nodes(node):
            key=prefix
            if isinstance(child,ast.FunctionDef):
                key=prefix+(child.name,);result.setdefault('.'.join(key),[]).append(child)
            walk(child,key)
    walk(tree,());return result


def check_python_body(candidate,reference,language):
    edited=ast.Module(body=candidate.body,type_ignores=[]);original=ast.Module(body=reference.body,type_ignores=[])
    old_nodes=list(ast.walk(original));local=bindings(candidate);old_local=bindings(reference)
    closure={n.id for n in old_nodes if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}-old_local
    if closure & FORBIDDEN:raise ValueError('Declared GPU emission body contains host runtime capabilities')
    call_names={n.func.id for n in old_nodes if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    attributes={n.attr for n in old_nodes if isinstance(n,ast.Attribute)}
    paths={dotted(n) for n in old_nodes if isinstance(n,ast.Attribute) and dotted(n)}
    strings={n.value for n in old_nodes if isinstance(n,ast.Constant) and isinstance(n.value,(str,bytes))}
    nested=Counter(signature(n) for n in old_nodes if isinstance(n,ast.FunctionDef))
    seen=Counter();fixed_calls={dump(n) for n in old_nodes if isinstance(n,ast.Call)
        and ((isinstance(n.func,ast.Name) and n.func.id=='hasattr')
             or isinstance(n.func,ast.Attribute) and n.func.attr.startswith('__'))}
    capabilities=closure|PURE|{'tl','triton','hasattr'}
    for node in ast.walk(edited):
        if type(node) not in SYNTAX:raise ValueError('Syntax outside the closed GPU language: '+type(node).__name__)
        if isinstance(node,ast.Name):
            if node.id in FORBIDDEN or node.id.startswith('__'):raise ValueError('Host introspection/runtime name is forbidden')
            if isinstance(node.ctx,ast.Store) and node.id in capabilities:raise ValueError('GPU capability rebinding is forbidden')
            if isinstance(node.ctx,ast.Load) and node.id not in local|closure|PURE|{'tl','hasattr'}:raise ValueError('Unknown GPU value/capability: '+node.id)
        if isinstance(node,ast.FunctionDef):
            token=signature(node);seen[token]+=1
            if language=='triton' or seen[token]>nested[token]:raise ValueError('New or changed nested host/device function is forbidden')
        if isinstance(node,ast.Attribute):
            if not isinstance(node.ctx,ast.Load):raise ValueError('Python object/module mutation is forbidden')
            path=dotted(node)
            if language=='triton':
                if isinstance(node.value,ast.Name) and node.value.id=='tl':
                    if node.attr not in TL_OPS:raise ValueError('Unknown Triton GPU operation/type')
                elif node.attr not in {'to','dtype','element_ty','shape','T'}:raise ValueError('Host attribute access is forbidden')
            else:
                if node.attr not in attributes:raise ValueError('Attribute outside immutable DSL surface')
                if path and path.split('.')[0] in closure and path not in paths:raise ValueError('Unknown DSL capability path')
        if isinstance(node,ast.Constant) and isinstance(node.value,(str,bytes)) and node.value not in strings:
            raise ValueError('New host strings/IR names are forbidden')
        if isinstance(node,ast.Subscript) and isinstance(node.ctx,ast.Store):
            root=node.value
            while isinstance(root,ast.Subscript):root=root.value
            if language=='triton' or not isinstance(root,ast.Name) or root.id not in local or root.id in capabilities:
                raise ValueError('Only local DSL value collections may be updated')
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Name):
                if node.func.id=='hasattr':
                    if dump(node) not in fixed_calls:raise ValueError('Compiler compatibility reflection is frozen')
                elif node.func.id not in PURE|call_names:raise ValueError('Unknown direct GPU/DSL call')
            elif isinstance(node.func,ast.Attribute):
                if node.func.attr.startswith('__') and dump(node) not in fixed_calls:raise ValueError('Compiler context methods are frozen')
                if language=='triton' and not (isinstance(node.func.value,ast.Name) and node.func.value.id=='tl') and node.func.attr!='to':
                    raise ValueError('Only GPU tensor casts are permitted method calls')
            else:raise ValueError('Indirect/computed Python calls are forbidden')


def validate_python(candidate,reference,targets,language):
    edited=ast.parse(candidate);original=ast.parse(reference)
    cdefs=functions(edited);rdefs=functions(original)
    for target in targets:
        if len(cdefs.get(target,[]))!=1 or len(rdefs.get(target,[]))!=1:raise ValueError('GPU body target missing or ambiguous: '+target)
        c,r=cdefs[target][0],rdefs[target][0]
        if language=='triton' and not any(dotted(node.func if isinstance(node,ast.Call) else node)=='triton.jit' for node in r.decorator_list):
            raise ValueError('A Triton edit target must be an actual frozen @triton.jit function')
        if language=='flydsl' and target!='compile_mixed_moe_gemm1_common._emit_moe_gemm1':
            raise ValueError('FlyDSL target is outside the reviewed device emitter')
        check_python_body(c,r,language)
        c.body=[ast.Pass()];r.body=[ast.Pass()]
    if dump(edited)!=dump(original):raise ValueError('Host functions, imports, decorators and GPU signatures are frozen')


def validate_sources(candidate_root,reference_root):
    candidate_root=Path(candidate_root).resolve();reference_root=Path(reference_root).resolve()
    policy=json.loads((reference_root/'ut/source_guard_policy.json').read_text())
    for relative,expected in policy.get('frozen_files',{}).items():
        path=candidate_root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(candidate_root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen host/build source changed: '+relative)
    for relative,entry in policy['sources'].items():
        path=candidate_root/relative;original=reference_root/entry['reference']
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(candidate_root):raise ValueError('Editable source must be regular and task-contained')
        if entry['language']=='hip':validate_cpp(path.read_text(),original.read_text(),entry['markers'])
        elif entry['language'] in ('triton','flydsl'):validate_python(path.read_text(),original.read_text(),entry['targets'],entry['language'])
        else:raise ValueError('No reviewed GPU implementation boundary for this source')
    return True

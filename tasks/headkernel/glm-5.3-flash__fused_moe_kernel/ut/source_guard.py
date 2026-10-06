"""Restrict editable Python to the GPU JIT body; all host code and ABI are frozen."""
import ast
from pathlib import Path

TARGETS = {'quantize_input','stage1','activate_quantize','stage2','reduce_routes'}
BUILTIN_CALLS = {'range','min','max','int','float'}
RESERVED = {'tl','triton',*BUILTIN_CALLS}
FORBIDDEN_NAMES = {'sys','os','builtins','globals','locals','vars','dir','eval','exec','compile','open','getattr','setattr','delattr','type','object','super'}
TL_TYPES = {'float32','float16','bfloat16','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float8e4nv','float8e5','constexpr'}
ALLOWED_TL = {'program_id','arange','load','store','zeros','dot','minimum','maximum','where','sum','max','exp','exp2','sigmoid','cdiv','multiple_of','max_contiguous','static_range','full','reshape','trans','broadcast_to','abs','cast','fdiv','div_rn'}
ALLOWED_TL |= {'fma','inline_asm_elementwise'}
GPU_ASM = {
    'v_rcp_f32 $0, $1;': ('=v,v',1),
}


def _approved_asm_strings(function):
    strings=set()
    for node in ast.walk(function):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='tl'
                and node.func.attr=='inline_asm_elementwise'):continue
        fields={item.arg:item.value for item in node.keywords}
        if node.args or len(fields)!=len(node.keywords) or set(fields)!={'asm','constraints','args','dtype','is_pure','pack'}:
            raise ValueError('GPU assembly requires the exact declared intrinsic signature')
        assembly=fields['asm'];constraints=fields['constraints'];arguments=fields['args']
        if not isinstance(assembly,ast.Constant) or assembly.value not in GPU_ASM:
            raise ValueError('GPU assembly opcode sequence is not declared')
        expected,count=GPU_ASM[assembly.value]
        if not isinstance(constraints,ast.Constant) or constraints.value!=expected or not isinstance(arguments,ast.List) or len(arguments.elts)!=count:
            raise ValueError('GPU assembly operands or constraints differ')
        dtype=fields['dtype']
        if not isinstance(dtype,ast.Attribute) or not isinstance(dtype.value,ast.Name) or dtype.value.id!='tl' or dtype.attr!='float32':
            raise ValueError('GPU assembly dtype must be float32')
        if not isinstance(fields['is_pure'],ast.Constant) or fields['is_pure'].value is not True or not isinstance(fields['pack'],ast.Constant) or type(fields['pack'].value) is not int or fields['pack'].value!=1:
            raise ValueError('GPU assembly purity and scalar packing are fixed')
        strings.update((id(assembly),id(constraints)))
    return strings


def _tree(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Editable source must be regular')
    return ast.parse(path.read_text(),filename=str(path))


def _body_guard(function):
    asm_strings=_approved_asm_strings(function)
    for node in ast.walk(ast.Module(body=function.body,type_ignores=[])):
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda,ast.With,ast.AsyncWith,ast.Try,ast.Raise,ast.Delete,ast.Yield,ast.YieldFrom,ast.Await)):
            raise ValueError('Only GPU tensor operations are editable')
        if isinstance(node,ast.Name):
            if node.id.startswith('__') or node.id in FORBIDDEN_NAMES:raise ValueError('Host Python names are forbidden')
            if isinstance(node.ctx,(ast.Store,ast.Del)) and node.id in RESERVED:raise ValueError('GPU namespaces and callable builtins are immutable')
            if isinstance(node.ctx,ast.Load) and node.id=='triton':raise ValueError('Host Triton runtime access is forbidden inside GPU code')
        if isinstance(node,ast.Attribute):
            if node.attr.startswith('__'):raise ValueError('Python introspection is forbidden')
            if isinstance(node.value,ast.Name) and node.value.id=='tl':
                if node.attr not in ALLOWED_TL | TL_TYPES:raise ValueError('Only declared Triton language operations/types are available')
            elif node.attr not in {'to','dtype','element_ty','shape','T'}:
                raise ValueError('Host object attribute access is forbidden')
        if isinstance(node,ast.Constant) and isinstance(node.value,(str,bytes)) and id(node) not in asm_strings:
            raise ValueError('String/byte payloads are not GPU values')
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if any(isinstance(x,(ast.Attribute,ast.Subscript)) for x in ast.walk(target)):raise ValueError('Host/global object mutation is forbidden')
        if isinstance(node,ast.Call):
            fn=node.func
            if isinstance(fn,ast.Name) and fn.id in BUILTIN_CALLS:continue
            if isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name) and fn.value.id=='tl' and fn.attr in ALLOWED_TL:continue
            if isinstance(fn,ast.Attribute) and fn.attr=='to':
                receiver=fn.value
                if isinstance(receiver,ast.Name) and receiver.id not in RESERVED | FORBIDDEN_NAMES:continue
                if isinstance(receiver,(ast.BinOp,ast.UnaryOp,ast.Subscript)):continue
                if isinstance(receiver,ast.Call) and isinstance(receiver.func,ast.Attribute) and isinstance(receiver.func.value,ast.Name) and receiver.func.value.id=='tl' and receiver.func.attr in ALLOWED_TL:continue
                raise ValueError('Cast receiver must be a GPU expression')
            raise ValueError('Untrusted Python call in GPU body')


def validate_sources(candidate_root, reference_root):
    candidate_root=Path(candidate_root).resolve();reference_root=Path(reference_root).resolve()
    path=candidate_root/'source/kernels.py'
    if not path.resolve().is_relative_to(candidate_root):raise ValueError('Source escapes task root')
    candidate=_tree(path);reference=_tree(reference_root/'ut/reference/kernels.py')
    cdefs={x.name:x for x in candidate.body if isinstance(x,ast.FunctionDef)}
    rdefs={x.name:x for x in reference.body if isinstance(x,ast.FunctionDef)}
    if set(cdefs)!=set(rdefs) or not TARGETS.issubset(cdefs):raise ValueError('GPU entrypoint set changed')
    for name in TARGETS:
        _body_guard(cdefs[name])
        cdefs[name].body=[ast.Pass()];rdefs[name].body=[ast.Pass()]
    if ast.dump(candidate,include_attributes=False)!=ast.dump(reference,include_attributes=False):
        raise ValueError('Imports, host launch, constants, decorators and kernel ABI are frozen')

"""CPU-only boundary: only the declared Triton GPU kernel bodies may change."""
import ast
import builtins
import hashlib
import json
import symtable
from pathlib import Path


def _require(ok, message):
    if not ok:
        raise ValueError(message)


# Keep this surface explicit: triton.language also exposes Python modules and
# compiler implementation objects that must never be reachable from a candidate.
_DSL_CALLS = frozenset({
    "assume",
    "abs", "add", "advance", "arange", "argmax", "argmin", "atomic_add", "atomic_and",
    "atomic_cas", "atomic_max", "atomic_min", "atomic_or", "atomic_xchg", "atomic_xor",
    "broadcast", "broadcast_to", "cast", "cat", "cdiv", "ceil", "clamp", "cos",
    "cumprod", "cumsum", "div_rn", "dot", "dot_scaled", "erf", "exp", "exp2",
    "expand_dims", "fdiv", "flip", "floor", "fma", "full", "full_like", "gather",
    "histogram", "inline_asm_elementwise", "interleave", "join", "load", "log", "log2",
    "make_block_ptr", "max", "max_constancy", "max_contiguous", "maximum", "min",
    "minimum", "multiple_of", "num_programs", "permute", "program_id", "range", "ravel",
    "reshape", "rsqrt", "sigmoid", "sin", "sort", "split", "sqrt", "sqrt_rn",
    "static_assert", "static_range", "store", "sum", "swizzle2d", "trans", "umulhi",
    "where", "xor_sum", "zeros", "zeros_like",
})
_DSL_VALUES = frozenset({
    "bfloat16", "constexpr", "float16", "float32", "float64", "float8e4b15", "float8e4b8",
    "float8e4nv", "float8e5", "float8e5b16", "int1", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
})
_BUILTIN_CALLS = frozenset({"range", "min", "max", "int", "float", "abs", "len"})
_TENSOR_METHODS = frozenset({"to", "reshape", "trans", "astype"})
_TENSOR_PROPERTIES = frozenset({"dtype", "type", "element_ty", "shape", "ndim", "numel", "T"})
_DEVICE_CALLS = frozenset({"pid_grid", "remap_xcd"})
_GPU_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.For,
                   ast.While, ast.If, ast.Break, ast.Continue, ast.Return, ast.Pass,
                   ast.Assert)


def _body_safety(function, module_names):
    body = ast.Module(body=function.body, type_ignores=[])
    nodes = list(ast.walk(body))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    arguments = function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    parameters = {arg.arg for arg in arguments}
    parameters.update(arg.arg for arg in (function.args.vararg, function.args.kwarg) if arg)
    locals_ = {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    # A name used before a later assignment can still resolve to a frozen global
    # during Triton compilation. Prevent all such shadowing, not just direct calls.
    reserved = set(module_names) | set(vars(builtins)) | {"tl"}
    _require(not locals_ & reserved, "rebinding a DSL, builtin or frozen global name is forbidden")
    for node in nodes:
        if isinstance(node, ast.stmt):
            _require(isinstance(node, _GPU_STATEMENTS), "host statements are forbidden in the editable GPU body")
        _require(not isinstance(node, (ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom)),
                 "host expressions are forbidden in the editable GPU body")
        parent = parents.get(node)
        is_call_target = isinstance(parent, ast.Call) and parent.func is node
        if isinstance(node, ast.Name):
            _require(not node.id.startswith("__"), "dunder access is forbidden")
            _require(not isinstance(node.ctx, ast.Del), "deleting names is forbidden")
            if node.id == "tl":
                _require(isinstance(parent, ast.Attribute) and parent.value is node,
                         "the DSL namespace may only be used for direct public members")
            elif node.id in _BUILTIN_CALLS | _DEVICE_CALLS:
                _require(is_call_target, "builtin callables may not be aliased")
            elif isinstance(node.ctx, ast.Load):
                _require(node.id in parameters | locals_, "undeclared global in GPU body: " + node.id)
        if isinstance(node, ast.Attribute):
            _require(isinstance(node.ctx, ast.Load), "attribute mutation is forbidden")
            if isinstance(node.value, ast.Name) and node.value.id == "tl":
                _require(node.attr in _DSL_CALLS | _DSL_VALUES,
                         "non-public or unapproved DSL member: " + node.attr)
                if node.attr in _DSL_CALLS:
                    _require(is_call_target, "DSL callables may not be aliased or inspected")
            else:
                _require(node.attr in _TENSOR_PROPERTIES or (node.attr in _TENSOR_METHODS and is_call_target),
                         "non-DSL attribute in GPU body: " + node.attr)
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                _require(target.id in _BUILTIN_CALLS | _DEVICE_CALLS, "undeclared host call in GPU body: " + target.id)
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id == "tl":
                    _require(target.attr in _DSL_CALLS, "non-callable DSL member: " + target.attr)
                else:
                    _require(target.attr in _TENSOR_METHODS, "non-DSL method call in GPU body: " + target.attr)
            else:
                raise ValueError("dynamic callable expression in GPU body")


def _protected_tree(path, targets):
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    module_names = symtable.symtable(source, str(path), "exec").get_identifiers()
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            found.add(node.name)
            _body_safety(node, module_names)
            node.body = [ast.Pass()]
    _require(found == set(targets), "declared GPU kernel definition is missing")
    return ast.dump(tree, include_attributes=False)


def validate_sources(candidate_root, reference_root):
    candidate_root, reference_root = Path(candidate_root), Path(reference_root)
    definition = json.loads((reference_root / "task_definition.json").read_text())
    candidate = candidate_root / definition["source_file"]
    reference = reference_root / definition["reference_file"]
    _require(candidate.is_file() and not candidate.is_symlink(), "candidate must be a regular source file")
    _require(reference.is_file() and not reference.is_symlink(), "frozen source must be a regular file")
    _require(hashlib.sha256(reference.read_bytes()).hexdigest() == definition["source_sha256"],
             "frozen SG520 source hash differs")
    _require(_protected_tree(candidate, definition["kernels"]) == _protected_tree(reference, definition["kernels"]),
             "imports, decorators, signatures, host launchers and helpers are frozen")
    return True

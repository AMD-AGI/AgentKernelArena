"""Constrained DSL emission edits inside frozen Python launch scaffolding.

The trusted reference defines the available DSL operations and closure values.
Candidates may compose those operations using explicit arithmetic/control-flow
syntax. Python reflection, arbitrary calls, new imports/decorators, and writes
to module/object attributes are outside this language. This guard complements
isolated compilation; it is not a general Python sandbox.
"""
import ast
import builtins
from collections import Counter
import copy
import json
from pathlib import Path


SYNTAX = {
    ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.If, ast.IfExp, ast.For, ast.While, ast.With, ast.withitem, ast.Return,
    ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Raise, ast.Assert,
    ast.Name, ast.Attribute, ast.Call, ast.keyword, ast.Constant, ast.Subscript,
    ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.ListComp, ast.comprehension,
    ast.Lambda, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.JoinedStr,
    ast.FormattedValue, ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult, ast.USub, ast.UAdd, ast.Not,
    ast.Invert, ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift,
    ast.RShift, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is,
    ast.IsNot, ast.In, ast.NotIn, ast.Starred,
}
BUILTIN_NAMES = frozenset(vars(builtins))
PURE_BUILTINS = {'bool', 'int', 'float', 'min', 'max', 'range', 'len', 'list',
                 'tuple', 'enumerate', 'zip', 'abs', 'sum', 'RuntimeError', 'ValueError'}


def dump(node):
    return ast.dump(node, include_attributes=False)


def body_tree(function):
    return ast.Module(body=function.body, type_ignores=[])


def signature(function):
    value = copy.deepcopy(function)
    value.body = [ast.Pass()]
    return dump(value)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return None if base is None else base + '.' + node.attr
    return None


def bindings(tree):
    result = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    result.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    result.update(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    return result


def validate_body(candidate, reference):
    original = body_tree(reference)
    edited = body_tree(candidate)
    original_nodes = list(ast.walk(original))
    local_names = bindings(candidate)
    original_locals = bindings(reference)
    closure_names = {node.id for node in original_nodes if isinstance(node, ast.Name)
                     and isinstance(node.ctx, ast.Load)} - original_locals
    original_calls = {node.func.id for node in original_nodes if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)}
    original_attributes = {node.attr for node in original_nodes if isinstance(node, ast.Attribute)}
    fixed_paths = {dotted(node) for node in original_nodes if isinstance(node, ast.Attribute)
                   and dotted(node) is not None}
    original_strings = {node.value for node in original_nodes if isinstance(node, ast.Constant)
                        and isinstance(node.value, (str, bytes))}
    fixed_imports = Counter(dump(node) for node in original_nodes if isinstance(node, (ast.Import, ast.ImportFrom)))
    # A handful of stock compiler compatibility/context operations are retained
    # exactly, not exposed as callable capabilities for new code.
    fixed_calls = {dump(node) for node in original_nodes if isinstance(node, ast.Call)
                   and (isinstance(node.func, ast.Name) and node.func.id == 'hasattr'
                        or isinstance(node.func, ast.Attribute) and node.func.attr.startswith('__'))}
    original_functions = {}
    for node in original_nodes:
        if isinstance(node, ast.FunctionDef):
            original_functions.setdefault(node.name, set()).add(signature(node))
    imports_seen = Counter()
    local_functions = {node.name for node in ast.walk(edited) if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(edited):
        if isinstance(node, ast.Module):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            token = dump(node)
            imports_seen[token] += 1
            if imports_seen[token] > fixed_imports[token]:
                raise ValueError('Only exact existing DSL imports may remain inside an emission body')
            continue
        if isinstance(node, ast.alias):
            continue  # Its containing import was checked as a complete AST.
        if type(node) not in SYNTAX:
            raise ValueError('Syntax outside the constrained DSL language: ' + type(node).__name__)
        if isinstance(node, ast.Name):
            if node.id in BUILTIN_NAMES and node.id not in PURE_BUILTINS | {'hasattr'}:
                raise ValueError('Builtin capability forbidden by the DSL allowlist: ' + node.id)
            if node.id.startswith('__'):
                raise ValueError('Runtime introspection names are outside the DSL language')
            if isinstance(node.ctx, ast.Store) and node.id in closure_names:
                raise ValueError('Cannot rebind a frozen closure or DSL capability: ' + node.id)
            if isinstance(node.ctx, ast.Load) and node.id not in local_names | closure_names | PURE_BUILTINS:
                raise ValueError('Unknown DSL value or capability: ' + node.id)
        if isinstance(node, ast.Attribute):
            if not isinstance(node.ctx, ast.Load):
                raise ValueError('Writes to Python object/module attributes are forbidden')
            if node.attr not in original_attributes:
                raise ValueError('Attribute outside the frozen DSL surface: ' + node.attr)
            path = dotted(node)
            if path and path.split('.')[0] in closure_names and path not in fixed_paths:
                raise ValueError('Unknown path through a frozen DSL capability: ' + path)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            if node.value not in original_strings:
                raise ValueError('New host strings/IR names are outside the frozen DSL surface')
        if isinstance(node, ast.FunctionDef):
            if node.name in original_functions:
                if signature(node) not in original_functions[node.name]:
                    raise ValueError('Existing local device helper signatures/decorators are frozen')
            elif node.decorator_list or node.returns or any(arg.annotation for arg in
                    [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]):
                raise ValueError('New local helpers must have no decorators or annotations')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'hasattr':
                    if dump(node) not in fixed_calls:
                        raise ValueError('Compiler compatibility reflection is frozen')
                elif node.func.id not in original_calls | local_functions | PURE_BUILTINS:
                    raise ValueError('Indirect or unapproved Python call: ' + node.func.id)
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith('__') and dump(node) not in fixed_calls:
                    raise ValueError('Compiler context calls are frozen')
            else:
                raise ValueError('Calls through subscripts, lambdas and computed objects are forbidden')


def checked_tree(path, allowed, reference=None):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    original_tree = ast.parse(Path(reference or path).read_text())
    originals = {}
    for node in ast.walk(original_tree):
        if isinstance(node, ast.FunctionDef) and node.name in allowed:
            originals.setdefault(node.name, []).append(node)
    found = Counter()
    # Materialize the walk before removing any selected body.
    selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in allowed]
    for node in selected:
        index = found[node.name]
        if index >= len(originals.get(node.name, [])):
            raise ValueError('Unexpected duplicate emission function: ' + node.name)
        validate_body(node, originals[node.name][index])
        found[node.name] += 1
    if set(found) != set(allowed) or any(found[name] != len(nodes) for name, nodes in originals.items()):
        raise ValueError('Declared emission functions are missing or duplicated')
    for node in selected:
        node.body = [ast.Pass()]
    return dump(tree)


def validate_sources(candidate_root, reference_root):
    candidate_root, reference_root = Path(candidate_root), Path(reference_root)
    policy = json.loads((reference_root / 'ut/source_guard_policy.json').read_text())
    for relative, entry in policy['sources'].items():
        candidate = candidate_root / relative
        reference = reference_root / entry['reference']
        if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(candidate_root.resolve()):
            raise ValueError('Editable source must be a regular task-local file: ' + relative)
        allowed = entry['device_functions']
        if checked_tree(candidate, allowed, reference) != checked_tree(reference, allowed, reference):
            raise ValueError('Edit outside protected device emission body: ' + relative)
    return True

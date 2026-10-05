"""CPU-only boundary: only the declared Triton GPU kernel bodies may change."""
import ast
import hashlib
import json
from pathlib import Path


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _body_safety(function):
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With,
                 ast.AsyncWith, ast.Try, ast.Raise, ast.Lambda, ast.ClassDef)
    for node in ast.walk(ast.Module(body=function.body, type_ignores=[])):
        _require(not isinstance(node, forbidden), "host effects are forbidden in the editable GPU body")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not function:
            raise ValueError("nested host functions are outside the GPU edit boundary")
        if isinstance(node, ast.Name):
            _require(not node.id.startswith("__"), "dunder access is forbidden")
        if isinstance(node, ast.Attribute):
            _require(not node.attr.startswith("__"), "dunder access is forbidden")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                _require(target.id in {"range", "min", "max", "int", "float", "abs", "len"},
                         "undeclared host call in GPU body: " + target.id)
            elif isinstance(target, ast.Attribute):
                root = target
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id == "tl":
                    continue
                _require(target.attr in {"to", "reshape", "trans", "astype"},
                         "non-DSL method call in GPU body: " + target.attr)
            else:
                raise ValueError("dynamic callable expression in GPU body")


def _protected_tree(path, targets):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            found.add(node.name)
            _body_safety(node)
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

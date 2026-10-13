"""Candidate declaration, initial state and source policy; no torch or GPU imports."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_IMPORTS = {"__future__", "typing", "collections", "dataclasses", "enum", "functools",
                   "itertools", "math", "operator", "numbers", "abc", "types", "torch", "flydsl"}
# Library matrix products and the normalizations these operators fuse.
LIBRARY_COMPUTE = {"matmul", "mm", "bmm", "einsum", "linear", "addmm", "addbmm", "baddbmm",
                   "tensordot", "_scaled_mm", "scaled_mm"}
# FlyDSL exposes math intrinsics such as rsqrt and exp under the same names, so
# these are rejected only when reached through a torch binding.
TORCH_COMPUTE = {"softmax", "log_softmax", "sigmoid", "rsqrt", "logsumexp", "rms_norm",
                 "layer_norm", "normalize", "silu"}


def _torch_aliases(tree):
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    aliases.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            aliases.update(alias.asname or alias.name for alias in node.names)
    return aliases


def _attribute_root(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def assert_source_independent(source):
    """Reject protected or library-operator imports, including members and aliases.

    This is a static guard, not proof against reflection or arbitrary hostile
    Python. The numerical checks of every timed invocation also apply.
    """
    tree = ast.parse(source)
    torch_names = _torch_aliases(tree)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise RuntimeError("candidate imports a protected task/local module")
            prefix = node.module or ""
            names = [prefix, *(f"{prefix}.{a.name}" for a in node.names)]
        for name in names:
            parts = name.split(".")
            if parts[0] == "scripts" or parts[0] not in ALLOWED_IMPORTS:
                raise RuntimeError(f"candidate imports a protected task or unsupported module: {name}")
            if parts[0] == "torch" and any(p in LIBRARY_COMPUTE | TORCH_COMPUTE for p in parts):
                raise RuntimeError(f"candidate imports library operator computation: {name}")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            raise RuntimeError("candidate uses the library matrix multiplication operator")
        if isinstance(node, ast.Attribute):
            if node.attr in LIBRARY_COMPUTE:
                raise RuntimeError(f"candidate references library operator computation: {node.attr}")
            if node.attr in TORCH_COMPUTE and _attribute_root(node) in torch_names:
                raise RuntimeError(f"candidate references torch operator computation: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("__import__", "eval", "exec", "compile", "open"):
                raise RuntimeError(f"candidate uses unsupported dynamic access: {node.func.id}")


def candidate_entry(config):
    entries = config["candidate"]["entrypoints"]
    if len(entries) != 1 or entries[0]["kind"] != "builder" or not entries[0]["symbol"].isidentifier():
        raise ValueError("This task requires one declared builder entrypoint")
    return entries[0], ROOT / entries[0]["file"]


def initial_state(config):
    """The declared candidate's state, read from its file without executing it."""
    entry, path = candidate_entry(config)
    if not path.exists():
        return "unimplemented"
    tree = ast.parse(path.read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"] for node in tree.body):
        return "implemented"
    # A shipped stub holds only a docstring and __future__ imports; anything
    # else is broken candidate code, not an initial generation target.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        raise RuntimeError("The candidate file contains code but does not define its builder")
    return "unimplemented"

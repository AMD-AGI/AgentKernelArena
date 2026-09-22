#!/usr/bin/env python3
"""Check declared image materialization before the framework freezes a baseline."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def within(path):
    resolved = (ROOT / path).resolve(strict=True)
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Source escapes the assigned workspace: {path}")
    return resolved


def verify_sources():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    for source in config.get("workspace", {}).get("sources", []):
        path = within(source["destination"])
        if not path.is_dir():
            raise ValueError(f"Declared image source was not materialized: {source['destination']}")
    result = {}
    for edit in config["candidate"]["editable"]:
        rel = edit if isinstance(edit, str) else edit["path"]
        path = within(rel)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Candidate source is absent or empty: {rel}")
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(), filename=rel)
            definitions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            if not definitions:
                raise ValueError(f"Candidate has no implementation definitions: {rel}")
        result[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in config["candidate"].get("entrypoints", []):
        path = within(entry["file"])
        if path.suffix == ".py":
            definitions = {n.name: n for n in ast.parse(path.read_text()).body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            node = definitions.get(entry["symbol"])
            if node is None:
                raise ValueError(f"Missing entrypoint {entry['file']}:{entry['symbol']}")
            body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
            def placeholder(n):
                if isinstance(n, (ast.Pass, ast.Raise)):
                    return True
                value = n.value if isinstance(n, (ast.Return, ast.Expr)) else None
                return isinstance(n, (ast.Return, ast.Expr)) and (
                    value is None or isinstance(value, ast.Constant) and value.value in (None, Ellipsis)
                )
            if not body or all(placeholder(n) for n in body):
                raise ValueError(f"Unimplemented entrypoint {entry['symbol']}")
    return result


def main():
    from task_adapter import setup
    verify_sources()
    setup()
    print("Declared task sources and setup: PASS")


if __name__ == "__main__":
    main()

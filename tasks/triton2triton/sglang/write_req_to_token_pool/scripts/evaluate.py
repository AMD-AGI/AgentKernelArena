#!/usr/bin/env python3
"""Task-owned seven-action runner; reports belong to the Arena framework."""
from pathlib import Path
import ast
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(argv):
    role, action = ("task", "validate-task") if argv == ["validate-task"] else argv
    if (role, action) != ("task", "validate-task") and (
        role not in {"baseline", "candidate"} or action not in {"compile", "correctness", "performance"}
    ):
        raise ValueError("Expected validate-task or ROLE ACTION")
    result = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "PASS", "cases": []}
    try:
        from test_kernel_harness import load_kernel, manifest, reference_controls, run_case
        rows = manifest()
        if action == "validate-task":
            tree = ast.parse((ROOT / "kernel.py").read_text())
            entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "write_req_to_token_pool_triton")
            if any(isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
                   and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NotImplementedError"
                   for n in entry.body):
                raise ValueError("Starting implementation is unimplemented")
            for path in ("README.md", "SOURCE.md", "LICENSE"):
                if not (ROOT / path).is_file():
                    raise FileNotFoundError(path)
            result.update(cases=[dict(row, status="PASS") for row in rows],
                          metadata={"candidate_state": "implemented",
                                    "oracle_device": "cpu",
                                    "reference_controls": reference_controls()})
        elif action == "compile":
            for path in ROOT.rglob("*.py"):
                compile(path.read_text(), str(path.relative_to(ROOT)), "exec")
            load_kernel()
        else:
            kernel = load_kernel()
            for row in rows:
                if action not in row["checks"]:
                    continue
                record = {key: row[key] for key in ("test_case_id", "shape", "dtype", "params")}
                record.update(run_case(row, action, kernel))
                result["cases"].append(record)
                print(json.dumps({"completed": row["test_case_id"], "action": action}), flush=True)
    except Exception as exc:
        traceback.print_exc()
        result.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
    print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))

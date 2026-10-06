#!/usr/bin/env python3
"""Export a framework-accepted candidate; never compute/write Arena scores.

This command is registered in config.yaml.exports for every optimizing agent.
It only reads task_result.yaml after Arena finalizes it. It neither publishes
to an external repository nor treats an agent's success message as acceptance.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math

import yaml

import task_contract


def accepted_result(result, case_count):
    for field in ("pass_compilation", "pass_correctness", "pass_tool_gate",
                  "workload_consistent", "benchmark_method_consistent"):
        if result.get(field) is not True:
            raise ValueError(f"Cannot export an unaccepted result: {field}")
    for field in ("valid_baseline_cases", "valid_optimized_cases"):
        if type(result.get(field)) is not int or result[field] != case_count:
            raise ValueError(f"Cannot export incomplete case coverage: {field}")
    time = result.get("best_optimized_execution_time")
    if type(time) not in (int, float) or not math.isfinite(time) or time <= 0:
        raise ValueError("Cannot export without completed candidate timing")


def tensor_entry(entry, workload):
    """Adapt the declared builder to SIKL's tensor-call interface in the artifact.

    No KernelForge naming rule is involved. The consumer receives the original
    candidate plus this explicit binding and can recreate per-shape launches.
    """
    header = f'''from functools import lru_cache
from pathlib import Path
import importlib.util
import sys

_path = Path(__file__).resolve().parent / {entry["file"]!r}
_spec = importlib.util.spec_from_file_location("_sikl_solution_candidate", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_builder = getattr(_module, {entry["symbol"]!r})

'''
    if workload["op_type"] == "gemm":
        return header + '''@lru_cache(None)
def _launch(m, n, k):
    return _builder(m=m, n=n, k=k)

def run(a, b):
    return _launch(a.shape[0], b.shape[0], a.shape[1])(a, b)
'''
    return header + '''@lru_cache(None)
def _launch(num_tokens, model_dim, inter_dim, num_experts, topk):
    return _builder(num_tokens=num_tokens, model_dim=model_dim,
                    inter_dim=inter_dim, num_experts=num_experts, topk=topk)

def run(hidden_states, w1, w2, topk_weights, topk_ids,
        activation=0, doweight_stage1=False, w1_scale=None, w2_scale=None):
    launch = _launch(hidden_states.shape[0], hidden_states.shape[1],
                     w1.shape[1] // 2, w1.shape[0], topk_ids.shape[1])
    return launch(hidden_states, w1, w2, topk_weights, topk_ids,
                  w1_scale, w2_scale, activation, doweight_stage1)
'''


def export():
    config = task_contract.load_config()
    workload = task_contract.load_workload(config)
    entry = task_contract.candidate_entry(config)
    exports = [item for item in config["exports"] if item["format"] == "sikl-solution"]
    if len(exports) != 1:
        raise ValueError("Expected one sikl-solution export declaration")
    result = yaml.safe_load(task_contract.task_path("task_result.yaml").read_text())
    accepted_result(result, len(workload["cases"]))
    source = task_contract.task_path(entry["file"]).read_text()
    task_contract.assert_source_independent(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"]
               for node in ast.parse(source).body):
        raise ValueError("The accepted candidate's declared builder is missing")
    binding_path = "sikl_entry.py"
    if entry["file"] == binding_path:
        raise ValueError("Candidate path conflicts with exported tensor binding")
    solution = json.loads(task_contract.task_path("solution.json").read_text())
    if solution["definition"] != workload["definition"]:
        raise ValueError("Solution template and workload definitions disagree")
    solution["spec"].update(language=config["candidate"]["language"],
                             entry_point=f"{binding_path}::run")
    solution["sources"] = [{"path": entry["file"], "content": source},
                           {"path": binding_path, "content": tensor_entry(entry, workload)}]
    digest = hashlib.sha256(source.encode()).hexdigest()
    solution["author"] = "AgentKernelArena"
    solution["description"] = (
        f"Arena-accepted FlyDSL candidate for {workload['definition']}; "
        f"all {len(workload['cases'])} cases evaluated. "
        f"Candidate SHA256 {digest}. The tensor entrypoint binds the declared "
        f"builder {entry['symbol']}. See framework task_result.yaml for scores "
        "and baseline diagnostics; no downstream SIKL acceptance is implied.")
    output = task_contract.task_path(exports[0]["output"], must_exist=False)
    if output == task_contract.task_path("solution.json"):
        raise ValueError("Export must preserve the protected solution template")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(solution, indent=2, allow_nan=False) + "\n")
    print(f"Exported {exports[0]['output']}")


if __name__ == "__main__":
    export()

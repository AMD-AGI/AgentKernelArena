#!/usr/bin/env python3
"""Export a framework-accepted candidate; never compute/write Arena scores.

This command is registered in config.yaml.exports for every optimizing agent.
It only reads task_result.yaml after Arena finalizes it. It neither publishes
to an external repository nor treats an agent's success message as acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.task_policy import assert_source_independent, candidate_entry, initial_state  # noqa: E402

BINDING_PATH = "sikl_entry.py"


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


def tensor_entry(entry, definition):
    """Adapt the declared builder to SIKL's tensor-call interface in the artifact.

    The axes a launch is built for are read from the input shapes the
    definition declares, so the consumer can recreate per-shape launches.
    """
    inputs = list(definition["inputs"])
    shapes = {name: spec["shape"] for name, spec in definition["inputs"].items() if spec.get("shape") is not None}
    const = {name: axis["value"] for name, axis in definition["axes"].items() if axis["type"] == "const"}
    return f'''from functools import lru_cache
from pathlib import Path
import importlib.util
import sys

_path = Path(__file__).resolve().parent / {entry["file"]!r}
_spec = importlib.util.spec_from_file_location("_sikl_solution_candidate", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_builder = getattr(_module, {entry["symbol"]!r})
_INPUTS = {inputs!r}
_SHAPES = {shapes!r}
_CONST_AXES = {const!r}


@lru_cache(None)
def _launch(axes):
    return _builder(**dict(axes))


def run(*args, **kwargs):
    values = dict(zip(_INPUTS, args))
    values.update(kwargs)
    axes = dict(_CONST_AXES)
    for name, shape in _SHAPES.items():
        for dim, size in zip(shape, values[name].shape):
            if isinstance(dim, str):
                axes[dim] = int(size)
    return _launch(tuple(sorted(axes.items())))(*(values[name] for name in _INPUTS))
'''


def export():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    contract = json.loads((ROOT / config["evaluation"]["workloads"]).read_text())
    definition = contract["definition"]
    entry, path = candidate_entry(config)
    exports = [item for item in config["exports"] if item["format"] == "sikl-solution"]
    if len(exports) != 1:
        raise ValueError("Expected one sikl-solution export declaration")
    result = yaml.safe_load((ROOT / "task_result.yaml").read_text())
    accepted_result(result, len(contract["cases"]))
    source = path.read_text()
    assert_source_independent(source)
    if initial_state(config) != "implemented":
        raise ValueError("The accepted candidate's declared builder is missing")
    if entry["file"] == BINDING_PATH:
        raise ValueError("Candidate path conflicts with exported tensor binding")
    solution = json.loads((ROOT / "solution.json").read_text())
    if solution["definition"] != definition["name"]:
        raise ValueError("Solution template and workload definitions disagree")
    solution["spec"].update(language=config["candidate"]["language"], entry_point=f"{BINDING_PATH}::run")
    solution["sources"] = [{"path": entry["file"], "content": source},
                           {"path": BINDING_PATH, "content": tensor_entry(entry, definition)}]
    digest = hashlib.sha256(source.encode()).hexdigest()
    solution["author"] = "AgentKernelArena"
    solution["description"] = (
        f"Arena-accepted FlyDSL candidate for {definition['name']}; "
        f"all {len(contract['cases'])} cases evaluated. "
        f"Candidate SHA256 {digest}. The tensor entrypoint binds the declared "
        f"builder {entry['symbol']}. See framework task_result.yaml for scores "
        "and baseline diagnostics; no downstream SIKL acceptance is implied.")
    output = (ROOT / exports[0]["output"]).resolve()
    if not output.is_relative_to(ROOT.resolve()) or output == (ROOT / "solution.json").resolve():
        raise ValueError("Export must stay in the workspace and preserve the protected solution template")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(solution, indent=2, allow_nan=False) + "\n")
    print(f"Exported {exports[0]['output']}")


if __name__ == "__main__":
    export()

#!/usr/bin/env python3
"""Verify the unchanged Lean starter and its one-task entry without GPU work."""
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
pins = json.loads((HERE / "INPUT-PINS.json").read_text())
ready = json.loads((HERE / "READY.json").read_text())
for name, expected in {**pins["files"], **pins["post_evaluation_files"]}.items():
    path = ROOT / name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Qualified starter input changed: " + name)
for key in ("entry_config", "validator_config"):
    config = yaml.safe_load((ROOT / ready[key]).read_text())
    if config["tasks"] != ready["ready_tasks"] or config["target_gpu_model"] != "MI355X":
        raise SystemExit("Ready entry scope changed: " + ready[key])
fixture = json.loads((ROOT / ready["fixtures"]["manifest"]).read_text())
if (fixture["oci_prefix"] != ready["fixtures"]["oci_prefix"]
        or len(fixture["assets"]) != 15 or sum(row["bytes"] for row in fixture["assets"]) != 632810514):
    raise SystemExit("Fixture inventory differs from the qualified pin")
if ready["scope"]["structural_cases"] != 1 or ready["scope"]["exhaustive_sequence_length_timing"] is not False:
    raise SystemExit("Single-case sampled-work scope changed")
print(json.dumps({"status": "QUALIFIED_LEAN_STARTER_PINS_MATCH", "qualified_commit": ready["qualified_commit"],
                  "tasks": ready["ready_tasks"], "files_verified": len(pins["files"]),
                  "post_evaluation_files_verified": len(pins["post_evaluation_files"]), "GPU_actions": False}, indent=2))

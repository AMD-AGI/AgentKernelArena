"""Static recognition of task-local timing protocols; never execute task code.

These protocols own their benchmark implementation and do not request generated
helpers. Recognition checks reviewed harness bytes, not task names, and does not
assert task correctness, GPU qualification, or score eligibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from src.perf_helper_materialization import _load_workspace_config
from src.task_contract import require, strict_json, validate_manifest


REGISTRY = Path(__file__).with_name("perf") / "custom_protocols.json"
CANONICAL_CONTRACT = Path(__file__).resolve().parents[1] / "task_contract.py"


def _read_file(task: Path, relative: str) -> bytes:
    path = PurePosixPath(relative)
    require(not path.is_absolute() and bool(path.parts) and ".." not in path.parts,
            "custom protocol file must stay inside the task")
    current = task
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "custom protocol file must not be a symlink: " + relative)
    return current.read_bytes()


def custom_protocol_family(task: Path, entrypoints: set[Path]) -> str | None:
    """Return a registered family or raise for a recognized but broken contract."""
    if len(entrypoints) != 1:
        return None
    entrypoint, = entrypoints
    relative = entrypoint.relative_to(task.resolve()).as_posix()
    digest = hashlib.sha256(_read_file(task, relative)).hexdigest()
    registry = strict_json(REGISTRY.read_text())
    require(registry["schema_version"] == 1, "unsupported custom protocol registry")
    matches = [record for record in registry["protocols"]
               if record["entrypoint"] == relative and digest in record["entrypoint_sha256"]]
    if not matches:
        return None
    require(len(matches) == 1, "ambiguous custom performance protocol")
    protocol = matches[0]
    require(_load_workspace_config(task).get("performance_command") ==
            ["python3 " + relative + " performance"],
            "custom protocol must invoke its performance phase exactly once")
    for name, expected in protocol["files_sha256"].items():
        require(hashlib.sha256(_read_file(task, name)).hexdigest() in expected,
                "custom protocol helper differs from its reviewed implementation: " + name)
    if protocol["family"] == "portable_case_contract":
        require(_read_file(task, "ut/evaluation_contract.py") == CANONICAL_CONTRACT.read_bytes(),
                "portable evaluation contract differs from src/task_contract.py")
        validate_manifest(strict_json(_read_file(task, "cases.json")))
    return protocol["family"]

"""Project normalized v2 task declarations into evaluation-tool inputs.

The shared task loader owns schema validation. These helpers only select the
declarations needed for profiling/evidence; they do not infer paths from task
families or baseline repositories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def is_v2_task(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == 2


def candidate_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    candidate = config["candidate"]
    paths = [
        item if isinstance(item, str) else item["path"]
        for item in candidate["editable"]
    ]
    paths.extend(item["file"] for item in candidate.get("entrypoints", ()))
    return tuple(dict.fromkeys(paths))


def candidate_symbols(config: Mapping[str, Any]) -> tuple[str, ...]:
    candidate = config["candidate"]
    entries = candidate.get("entrypoints", ())
    # A builder is a host interface, not an attested GPU dispatch symbol. Its
    # kind remains in profile.evidence; plugins still require exact artifacts.
    symbols = [item["symbol"] for item in entries if item.get("symbol")]
    if not entries:
        for item in candidate["editable"]:
            if isinstance(item, Mapping):
                symbols.extend(item.get("symbols", ()))
    return tuple(dict.fromkeys(symbols))


def candidate_framework(config: Mapping[str, Any], paths: tuple[str, ...]) -> str:
    evidence = list(paths)
    for source in config.get("workspace", {}).get("sources", ()):
        destination = Path(source["destination"])
        if any(Path(path).is_relative_to(destination) for path in paths):
            evidence.extend(str(source.get(key, "")) for key in ("url", "image_path"))
    joined = " ".join(evidence).lower()
    for framework in ("aiter", "sglang", "rocblas", "rccl"):
        if framework in joined:
            return framework
    # kernel_identity.source_owner describes provenance, not the candidate's
    # executable dependencies. A copied baseline is not candidate library use.
    return "standalone"


def protected_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Select conventional and explicitly declared protected task inputs.

    Capture callers additionally pass their authoritative protected-file list
    for task-owned helpers outside these declared/conventional locations.
    """
    paths = ["config.yaml", "README.md", "scripts", "tests", "eval_tools"]
    paths.extend(config.get("instructions", ()))
    paths.extend(config.get("baseline", {}).get("source_files", ()))
    evaluation = config.get("evaluation", {})
    if evaluation.get("workloads"):
        paths.append(evaluation["workloads"])
    return tuple(dict.fromkeys(paths))

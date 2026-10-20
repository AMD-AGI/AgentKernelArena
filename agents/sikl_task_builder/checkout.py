"""Read committed SIKL v5 schemas from a git checkout without importing or executing them."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .bundle import ImportProblem, parse_json

SCHEMA_PATH = re.compile(r"^sikl/ops/(?P<op>[a-z0-9_]+)/impls/(?P<language>flydsl|triton)/"
                         r"(?P<name>[a-z0-9_]+)/(?P<uuid>[0-9a-f]{16})\.json$")
SOURCE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# SIKL's code-lint check-added-large-files limit (default maxkb=500).
MAX_SCHEMA_BYTES = 500 * 1024


def is_checkout(path: Path) -> bool:
    return (Path(path) / "sikl" / "ops").is_dir()


def git(root: Path, *args: str) -> bytes:
    # Neither inherited GIT_* variables nor global/system configuration or hooks may steer these reads.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0")
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", f"safe.directory={root}", "-C", str(root), *args]
    result = subprocess.run(command, env=env, capture_output=True)
    if result.returncode:
        raise ImportProblem("not_a_sikl_checkout",
                            f"git {' '.join(args[:2])} in {root}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _validate_schema(path: str, match: re.Match, data) -> None:
    def invalid(reason):
        return ImportProblem("invalid_schema", f"{path}: {reason}")
    if not isinstance(data, dict):
        raise invalid("not a JSON object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 5:
        raise ImportProblem("unsupported_schema_version", f"{path}: schema_version {data.get('schema_version')!r}")
    for key, part in (("name", "name"), ("op_type", "op")):
        if data.get(key) != match[part]:
            raise invalid(f"{key} {data.get(key)!r} != {match[part]!r}")
    kernel = data.get("kernel")
    if not isinstance(kernel, dict) or kernel.get("language") != match["language"]:
        raise invalid(f"kernel.language is not {match['language']!r}")
    sources = kernel.get("sources")
    if not isinstance(sources, list) or not sources:
        raise invalid("kernel.sources is empty")
    paths = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "content"}:
            raise invalid("a kernel source is not {path, content}")
        name, content = source["path"], source["content"]
        if not isinstance(name, str) or not SOURCE_PATH.match(name) or name == "__init__.py":
            raise invalid(f"source path {name!r} is not one <identifier>.py segment")
        if not isinstance(content, str) or not content:
            raise invalid(f"source {name} is empty")
        paths.append(name)
    if paths != sorted(set(paths)):
        raise invalid("kernel.sources paths are not sorted and unique")
    entry = kernel.get("entry_point")
    file, _, symbol = entry.partition("::") if isinstance(entry, str) else ("", "", "")
    if file not in paths or not IDENTIFIER.match(symbol):
        raise invalid(f"entry_point {entry!r} is not <source path>::<identifier>")
    targets = kernel.get("targets")
    if not isinstance(targets, list) or not targets or any(
            not isinstance(t, dict) or not t or set(t) - {"arch", "hardware"}
            or any(not isinstance(v, str) or not v for v in t.values()) for t in targets):
        raise invalid("kernel.targets must be a non-empty list of {arch, hardware} subsets")
    workloads = data.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise invalid("workloads is empty")
    seen = set()
    for workload in workloads:
        axes = workload.get("axes") if isinstance(workload, dict) else None
        if not isinstance(axes, dict) or not axes or any(
                not isinstance(k, str) or type(v) is not int for k, v in axes.items()):
            raise invalid("a workload's axes is not a non-empty {str: int}")
        key = json.dumps(axes, sort_keys=True)
        if key in seen:
            raise invalid(f"duplicate workload axes {key}")
        seen.add(key)


@dataclass(frozen=True)
class SchemaTask:
    path: str
    op: str
    language: str
    name: str
    uuid: str
    oid: str
    text: str
    data: dict

    @property
    def task_id(self) -> str:
        return f"{self.name}/{self.uuid}"

    @property
    def cases(self) -> list[str]:
        return [",".join(f"{a}={v}" for a, v in w["axes"].items()) for w in self.data["workloads"]]

    def supports(self, arch: str | None, gpu_model: str) -> bool:
        # SIKL's DeviceTarget.normalize_hardware: drop the "AMD Instinct " prefix, compare upper case.
        targets = self.data["kernel"]["targets"]
        hardware = [t.get("hardware", "").strip().upper().removeprefix("AMD INSTINCT ") for t in targets]
        return any(t.get("arch") == arch for t in targets) or gpu_model.upper() in hardware

    def summary(self, arch: str | None, gpu_model: str) -> dict:
        return {"task_id": self.task_id, "path": self.path, "op_type": self.op, "language": self.language,
                "cases": len(self.cases), "targets": self.data["kernel"]["targets"],
                "platform_supported": self.supports(arch, gpu_model)}


def inspect_checkout(root: Path, selectors: list[str]) -> tuple[str, list[SchemaTask]]:
    """Return HEAD and its schemas; uncommitted files never become tasks."""
    root = Path(root).resolve()
    if not root.is_dir() or Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ImportProblem("not_a_sikl_checkout", f"{root} is not the top level of a git checkout")
    commit = git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tasks = []
    for entry in git(root, "ls-tree", "-r", "-z", "-l", "--full-tree", "HEAD", "--", "sikl/ops").split(b"\0"):
        meta, _, path = entry.decode(errors="replace").partition("\t")
        match = SCHEMA_PATH.match(path)
        if not match:
            continue
        mode, kind, oid, size = meta.split()
        if mode != "100644" or kind != "blob":
            raise ImportProblem("bad_mode", f"{path}: mode {mode} {kind}")
        if int(size) > MAX_SCHEMA_BYTES:
            raise ImportProblem("schema_too_large", f"{path}: {size} bytes > {MAX_SCHEMA_BYTES}")
        try:
            text = git(root, "cat-file", "blob", oid).decode("utf-8")
            data = parse_json(text, path)
        except (UnicodeDecodeError, ImportProblem) as exc:
            raise ImportProblem("invalid_schema", f"{path}: {exc}") from exc
        _validate_schema(path, match, data)
        tasks.append(SchemaTask(path, match["op"], match["language"], match["name"], match["uuid"], oid, text, data))
    if not tasks:
        raise ImportProblem("no_schemas", f"no sikl/ops/<op>/impls/<language>/<name>/<uuid>.json at {commit}")
    if selectors:
        missing = sorted(set(selectors) - {t.name for t in tasks} - {t.task_id for t in tasks})
        if missing:
            raise ValueError(f"Unknown task selectors: {missing}")
        tasks = [t for t in tasks if t.name in selectors or t.task_id in selectors]
    return commit, tasks


def check_mountable(root: Path) -> None:
    # The Docker runner mounts only this directory; its object store must be self-contained.
    if not (root / ".git").is_dir():
        raise ImportProblem("checkout_not_mountable", f"{root}/.git is not a directory (git worktree?)")
    if (root / ".git" / "objects" / "info" / "alternates").exists():
        raise ImportProblem("checkout_not_mountable",
                            f"{root} borrows objects through alternates; run `git repack -a -d` and remove them")

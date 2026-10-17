"""Read committed SIKL v5 schema files from a git checkout, without importing or executing them."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import BUILDER_VERSION
from .bundle import ImportProblem, parse_json

SCHEMA_PATH = re.compile(r"^sikl/ops/(?P<op>[a-z0-9_]+)/impls/(?P<language>flydsl|triton)/"
                         r"(?P<name>[a-z0-9_]+)/(?P<uuid>[0-9a-f]{16})\.json$")
SOURCE_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# SIKL's code-lint check-added-large-files limit (default maxkb=500).
MAX_SCHEMA_BYTES = 500 * 1024
TEMPLATES = Path(__file__).with_name("templates") / "sikl_v5"
RUNTIME_ROOT = "/opt/sikl"
DESCRIPTION = f"""\
Make the SIKL v5 kernel in source/ faster on every workload of start.json. start.json is the committed
schema: its kernel ABI, workloads, reference and error bounds are fixed. Edit only the files under source/;
do not add or remove files, and keep the entry point a top-level def. The kernel must be correct for every
scalar value inside its declared range.

Each evaluation rebuilds the schema from source/ and runs `python -m sikl validate --stages audit,bench`
at the pinned SIKL commit, as a SIKL pull request that adds this kernel would be judged.

A committed kernel is exempt from SIKL's source rules SA0-SA9 only while its bytes are unchanged. Once
edited it is audited as a new kernel, and committed kernels rarely pass that audit, so the first change
must make the whole file comply: no `global`/`nonlocal` (SA1), no module-level mutable state or caches
(SA2), no functools caching except the single `@functools.cache` compile function (SA3), imports limited to
a few standard modules, torch, flydsl and triton (SA4), no forbidden names (SA5), no mutation of non-local
objects or inputs (SA6, SA7), no allocation or other side effect in launch (SA8). The rules are in
{RUNTIME_ROOT}/sikl/ops/README.md; the authority is
{RUNTIME_ROOT}/sikl/core/validation/stages/_audit/_rules.py (AKA_SIKL_ROOT replaces {RUNTIME_ROOT}).
A correctness FAIL names every rejected line or failing workload.
"""


def is_sikl_checkout(path: Path) -> bool:
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


def _invalid(path: str, reason: str) -> ImportProblem:
    return ImportProblem("invalid_schema", f"{path}: {reason}")


def _check(path: str, match: re.Match, data) -> None:
    if not isinstance(data, dict):
        raise _invalid(path, "not a JSON object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 5:
        raise ImportProblem("unsupported_schema_version", f"{path}: schema_version {data.get('schema_version')!r}")
    for key, part in (("name", "name"), ("op_type", "op")):
        if data.get(key) != match[part]:
            raise _invalid(path, f"{key} {data.get(key)!r} != {match[part]!r}")
    kernel = data.get("kernel")
    if not isinstance(kernel, dict) or kernel.get("language") != match["language"]:
        raise _invalid(path, f"kernel.language is not {match['language']!r}")
    sources = kernel.get("sources")
    if not isinstance(sources, list) or not sources:
        raise _invalid(path, "kernel.sources is empty")
    paths = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "content"}:
            raise _invalid(path, "a kernel source is not {path, content}")
        name, content = source["path"], source["content"]
        if not isinstance(name, str) or not SOURCE_PATH.match(name) or name == "__init__.py":
            raise _invalid(path, f"source path {name!r} is not one <identifier>.py segment")
        if not isinstance(content, str) or not content:
            raise _invalid(path, f"source {name} is empty")
        paths.append(name)
    if paths != sorted(set(paths)):
        raise _invalid(path, "kernel.sources paths are not sorted and unique")
    entry = kernel.get("entry_point")
    file, _, symbol = entry.partition("::") if isinstance(entry, str) else ("", "", "")
    if file not in paths or not IDENTIFIER.match(symbol):
        raise _invalid(path, f"entry_point {entry!r} is not <source path>::<identifier>")
    targets = kernel.get("targets")
    if not isinstance(targets, list) or not targets or any(
            not isinstance(t, dict) or not t or set(t) - {"arch", "hardware"}
            or any(not isinstance(v, str) or not v for v in t.values()) for t in targets):
        raise _invalid(path, "kernel.targets must be a non-empty list of {arch, hardware} subsets")
    workloads = data.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise _invalid(path, "workloads is empty")
    seen = set()
    for workload in workloads:
        axes = workload.get("axes") if isinstance(workload, dict) else None
        if not isinstance(axes, dict) or not axes or any(
                not isinstance(k, str) or type(v) is not int for k, v in axes.items()):
            raise _invalid(path, "a workload's axes is not a non-empty {str: int}")
        key = json.dumps(axes, sort_keys=True)
        if key in seen:
            raise _invalid(path, f"duplicate workload axes {key}")
        seen.add(key)


@dataclass(frozen=True)
class SiklSchema:
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
        hardware = [t.get("hardware", "").strip().upper().removeprefix("AMD INSTINCT ")
                    for t in self.data["kernel"]["targets"]]
        return any(t.get("arch") == arch for t in self.data["kernel"]["targets"]) or gpu_model.upper() in hardware

    def summary(self, arch: str | None, gpu_model: str) -> dict:
        return {"task_id": self.task_id, "path": self.path, "op_type": self.op, "language": self.language,
                "cases": len(self.cases), "targets": self.data["kernel"]["targets"],
                "platform_supported": self.supports(arch, gpu_model)}

    def files(self, commit: str, arch: str, task_id: str, command_timeout: int) -> dict[str, str]:
        kernel = self.data["kernel"]
        entry_file, _, symbol = kernel["entry_point"].partition("::")
        config = {
            "schema_version": 2,
            "description": DESCRIPTION,
            "kernel_identity": {"logical_operator": self.name},
            "candidate": {
                "language": self.language, "initial_state": "implemented",
                "editable": [{"path": f"source/{s['path']}", "scope": "file"} for s in kernel["sources"]],
                "entrypoints": [{"file": f"source/{entry_file}", "kind": "function", "symbol": symbol}],
            },
            "baseline": {"kind": "provided", "language": self.language, "source_files": ["start.json"]},
            "evaluation": {"runner": ["python3", "-I", "-B", "scripts/task_runner.py"], "timeout_s": command_timeout},
            "platform_support": {"required_arch": arch, "status": "active"},
            "exports": [{"format": "sikl-schema", "output": "export.json",
                         "command": ["python3", "-I", "-B", "scripts/export_schema.py"], "timeout_s": 60}],
        }
        from src.task_spec import TaskSpec
        TaskSpec.from_mapping(config, task_id=task_id)
        files = {"config.yaml": yaml.safe_dump(config, sort_keys=False), "start.json": self.text}
        for source in kernel["sources"]:
            files[f"source/{source['path']}"] = source["content"]
        for name in ("task_runner.py", "export_schema.py"):
            files[f"scripts/{name}"] = (TEMPLATES / name).read_text()
        files["scripts/provenance.json"] = json.dumps({
            "builder_version": BUILDER_VERSION, "sikl_commit": commit, "schema_path": self.path,
            "oid": self.oid, "name": self.name, "uuid": self.uuid, "arch": arch,
        }, indent=2, sort_keys=True) + "\n"
        return files

    def materialize(self, destination: Path, commit: str, arch: str, task_id: str, command_timeout: int) -> None:
        files = self.files(commit, arch, task_id, command_timeout)
        destination.mkdir(parents=True)
        for relative, content in files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


@dataclass(frozen=True)
class SiklRepo:
    root: Path
    commit: str
    schemas: tuple[SiklSchema, ...]

    @classmethod
    def inspect(cls, root: Path, tasks: list[str]) -> "SiklRepo":
        root = Path(root).resolve()
        if not root.is_dir() or Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
            raise ImportProblem("not_a_sikl_checkout", f"{root} is not the top level of a git checkout")
        commit = git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        schemas = []
        for entry in git(root, "ls-tree", "-r", "-z", "-l", "--full-tree", "HEAD", "--", "sikl/ops").split(b"\0"):
            if not entry:
                continue
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
            except UnicodeDecodeError as exc:
                raise _invalid(path, f"not utf-8: {exc}") from exc
            try:
                data = parse_json(text, path)
            except ImportProblem as exc:
                raise _invalid(path, str(exc)) from exc
            _check(path, match, data)
            schemas.append(SiklSchema(path, match["op"], match["language"], match["name"], match["uuid"],
                                      oid, text, data))
        if not schemas:
            raise ImportProblem("no_schemas", f"no sikl/ops/<op>/impls/<language>/<name>/<uuid>.json at {commit}")
        if tasks:
            names = {s.name for s in schemas} | {s.task_id for s in schemas}
            missing = sorted(set(tasks) - names)
            if missing:
                raise ValueError(f"Unknown task selectors: {missing}")
            schemas = [s for s in schemas if s.name in tasks or s.task_id in tasks]
        return cls(root, commit, tuple(schemas))

    def check_mountable(self) -> None:
        if not (self.root / ".git").is_dir():
            raise ImportProblem("checkout_not_mountable", f"{self.root}/.git is not a directory (git worktree?)")
        if (self.root / ".git" / "objects" / "info" / "alternates").exists():
            raise ImportProblem("checkout_not_mountable",
                                f"{self.root} borrows objects through alternates; run `git repack -a -d` and remove them")

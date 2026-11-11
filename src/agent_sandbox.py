# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Confine an optimizing agent's process tree to its task workspace with Landlock.

The framework builds a policy before it launches an agent (``prepare``). A
launcher passes its agent command through ``wrap_command``; when a policy is
active, the command starts under this file run as a standalone script, which
applies the Landlock ruleset to itself, records an attestation and executes the
agent. Every child of the agent inherits the restriction and cannot lift it.

Inside the sandbox the agent can:

- read and write its own workspace, its HOME, ``/tmp`` and ``/dev/shm``;
- read the frozen baseline workspace and the framework's agent context files;
- read and execute everything outside the repository checkout (toolchains,
  interpreters, agent installations), except ``/dev``;
- use the terminal, null/random and GPU device nodes.

It cannot open, list or change anything else in the repository checkout:
framework sources, task packages, other workspaces, logs or other runs. The
network is not restricted. The script part uses only the standard library.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import secrets
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

MODES = ("none", "landlock")
CONFIG_KEY = "agent_sandbox"
POLICY_ENV = "ARENA_AGENT_SANDBOX_POLICY"
# Launchers that pass their agent command through wrap_command().
SUPPORTED_AGENTS = frozenset({"claude_code", "codex", "cursor", "deepseek_harness"})
# Agents that are part of the framework's control plane rather than optimizers.
EXEMPT_AGENTS = frozenset({"qa_auditor"})
POLICY_VERSION = 1
# ABI 3 adds LANDLOCK_ACCESS_FS_TRUNCATE; without it truncate(2) is unrestricted,
# so a sandboxed agent could still truncate repository files it owns.
MIN_ABI = 3

_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446  # same on x86_64 and aarch64
_CREATE_RULESET_VERSION = 1 << 0
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_SCOPE_SIGNAL = 1 << 1  # ABI 6: no signals to processes outside the sandbox

_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_REFER, _TRUNCATE, _IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_FILE_RIGHTS = _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE | _IOCTL_DEV
_ALL_RIGHTS = (1 << 16) - 1
ACCESS = {
    "ro": _EXECUTE | _READ_FILE | _READ_DIR,
    "rw": _ALL_RIGHTS,
    "dev": _READ_FILE | _WRITE_FILE | _READ_DIR | _TRUNCATE | _IOCTL_DEV,
}
# Character devices and directories an agent legitimately opens. /dev itself is
# excluded from the read-only tree: a privileged container exposes host disks.
DEVICE_PATHS = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom",
                "/dev/tty", "/dev/ptmx", "/dev/pts", "/dev/kfd", "/dev/dri")
CONTEXT_FILES = ("agent_context.json", "validation_context.json")


class SandboxError(RuntimeError):
    """The requested sandbox cannot be configured or enforced."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sandbox_mode(eval_config: Mapping[str, Any] | None) -> str:
    """Return the validated sandbox mode of a run configuration (default ``none``)."""
    value = (eval_config or {}).get(CONFIG_KEY, "none")
    if value is None:
        return "none"
    if not isinstance(value, str) or value not in MODES:
        raise ValueError(f"{CONFIG_KEY} must be one of {list(MODES)}, got {value!r}")
    return value


def check_agent_supported(mode: str, agent_name: str) -> None:
    """Refuse, before any work is spent, an agent whose launcher cannot be sandboxed."""
    if mode == "none" or agent_name in EXEMPT_AGENTS:
        return
    if agent_name not in SUPPORTED_AGENTS:
        raise ValueError(f"{CONFIG_KEY}: {mode} is not supported for agent {agent_name!r}; "
                         f"supported agents: {sorted(SUPPORTED_AGENTS)}")


# -- Landlock system calls ---------------------------------------------------------------------

class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64),
                ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def landlock_abi() -> int:
    """The kernel's Landlock ABI version, or 0 when Landlock is unavailable."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        return 0
    try:
        abi = _libc().syscall(ctypes.c_long(_SYS_CREATE_RULESET), None, ctypes.c_size_t(0),
                              ctypes.c_uint32(_CREATE_RULESET_VERSION))
    except (OSError, AttributeError):
        return 0
    return max(int(abi), 0)


def _handled_fs(abi: int) -> int:
    handled = (1 << 13) - 1  # ABI 1: EXECUTE .. MAKE_SYM
    if abi >= 2:
        handled |= _REFER
    if abi >= 3:
        handled |= _TRUNCATE
    if abi >= 5:
        handled |= _IOCTL_DEV
    return handled


def require_landlock(minimum: int = MIN_ABI) -> int:
    abi = landlock_abi()
    if abi < minimum:
        raise SandboxError(f"Landlock ABI {abi} is available; the agent sandbox needs ABI {minimum} "
                           "or later (Linux 6.2+ with Landlock enabled and not blocked by seccomp)")
    return abi


def _check(result: int, what: str) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise SandboxError(f"{what} failed: {os.strerror(error)} (errno {error})")
    return result


def _restrict(rules: Sequence[Mapping[str, str]], abi: int) -> dict:
    libc = _libc()
    handled = _handled_fs(abi)
    attr = _RulesetAttr(handled, 0, _SCOPE_SIGNAL if abi >= 6 else 0)
    ruleset = _check(libc.syscall(ctypes.c_long(_SYS_CREATE_RULESET), ctypes.byref(attr),
                                  ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0)),
                     "landlock_create_ruleset")
    applied, missing = 0, []
    try:
        for rule in rules:
            try:
                fd = os.open(rule["path"], os.O_PATH | os.O_CLOEXEC)
            except FileNotFoundError:
                missing.append(rule["path"])
                continue
            try:
                access = ACCESS[rule["access"]] & handled
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    access &= _FILE_RIGHTS
                beneath = _PathBeneathAttr(access, fd)
                _check(libc.syscall(ctypes.c_long(_SYS_ADD_RULE), ctypes.c_int(ruleset),
                                    ctypes.c_int(_RULE_PATH_BENEATH), ctypes.byref(beneath),
                                    ctypes.c_uint32(0)), f"landlock_add_rule({rule['path']})")
                applied += 1
            finally:
                os.close(fd)
        _check(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "prctl(PR_SET_NO_NEW_PRIVS)")
        _check(libc.syscall(ctypes.c_long(_SYS_RESTRICT_SELF), ctypes.c_int(ruleset), ctypes.c_uint32(0)),
               "landlock_restrict_self")
    finally:
        os.close(ruleset)
    return {"abi": abi, "handled_access_fs": handled, "signal_scoped": abi >= 6,
            "rules_applied": applied, "rules_missing": missing}


# -- Policy construction (framework side) ------------------------------------------------------

def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _complement(root: str, excluded: Sequence[str], seen: set[str] | None = None) -> list[str]:
    """Real paths covering ``root`` minus every excluded subtree.

    A rule names the real path it is opened at, so a symlink that resolves into
    an excluded tree is dropped instead of granting it.
    """
    seen = set() if seen is None else seen
    real = os.path.realpath(root)
    if real in seen or any(_within(real, item) for item in excluded):
        return []
    seen.add(real)
    if not any(_within(item, real) for item in excluded):
        return [real]
    if not os.path.isdir(real):
        return []
    paths: list[str] = []
    for entry in sorted(os.listdir(real)):
        paths.extend(_complement(os.path.join(real, entry), excluded, seen))
    return paths


def build_rules(*, workspace: Path, repo_root: Path, home: str | None = None,
                read_only: Sequence[Path] = (), read_write: Sequence[Path] = ()) -> list[dict]:
    """Landlock rules for an agent working in ``workspace`` below ``repo_root``."""
    repo = os.path.realpath(repo_root)
    workspace_real = os.path.realpath(workspace)
    if home and _within(repo, os.path.realpath(home)):
        # A HOME above the checkout could only be granted entry by entry, and agent
        # CLIs create new files there. The Docker runner gives each worker its own HOME.
        raise SandboxError(f"HOME {home} contains the repository {repo}; run agents with a separate HOME "
                           "(the Docker runner does)")
    excluded = [repo, "/dev"]
    rules = [{"path": path, "access": "ro"} for path in _complement("/", excluded)]
    for root in ("/tmp", "/dev/shm", *([home] if home else []), *map(str, read_write)):
        rules.extend({"path": path, "access": "rw"} for path in _complement(root, [repo, "/dev"])
                     if os.path.exists(path))
    rules.append({"path": workspace_real, "access": "rw"})
    for path in map(str, read_only):
        if os.path.exists(path):
            rules.append({"path": os.path.realpath(path), "access": "ro"})
    rules.extend({"path": path, "access": "dev"} for path in DEVICE_PATHS if os.path.exists(path))
    return rules


class Prepared:
    """One task's active sandbox: the policy file, its environment and attestations."""

    def __init__(self, mode: str, policy_path: Path, policy: dict):
        self.mode = mode
        self.policy_path = policy_path
        self.policy = policy

    @property
    def environment(self) -> dict[str, str]:
        return {POLICY_ENV: str(self.policy_path)}

    def attestations(self) -> list[dict]:
        records = []
        directory = Path(self.policy["attestation_directory"])
        for path in sorted(directory.glob(f"agent-sandbox-{self.policy['nonce']}-*.json")):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                records.append({"file": path.name, "error": "unreadable attestation"})
        return records

    def record(self) -> dict:
        """Result evidence; ``enforced`` only if every launch attested the policy."""
        attestations = self.attestations()
        valid = [row for row in attestations if row.get("nonce") == self.policy["nonce"]
                 and row.get("policy_sha256") == self.policy["policy_sha256"]
                 and row.get("self_test") == "denied"]
        status = "enforced" if attestations and len(valid) == len(attestations) else (
            "unattested" if not attestations else "invalid")
        return {"mode": self.mode, "status": status, "launches": len(attestations),
                "policy_sha256": self.policy["policy_sha256"],
                "abi": sorted({row.get("abi") for row in valid if row.get("abi") is not None}),
                "rules": len(self.policy["rules"])}


def prepare(mode: str, *, workspace: Path, state_directory: Path, baseline_workspace: Path,
            environ: Mapping[str, str] | None = None, repo_root: Path | None = None,
            self_test_path: Path | None = None) -> Prepared | None:
    """Write the policy for one agent launch; ``None`` when the sandbox is off.

    ``self_test_path`` is a protected repository file the wrapper must fail to
    open once restricted (default: this module).
    """
    if mode == "none":
        return None
    environ = os.environ if environ is None else environ
    repo_root = repository_root() if repo_root is None else repo_root
    self_test_path = Path(__file__).resolve() if self_test_path is None else Path(self_test_path).resolve()
    if not _within(str(self_test_path), os.path.realpath(repo_root)):
        raise SandboxError("The sandbox self-test path must be inside the protected repository")
    abi = require_landlock()
    read_only = [Path(baseline_workspace), *(Path(state_directory) / name for name in CONTEXT_FILES)]
    if environ.get("PYTHONUSERBASE"):
        read_only.append(Path(environ["PYTHONUSERBASE"]))
    read_write = [Path(environ[name]) for name in ("CODEX_HOME", "CLAUDE_CONFIG_DIR")
                  if environ.get(name)]
    rules = build_rules(workspace=workspace, repo_root=repo_root, home=environ.get("HOME"),
                        read_only=read_only, read_write=read_write)
    body = {"version": POLICY_VERSION, "mode": mode, "min_abi": MIN_ABI, "rules": rules,
            # The wrapper must fail to open this framework file after restricting itself.
            "self_test_path": str(self_test_path)}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    policy = {**body, "policy_sha256": digest, "nonce": secrets.token_hex(8), "host_abi": abi,
              "attestation_directory": str(Path(state_directory).resolve())}
    path = Path(state_directory) / f"agent-sandbox-policy-{policy['nonce']}.json"
    path.write_text(json.dumps(policy, indent=1, sort_keys=True) + "\n")
    return Prepared(mode, path, policy)


def wrap_command(cmd: Sequence[str], env: Mapping[str, str] | None = None) -> list[str]:
    """Return ``cmd`` started under the active sandbox policy, if any."""
    env = os.environ if env is None else env
    policy = env.get(POLICY_ENV)
    if not policy:
        return list(cmd)
    return [sys.executable, "-I", str(Path(__file__).resolve()), "--policy", policy, "--", *cmd]


# -- Wrapper (agent side) ----------------------------------------------------------------------

def _exec(policy_path: str, command: list[str]) -> None:
    with open(policy_path) as handle:
        policy = json.load(handle)
    if policy.get("version") != POLICY_VERSION or policy.get("mode") != "landlock":
        raise SandboxError("Unsupported agent sandbox policy")
    body = {key: policy[key] for key in ("version", "mode", "min_abi", "rules", "self_test_path")}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != policy["policy_sha256"]:
        raise SandboxError("Agent sandbox policy digest mismatch")
    environment = {key: value for key, value in os.environ.items() if key != POLICY_ENV}
    executable = command[0] if "/" in command[0] else shutil.which(command[0], path=environment.get("PATH"))
    if not executable:
        raise SandboxError(f"Agent command not found: {command[0]}")
    abi = require_landlock(int(policy["min_abi"]))
    attestation = Path(policy["attestation_directory"]) / f"agent-sandbox-{policy['nonce']}-{os.getpid()}.json"
    # Opened before restricting: the attestation directory is not writable afterwards.
    with open(os.open(attestation, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600), "w") as out:
        applied = _restrict(policy["rules"], abi)
        try:
            os.close(os.open(policy["self_test_path"], os.O_RDONLY | os.O_CLOEXEC))
        except PermissionError:
            self_test = "denied"
        else:
            raise SandboxError("Agent sandbox self-test could read a framework file")
        json.dump({"nonce": policy["nonce"], "policy_sha256": policy["policy_sha256"], "pid": os.getpid(),
                   "self_test": self_test, "time": time.time(), "command": os.path.basename(executable),
                   **applied}, out, sort_keys=True)
    os.execve(executable, command, environment)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "--policy" or args[2] != "--":
        print("usage: agent_sandbox.py --policy FILE -- COMMAND [ARG]...", file=sys.stderr)
        return 2
    try:
        _exec(args[1], args[3:])
    except (SandboxError, OSError, ValueError, KeyError) as exc:
        print(f"agent sandbox: {exc}", file=sys.stderr)
        return 126
    return 0  # not reached: execve replaces the process


if __name__ == "__main__":
    sys.exit(main())

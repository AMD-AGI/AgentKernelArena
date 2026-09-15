"""Process-local compatibility layer for Hyperloom's two KernelForge CLIs.

No installed package or shared source tree is edited. The changes below bind
upstream's private interfaces to a declared Arena task. The probe checks those
interfaces before any campaign starts; incompatible releases fail explicitly.
"""
from __future__ import annotations

import asyncio
import ast
from dataclasses import replace
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.forge.bridge import bound_candidate_root, load_plan
from agents.forge.task_context import TaskContext
from agents.forge.bundles import allow_candidate_paths, candidate_files, protected_paths

ADAPTER_API = 1


def verify_release() -> dict:
    """Reject unreviewed private implementations, even if signatures still fit."""
    release = json.loads(Path(__file__).with_name("upstream_compatibility.json").read_text())
    try:
        version = importlib.metadata.version("hyperloom-inference_optimizer")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    if version not in (release["package_version"], "source-checkout"):
        raise RuntimeError(f"Unreviewed KernelForge package version: {version}")
    sources = {}
    for name, expected in release["sources"].items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unreviewed KernelForge source: {name}; expected {expected}, got {actual}. "
                               "Review the adapter compatibility pin before using this engine.")
        sources[name] = actual
    return {"version": version, "reviewed_commit": release["commit"], "sources": sources}


def _modules():
    from kernelforge import cli
    from kernelforge.kernel_backends.constants import KERNEL_BACKENDS
    from kernelforge.loop import canonical_correctness
    from kernelforge.loop import new_path_allowlist
    from kernelforge.orchestrator import agent
    from kernelforge.rewrite_by_flydsl import driver_contract, optimize, port_loop, protocol, runner, seed, spec
    return SimpleNamespace(**locals())


def probe() -> dict:
    release = verify_release()
    modules = _modules()
    required = {
        "forge-loop": {"kernel", "driver", "workspace_dir", "deadline_unix", "prepare_task",
                       "source_files", "target_functions", "baseline_json", "lanes", "profiling_enabled"},
        "forge-rewrite-by-flydsl": {"source_kernel", "driver", "workspace_dir", "deadline_unix",
                                   "prepare_driver", "flydsl_kernel_name", "rewrite_kb"},
    }
    # Click parameter names can differ from Python argument names, so test
    # public option spellings for booleans whose dest has changed upstream.
    for name, parameters in required.items():
        command = modules.cli.main.commands.get(name)
        if command is None:
            raise RuntimeError(f"Installed KernelForge lacks {name}")
        names = {parameter.name for parameter in command.params}
        if "profiling" in names:
            names.add("profiling_enabled")
        missing = parameters - names
        if missing:
            raise RuntimeError(f"KernelForge {name} lacks adapter-required parameters: {sorted(missing)}")
    required_hooks = [
        (modules.canonical_correctness, "_run_canonical_suite", {"workspace_dir", "timeout_cap_sec"}),
        (modules.seed, "generate_seed", {"spec", "dest"}),
        (modules.port_loop, "build_port_program_md", {"spec", "driver_path"}),
        (modules.port_loop, "run_port_loop", {"spec", "driver_path", "config", "stop_at_unix"}),
        (modules.runner, "_ensure_git_committed", {"workspace", "message", "paths", "branch"}),
        (modules.agent, "make_agent_fn", {"source_files", "target_functions", "task_type", "correctness_only",
                                          "usage", "insession_gate", "interposed_driver_path",
                                          "validation_timeout_sec", "bench_timeout_sec"}),
        (modules.optimize, "_forge_loop_argv", set()),
        (modules.new_path_allowlist, "matches_commit_new_paths", {"path", "patterns"}),
    ]
    for module, name, parameters in required_hooks:
        function = getattr(module, name, None)
        if not callable(function) or not parameters <= inspect.signature(function).parameters.keys():
            raise RuntimeError(f"Incompatible KernelForge adapter interface: {module.__name__}.{name}")
    return {"adapter_api": ADAPTER_API, **release, "initialization_targets": ["flydsl", "hip", "triton"],
            "backends": sorted(modules.KERNEL_BACKENDS), "rewrite_target": "flydsl"}


def program_text(plan: dict, *, prefix: str = "", port: bool = False, initialize: bool = False) -> str:
    context = TaskContext.load(plan["context"])
    spec = context.spec

    config = spec.to_mapping()
    declarations = config["candidate"].copy()
    instructions = [config.get("description", "")]
    if plan.get("initial_candidate_failure"):
        instructions += [
            "Historical input assessment: an earlier candidate compiled but failed the complete numerical check. "
            "The protected driver determines whether the current candidate now passes. Optimization requires "
            "a currently passing implementation; these retained diagnostics do not override a later successful check:",
            json.dumps(plan["initial_candidate_failure"], ensure_ascii=False)[:4000],
        ]
    for name in dict.fromkeys(["README.md", *config.get("instructions", [])]):
        path = Path(plan["template"]) / name
        if path.is_file():
            instructions.append(f"\n## {name}\n" + path.read_text())
    return "\n".join([
        "# Arena task contract", "Task: " + spec.task_id,
        "Implement the task using " + spec.candidate.language + ".",
        "Task implementation and dependency constraints take precedence over backend guides, examples, "
        "and knowledge-base suggestions. Passing the driver does not waive these constraints. "
        "Do not treat an available library or backend example as permission to delegate an operator "
        "when the task forbids that delegation. A prohibited library operator remains prohibited "
        "even if it uses the target language internally. Wrapping that operator or tuning its launch "
        "parameters does not satisfy a requirement to implement the operator in candidate-owned kernels.",
        "The following paths are relative to " + (prefix or "the workspace root") + ".",
        "Editable declarations and real entrypoints (do not invent a factory convention):",
        json.dumps(declarations, indent=2),
        "Read the task's baseline/reference and evaluator for the exact interface and semantics:",
        json.dumps({"baseline": config["baseline"], "evaluation": config["evaluation"]}, indent=2),
        "The protected driver calls the task's compile and complete correctness commands.",
        "Correctness uses the task's own reference and comparison, not a generic SNR threshold.",
        "Run python3 arena_forge_driver.py for correctness; --bench-mode for candidate timing;",
        "--ref-bench-mode for the independent baseline. Task-owned cases and warmups are fixed.",
        "The driver reports allclose from the task verdict. Do not modify task configuration,",
        "harnesses, inputs, references, or import protected implementations into your candidate.",
        "INITIALIZE: implement the missing target-language code. Full task correctness and all declared "
        "implementation constraints are required; no speedup is required. Keep partial work between attempts and use compiler "
        "and correctness feedback. The same Forge loop optimizes the first valid implementation afterward."
        if initialize else "PORT first produces a correct implementation; the nested loop then optimizes it." if port else
        "Optimize the existing candidate. Keep all declared entrypoints and dependent source files.",
        *instructions,
    ])


def install_hooks(plan: dict) -> None:
    probe()  # No private-interface patches on unreviewed upstream code.
    from agents.forge.protected_inventory import install as install_inventory
    install_inventory()
    from agents.forge.gate_targets import install as install_gate_targets
    install_gate_targets()
    from agents.forge.deadline import install as install_deadline, bound_agent, bound_session
    install_deadline(plan)
    from agents.forge.incumbent import install as install_incumbent
    install_incumbent()
    if plan.get("agent_config", {}).get("codex_auth_mode") == "cli":
        from agents.forge.codex_auth import install_cli_auth
        install_cli_auth()
    modules = _modules()
    context = TaskContext.load(plan["context"])
    spec = context.spec

    original_agent = modules.agent.make_agent_fn
    agent_signature = inspect.signature(original_agent)

    def make_agent(*args, **kwargs):
        bound = agent_signature.bind_partial(*args, **kwargs)
        values = bound.arguments
        root = Path(values["config"].workspace).resolve()
        candidate_root = bound_candidate_root(plan, root)
        files = candidate_files(spec, candidate_root, required=False)
        values["source_files"] = list(map(str, files.values()))
        values["target_functions"] = [entry.symbol for entry in spec.candidate.entrypoints if entry.symbol]
        values["task_type"] = "image_kernel"  # upstream multi-file switch only
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, text=True).split("\0")
        editable = set(files.values())
        protected = [str(root / name) for name in filter(None, tracked) if root / name not in editable]
        values["extra_protected_paths"] = list(dict.fromkeys([*(values.get("extra_protected_paths") or []), *protected]))
        # Both CLI providers must run the root-level public bridge, even with a
        # nested anchor. Never infer the working directory from the anchor file.
        values["interposed_driver_path"] = str(root / "arena_forge_driver.py")
        configured_timeout = plan.get("agent_config", {}).get("session_timeout_seconds", 1200)
        timeout = values.get("session_timeout_sec")
        values["session_timeout_sec"] = min(timeout, configured_timeout) if timeout is not None else configured_timeout
        from agents.forge.action_budget import driver_limits
        limits = driver_limits(spec, plan)
        values["validation_timeout_sec"] = limits["validate_stage_timeout_sec"]
        values["bench_timeout_sec"] = limits["bench_timeout_sec"]
        return bound_agent(original_agent(*bound.args, **bound.kwargs), plan,
                           values["session_timeout_sec"])

    modules.agent.make_agent_fn = make_agent

    original_run_spec = modules.agent.AgentRunSpec
    def run_spec(*args, **kwargs):
        result = original_run_spec(*args, **kwargs)
        root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                    cwd=result.cwd, text=True).strip())
        candidate_root = bound_candidate_root(plan, root)
        prefix = candidate_root.relative_to(root)
        def literal(value):
            # fnmatch uses '*' across directory separators. Escape literal
            # metacharacters before adding our one declared-tree wildcard.
            return "".join({"[": "[[]", "*": "[*]", "?": "[?]"}.get(char, char) for char in value)
        allowed = [literal((prefix / scope.path).as_posix()) + ("/*" if scope.scope == "tree" else "")
                   for scope in spec.candidate.editable]
        return bound_session(replace(result, cwd=str(root),
                       ignored_untracked_globs=[*result.ignored_untracked_globs, *allowed]), plan)

    modules.agent.AgentRunSpec = run_spec

    # The upstream CLI has only single-directory globs. Use the task's already
    # validated declarations to admit newly authored nested helpers in a tree.
    # Upstream still applies its protected-measurement exclusions afterward.
    original_matches = modules.new_path_allowlist.matches_commit_new_paths
    def matches(path, patterns):
        relative = Path(path)
        if plan["workflow"] == "rewrite":
            if len(relative.parts) < 3 or relative.parts[0] != ".forge_rewrite":
                return False
            relative = Path(*relative.parts[2:])
        value = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or value in protected_paths(spec):
            return False
        return any(scope.contains(value) for scope in spec.candidate.editable)
    # Update imports taken before hook installation as well as future imports.
    for module in tuple(sys.modules.values()):
        if module and getattr(module, "__name__", "").startswith("kernelforge."):
            if getattr(module, "matches_commit_new_paths", None) is original_matches:
                module.matches_commit_new_paths = matches

    async def canonical(workspace_dir, *, timeout_cap_sec):
        root = Path(workspace_dir).resolve()
        # Canonical acceptance is a trusted subprocess bridge, including when
        # upstream uses a copied lane. It does not read legacy config fields.
        command = [sys.executable, "-c",
                   "from agents.forge.bridge import run_managed as run; import sys; "
                   "raise SystemExit(run(sys.argv[1], sys.argv[2], []))",
                   os.environ["ARENA_FORGE_PLAN"], str(root)]
        process = await asyncio.create_subprocess_exec(*command, cwd=root,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=True)
        from kernelforge.mcp_server.tools._subprocess import communicate_process_group
        try:
            stdout, stderr = await communicate_process_group(process, timeout=timeout_cap_sec)
        except asyncio.TimeoutError:
            return modules.canonical_correctness.CanonicalCorrectnessResult(
                passed=False, detail="Arena v2 correctness timed out", outcome="timeout")
        output = (stdout + stderr).decode(errors="replace")
        return modules.canonical_correctness.CanonicalCorrectnessResult(
            passed=process.returncode == 0,
            detail="Arena v2 compile + correctness", output=output[-4000:])

    modules.canonical_correctness._run_canonical_suite = canonical
    if plan["workflow"] != "rewrite":
        return

    # These are upstream hints only. No factory is derived from an operator ID.
    entry_symbol = next((entry.symbol for entry in spec.candidate.entrypoints if entry.symbol), "")
    modules.protocol.builder_symbol = lambda _operator: entry_symbol

    def seed(rewrite_spec, dest):
        destination = Path(dest)
        attempt = destination
        for _ in Path(plan["anchor"]).parts:
            attempt = attempt.parent
        files = candidate_files(spec, Path(plan["template"]), required=False)
        for relative, source in files.items():
            target = attempt / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Cross-language starting implementations cannot pass the candidate
        # probe merely by calling their original backend. The task source copy
        # remains available for orientation; generation starts explicitly empty.
        scope = next(scope for scope in spec.candidate.editable if scope.contains(plan["anchor"]))
        if scope.scope == "symbols":
            # Preserve a colocated harness. Only stub the explicitly editable
            # definitions; the port must not reconstruct protected tests.
            tree = ast.parse(destination.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in scope.symbols:
                    node.body = ast.parse('raise NotImplementedError("Arena candidate is not implemented")').body
            destination.write_text(ast.unparse(tree) + "\n")
        else:
            destination.write_text('"""Implement the declared Arena interface in FlyDSL."""\n'
                                   'raise NotImplementedError("Arena candidate is not implemented")\n')
        return str(destination)

    modules.seed.generate_seed = seed
    def port_program(rewrite_spec, driver_path):
        relative = Path(rewrite_spec.flydsl_kernel).relative_to(rewrite_spec.workspace)
        prefix = Path(*relative.parts[:-len(Path(plan["anchor"]).parts)])
        return program_text(plan, prefix=str(prefix), port=True)

    modules.port_loop.build_port_program_md = port_program
    from agents.forge.port_budget import bound_port
    modules.port_loop.run_port_loop = bound_port(modules.port_loop.run_port_loop, plan)
    modules.runner.run_port_loop = modules.port_loop.run_port_loop
    original_commit = modules.runner._ensure_git_committed

    def commit(workspace, message, paths, *, branch=""):
        root = bound_candidate_root(plan, Path(workspace))
        if message == "forge-rewrite: initial correct flydsl port":
            allow_candidate_paths(Path(workspace), spec, prefix=root.relative_to(workspace).as_posix())
            paths = [*paths, str(Path(workspace) / ".gitignore")]
        paths = list(dict.fromkeys([*paths, *(str(path) for path in candidate_files(spec, root, required=False).values())]))
        value = original_commit(workspace, message, paths, branch=branch)
        if message == "forge-rewrite: initial correct flydsl port":
            plan["port_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=workspace,
                                                  capture_output=True, text=True, check=True).stdout.strip()
        return value

    modules.runner._ensure_git_committed = commit
    modules.optimize._forge_loop_argv = lambda: [sys.executable, str(Path(__file__).resolve())]

    original_optimize = modules.runner.run_optimize
    def optimize(*args, **kwargs):
        value = original_optimize(*args, **kwargs)
        complete = (isinstance(value, dict) and value.get("llm_usage_complete") is True
                    and not value.get("terminated_for_deadline"))
        evidence = {"status": "COMPLETED" if complete else "FAILED", "result": value}
        Path(plan["result"]).with_name("nested_loop_status.json").write_text(
            json.dumps(evidence, indent=2) + "\n")
        if not complete:
            raise RuntimeError("Forge nested OPTIMIZE did not finish successfully; PORT evidence retained")
        return value

    modules.runner.run_optimize = optimize

    # Arena owns delivery; applying a patch to an upstream framework is not a
    # task requirement. Explicitly record it as unrequested, never as passed.
    modules.runner.DEFAULT_REWRITE_BUDGET = replace(modules.runner.DEFAULT_REWRITE_BUDGET,
                                                    applyback_reserve_sec=0)
    modules.runner.generate_applyback_patch = lambda *args, **kwargs: SimpleNamespace(
        ok=False, error="not_requested_by_arena", to_dict=lambda: {"ok": False, "error": "not_requested_by_arena"})
    original_result = modules.runner.report.build_result

    def result(**kwargs):
        kwargs["applyback_required"] = False
        if kwargs.get("port_ok") and plan.get("port_commit"):
            optimized = dict(kwargs.get("optimize_result") or {})
            if not optimized.get("best_commit"):
                optimized["best_commit"] = plan["port_commit"]
            kwargs["optimize_result"] = optimized
        return original_result(**kwargs)

    modules.runner.report.build_result = result


def configure_nested_loop(plan: dict, argv: list[str]) -> list[str]:
    if not argv or argv[0] != "forge-loop" or plan["workflow"] != "rewrite":
        return argv
    argv = list(argv)
    root = bound_candidate_root(plan, Path(plan["engine_root"]))
    context = TaskContext.load(plan["context"])
    sources = candidate_files(context.spec, root)
    program = Path(plan["program"])
    engine = Path(plan["engine_root"])
    if program != engine / "arena_program.md":
        raise ValueError("Unexpected adapter-generated program path")
    program.write_text(program_text(plan, prefix=str(root.relative_to(engine))))
    # PORT commits its candidate before this transition. Only the generated
    # phase instructions change here; do not sweep candidate or harness edits
    # into an adapter commit. The native campaign still rejects other dirt.
    dirty = subprocess.run(["git", "diff", "HEAD", "--quiet", "--", program.name], cwd=engine)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", program.name],
                             cwd=engine, capture_output=True)
    if dirty.returncode == 1 or (dirty.returncode == 0 and tracked.returncode == 1):
        subprocess.run(["git", "add", "--", program.name], cwd=engine, check=True)
        subprocess.run(["git", "commit", "--only", "-m", "Arena: prepare native optimization instructions",
                        "--", program.name], cwd=engine, check=True, capture_output=True, text=True)
    elif dirty.returncode or tracked.returncode:
        raise RuntimeError("Cannot inspect generated Forge phase instructions")
    for flag, value in (("--source-files", ",".join(map(str, sources.values()))),
                        ("--task-type", "image_kernel"), ("--target-functions", ",".join(
                            entry.symbol for entry in context.spec.candidate.entrypoints if entry.symbol))):
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv.extend([flag, value])
    argv.extend(["--lanes", "1", "--no-profiling", "--no-specialist-probe",
                 "--program-md-file", plan["program"]])
    return argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--arena-probe"]:
        print(json.dumps(probe()))
        return 0
    plan = load_plan(Path(os.environ["ARENA_FORGE_PLAN"]))
    probe()  # Reject unrecognized engine interfaces before patching anything.
    install_hooks(plan)
    from kernelforge.cli import main as cli
    from agents.forge.process_tree import managed_children
    with managed_children():
        initialization = None
        if argv and argv[0] == "--arena-initialize":
            from agents.forge.initialization import initialize, prepare_loop
            initialization = asyncio.run(initialize(plan))
            argv = prepare_loop(plan, argv[1:])
        exit_code = cli(args=configure_nested_loop(plan, argv), standalone_mode=False)
        # Click returns Exit.exit_code instead of raising SystemExit in this
        # mode. Preserve a failed engine exit, even if it wrote partial JSON.
        if isinstance(exit_code, int) and exit_code != 0:
            raise SystemExit(exit_code)
        if initialization is not None:
            from kernelforge.tracker.usage import combine_usage_totals
            result_path = Path(plan["result"])
            result = json.loads(result_path.read_text())
            result["initialization"] = initialization
            result["optimization_llm_usage"] = result.get("llm_usage")
            result["llm_usage"] = combine_usage_totals(initialization["llm_usage"], result.get("llm_usage"),
                incomplete=not isinstance(result.get("llm_usage"), dict) or result.get("llm_usage_complete") is False)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

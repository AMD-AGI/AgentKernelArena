"""Per-run compatibility copy of GEAK's real engine, pinned and fail-closed.

Only setup/benchmark plumbing and role contracts change. Planning, authoring,
engineer fan-out, verification, integration, and stopping stay upstream GEAK.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Callable

from .bridge import Bridge, copy_file, copy_tree, write_json


UPSTREAM_REVISION = "c0c0e2aee5e2bec70583253058382523bdf7a3ab"
SCRIPT_SHA256 = {
    "kernel_workflow.js": "9b4fef04fac1816db04ee2a29c414c5db3fefdbd6dda9d7154e6910a08af2bf6",
    "kernel_lane.js": "121db35f7e147261dec038b833c738f3a24dfab5da01b7a9d435ae2953d68e34",
}


def verify_upstream(checkout: Path, *, remaining: Callable[[], float] | None = None) -> None:
    def timeout() -> float:
        return min(10, remaining()) if remaining else 10

    revision = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=timeout(), check=True).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Unqualified GEAK revision; see agents/geak/compatibility.py")
    dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--",
                            "kernel_workflow", "perf_knowledge"],
                           capture_output=True, text=True, timeout=timeout(), check=True).stdout
    if dirty.strip():
        raise ValueError("GEAK engine/knowledge must match the clean pinned checkout")
    for name, expected in SCRIPT_SHA256.items():
        timeout()
        path = checkout / "kernel_workflow" / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("GEAK engine script does not match its pinned identity")
    timeout()


def adapt_lane(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != SCRIPT_SHA256["kernel_lane.js"]:
        raise ValueError("Refusing to adapt an unknown GEAK lane")
    # The Workflow runtime exposes filesystem execution through agents only.
    # Arena prepares these two factual inputs itself before starting that runtime.
    start = source.index("const setup = await agentT(")
    end = source.index("if (!setup || !setup.eval_dir)", start)
    source = source[:start] + "const setup = A.arena_setup;\n" + source[end:]
    start = source.index("const bench = await agentT(")
    end = source.index("if (!bench || !bench.baseline_per_case)", start)
    source = source[:start] + "const bench = A.arena_benchmark;\n" + source[end:]
    # Includes the inline engineer/integrator calls, not just roleAgent prompts.
    marker = "async function agentT(p, o) {"
    if source.count(marker) != 1:
        raise ValueError("GEAK agent extension point changed")
    source = source.replace(marker, marker + "\n  p += '\\n\\n' + A.arena_contract;"
                            "\n  if (A.arena_model) o = { ...o, model: A.arena_model };", 1)
    # Request a public decision summary, never private model deliberations.
    source = source.replace("reasoning: { type: 'string' }",
                            "decision_summary: { type: 'string', description: 'Brief decision summary only; no private deliberations.' }")
    source = source.replace("plan.reasoning", "plan.decision_summary")
    # A failed clock must never turn a bounded run into an unlimited one.
    source = source.replace("return Infinity;\n  }\n  return DEADLINE_EPOCH - r.epoch;",
                            "return 0;\n  }\n  return DEADLINE_EPOCH - r.epoch;")
    source = source.replace("treating time as unlimited for this check.",
                            "stopping new rounds because the clock check failed.")
    # Upstream publication is independent of warm_start/update_experience.
    # Disable its producer gate, even if a role reports a real GPU identity.
    source = source.replace("const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;",
                            "const KB_WRITE_OK = false; // Arena forbids external publication")
    return source


def arena_contract(bridge: Bridge, python: str) -> str:
    command = shlex.join([python, str(Path(__file__).with_name("bridge.py")),
                          "--job", str(bridge.job_path)])
    spec = bridge.spec.to_mapping()
    return f"""Arena task contract (takes precedence over upstream layout/harness examples)
Target language: {bridge.spec.candidate.language}. Initial state: {bridge.spec.candidate.initial_state}.
Read the task's config.yaml, README.md if present, declared instructions and entrypoints.
Keep all relative paths, multifile interfaces, and symbol-scoped restrictions exactly as declared:
{json.dumps(spec, indent=2)}

All task files outside candidate.editable are immutable, including colocated protected code.
Never create or change a runner, oracle, timing method, cases, tolerance, setup or task result.
Do not generate GEAK unittest.py/meta.json/kernel_src/ scaffolding. This task already has its
public CLI and independently frozen baseline. No task Python imports into GEAK are needed.
Never edit the Arena original workspace, baseline workspace, bridge, job, or COMMANDMENT.
Only edit candidate paths in the private workspace designated by your role's WORKSPACE
or KERNEL_PATH. When saving patches, stage every declared newly created file as well
as modifications; preserve the original relative paths and include added helpers.
Run these exact commands from that workspace (copies in round directories are supported):
COMPILE: {command} compile --workspace "$PWD"
CORRECTNESS: {command} correctness --workspace "$PWD"
FULL_BENCHMARK: {command} performance --workspace "$PWD"
PROFILE: unavailable through this adapter; report unavailable, never create a substitute harness.
BASELINE: {command} baseline
DIRECTOR_VALIDATION: {command} validate --workspace "$PWD"
The bridge rebuilds before correctness and rebuilds/checks before timing, enforcing the full
captured case manifest. GEAK_ARENA_RESULT is fresh JSON; nonzero exit means failure.
Baseline timings always come from Arena's frozen baseline role. Never use a generated seed
as the denominator. All actions share the invocation deadline and existing GPU visibility.
Do not nest gpu_lock around bridge commands or remap HIP/CUDA/ROCR device IDs.
The bridge serializes device actions across GEAK engineers using one deadline-bounded lock.
Any upstream examples mentioning specific vendor-tree symlinks or deleting/reusing output
directories are superseded: materialize_workspace.sh now makes fresh Arena task copies.
Never read credentials, dump the environment, install dependencies, access external services
other than the configured model, or publish/write learned knowledge outside this private run.
Upstream analysis/KB source is advisory; runtime dependencies are provisioned by Arena.
GEAK metrics guide search only. Arena independently evaluates delivered candidate files.
"""


def prepare_engine(checkout: Path, bridge: Bridge, *, python: str, options: dict) -> dict:
    verify_upstream(checkout, remaining=bridge.remaining)
    bridge.remaining()
    private = bridge.root / "engine"
    private.mkdir()
    workflow = private / "kernel_workflow"
    copy_tree(checkout / "kernel_workflow", workflow, remaining=bridge.remaining)
    copy_tree(checkout / "perf_knowledge", private / "perf_knowledge", remaining=bridge.remaining)
    lane = workflow / "kernel_lane.js"
    lane.write_text(adapt_lane(lane.read_text()))
    roles = Path(__file__).with_name("roles")
    for role in roles.glob("*.md"):
        copy_file(role, workflow / "roles" / role.name, remaining=bridge.remaining, overwrite=True)
    contract = arena_contract(bridge, python)
    for role in (workflow / "roles").glob("*.md"):
        bridge.remaining()
        content = role.read_text()
        if role.name == "tech_lead.md":
            content = content.replace('`reasoning`', '`decision_summary`').replace('"reasoning":', '"decision_summary":')
            content += "\nReturn only a brief decision summary and proposed actions, never private deliberations.\n"
        role.write_text(content + "\n\n" + contract)
    # The upstream copier special-cases vendor trees, excludes some potential
    # task inputs and clears existing destinations. Arena instead preserves all
    # task inputs and refuses overwrites. Only this private script is replaced.
    command = shlex.join([python, str(Path(__file__).with_name("bridge.py")),
                          "--job", str(bridge.job_path), "copy"])
    (workflow / "scripts/materialize_workspace.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nSRC=""\nDST=""\n'
        'while [[ $# -gt 0 ]]; do\n  case "$1" in\n'
        '    --src) SRC="$2"; shift 2;;\n    --dst) DST="$2"; shift 2;;\n'
        '    --shared-root|--soft-budget-bytes) shift 2;;\n'
        '    --link-aiter) shift;;\n    *) exit 2;;\n  esac\ndone\n'
        f'exec {command} --source "$SRC" --workspace "$DST"\n')
    (workflow / "scripts/reclaim_eval_artifacts.sh").write_text(
        '#!/usr/bin/env bash\n# Arena retains experiment artifacts for review.\n'
        'echo "Arena artifact retention: no files removed"\n')
    (bridge.eval_dir / "COMMANDMENT.md").write_text(contract)
    cases = bridge.job["baseline_performance"]["cases"]

    baseline = [{"name": row["test_case_id"], "baseline_ms": row["execution_time_ms"],
                 "ms": row["execution_time_ms"], "speedup": 1.0} for row in cases]
    candidate = bridge.spec.candidate
    author = candidate.initial_state == "unimplemented" or candidate.initial_language != candidate.language
    args = {
        "kernel_path": str(bridge.eval_dir / "original"),
        "workflow_dir": str(workflow), "kernel_lane_script": str(lane),
        "eval_dir": str(bridge.eval_dir), "exp_root": str(bridge.root / "runs"),
        "mode": "author" if author else "optimize", "target_language": candidate.language,
        "apply_to_original": "false", "budget": options["budget"],
        "min_improve": options["min_improve"], "deep_cost": options["deep_cost"],
        "gpu_ids": options["gpu_ids"], "gpu_mode": "pin",
        "deadline_epoch": bridge.job["deadline_epoch"],
        "agent_timeout_ms": max(1, int(bridge.remaining() * 1000)),
        "warm_start": "off", "update_experience": "off", "use_learned_kb": "false",
        "kb_remote": "off",
        "dra_enabled": "false", "use_expert_skills": "false", "frozen_oracle": "false",
        "arena_contract": contract, "task": contract, "arena_model": options.get("model"),
        "arena_setup": {"eval_dir": str(bridge.eval_dir),
                        "workspace": str(bridge.eval_dir / "workspace"),
                        "baseline_dir": str(bridge.context.baseline), "kernel_name": "arena_candidate",
                        "baseline_frozen": True, "baseline_callable": "",
                        "source_files": [edit.path for edit in candidate.editable]},
        "arena_benchmark": {"commandment_path": str(bridge.eval_dir / "COMMANDMENT.md"),
                            "baseline_per_case": baseline, "num_test_cases": len(baseline),
                            "baseline_geomean_ms": math.exp(sum(math.log(row["ms"]) for row in baseline) / len(baseline)),
                            "reliable": True},
    }
    write_json(bridge.root / "engine_identity.json", {
        "upstream_revision": UPSTREAM_REVISION, "upstream_scripts": SCRIPT_SHA256,
        "adapted_lane_sha256": hashlib.sha256(lane.read_bytes()).hexdigest(),
        "adapter_version": 1,
    })
    bridge.remaining()
    return {"script_path": str(workflow / "kernel_workflow.js"), "args": args}

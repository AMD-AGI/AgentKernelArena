#!/usr/bin/env python3
"""Select one captured head-kernel image, then use the standard Docker runner.

The run YAML selects tasks and an agent. Each task's headkernel.docker field
selects its capture image. A single container cannot mix these runtimes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("matrix", "plan", "preflight", "check-agents", "run", "parallel-run")


def runtime_contract_reader():
    helper = REPO_ROOT / "tasks/head_kernels/_support/runtime_preflight.py"
    spec = importlib.util.spec_from_file_location("top5_runtime_requirements", helper)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def environment_matrix(repo_root: Path = REPO_ROOT) -> str:
    """Render every task's declared environment from the preflight contract."""
    runtime = runtime_contract_reader()
    lines = [
        "# Head-kernel runtime environments", "",
        "Generated from task configs and protected capture metadata by",
        "`python3 src/scripts/top5_head_kernels.py matrix`.",
        "Regenerate this table after changing a task's runtime metadata.", "",
        "All tasks require an MI355X (`gfx950`). Versioned image tags are the",
        "capture runtime references. The Qwen metadata also records a captured",
        "Docker image/config ID, which is enforced when those tasks are selected.",
        "Docker image IDs are distinct from registry manifest digests. The runner",
        "records the actual launched image ID and available RepoDigests on each run.",
        "Registry digest qualification and unlisted package versions remain to be",
        "recorded during GPU qualification. An unlisted",
        "package version means the capture image supplies it; it is not permission",
        "to install a floating upgrade. This table is not a validation report.", "",
        "| Task | Capture image | Expected Docker image ID | ROCm/HIP | Package requirements | Captured source commits | Architecture registration | Required environment |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for path in sorted((repo_root / "tasks/head_kernels").rglob("config.yaml")):
        task = load_mapping(path)
        requirements = runtime.runtime_requirements(task, path.parent)
        versions = requirements["package_versions"]
        packages = "; ".join(f"{name}=={versions[name]}" if name in versions
                             else f"{name} (image version)"
                             for name in requirements["required_modules"])
        model_types = ", ".join(requirements["required_model_types"]) or "—"
        environment = "; ".join(f"`{name}={value}`" for name, value in requirements["environment"].items()) or "—"
        caches = "; ".join(f"`{name}`: isolated worker cache" for name in requirements["cache_environment"])
        environment = caches if environment == "—" else environment + "; " + caches
        image_id = f"`{requirements['expected_image_id']}`" if requirements["expected_image_id"] else "Not recorded"
        commits = "; ".join(f"{repo}: `{commit}`" for repo, commit in requirements["source_commits"].items()) or "Not recorded"
        relative_path = path.relative_to(repo_root).as_posix()
        selector = path.parent.relative_to(repo_root / "tasks").as_posix()
        lines.append(f"| [{selector}](../../{relative_path}) | "
                     f"`{requirements['image']}` | {image_id} | {requirements['hip_version']} | "
                     f"{packages} | {commits} | {model_types} | {environment} |")
    lines.extend(["", "See [the runtime guide](../how-to/top5-head-kernels-runtime.md) for",
                  "cohort launch commands, runtime enforcement, and remaining qualification.", ""])
    return "\n".join(lines)


def load_mapping(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def plan_run(config_path: Path, repo_root: Path = REPO_ROOT) -> dict:
    """Validate selectors and return the runtime without starting any process."""
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    if not config_path.is_relative_to(repo_root):
        raise ValueError("The run config must be inside the mounted repository")
    config = load_mapping(config_path)
    if config.get("target_gpu_model") != "MI355X":
        raise ValueError("Top-five head kernels require target_gpu_model: MI355X")
    validation_runtime = config.get("headkernel_validation_runtime")
    if validation_runtime is not None:
        if not isinstance(validation_runtime, str) or not validation_runtime:
            raise ValueError("headkernel_validation_runtime must be a declared runtime name")
        if (config.get("agent") or {}).get("template") != "task_validator":
            raise ValueError("An alternative validation runtime requires agent.template: task_validator")
    selectors = config.get("tasks")
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("The run config must select at least one head-kernel task")

    tasks_root = repo_root / "tasks"
    suite_root = (tasks_root / "head_kernels").resolve()
    selected: dict[str, str] = {}
    expected_image_ids = set()
    required_environment = {}
    runtime = runtime_contract_reader()
    for selector in selectors:
        if not isinstance(selector, str) or Path(selector).is_absolute():
            raise ValueError(f"Invalid task selector: {selector!r}")
        directory = (tasks_root / selector).resolve()
        if not directory.is_relative_to(suite_root):
            raise ValueError(f"Selector must be below tasks/head_kernels: {selector}")
        task_configs = sorted(directory.rglob("config.yaml"))
        if not task_configs:
            raise ValueError(f"Task selector matched no configs: {selector}")
        for task_path in task_configs:
            if not task_path.resolve().is_relative_to(suite_root):
                raise ValueError(f"Task config escapes the suite: {selector}")
            task = load_mapping(task_path)
            requirements = runtime.runtime_requirements(task, task_path.parent, validation_runtime)
            image = requirements["image"]
            if not isinstance(image, str) or not image.strip() or image != image.strip():
                raise ValueError(f"Missing headkernel.docker in {task_path}")
            reference = image.rsplit("/", 1)[-1]
            if (":" not in reference or reference.endswith(":latest")
                    or any(char.isspace() for char in image)):
                raise ValueError(f"A versioned image reference is required: {image!r}")
            selected[task_path.parent.relative_to(tasks_root).as_posix()] = image
            required_environment.update(requirements["environment"])
            if requirements["expected_image_id"]:
                expected_image_ids.add(requirements["expected_image_id"])

    images = sorted(set(selected.values()))
    if len(images) != 1:
        detail = "; ".join(f"{image}: {sum(v == image for v in selected.values())} tasks"
                           for image in images)
        raise ValueError(f"Mixed capture runtimes; use one cohort per run ({detail})")
    if len(expected_image_ids) > 1:
        raise ValueError("Selected tasks require conflicting captured Docker image IDs")
    if validation_runtime and len(selected) != 1:
        raise ValueError("An alternative validation runtime must select exactly one task")
    return {
        "config": config_path.relative_to(repo_root).as_posix(),
        "image": images[0],
        "validation_runtime": validation_runtime,
        "runtime_role": "validation_alternative" if validation_runtime else "capture",
        "expected_image_id": next(iter(expected_image_ids), None),
        "required_environment": required_environment,
        "target_gpu_model": "MI355X",
        "tasks": sorted(selected),
        "task_count": len(selected),
    }


def runtime_environment(plan: dict, environment: dict[str, str]) -> dict[str, str]:
    override = environment.get("AKA_DOCKER_IMAGE")
    if override and override != plan["image"]:
        raise ValueError("AKA_DOCKER_IMAGE conflicts with the selected tasks' "
                         f"runtime image: expected {plan['image']}, got {override}")
    validation_runtime = plan.get("validation_runtime") or ""
    inherited = environment.get("AKA_HEAD_KERNEL_VALIDATION_RUNTIME") or ""
    if inherited and inherited != validation_runtime:
        raise ValueError("AKA_HEAD_KERNEL_VALIDATION_RUNTIME must match the explicit run config selection")
    result = dict(environment)
    result["AKA_DOCKER_IMAGE"] = plan["image"]
    result["AKA_VERIFY_RUNTIME_IMAGE"] = "1"
    result["AKA_TOP5_ISOLATED_CACHES"] = "1"
    result["AKA_HEAD_KERNEL_VALIDATION_RUNTIME"] = validation_runtime
    result.update(plan.get("required_environment", {}))
    expected_id = plan.get("expected_image_id")
    if expected_id:
        if environment.get("AKA_EXPECTED_IMAGE_ID") not in (None, "", expected_id):
            raise ValueError("AKA_EXPECTED_IMAGE_ID conflicts with the selected capture metadata")
        result["AKA_EXPECTED_IMAGE_ID"] = expected_id
    # check-agents selects host architecture independently of the run config.
    result["AKA_GPU_ARCH"] = "gfx950"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", type=Path)
    args, runner_args = parser.parse_known_args(argv)
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    try:
        if args.action == "matrix":
            if args.config or runner_args:
                raise ValueError("matrix describes all task configs and accepts no runner arguments")
            print(environment_matrix(), end="")
            return 0
        if args.config is None:
            raise ValueError("--config is required for this action")
        if any(arg.split("=", 1)[0] in {"--config", "--config_name"}
               for arg in runner_args):
            raise ValueError("Pass the run config only with this launcher's --config option")
        plan = plan_run(args.config)
        environment = runtime_environment(plan, dict(os.environ))
        action = "run" if args.action == "plan" else args.action
        command = ["bash", "src/scripts/docker_benchmark.sh", action,
                   "--config_name", plan["config"], *runner_args]
        if args.action == "plan":
            identity_environment = {"AKA_VERIFY_RUNTIME_IMAGE": "1", "AKA_TOP5_ISOLATED_CACHES": "1"}
            if plan["validation_runtime"]:
                identity_environment["AKA_HEAD_KERNEL_VALIDATION_RUNTIME"] = plan["validation_runtime"]
            if environment.get("AKA_EXPECTED_IMAGE_ID"):
                identity_environment["AKA_EXPECTED_IMAGE_ID"] = environment["AKA_EXPECTED_IMAGE_ID"]
            print(json.dumps({**plan, "command": command,
                              "environment": {"AKA_DOCKER_IMAGE": plan["image"],
                                              "AKA_GPU_ARCH": "gfx950",
                                              **plan["required_environment"],
                                              **identity_environment}}, indent=2))
            return 0
        print(f"Top-five runtime: {plan['image']} ({plan['task_count']} tasks)",
              file=sys.stderr, flush=True)
        return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

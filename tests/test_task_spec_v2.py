"""Task contract tests independent of any agent, task family, or GPU runtime."""
from copy import deepcopy
import pytest

from src.task_spec import ACTIONS, TaskConfigError, TaskSpec, load_task_spec, resolve_task_path


def config():
    return {"schema_version": 2, "candidate": {"language": "hip", "editable": ["source/kernel.hip"]},
            "evaluation": {"runner": ["python3", "scripts/evaluate.py"]}}


def spec(value=None):
    return TaskSpec.from_mapping(config() if value is None else value, task_id="suite/operator")


def test_minimal_contract_expands_all_roles_without_task_type():
    task = spec()
    assert task.candidate.initial_state == "implemented"
    assert task.candidate.initial_language == "hip"
    assert task.baseline.kind == "initial_candidate"
    assert task.baseline.correctness_policy == "required"
    assert [(a.role, a.action) for a in task.actions] == list(ACTIONS)
    assert task.action("task", "validate-task").commands == (("python3", "scripts/evaluate.py", "validate-task"),)
    assert task.action("baseline", "correctness").commands[0][-2:] == ("baseline", "correctness")
    assert task.action("candidate", "performance").timeout_s == 3600
    restored = TaskSpec.from_mapping(task.to_mapping(), task_id=task.task_id)
    assert restored.candidate == task.candidate
    assert restored.baseline == task.baseline
    assert restored.actions == task.actions
    assert restored.to_mapping() == task.to_mapping()


def test_empty_candidate_and_cross_language_candidate_have_distinct_starting_states():
    raw = config()
    raw["candidate"]["initial_state"] = "unimplemented"
    assert spec(raw).baseline.kind == "provided"
    assert spec(raw).candidate.initial_language is None
    raw["candidate"].update(initial_state="implemented", initial_language="triton", language="flydsl")
    task = spec(raw)
    assert task.candidate.initial_language == "triton"
    assert task.candidate.language == "flydsl"
    assert task.baseline.kind == "initial_candidate"


def test_action_override_is_literal_argv_and_timeout_is_per_action():
    raw = config()
    commands = [["make", "a target"], ["python3", "scripts/build.py", "$(not a shell)"]]
    raw["evaluation"].update(timeout_s=400, candidate={"compile": {"commands": commands, "timeout_s": 90}})
    task = spec(raw)
    assert task.action("candidate", "compile").commands == tuple(tuple(c) for c in commands)
    assert task.action("candidate", "compile").timeout_s == 90
    assert task.action("candidate", "correctness").timeout_s == 400
    raw["evaluation"]["candidate"]["compile"] = {"timeout_s": 100}
    assert spec(raw).action("candidate", "compile").commands[0][-2:] == ("candidate", "compile")


def test_explicit_actions_need_all_seven_actions():
    raw = config()
    raw["evaluation"] = {"task": {"commands": [["validate"]]}}
    with pytest.raises(TaskConfigError, match="Missing commands"):
        spec(raw)
    for role in ("baseline", "candidate"):
        raw["evaluation"][role] = {action: {"commands": [["driver", role, action]]}
                                   for action in ("compile", "correctness", "performance")}
    assert len(spec(raw).actions) == 7


@pytest.mark.parametrize("update", [
    {"task_type": "hip2hip"}, {"compile_command": ["true"]}, {"schema_version": True},
    {"schema_version": 3}, {"environment": {}}, {"evaluation": {"runner": "python3 eval.py"}},
    {"evaluation": {"runner": ["python3"], "timeout_s": True}},
    {"evaluation": {"runner": ["python3"], "candidate": {"profile": {}}}},
    {"baseline": {"correctness_policy": "diagnostic"}},
    {"instructions": "Use a kernel"}, {"exports": [{"format": "sikl", "output": "out.json", "run": {}}]},
])
def test_rejects_mixed_or_ambiguous_contracts(update):
    raw = config()
    raw.update(update)
    with pytest.raises(TaskConfigError):
        spec(raw)


@pytest.mark.parametrize("path", ["/tmp/kernel.py", "../kernel.py", "src/../kernel.py", "src//kernel.py",
                                  "./kernel.py", "src\\kernel.py", "C:/kernel.py", "kernel.py/"])
def test_paths_reject_ambiguous_or_outside_spellings(path):
    raw = config()
    raw["candidate"]["editable"] = [path]
    with pytest.raises(TaskConfigError):
        spec(raw)


def test_nested_new_target_preserves_directory_and_checks_parent_symlinks(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    assert resolve_task_path(root, "source/new kernel.py") == root / "source/new kernel.py"
    (root / "source").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(TaskConfigError, match="within workspace"):
        resolve_task_path(root, "source/new kernel.py")
    with pytest.raises(FileNotFoundError):
        resolve_task_path(tmp_path / "missing", "kernel.py")


def test_entrypoints_and_scoped_edits_are_not_inferred_from_operator_names():
    raw = config()
    raw["kernel_identity"] = {"logical_operator": "no_relation_to_symbol", "source_owner": "aiter"}
    raw["candidate"] = {"language": "flydsl", "editable": [{"path": "source/kernel.py", "scope": "symbols",
                             "symbols": ["implementation"], "allow_new_helpers": True}],
                         "entrypoints": [{"file": "source/kernel.py", "kind": "builder", "symbol": "public_build"}]}
    task = spec(raw)
    assert task.candidate.entrypoints[0].symbol == "public_build"
    assert task.candidate.editable[0].symbols == ("implementation",)
    raw["candidate"]["entrypoints"][0]["file"] = "elsewhere/kernel.py"
    with pytest.raises(TaskConfigError, match="outside"):
        spec(raw)


@pytest.mark.parametrize("edits", [
    ["src/kernel.py", "src/kernel.py"],
    [{"path": "src", "scope": "tree"}, "src/kernel.py"],
    [{"path": "src/kernel.py", "scope": "symbols"}],
    [{"path": "src/kernel.py", "scope": "file", "allow_new_helpers": True}],
    ["config.yaml"],
    [{"path": "config.yaml", "scope": "symbols", "symbols": ["pretend"]}],
])
def test_edit_boundaries_cannot_silently_expand(edits):
    raw = config()
    raw["candidate"]["editable"] = edits
    with pytest.raises(TaskConfigError):
        spec(raw)


def test_declared_readonly_sources_cannot_be_candidate_files():
    raw = config()
    raw["baseline"] = {"kind": "provided", "source_files": ["source/kernel.hip"]}
    with pytest.raises(TaskConfigError, match="protected"):
        spec(raw)


def test_source_acquisition_exception_does_not_relax_candidate_paths():
    raw = config()
    raw["workspace"] = {"sources": [{"kind": "image", "image_path": "/runtime/aiter",
                                     "destination": "aiter_source", "exclude": ["jit", "__pycache__"]}],
                        "setup": [["python3", "scripts/setup.py"]]}
    spec(raw)
    raw["workspace"]["sources"].append({"kind": "git", "url": "https://example.invalid/upstream.git",
                                        "revision": "main", "destination": "other"})
    with pytest.raises(TaskConfigError, match="immutable"):
        spec(raw)
    raw["workspace"]["sources"][-1]["revision"] = "0" * 40
    spec(raw)
    raw["workspace"]["sources"][-1]["destination"] = "aiter_source/nested"
    with pytest.raises(TaskConfigError, match="overlap"):
        spec(raw)


def test_source_config_and_serialized_config_do_not_mutate_spec():
    raw = config()
    saved = deepcopy(raw)
    task = spec(raw)
    assert raw == saved
    raw["candidate"]["language"] = "corrupted"
    copy = task.to_mapping()
    copy["candidate"]["language"] = "also corrupted"
    assert task.to_mapping()["candidate"]["language"] == "hip"


def test_yaml_duplicate_fields_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 2\nschema_version: 1\n")
    with pytest.raises(TaskConfigError, match="Duplicate"):
        load_task_spec(path, task_id="suite/task")

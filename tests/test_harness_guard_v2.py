import logging
import shutil

import pytest
import yaml

from src.harness_guard import describe_workspace_harness, snapshot_workspace_harness, verify_workspace_harness


def workspace(tmp_path, *, symbols=False, helpers=False, tree=False):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source").mkdir()
    source = root / "source/kernel.py"
    source.write_text("import math\nTOLERANCE = 0.01\ndef compute(): return 1\ndef reference(): return 1\n")
    edit = "source/kernel.py"
    if tree:
        edit = {"path": "source", "scope": "tree"}
    elif symbols:
        edit = {"path": "source/kernel.py", "scope": "symbols", "symbols": ["compute"], "allow_new_helpers": helpers}
    config = {"schema_version": 2, "candidate": {"language": "triton", "editable": [edit]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    (root / "inputs.bin").write_bytes(b"task-specific inputs")
    return root, source


def test_file_edits_leave_all_non_candidate_task_material_protected(tmp_path):
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text("def compute(): return 2\n")
    verify_workspace_harness(before)
    (root / "inputs.bin").write_bytes(b"easier case")
    with pytest.raises(RuntimeError, match="inputs.bin"):
        verify_workspace_harness(before)


def test_read_only_guard_preserves_additions_but_evaluation_keeps_logged_cleanup(tmp_path, caplog):
    root, _ = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    extra = root / "debug_test.py"
    extra.write_text("# agent's diagnostic\n")
    with pytest.raises(RuntimeError, match="Added protected harness files require review"):
        verify_workspace_harness(before, discard_added=False)
    assert extra.read_text() == "# agent's diagnostic\n"

    verify_workspace_harness(before, logger=logging.getLogger(__name__))
    assert not extra.exists()
    assert "debug_test.py" in caplog.text


def test_symbol_edits_only_mask_named_implementation_and_permitted_new_helpers(tmp_path):
    root, source = workspace(tmp_path, symbols=True, helpers=True)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace("def compute(): return 1", "def compute(): return helper()")
                      + "def helper(): return 1\n")
    verify_workspace_harness(before)
    source.write_text(source.read_text() + "def reference(): return 999\n")
    with pytest.raises(RuntimeError, match="source/kernel.py"):
        verify_workspace_harness(before)


@pytest.mark.parametrize("change", [
    lambda text: text.replace("import math", "import random as math"),
    lambda text: text.replace("TOLERANCE = 0.01", "TOLERANCE = 1"),
    lambda text: text.replace("def reference(): return 1", "def reference(): return 2"),
    lambda text: text + "def new_helper(): return 2\n",
])
def test_symbol_scope_does_not_allow_import_reference_or_unpermitted_helper_edits(tmp_path, change):
    root, source = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    source.write_text(change(source.read_text()))
    with pytest.raises(RuntimeError, match="source/kernel.py"):
        verify_workspace_harness(before)


def test_tree_edit_does_not_override_harness_file_protection(tmp_path):
    root, source = workspace(tmp_path, tree=True)
    harness = root / "source/check_harness.py"
    harness.write_text("# trusted evaluation\n")
    before = snapshot_workspace_harness(root)
    source.write_text("optimized")
    verify_workspace_harness(before)
    harness.write_text("# weaker evaluation\n")
    with pytest.raises(RuntimeError, match="check_harness.py"):
        verify_workspace_harness(before)


def test_new_task_config_cannot_redefine_the_protected_boundary(tmp_path):
    root, source = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["candidate"]["editable"] = ["source/kernel.py"]
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    source.write_text("def reference(): return 'anything passes'\n")
    with pytest.raises(RuntimeError, match="config.yaml"):
        verify_workspace_harness(before)


def test_protected_symlink_escape_is_rejected_even_if_bytes_match(tmp_path):
    root, _ = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    original = root / "inputs.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(RuntimeError, match="escaped workspace"):
        verify_workspace_harness(before)


def test_missing_shipped_file_is_not_omitted_from_snapshot(tmp_path):
    root, _ = workspace(tmp_path)
    package = tmp_path / "task"
    shutil.copytree(root, package)
    (root / "inputs.bin").unlink()
    with pytest.raises(RuntimeError, match="inputs.bin"):
        snapshot_workspace_harness(root, task_root=package)


def test_validator_description_uses_v2_scope(tmp_path):
    root, _ = workspace(tmp_path, symbols=True)
    info = describe_workspace_harness(root)
    assert info["editable_entrypoint_targets"] == {"source/kernel.py": ["compute"]}
    assert "evaluate.py" in info["protected_paths"]


@pytest.mark.parametrize("name", ["yaml.py", "usercustomize.py"])
def test_new_import_shadow_files_follow_described_guard_policy(tmp_path, name):
    root, _ = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    info = describe_workspace_harness(root, snapshot=before)
    assert "scripts" in info["new_file_policy"]["protected_directories"]
    extra = root / "scripts" / name
    extra.parent.mkdir()
    extra.write_text("raise RuntimeError('unexpected import')\n")
    with pytest.raises(RuntimeError, match="Added protected harness files"):
        verify_workspace_harness(before, discard_added=False)
    assert extra.exists()
    verify_workspace_harness(before)
    assert not extra.exists()



LAUNCH_SOURCE = ("import math\n"
                 "def _launch_config(n):\n    return {'BLOCK': 64, 'num_warps': 4}\n"
                 "def launch(n):\n    cfg = _launch_config(n)\n    return cfg['BLOCK'] * n\n")
LAUNCH_RETURN = "    return {'BLOCK': 64, 'num_warps': 4}\n"


def launch_workspace(tmp_path, *, declared=True):
    root = tmp_path / "workspace"
    (root / "source").mkdir(parents=True)
    source = root / "source/kernel.py"
    source.write_text(LAUNCH_SOURCE)
    edit = {"path": "source/kernel.py", "scope": "symbols", "symbols": ["_launch_config"], "allow_new_helpers": True}
    if declared:
        edit["configuration_symbols"] = ["_launch_config"]
    config = {"schema_version": 2, "candidate": {"language": "triton", "editable": [edit]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    return root, source


def test_declared_configuration_symbol_may_be_retuned(tmp_path):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    assert describe_workspace_harness(root, snapshot=before)["protected_path_policies"]["source/kernel.py"][
        "configuration_symbols"] == ["_launch_config"]
    tuned = ("    if n >= 4096:\n        return {'BLOCK': 256, 'num_warps': 8}\n"
             "    block = 128\n    return {'BLOCK': block, 'num_warps': 4}\n")
    source.write_text(source.read_text().replace(LAUNCH_RETURN, tuned))
    verify_workspace_harness(before)


HOST_CODE = [
    "    math.sqrt(n)\n",
    "    import os\n",
    "    for _ in range(2):\n        pass\n",
    "    launch.cache = 1\n",
    "    globals()['TOLERANCE'] = 1\n",
    "    with open('x') as f:\n        pass\n",
]


@pytest.mark.parametrize("host", HOST_CODE)
def test_declared_configuration_symbol_cannot_run_host_code(tmp_path, host):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace(LAUNCH_RETURN, host + LAUNCH_RETURN))
    with pytest.raises(RuntimeError, match="only builds configuration data"):
        verify_workspace_harness(before)


@pytest.mark.parametrize("change", [
    lambda text: text.replace("def _launch_config(n):", "@staticmethod\ndef _launch_config(n):"),
    lambda text: text.replace("def _launch_config(n):\n" + LAUNCH_RETURN, ""),
])
def test_declared_configuration_symbol_must_remain_a_plain_function(tmp_path, change):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(change(source.read_text()))
    with pytest.raises(RuntimeError, match="source/kernel.py|kernel.py"):
        verify_workspace_harness(before)


def test_undeclared_editable_functions_keep_the_existing_definition_policy(tmp_path):
    # Python implementation tasks let protected code call editable host code.
    root, source = launch_workspace(tmp_path, declared=False)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace(LAUNCH_RETURN, HOST_CODE[0] + LAUNCH_RETURN))
    verify_workspace_harness(before)


@pytest.mark.parametrize("edit,message", [
    ({"symbols": ["compute"], "configuration_symbols": ["other"]}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": "compute"}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": ["compute"], "entry": True}, "entrypoint"),
    ({"scope": "file", "configuration_symbols": []}, "symbol scope"),
])
def test_configuration_symbols_schema(tmp_path, edit, message):
    from src.task_spec import TaskConfigError, load_task_spec
    entry = edit.pop("entry", False)
    raw = {"path": "source/kernel.py", "scope": "symbols", **edit}
    candidate = {"language": "triton", "initial_state": "implemented", "initial_language": "triton",
                 "editable": [raw]}
    if entry:
        candidate["entrypoints"] = [{"file": "source/kernel.py", "kind": "function", "symbol": "compute"}]
    config = {"schema_version": 2, "description": "d", "candidate": candidate,
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(TaskConfigError, match=message):
        load_task_spec(path, task_id="t")

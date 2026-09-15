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

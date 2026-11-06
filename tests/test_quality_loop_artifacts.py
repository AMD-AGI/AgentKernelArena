"""Keep runtime outputs out of task commits, including old accepted manifests."""
import logging
import subprocess

import pytest

from agents.quality_loop.config import GitHubConfig
from agents.quality_loop.filesystem import apply_changes, snapshot_tree
from agents.quality_loop.github import GitHubPublisher
from agents.quality_loop.orchestrator import _filtered_changes


# Framework- and auditor-written workspace files plus build outputs.
ARTIFACTS = {
    "config.yaml": b"schema_version: 2\n",
    "solution/kernel.py": b"def run(**kwargs): pass\n",
    "_aka_eval/evaluate.py": b"generated evaluator",
    "_aka_eval/_aka_benchmark.py": b"generated helper",
    "task_result.yaml": b"pass_correctness: true\n",
    "qa_report.yaml": b"status: PASS\n",
    ".qa_complete": b"{}\n",
    "qa_review.draft.json": b"{}\n",
    "quality_loop_review.yaml": b"accepted: true\n",
    "kernel.o": b"object",
    "kernel.hsaco": b"code object",
    "baseline/__pycache__/kernel.cpython-312.pyc": b"bytecode",
    "native_runner": b"\x7fELF" + bytes(60),
}
TASK = "tasks/elementwise/scale/f32"


def write_files(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_filtered_copy_keeps_package_files_and_fixtures_but_not_run_outputs(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    before = snapshot_tree(source)
    kept = {
        "task.yaml": b"schema_version: 3\n",
        "workloads.json": b'{"cases": []}\n',
        "inputs.py": b"def make_inputs(case, inputs, generator, device): return {}\n",
        "baseline/kernel.py": b"def run(**kwargs): pass\n",
        "baseline/LICENSE": b"license\n",
        "fixtures/reference.pt": b"PK intentional tensor fixture",
    }
    write_files(source, {**kept, **ARTIFACTS})
    after = snapshot_tree(source)
    # Filtering proposals must not hide candidate/evaluator/report tampering
    # from reviewer snapshots; only transient trees are invisible there.
    assert set(ARTIFACTS) - {"baseline/__pycache__/kernel.cpython-312.pyc"} <= after.keys()
    changes = _filtered_changes(before, after, source_root=source)
    assert set(changes.paths) == set(kept)
    apply_changes(source, destination, changes)
    assert snapshot_tree(destination) == {p: after[p] for p in kept}


def test_modified_outputs_are_filtered(tmp_path):
    write_files(tmp_path, ARTIFACTS)
    before = snapshot_tree(tmp_path)
    write_files(tmp_path, {p: data + b"updated" for p, data in ARTIFACTS.items()})
    changes = _filtered_changes(before, snapshot_tree(tmp_path), source_root=tmp_path)
    assert changes.empty


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.email", "quality-loop@example.invalid")
    git("config", "user.name", "quality_loop test")
    write_files(tmp_path, {f"{TASK}/task.yaml": b"schema_version: 3\n"})
    git("add", ".")
    git("commit", "-m", "base")
    publisher = GitHubPublisher(tmp_path, GitHubConfig(), logging.getLogger(__name__))
    return tmp_path, git, publisher


@pytest.mark.parametrize("relative", [p for p in ARTIFACTS if "__pycache__" not in p])
@pytest.mark.parametrize("staged", [False, True])
def test_host_rejects_outputs_even_if_old_manifest_accepts_them(repository, relative, staged):
    root, git, publisher = repository
    path = f"{TASK}/{relative}"
    write_files(root, {path: ARTIFACTS[relative]})
    if staged:
        git("add", "--", path)
    base = git("rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="generated task artifacts"):
        publisher.verify_pending_changes(
            worktree=root, branch=git("branch", "--show-current"),
            base_sha=base, expected_paths={path},
        )
    # Direct publication must have the same guard, not just deferred host mode.
    with pytest.raises(RuntimeError, match="generated task artifacts"):
        publisher.commit_task(root, "elementwise/scale/f32")
    assert git("rev-parse", "HEAD") == base


def test_host_allows_fixture_addition_and_legacy_artifact_removal(repository):
    root, git, publisher = repository
    artifact = f"{TASK}/task_result.yaml"
    write_files(root, {artifact: b"correctness: true\n"})
    git("add", ".")
    git("commit", "-m", "legacy output")
    (root / artifact).unlink()
    fixture = f"{TASK}/workloads.json"
    write_files(root, {fixture: b'{"cases": []}\n'})
    publisher.verify_pending_changes(
        worktree=root, branch=git("branch", "--show-current"),
        base_sha=git("rev-parse", "HEAD"), expected_paths={artifact, fixture},
    )
    assert publisher.commit_task(root, "elementwise/scale/f32")
    assert git("status", "--porcelain") == ""
    assert git("ls-files", "--", artifact) == ""
    assert git("ls-files", "--", fixture) == fixture

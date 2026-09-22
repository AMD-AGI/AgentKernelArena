"""Keep runtime outputs out of task commits, including old accepted manifests."""
import logging
import subprocess

import pytest

from agents.quality_loop.config import GitHubConfig
from agents.quality_loop.filesystem import apply_changes, snapshot_tree
from agents.quality_loop.github import GitHubPublisher
from agents.quality_loop.orchestrator import _filtered_changes


ARTIFACTS = {
    "perf_report.json": b"[]",
    "perf/benchmark_results.json": b"[]",
    "test_kernel_py.pt": b"PK tensor output",
    "eval_result.yaml": b"correctness: true\n",
    "correctness_report.json": b'{}',
    "scripts/_aka_benchmark.py": b"generated helper",
    "scripts/native/hip_graph_benchmark.hpp": b"generated header",
    "kernel.o": b"object",
    "kernel.hsaco": b"code object",
    "native_runner": b"\x7fELF" + bytes(60),
}


def write_files(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_filtered_copy_keeps_sources_and_fixtures_but_not_run_outputs(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    before = snapshot_tree(source)
    kept = {
        "kernel.py": b"def kernel(): return 1\n",
        "tests/reference.pt": b"PK intentional tensor fixture",
        "inputs/data.bin": b"input bytes",
        "scripts/run.sh": b"#!/bin/sh\nexit 1\n",
    }
    write_files(source, {**kept, **ARTIFACTS})
    after = snapshot_tree(source)
    # Filtering proposals must not hide artifact tampering from reviewer snapshots.
    assert set(ARTIFACTS) <= after.keys()
    changes = _filtered_changes(before, after, source_root=source)
    assert set(changes.paths) == set(kept)
    apply_changes(source, destination, changes)
    assert snapshot_tree(destination) == {p: after[p] for p in kept}


@pytest.mark.parametrize("destination", ["upstream", "vendor/upstream"])
def test_modified_outputs_and_cloned_dependencies_are_filtered(tmp_path, destination):
    write_files(tmp_path, ARTIFACTS)
    before = snapshot_tree(tmp_path)
    write_files(tmp_path, {p: data + b"updated" for p, data in ARTIFACTS.items()})
    write_files(tmp_path, {f"{destination}/source.py": b"downloaded source"})
    changes = _filtered_changes(
        before, snapshot_tree(tmp_path), materialized=(destination,), source_root=tmp_path
    )
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
    write_files(tmp_path, {"tasks/hip2hip/sample/kernel.py": b"pass\n"})
    git("add", ".")
    git("commit", "-m", "base")
    publisher = GitHubPublisher(tmp_path, GitHubConfig(), logging.getLogger(__name__))
    return tmp_path, git, publisher


@pytest.mark.parametrize("relative", list(ARTIFACTS))
@pytest.mark.parametrize("staged", [False, True])
def test_host_rejects_outputs_even_if_old_manifest_accepts_them(repository, relative, staged):
    root, git, publisher = repository
    path = f"tasks/hip2hip/sample/{relative}"
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
        publisher.commit_task(root, "hip2hip/sample")
    assert git("rev-parse", "HEAD") == base


def test_host_allows_fixture_addition_and_legacy_artifact_removal(repository):
    root, git, publisher = repository
    artifact = "tasks/hip2hip/sample/eval_result.yaml"
    write_files(root, {artifact: b"correctness: true\n"})
    git("add", ".")
    git("commit", "-m", "legacy output")
    (root / artifact).unlink()
    fixture = "tasks/hip2hip/sample/tests/reference.pt"
    write_files(root, {fixture: b"PK intentional tensor fixture"})
    publisher.verify_pending_changes(
        worktree=root, branch=git("branch", "--show-current"),
        base_sha=git("rev-parse", "HEAD"), expected_paths={artifact, fixture},
    )
    assert publisher.commit_task(root, "hip2hip/sample")
    assert git("status", "--porcelain") == ""
    assert git("ls-files", "--", artifact) == ""
    assert git("ls-files", "--", fixture) == fixture


def test_all_v2_materialized_destinations_are_excluded_without_prefix_overreach(tmp_path):
    before = snapshot_tree(tmp_path)
    files = {"vendor/one/kernel.py": b"download", "vendor/two/lib.py": b"download",
             "vendor/one_more/fixture.py": b"fixture", "kernel.py": b"candidate"}
    write_files(tmp_path, files)
    changes = _filtered_changes(before, snapshot_tree(tmp_path),
                                materialized=("vendor/one", "vendor/two"), source_root=tmp_path)
    assert set(changes.paths) == {"vendor/one_more/fixture.py", "kernel.py"}

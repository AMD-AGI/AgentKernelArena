from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


MODULE = Path(__file__).resolve().parents[1] / "src/scripts/forge_runtime.py"
spec = importlib.util.spec_from_file_location("forge_runtime_installation", MODULE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def commit(source, value):
    source.mkdir(exist_ok=True)
    if not (source / ".git").exists():
        subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "engine_version.py").write_text(f"VERSION = {value!r}\n")
    subprocess.run(["git", "-C", str(source), "add", "engine_version.py"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.com", "commit", "-qm", value], check=True)
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def fake_pip(monkeypatch, *, fail=False):
    real_run = subprocess.run
    calls = []

    def run(argv, **kwargs):
        if argv[1:4] == ["-m", "pip", "install"]:
            calls.append(argv)
            time.sleep(0.03)
            if fail:
                raise subprocess.CalledProcessError(1, argv)
            source = Path(argv[6].removesuffix("[forge]"))
            shutil.copytree(source, argv[5])
            return subprocess.CompletedProcess(argv, 0)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    return calls


def read_version(path):
    return subprocess.check_output([sys.executable, "-c",
                                    "import sys; sys.path.insert(0, sys.argv[1]); "
                                    "import engine_version; print(engine_version.VERSION)", str(path)],
                                   text=True).strip()


def test_two_campaigns_keep_their_own_engine_after_head_moves(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    first = commit(source, "one")
    calls = fake_pip(monkeypatch)
    first_path = runtime.install(cache, source, first, "image-one")
    second = commit(source, "two")
    second_path = runtime.install(cache, source, second, "image-one")
    assert first_path != second_path
    assert read_version(first_path) == "one"
    assert read_version(second_path) == "two"
    assert runtime.install(cache, source, first, "image-one") == first_path
    assert len(calls) == 2


def test_export_uses_the_pinned_revision_not_current_head(tmp_path, monkeypatch):
    source = tmp_path / "source"
    first = commit(source, "one")
    commit(source, "two")
    (source / "engine_version.py").write_text("VERSION = 'uncommitted'\n")
    fake_pip(monkeypatch)
    target = runtime.install(tmp_path / "cache", source, first, "image-one")
    assert read_version(target) == "one"
    metadata = json.loads((target / ".aka-forge-runtime.json").read_text())
    assert metadata == {"revision": first, "image_id": "image-one",
                        "python_cache_tag": sys.implementation.cache_tag}


def test_same_revision_is_separate_for_different_runtime_images(tmp_path, monkeypatch):
    source = tmp_path / "source"
    revision = commit(source, "one")
    calls = fake_pip(monkeypatch)
    a = runtime.install(tmp_path / "cache", source, revision, "image-a")
    b = runtime.install(tmp_path / "cache", source, revision, "image-b")
    assert a != b and len(calls) == 2


def test_simultaneous_setup_publishes_only_one_complete_install(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    revision = commit(source, "one")
    calls = fake_pip(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: runtime.install(cache, source, revision, "image-one"), range(2)))
    assert results[0] == results[1] and len(calls) == 1
    assert read_version(results[0]) == "one"


def test_failed_install_never_publishes_or_changes_an_existing_revision(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    first = commit(source, "one")
    with monkeypatch.context() as patch:
        fake_pip(patch)
        first_path = runtime.install(cache, source, first, "image-one")
    second = commit(source, "two")
    with monkeypatch.context() as patch:
        fake_pip(patch, fail=True)
        with pytest.raises(subprocess.CalledProcessError):
            runtime.install(cache, source, second, "image-one")
    second_path = runtime.installation_path(cache, second, "image-one")
    assert not second_path.exists()
    assert not list(second_path.parent.glob(".forge-install-*"))
    assert read_version(first_path) == "one"


@pytest.mark.parametrize("revision,image", [("HEAD", "image"), ("../escape", "image"), ("a" * 40, "")])
def test_install_identity_must_be_explicit_and_safe(tmp_path, revision, image):
    with pytest.raises(ValueError):
        runtime.installation_path(tmp_path, revision, image)

"""Complete JIT cache parity and root-only initialization boundaries."""

import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src.tools import seed_aiter_jit_cache as seed


@pytest.fixture
def image_package(tmp_path):
    if shutil.which("rclone") is None:
        pytest.skip("cache payload staging requires rclone")
    package = tmp_path / "image/aiter"
    jit = package / "jit"
    (jit / "flydsl_cache").mkdir(parents=True)
    (jit / "empty").mkdir()
    (jit / "module_quant.so").write_bytes(b"complete precompiled native extension")
    (jit / "flydsl_cache/compiled.pkl").write_bytes(b"restricted image cache")
    (jit / "flydsl_cache/compiled.pkl").chmod(0o600)
    (jit / "alias.so").symlink_to("module_quant.so")
    return package


def test_initializer_preserves_all_bytes_empty_dirs_links_and_restricted_cache(image_package, tmp_path, monkeypatch):
    monkeypatch.setattr(seed.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(image_package)]))
    destination = tmp_path / "seed"
    destination.mkdir()
    proof = seed.initialize_inside(destination, os.getuid(), os.getgid(), shutil.which("rclone"))
    assert proof["complete_parity"] is True
    assert proof["entries"] == seed.tree_manifest(image_package / "jit") == seed.tree_manifest(destination / "jit")
    assert proof["file_count"] == 2
    assert (destination / "jit/flydsl_cache/compiled.pkl").stat().st_uid == os.getuid()
    assert (destination / "jit/module_quant.so").read_bytes() == b"complete precompiled native extension"
    assert (destination / "jit/alias.so").is_symlink()
    assert (destination / "jit/empty").is_dir()


def test_root_initializer_has_no_gpu_network_or_agent_auth_mounts(image_package, tmp_path, monkeypatch):
    monkeypatch.setattr(seed.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(image_package)]))
    destination = tmp_path / "seed"
    original_run = subprocess.run
    commands = []

    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            commands.append(command)
            seed.initialize_inside(destination, os.getuid(), os.getgid(), shutil.which("rclone"))
            return None
        if command[:2] == ["docker", "rm"]:
            return None
        assert command[:2] != ["docker", "run"]
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    image = "registry/image@sha256:" + "a" * 64
    proof = seed.seed_image_cache(image, destination, tmp_path / "init.log")
    command = commands[0]
    assert command[command.index("--user") + 1] == "0:0"
    assert "--network=none" in command and "--read-only" in command
    assert "--device" not in command and "--privileged" not in command
    assert "--cap-add=CHOWN" in command and "--cap-add=DAC_OVERRIDE" in command
    assert "GOMAXPROCS=1" in command
    assert command.count("--mount") == 3
    assert not any("docker.sock" in part or ".codex" in part for part in command)
    assert image in command and proof["complete_parity"] is True


def test_bulk_copy_bounds_go_threads_and_buffers(image_package, tmp_path, monkeypatch):
    original = subprocess.run
    monkeypatch.setenv("GOMAXPROCS", "128")

    def run(command, **kwargs):
        assert command[command.index("--transfers") + 1] == "64000"
        assert "--progress" in command
        assert command[command.index("--buffer-size") + 1] == "0"
        assert kwargs["env"]["GOMAXPROCS"] == "1"
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    seed.copy_cache(image_package / "jit", tmp_path / "bounded-copy")


def test_missing_file_after_transfer_rejects_cache_parity(image_package, tmp_path, monkeypatch):
    original = subprocess.run

    def lose_file(command, **kwargs):
        result = original(command, **kwargs)
        (tmp_path / "copy/module_quant.so").unlink()
        return result

    monkeypatch.setattr(subprocess, "run", lose_file)
    with pytest.raises(ValueError, match="parity failed"):
        seed.copy_cache(image_package / "jit", tmp_path / "copy")


def test_seed_verification_rejects_forged_manifest(tmp_path, monkeypatch):
    destination = tmp_path / "seed"

    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            (destination / "jit").mkdir()
            (destination / "jit/module_quant.so").write_bytes(b"wrong")
            (destination / "manifest.json").write_text(json.dumps({
                "complete_parity": True, "owner_uid": os.getuid(), "owner_gid": os.getgid(), "entries": {},
            }))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rclone")
    with pytest.raises(ValueError, match="parity/ownership evidence"):
        seed.seed_image_cache("registry/image@sha256:" + "a" * 64, destination, tmp_path / "init.log")

"""CPU-only opt-in copy batching and real descriptor-limit coverage."""
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.tools import seed_aiter_jit_cache as seed
from test_seed_aiter_jit_cache import image_package


@pytest.mark.parametrize("requested,soft,expected", [(None,1024,64), (512,1048576,512),
                                                     (512,1024,240), (512,256,48), (512,-1,512)])
def test_runtime_descriptor_budget(monkeypatch, requested, soft, expected):
    monkeypatch.delenv(seed.COPY_BATCH_ENV, raising=False)
    monkeypatch.setenv(seed.COPY_THREAD_ENV, "2048")
    monkeypatch.setattr(seed, "copy_thread_headroom", lambda: {"headroom": 4096})
    monkeypatch.setattr(seed.resource, "getrlimit", lambda which: (soft, soft))
    assert seed.copy_batch_size(requested) == (expected, soft)


@pytest.mark.parametrize("value", [0, -1, 513, True, "bad", "1.5"])
def test_invalid_request_does_not_create_destination(tmp_path, value):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="batch size"):
        seed.copy_cache(tmp_path, output, batch_size=value)
    assert not output.exists()


def test_exhausted_descriptor_budget_rejects_before_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(seed.resource, "getrlimit", lambda which: (64,64))
    with pytest.raises(ValueError, match="RLIMIT_NOFILE"):
        seed.copy_cache(tmp_path, tmp_path / "never-created", batch_size=512)
    assert not (tmp_path / "never-created").exists()


def test_larger_batches_keep_flags_three_hash_passes_links_and_modes(image_package, tmp_path, monkeypatch):
    source = image_package / "jit"
    for index in range(509):
        (source / "flydsl_cache" / f"item-{index:04d}").write_bytes(str(index).encode())
    (source / "check.py").write_text("print('source')\n")
    (source / "check.py").chmod(0o755)
    output = tmp_path / "larger-copy"
    monkeypatch.setenv(seed.COPY_BATCH_ENV, "512")
    monkeypatch.setenv(seed.COPY_THREAD_ENV, "2048")
    monkeypatch.setattr(seed, "copy_thread_headroom", lambda: {"headroom": 4096})
    monkeypatch.setattr(seed.resource, "getrlimit", lambda which: (1048576,1048576))
    original_run, original_manifest = subprocess.run, seed.tree_manifest
    batches, hashes = [], []
    def run(command, **kwargs):
        batches.append(Path(command[command.index("--files-from-raw")+1]).read_text().splitlines())
        for flag,value in [("--transfers","64000"),("--buffer-size","0"),("--multi-thread-streams","0"),("--config",os.devnull)]:
            assert command[command.index(flag)+1] == value
        assert all(flag in command for flag in ["--progress","--links","--no-traverse"])
        assert kwargs["env"]["GOMAXPROCS"] == "1"
        return original_run(command, **kwargs)
    def manifest(path):
        hashes.append(Path(path)); return original_manifest(path)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(seed, "tree_manifest", manifest)
    expected = seed.copy_cache(source, output)
    assert [len(batch) for batch in batches] == [512,1]
    assert hashes == [source,output,source]
    assert original_manifest(output) == expected == original_manifest(source)
    assert (output / "alias.so").is_symlink() and (output / "empty").is_dir()
    assert stat.S_IMODE((output / "check.py").stat().st_mode) == 0o755


def test_real_copy_under_lower_nofile_limit(image_package, tmp_path):
    source = image_package / "jit"
    for index in range(257):
        (source / "flydsl_cache" / f"low-{index}").write_bytes(b"data")
    output = tmp_path / "low-limit-copy"
    script = """import resource, sys
from src.tools.seed_aiter_jit_cache import copy_cache
resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
copy_cache(sys.argv[1], sys.argv[2], batch_size=512)
"""
    result = subprocess.run([sys.executable,"-B","-c",script,str(source),str(output)],
                            capture_output=True,text=True,check=True,timeout=30)
    assert "effective=48 rlimit_nofile=256" in result.stdout
    assert seed.tree_manifest(output) == seed.tree_manifest(source)


def test_seed_forwards_request_for_container_local_recapping(image_package, tmp_path, monkeypatch):
    monkeypatch.setenv(seed.COPY_BATCH_ENV, "512")
    monkeypatch.setenv(seed.COPY_THREAD_ENV, "2048")
    monkeypatch.setattr(seed, "copy_thread_headroom", lambda: {"headroom": 4096})
    monkeypatch.setattr(seed.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(image_package)]))
    destination = tmp_path / "seed"
    original = subprocess.run
    seen = []
    def run(command, **kwargs):
        if command[:2] == ["docker","run"]:
            requested = int(command[command.index("--batch-size")+1]); seen.append(requested)
            with monkeypatch.context() as inner:
                inner.setattr(seed.resource,"getrlimit",lambda which:(256,256))
                assert seed.copy_batch_size(requested) == (48,256)
                seed.initialize_inside(destination,os.getuid(),os.getgid(),shutil.which("rclone"),batch_size=requested)
            return None
        if command[:2] == ["docker","rm"]: return None
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess,"run",run)
    proof = seed.seed_image_cache("registry/image@sha256:"+"a"*64,destination,tmp_path/"seed.log")
    assert seen == [512] and proof["complete_parity"] is True


@pytest.mark.parametrize("headroom,allowance,expected", [(None,"2048",64),({"headroom":4096},None,64),({"headroom":128},"2048",32),({"headroom":4096},"1088",512)])
def test_large_batches_require_observed_headroom_and_reserved_allowance(monkeypatch, headroom, allowance, expected):
    monkeypatch.setattr(seed.resource,"getrlimit",lambda which:(1048576,1048576))
    monkeypatch.setattr(seed,"copy_thread_headroom",lambda:headroom)
    if allowance is None:monkeypatch.delenv(seed.COPY_THREAD_ENV,raising=False)
    else:monkeypatch.setenv(seed.COPY_THREAD_ENV,allowance)
    assert seed.copy_batch_size(512)[0] == expected
    assert seed.copy_batch_size(64)[0] <= 64


def test_actual_accounting_parser_takes_tightest_uid_or_cgroup_limit(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'; group = tmp_path / 'cgroup'
    (proc/'self').mkdir(parents=True); (proc/'self/cgroup').write_text('0::/scope/worker\n')
    (proc/'123').mkdir(); (proc/'123/status').write_text('Uid:\t500 500 500 500\nThreads:\t20\n')
    (group/'scope/worker').mkdir(parents=True)
    for path, limit, current in [(group/'scope',512,50),(group/'scope/worker',128,20)]:
        (path/'pids.max').write_text(str(limit)); (path/'pids.current').write_text(str(current))
    original_path = Path
    def mapped(path):
        return proc if path == '/proc' else proc/'self/cgroup' if path == '/proc/self/cgroup' else group if path == '/sys/fs/cgroup' else original_path(path)
    monkeypatch.setattr(seed,'Path',mapped)
    monkeypatch.setattr(seed.os,'getuid',lambda:500)
    monkeypatch.setattr(seed.resource,'getrlimit',lambda which:(1000,1000))
    result = seed.copy_thread_headroom()
    assert result['visible_uid_threads'] == 20 and result['headroom'] == 108
    assert len(result['cgroups']) == 2
    (proc/'self/cgroup').unlink()
    assert seed.copy_thread_headroom() is None

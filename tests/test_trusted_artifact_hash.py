"""Streaming artifact checks preserve bounds and the trusted path boundary."""
import hashlib

import pytest

from src.tools import trusted_native_eval as trusted


def test_stream_hash_matches_bytes_and_enforces_declared_size(tmp_path):
    path = tmp_path / "artifact.pt"
    path.write_bytes(b"original bytes")
    assert trusted.hash_regular_file(path, max_bytes=14) == hashlib.sha256(b"original bytes").hexdigest()
    with pytest.raises(ValueError, match="declared size"):
        trusted.hash_regular_file(path, max_bytes=13)
    with pytest.raises(ValueError, match="2 GiB"):
        trusted.hash_regular_file(path, max_bytes=(2 << 30) + 1)


@pytest.mark.parametrize("parent", [False, True])
def test_stream_hash_rejects_symlinks(tmp_path, parent):
    directory = tmp_path / "real"
    directory.mkdir()
    path = directory / "artifact.pt"
    path.write_bytes(b"data")
    link = tmp_path / "link"
    link.symlink_to(directory if parent else path, target_is_directory=parent)
    with pytest.raises(OSError):
        trusted.hash_regular_file(link / "artifact.pt" if parent else link, max_bytes=4)


def test_stream_hash_rejects_growth_during_read(tmp_path, monkeypatch):
    path = tmp_path / "artifact.pt"
    path.write_bytes(b"data")
    digest = hashlib.sha256()

    class GrowingHash:
        def update(self, block):
            with path.open("ab") as stream:
                stream.write(b"extra")
            digest.update(block)

        def hexdigest(self):
            return digest.hexdigest()

    monkeypatch.setattr(trusted.hashlib, "sha256", GrowingHash)
    with pytest.raises(ValueError, match="grew beyond"):
        trusted.hash_regular_file(path, max_bytes=4)

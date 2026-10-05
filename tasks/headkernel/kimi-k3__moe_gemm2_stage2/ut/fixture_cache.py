"""Protected per-case CPU blob cache; cached bytes never enter native arguments."""
import hashlib
from pathlib import Path


class CpuFixtureCache:
    def __init__(self, root):
        self._root = Path(root).resolve()
        self._blobs = {}

    def read_blob(self, relative, expected_sha256, expected_bytes):
        key = (relative, expected_sha256, expected_bytes)
        if key not in self._blobs:
            path = self._root / relative
            if (path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(self._root)):
                raise ValueError('Fixture must be a regular task-local file')
            data = path.read_bytes()
            if len(data) != expected_bytes or hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError('Fixture byte size or SHA256 differs: ' + relative)
            # bytes is immutable. Tensor construction copies from this cache;
            # neither candidate nor reference receives its backing storage.
            self._blobs[key] = data
        return self._blobs[key]

    def copy_into(self, relative, expected_sha256, expected_bytes, destination, torch):
        data = self.read_blob(relative, expected_sha256, expected_bytes)
        for offset in range(0, len(data), 8 << 20):
            owned = bytearray(data[offset:offset + (8 << 20)])
            destination[offset:offset + len(owned)].copy_(torch.frombuffer(owned, dtype=torch.uint8))

"""Trusted, GPU-free initialization of a complete image-owned AITER JIT tree."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile


COPY_BATCH_SIZE = 128


def copy_batches(source, destination, entries, rclone):
    """Bound local blocking I/O despite the required 64,000-transfer setting."""
    destination.mkdir()
    payloads = []
    for relative, entry in sorted(entries.items()):
        if "\n" in relative or "\r" in relative:
            raise RuntimeError("Cache entry cannot be represented in a raw file list")
        if entry["kind"] == "directory":
            (destination / relative).mkdir(parents=True, exist_ok=True)
        else:
            # With --links, rclone exposes a symlink as this virtual file name.
            payloads.append(relative + (".rclonelink" if entry["kind"] == "symlink" else ""))
    with tempfile.TemporaryDirectory(prefix="aka-aiter-filelists-") as temporary:
        listing = Path(temporary) / "batch.txt"
        for offset in range(0, len(payloads), COPY_BATCH_SIZE):
            batch = payloads[offset:offset + COPY_BATCH_SIZE]
            listing.write_text("\n".join(batch) + "\n")
            subprocess.run(
                [str(rclone), "copy", str(source), str(destination), "--transfers", "64000",
                 "--progress", "--buffer-size", "0", "--links", "--files-from-raw", str(listing),
                 "--no-traverse", "--checkers", "1", "--multi-thread-streams", "0"],
                env={**os.environ, "GOMAXPROCS": "1"}, check=True,
            )


def manifest(root):
    entries = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in sorted(directories + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                entries[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif stat.S_ISDIR(mode):
                entries[relative] = {"kind": "directory"}
            elif stat.S_ISREG(mode):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                entries[relative] = {"kind": "file", "sha256": digest.hexdigest(), "size": path.stat().st_size}
            else:
                raise RuntimeError(f"Unsupported cache entry: {relative}")
    return entries


def assign_owner(root, uid, gid, source=None, destination=None):
    # lchown avoids following image symlinks. Symlink targets are not rewritten.
    paths = [root]
    for directory, directories, files in os.walk(root, followlinks=False):
        paths.extend(Path(directory) / name for name in directories + files)
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        os.chown(path, uid, gid, follow_symlinks=False)
        if not path.is_symlink():
            bits = 0o700 if path.is_dir() else 0o600
            mode = stat.S_IMODE(path.stat().st_mode)
            if source is not None and path.is_relative_to(destination):
                original = source / path.relative_to(destination)
                if original.exists() and not original.is_symlink():
                    mode = stat.S_IMODE(original.stat().st_mode)
            os.chmod(path, mode | bits)


def seed(source, destination, uid, gid, rclone):
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("Image AITER JIT source must be a real directory")
    if destination.exists():
        raise RuntimeError("AITER seed destination must be fresh")
    expected = manifest(source)
    if not expected:
        raise RuntimeError("Refusing an empty image JIT seed")
    try:
        copy_batches(source, destination, expected, rclone)
        actual = manifest(destination)
        if actual != expected:
            raise RuntimeError("Seeded AITER cache differs from complete image source")
        record = {"source": str(source), "entries": expected,
                  "manifest_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()}
        (destination.parent / "SEED-MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"aiter_seed_verified entries={len(expected)} manifest_sha256={record['manifest_sha256']}", flush=True)
    finally:
        # Return partial initialization files to the caller too, so failure
        # cleanup does not need a second privileged process.
        assign_owner(destination.parent, uid, gid, source, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--rclone", type=Path, required=True)
    args = parser.parse_args()
    seed(args.source, args.destination, args.uid, args.gid, args.rclone)

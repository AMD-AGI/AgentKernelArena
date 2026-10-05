"""Copy and attest the complete image AITER JIT tree without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
from pathlib import Path


def tree_manifest(root):
    root = Path(root).resolve()
    entries = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("JIT cache contains an escaping symlink: " + name)
            entries[name] = {"type": "symlink", "target": str(path.readlink())}
        elif stat.S_ISDIR(info.st_mode):
            entries[name] = {"type": "directory"}
        elif stat.S_ISREG(info.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries[name] = {"type": "file", "bytes": info.st_size, "sha256": digest.hexdigest()}
        else:
            raise ValueError("unsupported JIT cache node: " + name)
    return entries


def copy_cache(source, destination, *, rclone="rclone", timeout=1800):
    destination = Path(destination)
    destination.mkdir()
    before = tree_manifest(source)
    copy_env = os.environ.copy()
    copy_env["GOMAXPROCS"] = "1"
    subprocess.run([
        rclone, "copy", str(Path(source).resolve()), str(destination.resolve()),
        "--transfers", "64000", "--progress", "--config", os.devnull,
        "--buffer-size", "0", "--links", "--create-empty-src-dirs",
    ], check=True, timeout=timeout, env=copy_env)
    try:
        copied = tree_manifest(destination)
        unchanged = tree_manifest(source) == before
    except (OSError, RuntimeError) as exc:
        raise ValueError("complete AITER JIT cache parity failed") from exc
    if not unchanged or copied != before:
        raise ValueError("complete AITER JIT cache parity failed")
    return before


def initialize_inside(destination, uid, gid, rclone):
    """Called as root only in a fresh, isolated copy of the pinned image."""
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("pinned image has no AITER package")
    source = Path(next(iter(spec.submodule_search_locations))) / "jit"
    destination = Path(destination)
    cache = destination / "jit"
    try:
        entries = copy_cache(source, cache, rclone=rclone)
        files = [entry for entry in entries.values() if entry["type"] == "file"]
        if not files or not any(name.endswith(".so") for name in entries):
            raise ValueError("complete image JIT cache must include its precompiled native modules")
    finally:
        # Also transfer partial output on failure so the unprivileged host can
        # clean it up. No successful manifest exists until parity is proven.
        if cache.exists():
            for path in [*cache.rglob("*"), cache]:
                info = path.lstat()
                if not path.is_symlink():
                    owner_bits = 0o700 if path.is_dir() else 0o600
                    path.chmod(stat.S_IMODE(info.st_mode) | owner_bits)
                os.chown(path, uid, gid, follow_symlinks=False)
    if tree_manifest(cache) != entries:
        raise ValueError("cache contents changed while transferring ownership")
    manifest = {"schema_version": 1, "complete_parity": True, "source": str(source),
                "owner_uid": uid, "owner_gid": gid, "entries": entries,
                "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files)}
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    os.chown(manifest_path, uid, gid)
    return manifest


def seed_image_cache(image, destination, log_path, *, timeout=1800):
    """Return a complete, hash-verified image cache owned by the host UID."""
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("image-cache seeding requires a digest-pinned image")
    destination = Path(destination).resolve()
    destination.mkdir(mode=0o700)
    rclone = shutil.which("rclone")
    if rclone is None:
        raise RuntimeError("rclone is required for complete image-cache seeding")
    helper = Path(__file__).resolve()
    for path in (destination, helper, Path(rclone)):
        if "," in str(path):
            raise ValueError("Docker bind paths cannot contain commas")
    name = "aka-jit-seed-" + uuid.uuid4().hex
    command = [
        "docker", "run", "--rm", "--pull=never", "--name", name,
        "--network=none", "--read-only", "--user", "0:0", "--cap-drop=ALL",
        "--cap-add=CHOWN", "--cap-add=DAC_OVERRIDE", "--security-opt=no-new-privileges",
        "--mount", f"type=bind,src={destination},dst=/seed",
        "--mount", f"type=bind,src={helper},dst=/seed_helper.py,readonly",
        "--mount", f"type=bind,src={Path(rclone).resolve()},dst=/seed_rclone,readonly",
        "--tmpfs", "/tmp:rw,nosuid,mode=1777", "--env", "HOME=/tmp",
        "--env", "GOMAXPROCS=1",
        "--entrypoint", "python3", image, "-I", "-B", "/seed_helper.py",
        "--initialize-inside", "--destination", "/seed", "--uid", str(os.getuid()),
        "--gid", str(os.getgid()), "--rclone", "/seed_rclone",
    ]
    primary = None
    try:
        with Path(log_path).open("xb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        manifest = json.loads((destination / "manifest.json").read_text())
        if (manifest.get("complete_parity") is not True
                or manifest.get("owner_uid") != os.getuid()
                or manifest.get("owner_gid") != os.getgid()
                or tree_manifest(destination / "jit") != manifest.get("entries")):
            raise ValueError("image-cache parity/ownership evidence is invalid")
        return manifest
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            if primary is None:
                raise
            primary.add_note("Cache-seed container cleanup failed: " + str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-inside", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--log")
    parser.add_argument("--uid", type=int)
    parser.add_argument("--gid", type=int)
    parser.add_argument("--rclone", default="rclone")
    args = parser.parse_args()
    if args.initialize_inside:
        if os.geteuid() != 0 or args.uid is None or args.gid is None:
            parser.error("image initializer requires root and an explicit receiving UID/GID")
        initialize_inside(args.destination, args.uid, args.gid, args.rclone)
    else:
        if not args.image or not args.log:
            parser.error("host seeding requires --image and --log")
        seed_image_cache(args.image, args.destination, args.log)


if __name__ == "__main__":
    main()

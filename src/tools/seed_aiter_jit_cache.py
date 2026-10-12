"""Copy and attest the complete image AITER JIT tree without using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path

COPY_BATCH_ENV = "AKA_AITER_JIT_COPY_BATCH_SIZE"
DEFAULT_COPY_BATCH_SIZE = 64
MAX_COPY_BATCH_SIZE = 512
# Local transfers need a source and destination descriptor. Budget twice that
# amount per offered file plus space for stdio, file lists and checker/runtime
# descriptors. Do not raise the caller's descriptor or other resource limits.
COPY_FDS_PER_FILE = 4
COPY_FD_RESERVE = 64
COPY_THREAD_ENV = "AKA_AITER_JIT_COPY_THREAD_ALLOWANCE"
COPY_THREADS_PER_FILE = 2
COPY_THREAD_RESERVE = 64


def copy_thread_headroom():
    """Observe local ceilings, not an exclusive allocation of shared capacity.

    Unreadable accounting is unknown. A trusted caller must separately reserve
    an allowance for this one copy (including any hidden/shared ancestors).
    """
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        ceilings = []
        uid_threads = None
        if os.getuid() != 0 and soft != resource.RLIM_INFINITY:
            uid_threads = 0
            for path in Path("/proc").glob("[0-9]*/status"):
                try:
                    values = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
                except FileNotFoundError:
                    continue
                if int(values["Uid"].split()[0]) == os.getuid():
                    uid_threads += int(values["Threads"])
            ceilings.append(soft - uid_threads)
        groups = [line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::")]
        if len(groups) != 1:
            return None
        base = Path("/sys/fs/cgroup")
        current = base / groups[0].lstrip("/")
        if not current.is_dir():
            return None
        observed = []
        for group in [current, *current.parents]:
            if group != base and not group.is_relative_to(base):
                break
            limit_path, usage_path = group / "pids.max", group / "pids.current"
            if limit_path.exists():
                limit, usage = limit_path.read_text().strip(), int(usage_path.read_text())
                observed.append({"path": str(group), "limit": limit, "current": usage})
                if limit != "max":
                    ceilings.append(int(limit) - usage)
        if not observed:
            return None
        return {"headroom": min(ceilings) if ceilings else None, "nproc_soft": soft,
                "visible_uid_threads": uid_threads, "cgroups": observed}
    except (OSError, ValueError, KeyError):
        return None


def copy_thread_cap():
    observed = copy_thread_headroom()
    # Existing default remains the ceiling when an allocator has not assigned
    # an exclusive per-copy budget. Shared free counters alone are insufficient.
    allowance = os.environ.get(COPY_THREAD_ENV)
    if observed is None:
        return DEFAULT_COPY_BATCH_SIZE
    available = observed["headroom"]
    local_cap = MAX_COPY_BATCH_SIZE if available is None else (available - COPY_THREAD_RESERVE) // COPY_THREADS_PER_FILE
    if local_cap < 1:
        raise ValueError("PID/thread headroom leaves no safe capacity for a cache copy")
    if allowance is None:
        return min(DEFAULT_COPY_BATCH_SIZE, local_cap)
    if not allowance.isdecimal() or int(allowance) <= 0:
        raise ValueError("copy thread allowance must be a positive reserved-slot count")
    slots = int(allowance)
    if available is not None:
        slots = min(slots, available)
    cap = (slots - COPY_THREAD_RESERVE) // COPY_THREADS_PER_FILE
    if cap < 1:
        raise ValueError("Reserved PID/thread allowance leaves no safe copy capacity")
    return cap


def requested_copy_batch_size(value=None):
    if value is None:
        value = os.environ.get(COPY_BATCH_ENV, str(DEFAULT_COPY_BATCH_SIZE))
    if type(value) is str and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 1 <= value <= MAX_COPY_BATCH_SIZE:
        raise ValueError(f"copy batch size must be an integer between 1 and {MAX_COPY_BATCH_SIZE}")
    return value


def copy_batch_size(value=None):
    requested = requested_copy_batch_size(value)
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    cap = MAX_COPY_BATCH_SIZE if soft == resource.RLIM_INFINITY else (soft - COPY_FD_RESERVE) // COPY_FDS_PER_FILE
    if cap < 1:
        raise ValueError("RLIMIT_NOFILE leaves no safe descriptors for a cache copy batch")
    return min(requested, cap, copy_thread_cap()), soft


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


def copy_cache(source, destination, *, rclone="rclone", timeout=1800, batch_size=None):
    effective_batch_size, nofile = copy_batch_size(batch_size)
    if batch_size is not None or COPY_BATCH_ENV in os.environ:
        print(f"aiter_copy_batch requested={requested_copy_batch_size(batch_size)} "
              f"effective={effective_batch_size} rlimit_nofile={nofile}", flush=True)
    destination = Path(destination)
    destination.mkdir()
    before = tree_manifest(source)
    executable_bits = {name: (Path(source) / name).stat().st_mode & 0o111
                       for name, entry in before.items() if entry["type"] == "file"}
    copy_env = os.environ.copy()
    copy_env["GOMAXPROCS"] = "1"
    files = []
    for name, entry in before.items():
        if "\n" in name or "\r" in name:
            raise ValueError("JIT cache names cannot contain line breaks")
        if entry["type"] == "directory":
            (destination / name).mkdir(parents=True, exist_ok=True)
        else:
            # --links exposes symlinks as these virtual transfer filenames.
            files.append(name + ".rclonelink" if entry["type"] == "symlink" else name)
    if len(set(files)) != len(files):
        raise ValueError("JIT cache has conflicting rclone virtual filenames")
    # Keep the mandated transfer setting while bounding the available jobs.
    # GOMAXPROCS alone does not prevent blocking file I/O from growing Go's
    # OS-thread pool when thousands of payload files are offered at once.
    for start in range(0, len(files), effective_batch_size):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="aka-jit-batch-") as batch:
            batch.write("\n".join(files[start:start + effective_batch_size]) + "\n")
            batch.flush()
            subprocess.run([
                rclone, "copy", str(Path(source).resolve()), str(destination.resolve()),
                "--transfers", "64000", "--progress", "--config", os.devnull,
                "--buffer-size", "0", "--multi-thread-streams", "0", "--links", "--no-traverse",
                "--files-from-raw", batch.name,
            ], check=True, timeout=timeout, env=copy_env)
    try:
        copied = tree_manifest(destination)
        unchanged = tree_manifest(source) == before
    except (OSError, RuntimeError) as exc:
        raise ValueError("complete AITER JIT cache parity failed") from exc
    if not unchanged or copied != before:
        raise ValueError("complete AITER JIT cache parity failed")
    # rclone's local copy creates regular files without executable bits. Task
    # packages also use this helper, so preserve those trusted source bits
    # after the complete byte/path/link parity check, without changing its schema.
    for name, expected in executable_bits.items():
        if (Path(source) / name).stat().st_mode & 0o111 != expected:
            raise ValueError("source executable mode changed during copy")
        target = destination / name
        mode = stat.S_IMODE(target.stat().st_mode)
        if mode & 0o111 != expected:
            target.chmod((mode & ~0o111) | expected)
        if target.stat().st_mode & 0o111 != expected:
            raise ValueError("copied executable mode differs from source")
    return before


def initialize_inside(destination, uid, gid, rclone, *, batch_size=None):
    """Called as root only in a fresh, isolated copy of the pinned image."""
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("pinned image has no AITER package")
    source = Path(next(iter(spec.submodule_search_locations))) / "jit"
    destination = Path(destination)
    cache = destination / "jit"
    try:
        entries = copy_cache(source, cache, rclone=rclone, batch_size=batch_size)
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


def seed_image_cache(image, destination, log_path, *, timeout=1800, batch_size=None):
    """Return a complete, hash-verified image cache owned by the host UID."""
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("image-cache seeding requires a digest-pinned image")
    requested_batch_size = requested_copy_batch_size(batch_size)
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
    if batch_size is not None or COPY_BATCH_ENV in os.environ:
        # Pass the request, not a host-capped value: the initializer must apply
        # its own inherited RLIMIT_NOFILE inside the fresh seed container.
        command.extend(["--batch-size", str(requested_batch_size)])
    if COPY_THREAD_ENV in os.environ:
        # This is a trusted allocator's exclusive allowance, not a free-space
        # estimate. The child observes its own local NPROC/cgroup ceilings too.
        command[command.index(image):command.index(image)] = ["--env", COPY_THREAD_ENV + "=" + os.environ[COPY_THREAD_ENV]]
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
    parser.add_argument("--destination")
    parser.add_argument("--log")
    parser.add_argument("--uid", type=int)
    parser.add_argument("--gid", type=int)
    parser.add_argument("--rclone", default="rclone")
    parser.add_argument("--batch-size", type=int, help="opt-in local copy batch (1-512), capped by RLIMIT_NOFILE")
    parser.add_argument("--show-copy-budget", action="store_true", help="CPU-only current descriptor/PID budget observation")
    args = parser.parse_args()
    if args.show_copy_budget:
        effective, nofile = copy_batch_size(args.batch_size)
        print(json.dumps({"rlimit_nofile": nofile, "pid_thread_observation": copy_thread_headroom(),
                          "caller_reserved_thread_allowance": os.environ.get(COPY_THREAD_ENV),
                          "effective_batch_size": effective, "shared_capacity_reservation_proven_by_tool": False}))
        return
    if not args.destination:
        parser.error("--destination is required for cache copying")
    if args.initialize_inside:
        if os.geteuid() != 0 or args.uid is None or args.gid is None:
            parser.error("image initializer requires root and an explicit receiving UID/GID")
        initialize_inside(args.destination, args.uid, args.gid, args.rclone, batch_size=args.batch_size)
    else:
        if not args.image or not args.log:
            parser.error("host seeding requires --image and --log")
        seed_image_cache(args.image, args.destination, args.log, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

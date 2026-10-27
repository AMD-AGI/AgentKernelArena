"""Install a committed Forge engine into an immutable, runtime-specific cache."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def installation_path(cache: Path, revision: str, image_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Forge requires a full committed Git revision")
    if not image_id.strip():
        raise ValueError("Forge requires the selected container image identity")
    image_key = hashlib.sha256(image_id.encode()).hexdigest()
    return cache / sys.implementation.cache_tag / image_key / revision


def install(cache: Path, source: Path, revision: str, image_id: str) -> Path:
    target = installation_path(cache, revision, image_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    # All callers of this revision serialize; other revisions have independent
    # directories, so setup never replaces code a running campaign imports.
    with (target.parent / (revision + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / ".aka-hyperloom-revision"
        if target.exists():
            if marker.is_file() and marker.read_text().strip() == revision:
                return target
            raise RuntimeError(f"Incomplete Forge installation requires review: {target}")
        stage = Path(tempfile.mkdtemp(prefix=".forge-install-", dir=target.parent))
        try:
            archive = stage / "source.tar"
            exported = stage / "source"
            packages = stage / "packages"
            exported.mkdir()
            # Export exactly the host-resolved revision, even if HEAD moves.
            with archive.open("wb") as output:
                subprocess.run(["git", "-C", str(source), "archive", revision],
                               stdout=output, check=True)
            subprocess.run(["tar", "-xf", str(archive), "-C", str(exported)], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(packages),
                            f"{exported}[forge]"], check=True)
            (packages / ".aka-hyperloom-revision").write_text(revision + "\n")
            (packages / ".aka-forge-runtime.json").write_text(json.dumps({
                "revision": revision, "image_id": image_id,
                "python_cache_tag": sys.implementation.cache_tag,
            }, sort_keys=True) + "\n")
            # Nothing under the published target changes after this rename.
            os.replace(packages, target)
        finally:
            shutil.rmtree(stage)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("path", "install"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.action == "install":
        if args.source is None:
            parser.error("install requires --source")
        path = install(args.cache, args.source, args.revision, args.image_id)
    else:
        path = installation_path(args.cache, args.revision, args.image_id)
    print(path)


if __name__ == "__main__":
    main()

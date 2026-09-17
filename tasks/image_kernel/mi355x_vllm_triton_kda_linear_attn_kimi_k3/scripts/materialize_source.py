"""Stage the package from the framework-acquired, pinned vLLM repository.

This performs no downloads. A completed staging receipt permits harmless setup
repetition while preserving candidate edits; unknown collisions fail closed.
"""
import hashlib
import json
from pathlib import Path
import shutil

import yaml


RECEIPT = ".arena-source-staging.json"


def source_identity(root, checkout, source):
    config = yaml.safe_load((root / "config.yaml").read_text())
    declarations = [item for item in config["workspace"]["sources"]
                    if item["destination"] == "upstream/vllm"]
    if len(declarations) != 1 or declarations[0]["kind"] != "git":
        raise ValueError("Expected one declared Git source for the vLLM package")
    entries = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(checkout):
                raise ValueError("vLLM package contains an external source symlink")
            # File links are hashed by their copied content. Reject a directory
            # link rather than omit its contents from the identity.
            if path.is_dir():
                raise ValueError("vLLM package contains an unsupported directory symlink")
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.name.endswith(".pyc"):
            continue
        if path.is_file():
            entries.append([relative.as_posix(), "file", hashlib.sha256(path.read_bytes()).hexdigest()])
        elif path.is_dir():
            entries.append([relative.as_posix(), "directory"])
    return {"version": 1, "source": declarations[0], "package": "vllm",
            "tree_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest()}


def materialize(root):
    root = Path(root).resolve(strict=True)
    checkout = (root / "upstream/vllm").resolve(strict=True)
    source = (checkout / "vllm").resolve(strict=True)
    if not checkout.is_relative_to(root) or not source.is_relative_to(checkout) or not (source / "__init__.py").is_file():
        raise ValueError("Pinned vLLM package is absent or outside the task")
    if (source / RECEIPT).exists() or (source / RECEIPT).is_symlink():
        raise ValueError("Upstream package collides with the reserved staging receipt")
    identity = source_identity(root, checkout, source)
    destination = root / "vllm"
    if destination.exists() or destination.is_symlink():
        receipt = destination / RECEIPT
        if destination.is_symlink() or not destination.is_dir() or receipt.is_symlink() or not receipt.is_file():
            raise FileExistsError("Unrecognized existing candidate package; refusing to replace it")
        try:
            recorded = json.loads(receipt.read_text())
        except (OSError, ValueError) as error:
            raise FileExistsError("Existing candidate has an invalid staging receipt") from error
        if recorded != identity:
            raise FileExistsError("Existing candidate staging source identity does not match")
        for path in destination.rglob("*"):
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("Existing candidate package contains an external symlink")
        return  # Never compare editable bytes to upstream or overwrite edits.
    shutil.copytree(source, destination, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Only a successfully completed copy gets a receipt. Partial copies remain
    # explicit collisions on retry, not accepted candidates or silently reset.
    with (destination / RECEIPT).open("x") as handle:
        json.dump(identity, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    materialize(Path(__file__).resolve().parents[1])

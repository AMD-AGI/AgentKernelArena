"""Stage the package from the framework-acquired, pinned SGLang repository.

This performs no downloads and never replaces an existing candidate on resume.
The framework runs setup once, before freezing the initial candidate baseline.
"""
from pathlib import Path
import shutil


def materialize(root):
    root = Path(root).resolve(strict=True)
    checkout = (root / "upstream/sglang").resolve(strict=True)
    source = (checkout / "python/sglang").resolve(strict=True)
    if not checkout.is_relative_to(root) or not source.is_relative_to(checkout) or not (source / "__init__.py").is_file():
        raise ValueError("Pinned SGLang package is absent or outside the task")
    destination = root / "sglang"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to replace an existing candidate package")
    for path in source.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(checkout):
            raise ValueError("SGLang package contains an external source symlink")
    shutil.copytree(source, destination, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


if __name__ == "__main__":
    materialize(Path(__file__).resolve().parents[1])

"""Normalize the pinned image's uv Python and ROCm SDK layout at build time."""
from pathlib import Path
import shutil
import sys
import sysconfig


def prepare_runtime(venv: Path, base: Path, sdk: Path, prefix: Path) -> None:
    """Relocate only the base interpreter; keep installed packages in the venv."""
    base = base.resolve()
    destination = prefix / "aka-python-runtime"
    aliases = {prefix / "venv": venv, prefix / "rocm": sdk}
    for path in (destination, *aliases):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to replace existing runtime path: {path}")
    if not sdk.is_dir() or not (sdk / "bin/hipcc").is_file():
        raise ValueError(f"Missing ROCm SDK: {sdk}")

    config_path = venv / "pyvenv.cfg"
    config = config_path.read_text().splitlines()
    if sum(line.partition("=")[0].strip() == "home" for line in config) != 1:
        raise ValueError("Expected exactly one Python home in pyvenv.cfg")

    # Resolve links before changing any of them (python3 may point to python).
    links = {
        link: link.resolve().relative_to(base)
        for link in (venv / "bin").glob("python*")
        if link.is_symlink()
    }
    if venv / "bin/python" not in links:
        raise ValueError("Expected the pinned image's Python interpreter symlink")

    shutil.copytree(base, destination, symlinks=True)
    for link, relative_target in links.items():
        link.unlink()
        link.symlink_to(destination / relative_target)
    config_path.write_text("\n".join(
        f"home = {destination / 'bin'}"
        if line.partition("=")[0].strip() == "home" else line
        for line in config
    ) + "\n")
    for alias, target in aliases.items():
        alias.symlink_to(target, target_is_directory=True)


if __name__ == "__main__":
    prepare_runtime(
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sysconfig.get_path("purelib")) / "_rocm_sdk_devel",
        Path("/opt"),
    )

"""Locate shipped generated-input helpers without maintaining test copies."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "minimax": "minimax-m3-mxfp4",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.8-2.4t-a95b-mxfp4",
    "glm": "glm-5.3-flash",
    "kimi": "kimi-k3",
}


def generated_task(family):
    candidates = sorted(
        path.parent.parent
        for path in (ROOT / "tasks/head_kernels" / MODELS[family]).rglob("ut/generated_contract.py")
    )
    if not candidates:
        raise FileNotFoundError(f"no shipped generated-input task for {family}")
    return candidates[0]


def generated_helper(family, filename):
    if filename == "extract_contract.py":
        return ROOT / "src/tools/head_kernel_archives" / f"{family}.py"
    directory = "ut" if filename in {"generated_contract.py", "cases.py"} else "scripts"
    path = generated_task(family) / directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

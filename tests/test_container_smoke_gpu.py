"""Exercise the runtime smoke's numerical gate without a GPU dependency."""
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def smoke_program():
    source = (ROOT / "src/scripts/docker_benchmark.sh").read_text()
    block = source.split("container_smoke() {\n    python - <<'PY'\n", 1)[1]
    return block.split("\nPY\n}", 1)[0]


@pytest.mark.parametrize("behavior", ["correct", "wrong", "kernel_error"])
def test_smoke_requires_successful_gpu_arithmetic(monkeypatch, capsys, behavior):
    events = []

    class Tensor:
        def __init__(self, values, device="cpu"):
            self.values, self.device = values, device

        def to(self, device):
            events.append(("copy", device))
            return Tensor(self.values[:], device)

        def __add__(self, scalar):
            if self.device != "cpu":
                events.append(("kernel", self.device))
                if behavior == "kernel_error":
                    raise RuntimeError("simulated GPU execution failure")
            increment = 0 if behavior == "wrong" and self.device != "cpu" else scalar
            return Tensor([value + increment for value in self.values], self.device)

        def cpu(self):
            return self.to("cpu")

    torch = types.ModuleType("torch")
    torch.__version__ = "test"
    torch.float32 = object()
    torch.arange = lambda count, **kwargs: Tensor(list(range(count)))
    torch.equal = lambda left, right: left.values == right.values
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "test GPU",
        get_device_properties=lambda index: types.SimpleNamespace(gcnArchName="gfx950"),
        synchronize=lambda: events.append(("synchronize",)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("triton", "pytest", "yaml", "numpy", "flydsl"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr("shutil.which", lambda name: "/runtime/bin/" + name)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_ARCH", "gfx950")
    monkeypatch.delenv("AKA_REQUIRED_PROFILERS", raising=False)

    if behavior == "correct":
        exec(compile(smoke_program(), "runtime_smoke", "exec"), {})
        assert "torch_gpu_numerical_smoke=PASS" in capsys.readouterr().out
        assert events == [("copy", "cuda:0"), ("kernel", "cuda:0"),
                          ("synchronize",), ("copy", "cpu")]
    else:
        error = SystemExit if behavior == "wrong" else RuntimeError
        with pytest.raises(error):
            exec(compile(smoke_program(), "runtime_smoke", "exec"), {})
        assert "torch_gpu_numerical_smoke=PASS" not in capsys.readouterr().out
        assert ("kernel", "cuda:0") in events

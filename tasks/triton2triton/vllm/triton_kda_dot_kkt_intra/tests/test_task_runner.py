from types import SimpleNamespace

import torch

from scripts import task_runner


def test_correctness_rejects_empty_result_tuple(monkeypatch):
    mod = SimpleNamespace(kda_dot_kkt_intra=lambda *args, **kwargs: ())
    refs = (torch.zeros(1), torch.zeros(1))

    monkeypatch.setattr(task_runner, "SEEDS", [42])
    monkeypatch.setattr(task_runner, "load_module", lambda: mod)
    monkeypatch.setattr(task_runner, "gen_inputs", lambda seed, device: ((0,), {}))
    monkeypatch.setattr(task_runner, "reference", lambda *args, **kwargs: refs)

    ok, error = task_runner.run_correctness()

    assert not ok
    assert error == "Shape 1: expected 2 outputs, got 0"

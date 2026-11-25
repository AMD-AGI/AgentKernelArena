"""Architecture eligibility and runtime identity; CPU fixtures, not GPU evidence."""
import argparse
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.quality_loop.orchestrator import QualityLoop
from main import _build_context, should_run_task_for_platform
from src import gpu_identity
from src.gpu_identity import (
    GpuAgent,
    identity_error,
    load_model_identity,
    parse_rocminfo,
    smoke_identity_lines,
    write_identity_record,
)
from src.task_runtime import bind_session_runtime
from src.task_spec import TaskConfigError, TaskSpec

# The GPU agents of a rocminfo capture from a host with four Radeon PRO W7900 Dual Slot
# boards (amdgpu 6.19.0), cut after "Compute Unit"; GPU UUIDs are placeholders.
NAVI31_ROCMINFO = Path(__file__).parent / "fixtures/rocminfo_navi31_4x_w7900_dual_slot.txt"
W7900 = load_model_identity("W7900")


def task(required):
    return TaskSpec.from_mapping({
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["kernel.hip"]},
        "evaluation": {"runner": ["python3", "evaluate.py"]},
        "platform_support": {"required_arch": required},
    }, task_id="example/operator")


@pytest.mark.parametrize("required", ["gfx950", ["gfx950"], ["gfx942", "gfx950"]])
@pytest.mark.parametrize("actual", ["gfx942", "gfx950", "gfx1100", None])
def test_scheduling_quality_loop_and_runtime_agree(required, actual, tmp_path, monkeypatch):
    spec = task(required)
    allowed = required if isinstance(required, list) else [required]
    expected = actual in allowed
    assert spec.to_mapping()["platform_support"]["required_arch"] == required
    assert should_run_task_for_platform(spec.task_id, spec.to_mapping(), actual,
                                        logging.getLogger(__name__)) is expected
    assert QualityLoop._platform_matches(spec.to_mapping(), actual) is expected
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": actual})
    session = SimpleNamespace(spec=spec, state_directory=tmp_path)
    if expected:
        assert bind_session_runtime(session)["gpu_arch"] == actual
    else:
        with pytest.raises(RuntimeError, match="Task requires"):
            bind_session_runtime(session)
        assert not (tmp_path / "runtime_identity.json").exists()


@pytest.mark.parametrize("invalid", [None, [], ["gfx950", "gfx950"], ["gfx950", 950],
                                     [""], ["gfx950", ["gfx942"]], True, "gfx9*", "gfx942 gfx950"])
def test_invalid_architecture_declarations_fail_before_execution(invalid):
    with pytest.raises(TaskConfigError, match="required_arch"):
        task(invalid)


def test_allowed_architectures_do_not_allow_reusing_baseline_on_other_gpu(tmp_path, monkeypatch):
    session = SimpleNamespace(spec=task(["gfx942", "gfx950"]), state_directory=tmp_path)
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx942"})
    bind_session_runtime(session)
    recorded = (tmp_path / "runtime_identity.json").read_bytes()
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx950"})
    with pytest.raises(RuntimeError, match="Scoring runtime changed"):
        bind_session_runtime(session)
    assert (tmp_path / "runtime_identity.json").read_bytes() == recorded
    assert json.loads(recorded)["gpu_arch"] == "gfx942"


def test_explicit_skip_overrides_architecture_match():
    config = task(["gfx942", "gfx950"]).to_mapping()
    config["platform_support"].update(status="skip", skip_reason="Pending task maintenance")
    assert not should_run_task_for_platform("example/operator", config, "gfx950", logging.getLogger(__name__))
    assert not QualityLoop._platform_matches(config, "gfx950")


def navi31_agents(name=None, gfx=None):
    """The captured agents, optionally with every GPU's name or ISA edited."""
    text = NAVI31_ROCMINFO.read_text()
    if name is not None:
        text = re.sub(r"(?m)^(  Marketing Name:\s+)AMD Radeon Pro W7900 Dual Slot$",
                      lambda m: m.group(1) + name, text)
    if gfx is not None:
        text = re.sub(r"(?m)^(  Name:\s+)gfx1100$", lambda m: m.group(1) + gfx, text)
    return parse_rocminfo(text)


def agent(gfx, name, pci="1002:744a"):
    return GpuAgent(gfx_arch=gfx, marketing_name=name, pci_device_id=pci, node=1)


def test_navi31_capture_yields_the_four_gpu_agents():
    agents = navi31_agents()
    assert [a.node for a in agents] == [1, 2, 3, 4]
    assert {a.gfx_arch for a in agents} == {"gfx1100"}
    assert {a.marketing_name for a in agents} == {"AMD Radeon Pro W7900 Dual Slot"}
    assert {a.pci_device_id for a in agents} == {"1002:744a"}
    assert all(a.uuid.startswith("GPU-") for a in agents)


def test_cpu_agent_and_isa_entries_do_not_become_gpu_agents():
    text = """\
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Node:                    0
  Device Type:             CPU
  Chip ID:                 0(0x0)
  ISA Info:
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon Pro W7900 Dual Slot
  Node:                    1
  Device Type:             GPU
  Chip ID:                 29770(0x744a)
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1100
    ISA 2
      Name:                    amdgcn-amd-amdhsa--gfx11-generic
"""
    assert parse_rocminfo(text) == [agent("gfx1100", "AMD Radeon Pro W7900 Dual Slot")]


def test_w7900_entry_lists_both_marketing_names():
    assert W7900.gfx_arch == "gfx1100"
    assert W7900.marketing_names == ("AMD Radeon Pro W7900", "AMD Radeon Pro W7900 Dual Slot")
    assert load_model_identity("w7900").marketing_names == W7900.marketing_names


@pytest.mark.parametrize("name", [
    "AMD Radeon Pro W7900", "AMD Radeon PRO W7900",
    "AMD Radeon Pro W7900 Dual Slot", "AMD Radeon PRO W7900 Dual Slot", "amd radeon pro w7900 dual slot",
])
def test_w7900_names_are_accepted_case_insensitively(name):
    assert identity_error(W7900, navi31_agents(name=name)) is None


def test_accepted_identity_records_observed_name_and_pci(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    path = write_identity_record(tmp_path, "worker3", W7900, navi31_agents()[:1])
    record = json.loads(path.read_text())
    assert path == tmp_path / "gpu_identity" / "worker3.json"
    assert record["target_gpu_model"] == "W7900"
    assert record["rocr_visible_devices"] == "2"
    assert record["agents"][0]["marketing_name"] == "AMD Radeon Pro W7900 Dual Slot"
    assert record["agents"][0]["pci_device_id"] == "1002:744a"


@pytest.mark.parametrize("name", [
    "AMD Radeon Pro W7800", "AMD Radeon PRO W7800 48GB", "AMD Radeon RX 7900 XTX",
    "AMD Radeon RX 7900 XT", "AMD Radeon RX 7900 GRE", "", "AMD Radeon Pro W7900X",
    "AMD Radeon Pro W7900 Dual", "Radeon Pro W7900",
])
def test_other_gfx1100_boards_and_empty_names_are_refused(name):
    error = identity_error(W7900, navi31_agents(name=name))
    assert error is not None
    assert f"'{name or '<empty>'}' gfx1100 PCI 1002:744a" in error


def test_every_visible_gpu_must_be_a_w7900():
    error = identity_error(W7900, [agent("gfx1100", "AMD Radeon Pro W7900"),
                                   agent("gfx1100", "AMD Radeon RX 7900 XTX", "1002:744c")])
    assert "'AMD Radeon RX 7900 XTX' gfx1100 PCI 1002:744c" in error
    assert "'AMD Radeon Pro W7900' gfx1100" not in error


@pytest.mark.parametrize("gfx", ["gfx1101", "gfx1102", "gfx1201"])
def test_w7900_name_on_another_isa_is_refused(gfx):
    error = identity_error(W7900, navi31_agents(gfx=gfx))
    assert f"'AMD Radeon Pro W7900 Dual Slot' {gfx}" in error


@pytest.mark.parametrize("model", ["RDNA4", "MI300", "MI355X", "UNKNOWN_GPU"])
def test_gfx11_board_under_a_model_without_product_identity_is_refused(model):
    error = identity_error(load_model_identity(model), navi31_agents())
    assert "does not identify a qualified gfx11 board" in error
    assert "Use target_gpu_model W7900" in error
    error = identity_error(load_model_identity(model), navi31_agents(gfx="gfx1101"))
    assert "No AKA target_gpu_model qualifies this board" in error


def test_missing_rocminfo_or_no_gpu_refuses_only_identity_gated_models():
    assert "rocminfo was not found on PATH or failed" in identity_error(W7900, None)
    assert "reported no GPU agents" in identity_error(W7900, [])
    for model in ("MI300", "MI355X", "RDNA4"):
        assert identity_error(load_model_identity(model), None) is None
        assert identity_error(load_model_identity(model), []) is None


def test_evaluation_tools_are_refused_on_w7900_only():
    error = identity_error(W7900, navi31_agents(), ("gpu_asan", "rocjitsu"))
    assert "verified only for gfx950" in error
    assert "gpu_asan, rocjitsu" in error
    mi355x = [agent("gfx950", "AMD Instinct MI355X", "1002:75a3")]
    assert identity_error(load_model_identity("MI355X"), mi355x, ("gpu_asan",)) is None


@pytest.mark.parametrize("model,gfx,name", [
    ("MI300", "gfx942", "AMD Instinct MI300X"),
    ("MI325X", "gfx942", "AMD Instinct MI325X"),
    ("MI355X", "gfx950", "AMD Instinct MI355X"),
    ("RDNA4", "gfx1201", "AMD Radeon AI PRO R9700"),
    ("RDNA4", "gfx1201", ""),
])
def test_existing_models_on_their_hardware_are_unchanged(model, gfx, name):
    assert identity_error(load_model_identity(model), [agent(gfx, name)]) is None


def build_context(config, agents, monkeypatch, *, role=None, resolve=None):
    monkeypatch.delenv("AKA_EVAL_TOOLS_SELECTED", raising=False)
    args = argparse.Namespace(run_suffix=None, config_name="unused.yaml", worker_id=None)
    output = io.StringIO()
    with patch("main._load_config", return_value=config), \
            patch("main.probe_gpu_agents", return_value=agents), \
            patch("main._resolve_run", side_effect=resolve or AssertionError("run directory resolved")), \
            patch("main._configure_logging", return_value=logging.getLogger("aka-identity-test")), \
            patch("main.setup_rocm_env"), \
            patch("main.apply_subprocess_python_path", return_value=""), \
            patch("main._discover_tasks", return_value={}), \
            redirect_stdout(output):
        context = _build_context(args, need_agent_launcher=False, role=role)
    return context, output.getvalue()


W7900_CONFIG = {"tasks": [], "agent": {"template": "claude_code"}, "log_directory": "logs",
                "workspace_directory_prefix": "workspace", "target_gpu_model": "W7900"}


@pytest.mark.parametrize("config,agents,expected", [
    (W7900_CONFIG, navi31_agents(name="AMD Radeon Pro W7800"), "'AMD Radeon Pro W7800' gfx1100 PCI 1002:744a"),
    (W7900_CONFIG, None, "rocminfo was not found"),
    ({**W7900_CONFIG, "evaluation_tools": {"enabled": ["gpu_asan"]}}, navi31_agents(), "verified only for gfx950"),
    ({**W7900_CONFIG, "target_gpu_model": "RDNA4"}, navi31_agents(), "does not identify a qualified gfx11 board"),
])
def test_build_context_refuses_before_any_run_directory(config, agents, expected, monkeypatch):
    context, output = build_context(config, agents, monkeypatch)
    assert context is None
    assert output.startswith("Error: ")
    assert expected in output


def test_build_context_records_identity_per_role(tmp_path, monkeypatch):
    run_directory = tmp_path / "run_20261112_000000"
    run_directory.mkdir()
    resolved = lambda args, workspace: (run_directory, run_directory.name, "20261112_000000", False)
    context, output = build_context(W7900_CONFIG, navi31_agents()[2:3], monkeypatch,
                                    role="worker1", resolve=resolved)
    assert context is not None and output == ""
    record = json.loads((run_directory / "gpu_identity" / "worker1.json").read_text())
    assert record["role"] == "worker1"
    assert record["agents"][0]["node"] == 3
    assert record["agents"][0]["marketing_name"] == "AMD Radeon Pro W7900 Dual Slot"


def test_build_context_writes_no_identity_record_for_existing_models(tmp_path, monkeypatch):
    run_directory = tmp_path / "run_20261112_000000"
    run_directory.mkdir()
    resolved = lambda args, workspace: (run_directory, run_directory.name, "20261112_000000", False)
    context, _ = build_context({**W7900_CONFIG, "target_gpu_model": "MI355X"},
                               [agent("gfx950", "AMD Instinct MI355X")], monkeypatch, resolve=resolved)
    assert context is not None
    assert not (run_directory / "gpu_identity").exists()


def test_runtime_smoke_reports_and_checks_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_identity, "probe_gpu_agents", lambda: navi31_agents()[:1])
    lines = smoke_identity_lines("gfx1100")
    assert "gpu_marketing_name=AMD Radeon Pro W7900 Dual Slot" in lines
    assert "gpu_pci_device_id=1002:744a" in lines
    assert "gpu_identity=PASS target_gpu_model=W7900" in lines

    monkeypatch.setattr(gpu_identity, "probe_gpu_agents",
                        lambda: navi31_agents(name="AMD Radeon RX 7900 XTX")[:1])
    with pytest.raises(SystemExit, match="'AMD Radeon RX 7900 XTX' gfx1100"):
        smoke_identity_lines("gfx1100")

    config = tmp_path / "w7900-eval.yaml"
    config.write_text("target_gpu_model: W7900\nevaluation_tools:\n  enabled: [gpu_asan]\n")
    monkeypatch.setattr(gpu_identity, "probe_gpu_agents", lambda: navi31_agents()[:1])
    with pytest.raises(SystemExit, match="verified only for gfx950"):
        smoke_identity_lines("gfx1100", str(config))

    monkeypatch.setattr(gpu_identity, "probe_gpu_agents", lambda: None)
    assert smoke_identity_lines("gfx950")[0].startswith("gpu_marketing_name=unavailable")
    with pytest.raises(SystemExit, match="rocminfo was not found"):
        smoke_identity_lines("gfx1100")

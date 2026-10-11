"""Official Spur command and explicit recorded reservation evidence; CPU only."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import case_gpu_plan as admission


FIXTURE = Path(__file__).with_name("fixtures") / "spur_running_207815.json"


@pytest.fixture
def recorded(monkeypatch):
    records = json.loads(FIXTURE.read_text())["records"]
    for record in records.values():
        assert hashlib.sha256(record["raw_json"].encode()).hexdigest() == record["sha256"]
    raw = json.loads(records["scheduler"]["raw_json"])
    state = {**admission.spur_fields(raw["stdout"]), "_provider": "spur", "_receipt": raw}
    ready = json.loads(records["ready"]["raw_json"])
    bindings = json.loads(records["binding"]["raw_json"])["binding"]
    devices = json.loads(records["gpu_admission"]["raw_json"])["devices"]
    evidence = {"schema": "spur-recorded-reservation-v1"}
    for name in ("ready", "assignment", "binding", "renders"):
        evidence[name + "_json"] = records[name]["raw_json"]
        evidence[name + "_sha256"] = records[name]["sha256"]
    plan = {"scheduler_provider": "spur", "job_id": "207815", "node": ready["node"],
            "restart_generation": 1, "work_stop_unix": ready["work_stop_unix"],
            "allocation_receipt_sha256": records["ready"]["sha256"], "spur_allocation": evidence,
            "allocation_capacity": {"cpus": 192, "memory_gib": 1536, "gpus": 8},
            "bindings": [{"worker_id": i, "render_device": row["render_device"],
                "drm_minor": int(row["render_device"].removeprefix("/dev/dri/renderD")),
                "pci_bus_id": row["bdf"], "rocr_uuid": row["ROCR_UUID"],
                "kfd_node": str(next(d["node_id"] for d in devices if d["bdf"] == row["bdf"]))}
                for i, row in enumerate(bindings)]}
    monkeypatch.setattr(admission.time, "time", lambda: ready["observed_unix"] + 10)
    return plan, state


def test_actual_spur_receipt_retains_requested_fields_and_has_no_alloctres(recorded):
    plan, state = recorded
    before = copy.deepcopy(state)
    assert state["CPUs/Task"] == "192" and "AllocTRES" not in state
    result = admission.validate_scheduler(plan, state, plan["node"])
    assert result["recorded_profile"]["memory_MiB"] == 1572864
    assert result["capacity_basis"] == "root_recorded_single_node_single_task_reservation"
    assert result["synthetic_AllocTRES_created"] is False
    assert result["individual_registry_to_ordinal_mapping_assumed"] is False
    assert state == before


def test_spur_request_fields_never_become_a_native_slurm_capacity_fallback(recorded):
    plan, state = recorded
    del plan["scheduler_provider"]
    with pytest.raises(ValueError, match="allocated CPU/memory/GPU"):
        admission.validate_scheduler(plan, state, plan["node"])


def test_official_spur_binary_and_known_read_command_only(recorded, monkeypatch):
    plan, state = recorded
    calls = []
    def run(argv, **options):
        calls.append((argv, options))
        return SimpleNamespace(returncode=0, stdout=state["_receipt"]["stdout"], stderr="")
    monkeypatch.setattr(admission.subprocess, "run", run)
    observed = admission.plan_scheduler_state(plan)
    assert calls[0][0] == ["/usr/local/bin/spur", "show", "job", "207815"]
    assert calls[0][1]["env"]["TOKIO_WORKER_THREADS"] == "1"
    assert observed["CPUs/Task"] == "192" and observed["_receipt"]["returncode"] == 0
    assert "/home/sapmajum/bin/spur" not in str(calls)
    calls.clear()
    admission.plan_scheduler_state({"job_id": "12"})
    assert calls[0][0] == ["scontrol", "show", "job", "-o", "12"]


def test_command_body_cannot_rewrite_controller_fields():
    assert admission.spur_fields("JobId=12\n Command=echo JobId=99\n SubmitLine=sbatch JobId=99\n") == {"JobId": "12"}
    with pytest.raises(ValueError, match="duplicate"):
        admission.spur_fields("JobId=12 JobId=99")


@pytest.mark.parametrize("attack", ["missing_evidence", "missing_capacity", "wrong_ready_hash", "wrong_assignment",
    "duplicate_gpu", "foreign_gpu", "fake_minor", "missing_render", "cpu_request", "memory_request", "gpu_request",
    "oversized_cpu", "oversized_memory", "partial_gpu_pool", "multi_task", "multi_node", "restarted", "wrong_start",
    "cancelled", "expired_lease", "wrong_provider"])
def test_incomplete_or_foreign_spur_evidence_rejects(recorded, attack):
    plan, state = recorded
    def mutate(name, function):
        evidence = plan["spur_allocation"]
        data = json.loads(evidence[name + "_json"])
        function(data)
        raw = json.dumps(data)
        evidence[name + "_json"] = raw
        evidence[name + "_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        if name == "ready": plan["allocation_receipt_sha256"] = evidence[name + "_sha256"]
    if attack == "missing_evidence": plan.pop("spur_allocation")
    elif attack == "missing_capacity": mutate("ready", lambda d: d["ready"].pop("resource_profile"))
    elif attack == "wrong_ready_hash": plan["allocation_receipt_sha256"] = "0" * 64
    elif attack == "wrong_assignment": mutate("binding", lambda d: d.update(original_assignment_sha256="0" * 64))
    elif attack == "duplicate_gpu": mutate("binding", lambda d: d["binding"].__setitem__(1, d["binding"][0]))
    elif attack == "foreign_gpu": plan["bindings"][0]["rocr_uuid"] = "GPU-0000000000000000"
    elif attack == "fake_minor": mutate("renders", lambda d: d["renders"][0].update(major_minor="226:1"))
    elif attack == "missing_render": mutate("renders", lambda d: d["renders"].pop())
    elif attack == "cpu_request": state["CPUs/Task"] = "128"
    elif attack == "memory_request": state["MinMemoryNode"] = "960G"
    elif attack == "gpu_request": state["ReqGPUs"] = "4"
    elif attack == "oversized_cpu": plan["allocation_capacity"]["cpus"] = 193
    elif attack == "oversized_memory": plan["allocation_capacity"]["memory_gib"] = 1537
    elif attack == "partial_gpu_pool": plan["bindings"].pop()
    elif attack == "multi_task": state["NumTasks"] = "2"
    elif attack == "multi_node": state["NumNodes"] = "2"
    elif attack == "restarted": state["Restarts"] = "2"
    elif attack == "wrong_start": state["StartTime"] = "2026-10-11T14:00:00"
    elif attack == "cancelled": state["JobState"] = "CANCELLED"
    elif attack == "expired_lease": state["EndTime"] = "2026-10-11T13:30:00"
    else: plan["scheduler_provider"] = "auto"
    with pytest.raises(ValueError):
        admission.validate_scheduler(plan, state, plan["node"])

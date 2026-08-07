"""CPU-only tests for fail-closed direct ROCm SMI process evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.kfd_process_inventory import (
    KfdProcessInventoryError,
    parse_inventory,
    query_process_inventory,
)


class _FakeApi:
    def __init__(
        self,
        *,
        pids: list[int] | None = None,
        init_status: int = 0,
        count_status: int = 0,
        fetch_status: int = 0,
        shutdown_status: int = 0,
    ) -> None:
        self.pids = pids or []
        self.init_status = init_status
        self.count_status = count_status
        self.fetch_status = fetch_status
        self.shutdown_status = shutdown_status
        self.calls: list[str] = []

    def init(self) -> int:
        self.calls.append("init")
        return self.init_status

    def process_count(self) -> tuple[int, int]:
        self.calls.append("count")
        return self.count_status, len(self.pids)

    def process_pids(self, capacity: int) -> tuple[int, list[int]]:
        self.calls.append(f"fetch:{capacity}")
        return self.fetch_status, list(self.pids)

    def shutdown(self) -> int:
        self.calls.append("shutdown")
        return self.shutdown_status


def _query(api: _FakeApi) -> dict:
    return query_process_inventory(
        api,
        library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
        library_sha256="c" * 64,
        observed_at_ns=123,
    )


def _encode(document: dict) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _document_digest(document: dict) -> str:
    material = {key: value for key, value in document.items() if key != "sha256"}
    return hashlib.sha256(_encode(material).rstrip(b"\n")).hexdigest()


def test_empty_inventory_checks_both_queries_and_shutdown() -> None:
    api = _FakeApi()
    document = _query(api)

    parsed = parse_inventory(_encode(document))

    assert parsed["verified_empty"] is True
    assert parsed["pids"] == []
    assert api.calls == ["init", "count", "fetch:64", "shutdown"]


def test_active_processes_are_preserved_as_authoritative_evidence() -> None:
    document = _query(_FakeApi(pids=[9002, 9001]))

    parsed = parse_inventory(_encode(document))

    assert parsed["verified_empty"] is False
    assert parsed["pids"] == [9001, 9002]


@pytest.mark.parametrize(
    ("api", "message"),
    [
        (_FakeApi(init_status=4), "rsmi_init.*status=4"),
        (_FakeApi(count_status=4), "count failed.*status=4"),
        (_FakeApi(fetch_status=11), "fetch failed.*status=11"),
        (_FakeApi(shutdown_status=6), "rsmi_shut_down failed.*status=6"),
    ],
)
def test_every_rocm_smi_failure_is_fail_closed(api: _FakeApi, message: str) -> None:
    with pytest.raises(KfdProcessInventoryError, match=message):
        _query(api)

    if api.init_status == 0:
        assert api.calls[-1] == "shutdown"


def test_signed_failed_status_cannot_claim_an_empty_inventory() -> None:
    document = _query(_FakeApi())
    document["query"]["count_status"] = 4
    document["sha256"] = _document_digest(document)

    with pytest.raises(KfdProcessInventoryError, match="failed API status"):
        parse_inventory(_encode(document))

"""Imported workload bytes and provenance are an independently checked contract."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1] / "tasks/e2e"


def test_complete_pinned_catalog():
    manifest = json.loads((ROOT / "upstream_manifest.json").read_text())
    assert len(manifest["revision"]) == 40
    assert manifest["files"]
    paths = set()
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path not in paths
        paths.add(path)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert isinstance(yaml.safe_load(data)["benchmark"], dict)
        assert entry["status"] == "imported"
        assert entry["used_by"]
        for consumer in entry["used_by"]:
            assert (ROOT / consumer).read_bytes() == data
            assert (ROOT / consumer).parent.joinpath("config.yaml").is_file()
    imported = set(ROOT.glob("*/benchmark_*.yaml")) | set(ROOT.glob("*/*/benchmark_*.yaml"))
    assert imported == paths
    assert (ROOT / "MAGPIE_LICENSE").is_file()

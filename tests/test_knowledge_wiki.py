"""The run-level knowledge base: configuration, the logged query command and its instructions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.geak.compatibility import arena_contract
from src import knowledge_wiki

CLI = Path(knowledge_wiki.__file__).resolve()
ENABLED = {"knowledge_wiki": {"enabled": True}}

GET_PAGE = ('import sys; print("# page " + sys.argv[1]); '
            'sys.exit(0 if sys.argv[1] == "--help" else 3 if sys.argv[1] == "missing" else 0)')
# Output shapes of the real query.py, get_page.py and grep_wiki.py of ROCmKernelWiki and InstinctWiki.
FAKE_SCRIPTS = {
    "rocm-kernel-wiki": {
        "query.py": 'print("# 2 result(s)\\n\\n'
                    '  [wiki-technique] technique-mfma-pipelining: MFMA  (wiki/techniques/mfma-pipelining.md)\\n'
                    '  [pr-source] pr-aiter-12: Fix  (sources/prs/aiter/12.md)")',
        "get_page.py": GET_PAGE,
        "grep_wiki.py": 'print("=== sources/docs/doc-llvm-amdgpu.md (1 match) ===\\n>> 22: global_load_lds")',
    },
    "instinct-wiki": {
        "query.py": 'print("# 2 result(s)\\n\\n## MFMA\\n- **id**: `technique-mfma-pipelining`\\n'
                    '- **type**: `wiki-technique`\\n\\n## Fix\\n- **id**: `pr-aiter-12`\\n- **type**: `source-pr`")',
        "get_page.py": GET_PAGE,
        "grep_wiki.py": 'print("# 1 file(s) match\\n\\n## sources/docs/doc-llvm-amdgpu.md  (1 matches)\\n'
                        '```\\n22: global_load_lds\\n```")',
    },
}


@pytest.fixture(params=sorted(FAKE_SCRIPTS))
def wiki_root(request, tmp_path):
    root = tmp_path / "wiki"
    (root / "scripts").mkdir(parents=True)
    for name, body in FAKE_SCRIPTS[request.param].items():
        (root / "scripts" / name).write_text(body + "\n")
    return root


def test_resolve_is_off_unless_enabled(wiki_root):
    env = {knowledge_wiki.ROOT_ENV: str(wiki_root)}
    assert knowledge_wiki.resolve({}, env) is None
    assert knowledge_wiki.resolve({"knowledge_wiki": {"enabled": False}}, env) is None
    assert knowledge_wiki.resolve(ENABLED, env).root == wiki_root.resolve()


@pytest.mark.parametrize("value", [True, {"enabled": "yes"}, {"enabled": True, "root": "/elsewhere"}])
def test_resolve_rejects_malformed_config(wiki_root, value):
    with pytest.raises(ValueError, match="knowledge_wiki"):
        knowledge_wiki.resolve({"knowledge_wiki": value}, {knowledge_wiki.ROOT_ENV: str(wiki_root)})


def test_resolve_fails_closed_without_a_checkout(tmp_path):
    with pytest.raises(ValueError, match="not set"):
        knowledge_wiki.resolve(ENABLED, {})
    with pytest.raises(ValueError, match="missing scripts/query.py"):
        knowledge_wiki.resolve(ENABLED, {knowledge_wiki.ROOT_ENV: str(tmp_path)})


def test_run_rejects_agents_that_would_ignore_the_knowledge_base(wiki_root, monkeypatch):
    monkeypatch.setenv(knowledge_wiki.ROOT_ENV, str(wiki_root))
    for agent in knowledge_wiki.SUPPORTED_AGENTS:
        assert knowledge_wiki.validate_run(ENABLED, agent) is not None
    with pytest.raises(ValueError, match="supported only"):
        knowledge_wiki.validate_run(ENABLED, "claude_code")
    assert knowledge_wiki.validate_run({"knowledge_wiki": {"enabled": False}}, "claude_code") is None


def run_cli(root: Path, log: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CLI), "--root", str(root), "--log", str(log), *args],
                          capture_output=True, text=True, check=False)


def test_cli_passes_output_through_and_logs_each_call(wiki_root, tmp_path):
    log = tmp_path / "calls.jsonl"
    search = run_cli(wiki_root, log, "search", "mfma pipelining", "--limit", "2")
    page = run_cli(wiki_root, log, "page", "missing", "--body-only")
    grep = run_cli(wiki_root, log, "grep", "global_load_lds", "--only", "sources")
    usage = run_cli(wiki_root, log, "page", "--help")
    assert search.returncode == 0 and "technique-mfma-pipelining" in search.stdout
    assert page.returncode == 3 and page.stdout == "# page missing\n"
    assert grep.returncode == 0 and "sources/docs/doc-llvm-amdgpu.md" in grep.stdout
    assert usage.returncode == 0 and usage.stdout == "# page --help\n"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["command"] for record in records] == ["search", "page", "grep", "page"]
    assert records[0]["args"] == ["mfma pipelining", "--limit", "2"]
    assert records[0]["results"] == ["technique-mfma-pipelining", "pr-aiter-12"]
    assert records[1]["exit_code"] == 3 and records[1]["results"] == ["missing"]
    assert records[2]["results"] == ["sources/docs/doc-llvm-amdgpu.md"]


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_identity_records_the_checkout_commit_and_instruction_version(wiki_root):
    git(wiki_root, "init", "-q")
    git(wiki_root, "add", "-A")
    git(wiki_root, "commit", "-q", "-m", "wiki")
    identity = knowledge_wiki.KnowledgeWiki(wiki_root).identity()
    assert identity["root"] == str(wiki_root)
    assert identity["commit"] == git(wiki_root, "rev-parse", "HEAD") and identity["dirty"] is False
    assert identity["instructions_version"] == knowledge_wiki.INSTRUCTIONS_VERSION
    assert len(identity["instructions_sha256"]) == 64


def test_identity_ignores_an_enclosing_repository(tmp_path, wiki_root):
    git(tmp_path, "init", "-q")
    git(tmp_path, "commit", "-q", "--allow-empty", "-m", "outer")
    identity = knowledge_wiki.KnowledgeWiki(wiki_root).identity()
    assert identity["commit"] is None and identity["dirty"] is None


def test_geak_contract_ends_with_the_shared_instructions(wiki_root, tmp_path):
    wiki = knowledge_wiki.resolve(ENABLED, {knowledge_wiki.ROOT_ENV: str(wiki_root)})
    log = tmp_path / "calls.jsonl"
    text = wiki.instructions(sys.executable, log)
    assert wiki.command(sys.executable, log) in text
    spec = SimpleNamespace(to_mapping=lambda: {"task_id": "t"},
                           candidate=SimpleNamespace(language="hip", initial_state="implemented"))
    bridge = SimpleNamespace(job_path=tmp_path / "job.json", spec=spec)
    assert arena_contract(bridge, sys.executable, text).endswith("\n" + text)
    assert "Kernel knowledge base" not in arena_contract(bridge, sys.executable)

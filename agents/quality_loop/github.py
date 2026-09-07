# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import GitHubConfig


class CommandError(RuntimeError):
    pass


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = " ".join(args)
        detail = (result.stderr or result.stdout).strip()
        raise CommandError(f"command failed ({result.returncode}): {rendered}\n{detail}")
    return result


def parse_github_slug(remote: str) -> str:
    remote = remote.strip()
    patterns = (
        r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$",
        r"^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$",
    )
    for pattern in patterns:
        match = re.match(pattern, remote)
        if match:
            return match.group(1)
    raise ValueError(f"origin is not a supported GitHub remote: {remote!r}")


@dataclass(frozen=True)
class PreflightResult:
    repo_slug: str
    default_branch: str
    base_sha: str
    viewer_permission: str


class GitHubPublisher:
    """The only quality_loop component allowed to use GitHub credentials."""

    def __init__(self, repo_root: Path, config: GitHubConfig, logger: logging.Logger):
        self.repo_root = repo_root.resolve()
        self.config = config
        self.logger = logger

    def preflight(self) -> PreflightResult:
        for command in ("git", "gh", "codex"):
            if not shutil.which(command):
                raise RuntimeError(f"required command not found: {command}")

        status = run_command(["git", "status", "--porcelain"], cwd=self.repo_root)
        if status.stdout.strip():
            raise RuntimeError(
                "quality_loop requires a clean source worktree before creating "
                "its isolated audit worktree"
            )
        run_command(["git", "var", "GIT_AUTHOR_IDENT"], cwd=self.repo_root)

        run_command(["gh", "auth", "status", "-h", "github.com"], cwd=self.repo_root)
        remote = run_command(
            ["git", "remote", "get-url", "origin"], cwd=self.repo_root
        ).stdout.strip()
        slug = parse_github_slug(remote)
        repo_data = json.loads(
            run_command(["gh", "api", f"repos/{slug}"], cwd=self.repo_root).stdout
        )
        permissions = repo_data.get("permissions") or {}
        permission = str(repo_data.get("viewer_permission") or "").upper()
        push_allowed = bool(permissions.get("push")) or permission in {
            "WRITE",
            "MAINTAIN",
            "ADMIN",
        }
        if not push_allowed:
            raise RuntimeError(
                f"authenticated GitHub user lacks write permission for {slug} "
                f"(viewer_permission={permission or 'unknown'})"
            )
        default_branch = self.config.base_branch or repo_data.get("default_branch")
        if not default_branch:
            raise RuntimeError(f"could not determine the default branch for {slug}")
        run_command(["git", "fetch", "origin", default_branch], cwd=self.repo_root, timeout=600)
        base_sha = run_command(
            ["git", "rev-parse", f"origin/{default_branch}"], cwd=self.repo_root
        ).stdout.strip()
        return PreflightResult(slug, str(default_branch), base_sha, permission or "WRITE")

    def create_worktree(
        self,
        *,
        path: Path,
        branch: str,
        base_branch: str,
    ) -> None:
        if path.exists():
            raise RuntimeError(f"quality_loop worktree already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch,
                str(path),
                f"origin/{base_branch}",
            ],
            cwd=self.repo_root,
            timeout=600,
        )

    def commit_task(self, worktree: Path, task_id: str) -> str | None:
        relative = f"tasks/{task_id}"
        run_command(["git", "add", "--", relative], cwd=worktree)
        staged = run_command(
            ["git", "diff", "--cached", "--quiet", "--", relative],
            cwd=worktree,
            check=False,
        )
        if staged.returncode == 0:
            return None
        if staged.returncode != 1:
            raise CommandError(f"could not inspect staged changes for {task_id}")
        run_command(
            ["git", "commit", "-m", f"fix(tasks): quality audit {task_id}"],
            cwd=worktree,
            timeout=600,
        )
        return run_command(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()

    def verify_pending_changes(
        self,
        *,
        worktree: Path,
        branch: str,
        base_sha: str,
        expected_paths: set[str],
    ) -> None:
        top = Path(
            run_command(["git", "rev-parse", "--show-toplevel"], cwd=worktree)
            .stdout.strip()
        ).resolve()
        if top != worktree.resolve():
            raise RuntimeError(f"quality_loop worktree identity changed: {top}")
        current_branch = run_command(
            ["git", "branch", "--show-current"], cwd=worktree
        ).stdout.strip()
        if current_branch != branch:
            raise RuntimeError(
                f"quality_loop worktree branch changed: expected {branch}, got {current_branch}"
            )
        ancestor = run_command(
            ["git", "merge-base", "--is-ancestor", base_sha, "HEAD"],
            cwd=worktree,
            check=False,
        )
        if ancestor.returncode != 0:
            raise RuntimeError("quality_loop worktree no longer descends from its recorded base")

        changed: set[str] = set()
        for args in (
            ["git", "diff", "--name-only", "HEAD"],
            ["git", "diff", "--cached", "--name-only"],
            ["git", "ls-files", "--others", "--exclude-standard"],
        ):
            output = run_command(args, cwd=worktree).stdout
            changed.update(line for line in output.splitlines() if line)
        if changed != expected_paths:
            unexpected = sorted(changed - expected_paths)
            missing = sorted(expected_paths - changed)
            raise RuntimeError(
                "quality_loop worktree diff does not match accepted task changes; "
                f"unexpected={unexpected}, missing={missing}"
            )

    def publish_draft_pr(
        self,
        *,
        worktree: Path,
        repo_slug: str,
        branch: str,
        base_branch: str,
        title: str,
        body: str,
        artifact_dir: Path,
    ) -> str | None:
        ahead = int(
            run_command(
                ["git", "rev-list", "--count", f"origin/{base_branch}..HEAD"],
                cwd=worktree,
            ).stdout.strip()
            or "0"
        )
        if ahead == 0:
            self.logger.info("No accepted task changes; skipping empty pull request")
            return None

        # Use gh's credential helper explicitly so an SSH origin does not require
        # forwarding private SSH keys into the GPU container.
        https_remote = f"https://github.com/{repo_slug}.git"
        run_command(
            [
                "git",
                "-c",
                "credential.helper=!gh auth git-credential",
                "push",
                "--set-upstream",
                https_remote,
                branch,
            ],
            cwd=worktree,
            timeout=1200,
        )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        body_path = artifact_dir / "pull_request_body.md"
        body_path.write_text(body.rstrip() + "\n", encoding="utf-8")
        args = [
            "gh",
            "pr",
            "create",
            "--repo",
            repo_slug,
            "--head",
            branch,
            "--base",
            base_branch,
            "--title",
            title,
            "--body-file",
            str(body_path),
            "--draft",
        ]
        existing = json.loads(
            run_command(
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    repo_slug,
                    "--head",
                    branch,
                    "--state",
                    "all",
                    "--limit",
                    "1",
                    "--json",
                    "url",
                ],
                cwd=worktree,
            ).stdout
            or "[]"
        )
        if existing:
            return str(existing[0].get("url") or "")
        return run_command(args, cwd=worktree, timeout=300).stdout.strip()

"""Adapt quality_loop to the shared TaskSession; never implement local scoring.

The shared lifecycle owns snapshots, manifests, phase policy, and command gates.
Final scoring/tool execution is delegated to the evaluator's session entrypoint.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.task_spec import TaskSpec


class InitialValidationRejected(RuntimeError):
    """The shared lifecycle rejected this package; no candidate gate may pass."""


class EvaluationSession:
    def __init__(self, session, *, eval_config: dict[str, Any], logger: logging.Logger):
        self.session = session
        self.eval_config = eval_config
        self.logger = logger
        self.baseline_cases: list[Any] = []

    @property
    def agent_context_path(self) -> Path:
        return self.session.agent_context_path

    def prepare(self) -> None:
        initial = self.session.validate_initial()
        if not initial.accepted:
            raise InitialValidationRejected("Initial task validation failed: " + "; ".join(initial.errors))
        # These are the framework-validated protocol rows, not recomputed timings.
        self.baseline_cases = list(self.session.results[
            ("task_validation", "baseline", "performance")
        ].result.cases)

    def evaluate_candidate(self) -> dict[str, Any]:
        try:
            from src.evaluator import evaluate_task_session
        except ImportError as exc:
            raise RuntimeError(
                "quality_loop requires src.evaluator.evaluate_task_session(session, "
                "eval_config=..., logger=...) to score and write task_result.yaml; "
                "legacy evaluation is not a compatible fallback"
            ) from exc
        return evaluate_task_session(self.session, eval_config=self.eval_config, logger=self.logger)

    def check_candidate(self) -> bool:
        for action in ("compile", "correctness"):
            if not self.session.candidate_action(action).result.passed:
                return False
        return True


def create_session(*, spec: TaskSpec, workspace: Path, artifact_dir: Path,
                   eval_config: dict[str, Any], logger: logging.Logger) -> EvaluationSession:
    try:
        from src.task_session import TaskSession
    except ImportError as exc:
        raise RuntimeError("quality_loop v2 requires the shared src.task_session.TaskSession") from exc
    session = TaskSession.create(spec, workspace, artifact_dir, logger=logger)
    return EvaluationSession(session, eval_config=eval_config, logger=logger)

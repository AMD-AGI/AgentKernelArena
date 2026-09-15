"""Adapt quality_loop to the shared TaskSession; never implement local scoring.

The shared lifecycle owns snapshots, manifests, phase policy, and command gates.
Final scoring/tool execution is delegated to the evaluator's session entrypoint.
"""
from __future__ import annotations

import json
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
        self._review_context_files = {}
        self._review_result = None
        self._review_actions = "{}"

    @property
    def agent_context_path(self) -> Path:
        return self.session.agent_context_path

    def prepare(self) -> None:
        from src.task_runtime import bind_session_runtime

        bind_session_runtime(self.session)
        initial = self.session.validate_initial()
        if not initial.accepted:
            raise InitialValidationRejected("Initial task validation failed: " + "; ".join(initial.errors))
        # These are the framework-validated protocol rows, not recomputed timings.
        self.baseline_cases = list(self.session.results[
            ("task_validation", "baseline", "performance")
        ].result.cases)
        from .review_evidence import snapshot_context_files

        self._review_context_files = snapshot_context_files(self.session)

    def evaluate_candidate(self) -> dict[str, Any]:
        try:
            from src.evaluator import evaluate_task_session
        except ImportError as exc:
            raise RuntimeError(
                "quality_loop requires src.evaluator.evaluate_task_session(session, "
                "eval_config=..., logger=...) to score and write task_result.yaml; "
                "legacy evaluation is not a compatible fallback"
            ) from exc
        from .review_evidence import snapshot_candidate_actions

        previous = {action.invocation_id for action in self.session.results.values()}
        result = evaluate_task_session(self.session, eval_config=self.eval_config, logger=self.logger)
        self._review_result = json.dumps(result, sort_keys=True, allow_nan=False)
        self._review_actions = snapshot_candidate_actions(self.session, previous)
        return result

    def review_evidence(self, result):
        from .review_evidence import expose_review_evidence

        if not self._review_context_files or self._review_result is None:
            raise RuntimeError("Review requires the controller's prepared and evaluated session")
        return expose_review_evidence(
            self.session, context_files=self._review_context_files,
            evaluated_result=self._review_result, candidate_actions=self._review_actions, result=result)

    def verify_review_evidence(self, evidence) -> None:
        evidence.verify()
        self.session.verify_baseline_sources()

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

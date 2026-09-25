"""Compose the pinned upstream optimizer with Arena's evaluator authority."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from apex.bootstrap import build_application
from apex.core import sha256_json
from apex.evaluation import (
    DigestBoundEvaluationAuthorizer, EvaluationAuthorityIdentity, EvaluationAuthorityKind,
)
from apex.intake import TaskSpec
from apex.optimization.kernel import KernelOptimizeRequest


def run(job_path: Path) -> int:
    job = json.loads(job_path.read_text())
    task = TaskSpec.from_mapping(job["task"])
    optimizer = build_application().kernel_optimizer
    preview = optimizer.preview_evaluation_contract(task)
    (job_path.parent / "evaluation_contract.json").write_text(json.dumps(preview.to_dict(), indent=2))
    authorizer = DigestBoundEvaluationAuthorizer(preview.draft.digest, EvaluationAuthorityIdentity(
        authority_id="arena-v2", kind=EvaluationAuthorityKind.EXTERNAL_EVALUATOR,
        issuer="AgentKernelArena", policy_sha256=sha256_json(job["context"]["task_config"]),
        template_sha256=sha256_json(job["context"]["manifest"]),
    ))
    optimizer = build_application(kernel_evaluation_authorizer=authorizer).kernel_optimizer
    result = optimizer.run(KernelOptimizeRequest(task=task, result_json=Path(job["result"])))
    print(json.dumps({"status": result.status.value, "reason_code": result.reason_code}))
    return 0 if result.status.value in {"candidate_ready", "no_gain"} else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))

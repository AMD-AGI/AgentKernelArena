# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Post-processing for the forge_rewrite agent.

Runs Arena's normal run aggregation and then fills each task's SIKL solution
slot from the scored result. Post-processing is the first point where every
task's ``task_result.yaml`` exists, which is what makes it the right place to
record what a rewrite produced.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from agents.forge_rewrite.solution_backfill import backfill_solutions
from src.postprocessing import general_post_processing


def forge_rewrite_post_processing(
    workspace_paths: Union[str, List[str]], logger: Optional[logging.Logger]
) -> None:
    """Aggregate the run, then fill the solution slot of every ported task."""
    logger = logger or logging.getLogger(__name__)
    general_post_processing(workspace_paths, logger)
    paths = [workspace_paths] if isinstance(workspace_paths, str) else list(workspace_paths)
    backfill_solutions(paths, logger)

# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deprecated name for common aggregation. Exports belong to the evaluator."""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from src.postprocessing import general_post_processing


def forge_operator2flydsl_post_processing(
    workspace_paths: Union[str, List[str]], logger: Optional[logging.Logger]
) -> None:
    """Aggregate normally; do not perform provider-specific task exports."""
    logger = logger or logging.getLogger(__name__)
    general_post_processing(workspace_paths, logger)

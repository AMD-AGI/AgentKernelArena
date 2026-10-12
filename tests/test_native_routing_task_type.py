"""The literal routing contract is admitted without pretending to edit Triton."""
from copy import deepcopy
import logging
from pathlib import Path

import pytest
import yaml

from agents.task_validator.validation_prompt import build_validation_prompt
from src.prompt_builder import _load_cheatsheet, prompt_builder
from src.tools import trusted_qwen_eval


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster"
LOGGER = logging.getLogger(__name__)


def test_native_routing_builds_normal_isolated_task_prompt():
    config = TASK / "config.yaml"
    text = prompt_builder(str(config), TASK, {"target_gpu_model": "MI355X"}, LOGGER)
    assert "literal Python routing policy" in text
    assert "Only the declared policy body is editable" in text
    guide, arch = _load_cheatsheet("native_routing", "MI355X", ROOT, {}, LOGGER)
    assert arch == "gfx950"
    assert (ROOT / "src/prompts/cheatsheet/native_routing_cheatsheet.md").read_text() in guide
    validation = build_validation_prompt(str(config), str(TASK), {"target_gpu_model": "MI355X"})
    assert "`native_routing` is an isolated task" in validation
    assert "Apply all twelve" in validation


def test_dense_native_routing_admission_preserves_case_and_source_contract():
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    contract = trusted_qwen_eval.package_contract(TASK, config)
    assert config["task_type"] == "native_routing"
    assert contract["sources"] == ["source/routing.py"]
    assert len(contract["manifest"]["cases"]) == 5
    assert config["harness_protection"]["freeze_task_inputs"] is True
    assert config["harness_protection"]["reject_new_source_symlinks"] is True
    wrong_type = deepcopy(config)
    wrong_type["task_type"] = "triton2triton"
    with pytest.raises(ValueError, match="Qwen target contract differs"):
        trusted_qwen_eval.package_contract(TASK, wrong_type)

from __future__ import annotations

import pytest

from src.eval_tools.config import EvalToolsConfig, merge_task_tool_config


def _run_config() -> EvalToolsConfig:
    return EvalToolsConfig.from_mapping(
        {
            "evaluation_tools": {
                "policy": "required",
                "enabled": ["gpu_asan"],
                "timeout_s": 100,
                "tools": {
                    "gpu_asan": {
                        "runtime_ref": "image@sha256:abc",
                        "options": {},
                    }
                },
            }
        }
    )


def test_task_can_add_adapter_argv_but_not_change_policy_or_image() -> None:
    merged = merge_task_tool_config(
        _run_config(),
        {
            "evaluation_tools": {
                "tools": {
                    "gpu_asan": {
                        "timeout_s": 40,
                        "options": {"command": ["python3", "asan_case.py"]},
                    }
                }
            }
        },
    )
    tool = merged.tools[0]
    assert merged.policy.value == "required"
    assert tool.runtime_ref == "image@sha256:abc"
    assert tool.timeout_s == 40
    assert tool.options["positive_control_required"] is True
    assert tool.options["command"] == ["python3", "asan_case.py"]


def test_task_cannot_enable_tool_or_increase_timeout() -> None:
    with pytest.raises(ValueError, match="not enabled"):
        merge_task_tool_config(
            _run_config(),
            {"evaluation_tools": {"tools": {"rocjitsu": {"options": {}}}}},
        )
    with pytest.raises(ValueError, match="must be in"):
        merge_task_tool_config(
            _run_config(),
            {
                "evaluation_tools": {
                    "tools": {"gpu_asan": {"timeout_s": 101}}
                }
            },
        )


def test_task_cannot_disable_required_positive_control_through_options() -> None:
    with pytest.raises(ValueError, match="reserved options"):
        merge_task_tool_config(
            _run_config(),
            {
                "evaluation_tools": {
                    "tools": {
                        "gpu_asan": {
                            "options": {"positive_control_required": False}
                        }
                    }
                }
            },
        )


def test_run_cannot_override_framework_positive_control_state() -> None:
    with pytest.raises(ValueError, match="reserved framework keys"):
        EvalToolsConfig.from_mapping(
            {
                "evaluation_tools": {
                    "positive_control": "required",
                    "enabled": ["gpu_asan"],
                    "tools": {
                        "gpu_asan": {
                            "options": {"positive_control_required": False}
                        }
                    },
                }
            }
        )


def test_runtime_asset_paths_are_reserved_at_run_and_task_levels() -> None:
    with pytest.raises(ValueError, match="asan_runtime_dir"):
        EvalToolsConfig.from_mapping(
            {
                "evaluation_tools": {
                    "enabled": ["gpu_asan"],
                    "tools": {
                        "gpu_asan": {"options": {"asan_runtime_dir": "/fake"}}
                    },
                }
            }
        )
    with pytest.raises(ValueError, match="reserved options"):
        merge_task_tool_config(
            _run_config(),
            {
                "evaluation_tools": {
                    "tools": {
                        "gpu_asan": {"options": {"asan_runtime_dir": "/fake"}}
                    }
                }
            },
        )


def test_host_selected_tool_subset_is_authoritative(monkeypatch) -> None:
    monkeypatch.setenv("AKA_EVAL_TOOLS_SELECTED", "rocjitsu,gpu-asan")
    parsed = EvalToolsConfig.from_mapping(
        {
            "evaluation_tools": {
                "enabled": ["triton_fpsan"],
                "tools": {},
            }
        }
    )
    assert parsed.enabled == ("rocjitsu", "gpu_asan")


def test_host_selected_image_id_is_part_of_parsed_config(monkeypatch) -> None:
    monkeypatch.setenv("AKA_EVAL_TOOL_RUNTIME_REF_GPU_ASAN", "sha256:selected")
    parsed = EvalToolsConfig.from_mapping(
        {"evaluation_tools": {"enabled": ["gpu_asan"]}}
    )
    assert parsed.tools[0].runtime_ref == "sha256:selected"


def test_explicit_runtime_ref_must_match_host_selected_image(monkeypatch) -> None:
    monkeypatch.setenv("AKA_EVAL_TOOL_RUNTIME_REF_GPU_ASAN", "sha256:selected")
    with pytest.raises(ValueError, match="does not match selected"):
        EvalToolsConfig.from_mapping(
            {
                "evaluation_tools": {
                    "enabled": ["gpu_asan"],
                    "tools": {
                        "gpu_asan": {"runtime_ref": "sha256:different"}
                    },
                }
            }
        )

import subprocess
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.codex.auth_preflight import CodexAuthError, _failure_signals, _model_settings, _preflight_timeout, check_codex_access


class CodexAuthPreflightTests(unittest.TestCase):
    def custom_config(self, **provider):
        return {"model_provider": "example", "model": "example-model",
                "model_providers": {"example": {"http_headers": {"key": "SECRET"}, **provider}}}

    @staticmethod
    def successful_inference(command, *args, **kwargs):
        output = Path(command[command.index("--output-last-message") + 1])
        output.write_text("AKA_CODEX_PROVIDER_ACCESS_OK\n")
        return subprocess.CompletedProcess(command, 0, "private transcript", "private diagnostics")

    def test_default_provider_requires_openai_login(self):
        with patch("agents.codex.auth_preflight._user_config", return_value={}), \
             patch("agents.codex.auth_preflight.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(check_codex_access(), "codex_status=OpenAI_login_verified")
            self.assertEqual(run.call_args.args[0], ["codex", "login", "status"])

    def test_custom_headers_require_real_inference_not_openai_login(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config()), \
             patch("agents.codex.auth_preflight._run_inference", side_effect=self.successful_inference) as run:
            self.assertEqual(check_codex_access(), "codex_status=custom_provider_inference_verified")
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertNotIn("SECRET", " ".join(command))
            self.assertEqual(command[command.index("--model") + 1], "example-model")
            self.assertIn("--ephemeral", command)
            self.assertIn("read-only", command)

    def test_custom_provider_can_explicitly_require_openai_login(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config(requires_openai_auth=True)), \
             patch("agents.codex.auth_preflight.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "SECRET", "SECRET")):
            with self.assertRaisesRegex(CodexAuthError, "requires OpenAI authentication") as error:
                check_codex_access()
            self.assertNotIn("SECRET", str(error.exception))

    def test_static_credentials_do_not_make_a_failed_provider_pass(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config()), \
             patch("agents.codex.auth_preflight._run_inference", return_value=subprocess.CompletedProcess([], 7, "SECRET", "SECRET")):
            with self.assertRaisesRegex(CodexAuthError, "exit 7") as error:
                check_codex_access()
            self.assertNotIn("SECRET", str(error.exception))

    def test_exit_zero_without_expected_inference_response_fails(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config()), \
             patch("agents.codex.auth_preflight._run_inference", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(CodexAuthError, "expected response"):
                check_codex_access()

    def test_timeout_diagnostics_are_redacted(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config()), \
             patch("agents.codex.auth_preflight._run_inference", side_effect=subprocess.TimeoutExpired(["SECRET"], 120, output="SECRET")):
            with self.assertRaisesRegex(CodexAuthError, "timed out") as error:
                check_codex_access()
            self.assertNotIn("SECRET", str(error.exception))

    def test_validator_run_model_and_effort_override_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "run.yaml"
            config.write_text("agent:\n  template: task_validator\n  backend: codex\n  model: selected-model\n  effort: max\n")
            self.assertEqual(_model_settings(str(config), {"model": "cli-model"}), ("selected-model", "max"))

    def test_invalid_auth_setting_fails_closed(self):
        with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config(requires_openai_auth="false")):
            with self.assertRaisesRegex(CodexAuthError, "must be boolean"):
                check_codex_access()

    def test_preflight_timeout_config_is_forwarded_without_model_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "run.yaml"
            config.write_text("agent:\n  template: task_validator\n  model: selected-model\n  effort: max\n  preflight_timeout_seconds: 300\n")
            with patch("agents.codex.auth_preflight._user_config", return_value=self.custom_config()), \
                 patch("agents.codex.auth_preflight._run_inference", side_effect=self.successful_inference) as run:
                check_codex_access(str(config))
                self.assertEqual(run.call_args.args[2], 300)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--model") + 1], "selected-model")
                self.assertIn('model_reasoning_effort="max"', command)
                self.assertEqual(_preflight_timeout(str(config), 60), 60)

    def test_preflight_timeout_cannot_disable_or_escape_the_bound(self):
        self.assertEqual(_preflight_timeout(None, None), 120)
        for value in (0, -1, 29, 301, True, "300", 300.0):
            with self.subTest(value=value), self.assertRaises(CodexAuthError):
                _preflight_timeout(None, value)

    def test_failure_categories_never_include_provider_details(self):
        summary = _failure_signals(b"429 retrying connection to https://SECRET.invalid/?key=SECRET")
        self.assertEqual(summary, "; diagnostic_signals=rate_limit,transport,retry")
        self.assertNotIn("SECRET", summary)
        self.assertEqual(_failure_signals(None, "private provider text"), "; diagnostic_signals=unclassified")

    def test_container_call_forwards_config_and_agent_selection(self):
        runner = Path(__file__).parents[1] / "src/scripts/docker_benchmark.sh"
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text('#!/bin/sh\nprintf "%s|%s\\n" "$AKA_CHECK_CONFIG" "$AKA_CHECK_AGENTS"\n')
            python.chmod(0o755)
            environment = {**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]}
            result = subprocess.run(
                ["bash", str(runner), "_container_check_agents", "--config_name", "selected.yaml", "codex"],
                env=environment, capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), "selected.yaml|codex")


if __name__ == "__main__":
    unittest.main()

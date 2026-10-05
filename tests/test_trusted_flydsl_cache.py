import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.tools import trusted_task_eval as trusted


class TrustedFlydslCacheTests(unittest.TestCase):
    def run_case(self, seeded):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, staging, output = root / "task", root / "staging", root / "output"
            for path in (task, staging, output):
                path.mkdir()
            image = "registry/image@sha256:" + "a" * 64
            commands = []

            def invoke(command, **_kwargs):
                if command[:2] == ["docker", "run"]:
                    commands.append(command)
                    build = staging / "candidate_compile_build"
                    (build / "gpu_preflight.json").write_text("{}")
                    (build / "compile_report.json").write_text(json.dumps({"unit_test": True}))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(trusted, "copy_cache"), \
                 patch.object(trusted, "docker_command", return_value=["docker", "run", "--network=none", image, "runner"]), \
                 patch.object(trusted, "command_with_binding", side_effect=lambda command, *_args: command), \
                 patch.object(trusted, "validate_preflight"), \
                 patch.object(trusted, "preserve_diagnostics"), \
                 patch.object(trusted.subprocess, "run", side_effect=invoke):
                trusted.run_phase(image, task, staging, output, "candidate", {"phase": "compile", "gpu": {}},
                                  "/dev/dri/renderD128", 30, root / "verified-cache" if seeded else None)
            self.assertEqual(len(commands), 1)
            return commands[0], image

    def test_seeded_phase_selects_private_flydsl_cache_before_image(self):
        command, image = self.run_case(True)
        setting = "FLYDSL_RUNTIME_CACHE_DIR=/aiter-jit/flydsl_cache"
        self.assertIn(setting, command)
        self.assertLess(command.index(setting), command.index(image))
        self.assertEqual(command[command.index(setting) - 1], "--env")

    def test_unseeded_phase_does_not_redirect_flydsl_cache(self):
        command, _image = self.run_case(False)
        self.assertFalse(any(value.startswith("FLYDSL_RUNTIME_CACHE_DIR=") for value in command))


if __name__ == "__main__":
    unittest.main()

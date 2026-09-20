"""CPU checks for direct evidence; these never claim GPU qualification."""
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

import yaml
from src.tools import verify_head_kernels as verifier
from src.scripts import top5_head_kernels as launcher

ROOT = Path(__file__).resolve().parents[1]


class DirectVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.source = self.repo / 'tasks/head_kernels/test'
        self.source.mkdir(parents=True)
        self.workspace = self.repo / 'run/task'
        self.workspace.parent.mkdir()
        self.helper = mock.patch.object(verifier, 'materialize_perf_helpers_in_workspace', return_value=[])
        self.helper.start()
        self.addCleanup(self.helper.stop)

    def task(self, *, failure=None, native_failure=None, timeout=None):
        config = {}
        for phase in verifier.PHASES:
            code = "import json,pathlib; p=pathlib.Path('build'); p.mkdir(exist_ok=True); "
            status = 'fail' if phase == native_failure else 'ok'
            code += f"(p/'{phase}_report.json').write_text(json.dumps({{'status':{status!r}}})); "
            if phase == timeout:
                code += 'import time; time.sleep(30); '
            code += f"raise SystemExit({19 if phase == failure else 0})"
            config[phase + '_command'] = [shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)]
            config[phase + '_timeout'] = .05 if phase == timeout else 5
        (self.source / 'config.yaml').write_text(yaml.safe_dump(config))
        (self.source / 'kernel.py').write_text('value = 1\n')
        (self.source / 'alias.py').symlink_to('kernel.py')

    def test_copy_preserves_binding_and_removes_stale_reports(self):
        self.task()
        (self.source / 'build').mkdir()
        (self.source / 'build/performance_report.json').write_text('{"status":"ok"}')
        (self.source / 'validation_report.yaml').write_text('overall_status: PASS')
        identity = verifier.copy_task(self.source, self.workspace, self.repo)
        (self.workspace / 'kernel.py').write_text('value = 2\n')
        self.assertTrue((self.workspace / 'alias.py').is_symlink())
        self.assertEqual((self.workspace / 'alias.py').read_text(), 'value = 2\n')
        self.assertEqual((self.source / 'kernel.py').read_text(), 'value = 1\n')
        self.assertFalse((self.workspace / 'build').exists())
        self.assertFalse((self.workspace / 'validation_report.yaml').exists())
        self.assertIn('alias.py', identity['files'])
        verifier.materialize_perf_helpers_in_workspace.assert_called_once()

    def test_escaping_alias_is_refused(self):
        self.task()
        outside = self.repo / 'outside.py'
        outside.write_text('')
        (self.source / 'escape.py').symlink_to(outside)
        with self.assertRaises(ValueError):
            verifier.copy_task(self.source, self.workspace, self.repo)

    def test_command_failure_preserves_exit_and_skips_performance(self):
        self.task(failure='correctness')
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'command_failed')
        self.assertEqual(len(result['phases']), 2)
        self.assertEqual(result['phases'][-1]['commands'][0]['returncode'], 19)
        self.assertEqual(result['framework_task_validator'], 'NOT_RUN')
        self.assertFalse(result['framework_PASS_claimed'])
        self.assertFalse((self.workspace / 'task_result.yaml').exists())

    def test_native_failure_is_fatal_even_when_command_exits_zero(self):
        self.task(native_failure='correctness')
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'native_report_failed_or_missing')
        self.assertEqual(len(result['phases']), 2)
        self.assertEqual(result['phases'][-1]['native_status'], 'fail')

    def test_timeout_preserves_partial_evidence_and_stops_later_phases(self):
        self.task(timeout='correctness')
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'timeout')
        execution = result['phases'][-1]['commands'][0]
        self.assertTrue(execution['timed_out'])
        self.assertEqual(execution['returncode'], -signal.SIGKILL)
        self.assertEqual(len(result['phases']), 2)

    def test_timeout_stops_worker_in_a_separate_session(self):
        self.workspace.mkdir()
        pid_file = self.workspace / 'child.pid'
        child_code = "import pathlib,os,time; pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(30)"
        parent_code = ("import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
                       + repr(child_code) + "],start_new_session=True); time.sleep(30)")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            execution = verifier.run_command(shlex.quote(sys.executable) + ' -c ' + shlex.quote(parent_code),
                                             self.workspace, stdout, stderr, .3)
        self.assertTrue(execution['timed_out'])
        pid = int(pid_file.read_text())
        # A killed child may briefly remain a zombie until its new parent reaps it.
        status = Path(f'/proc/{pid}/stat')
        for _ in range(100):
            if not status.exists() or status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                break
            time.sleep(.01)
        else:
            os.kill(pid, signal.SIGKILL)
            self.fail('Detached native worker survived command timeout')

    def test_one_phase_budget_is_shared_by_all_commands(self):
        self.task()
        config = yaml.safe_load((self.source / 'config.yaml').read_text())
        config['compile_command'] = ['sleep .15', 'sleep .15']
        config['compile_timeout'] = .2
        (self.source / 'config.yaml').write_text(yaml.safe_dump(config))
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(len(result['phases']), 1)

    def test_public_task_copy_materializes_helpers_and_passes_ast_check(self):
        self.helper.stop()
        config_path = ROOT / 'example_configs/top5_validator_glm_bf16_public_mi355x.yaml'
        plan = launcher.plan_run(config_path)
        source = ROOT / 'tasks' / plan['tasks'][0]
        identity = verifier.copy_task(source, self.workspace, ROOT)
        self.assertIn('scripts/_aka_benchmark.py', identity['materialized_perf_helpers'])
        config = yaml.safe_load((self.workspace / 'config.yaml').read_text())
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            execution = verifier.run_command(config['compile_command'][0], self.workspace,
                                             stdout, stderr, 10)
        self.assertEqual(execution['returncode'], 0)
        self.assertEqual(json.loads((self.workspace / 'build/compile_report.json').read_text())['status'], 'ok')

    def test_success_keeps_native_snapshots_and_no_framework_report(self):
        self.task()
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'all_native_phases_succeeded')
        for phase in result['phases']:
            report = self.workspace / phase['native_report']['path']
            self.assertEqual(verifier.fingerprint(report)['sha256'], phase['native_report']['sha256'])
        self.assertFalse((self.workspace / 'validation_report.yaml').exists())

    def test_missing_report_is_not_success(self):
        self.task()
        config = yaml.safe_load((self.source / 'config.yaml').read_text())
        config['compile_command'] = ['true']
        (self.source / 'config.yaml').write_text(yaml.safe_dump(config))
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        self.assertEqual(result['status'], 'native_report_failed_or_missing')

    def test_later_phase_cannot_overwrite_retained_correctness_report(self):
        self.task()
        config = yaml.safe_load((self.source / 'config.yaml').read_text())
        config['performance_command'].append("printf '%s' '{\"status\":\"changed\"}' > build/correctness_report.json")
        (self.source / 'config.yaml').write_text(yaml.safe_dump(config))
        result = verifier.verify_task(self.source, self.workspace, self.repo)
        retained = self.workspace / result['phases'][1]['native_report']['path']
        self.assertEqual(json.loads(retained.read_text())['status'], 'ok')
        self.assertEqual(json.loads((self.workspace / 'build/correctness_report.json').read_text())['status'], 'changed')

    def test_cohort_creates_unique_runs_and_propagates_failure(self):
        self.task(failure='correctness')
        config = self.repo / 'run.yaml'
        config.write_text(yaml.safe_dump({'target_gpu_model': 'MI355X', 'tasks': ['head_kernels/test']}))
        task_config = yaml.safe_load((self.source / 'config.yaml').read_text())
        task_config['headkernel'] = {'docker': 'example.invalid/runtime:v1'}
        (self.source / 'config.yaml').write_text(yaml.safe_dump(task_config))
        environment = {'AGENT_KERNEL_ARENA_DOCKER': '1',
                       'AGENT_KERNEL_ARENA_DOCKER_IMAGE': 'example.invalid/runtime:v1',
                       'AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID': 'sha256:' + 'a' * 64,
                       'AGENT_KERNEL_ARENA_HEAD_KERNEL_VALIDATION_RUNTIME': ''}
        with mock.patch.dict(os.environ, environment):
            code, first = verifier.verify(config, self.repo)
            second_code, second = verifier.verify(config, self.repo)
        self.assertEqual((code, second_code), (1, 1))
        self.assertNotEqual(first, second)
        report = json.loads(first.read_text())
        self.assertEqual(report['status'], 'direct_verification_failed')
        self.assertEqual(report['runtime_identity']['AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID'], 'sha256:' + 'a' * 64)

    def test_host_execution_is_refused(self):
        config = ROOT / 'example_configs/top5_validator_glm_bf16_public_mi355x.yaml'
        with mock.patch.dict(os.environ, {'AGENT_KERNEL_ARENA_DOCKER': ''}):
            with self.assertRaisesRegex(ValueError, 'Docker runner'):
                verifier.verify(config)

    def test_verify_launcher_preserves_public_runtime_and_exit_status(self):
        config = ROOT / 'example_configs/top5_validator_glm_bf16_public_mi355x.yaml'
        with mock.patch.dict(os.environ, {'AKA_DOCKER_IMAGE': '', 'AKA_EXPECTED_IMAGE_ID': '',
                                         'AKA_HEAD_KERNEL_VALIDATION_RUNTIME': ''}), \
                mock.patch.object(launcher.subprocess, 'run') as run:
            run.return_value.returncode = 23
            self.assertEqual(launcher.main(['verify', '--config', str(config)]), 23)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ['bash', 'src/scripts/docker_benchmark.sh', 'verify'])
        self.assertEqual(run.call_args.kwargs['env']['AKA_HEAD_KERNEL_VALIDATION_RUNTIME'],
                         'public_hyperloom_rocm720')


if __name__ == '__main__':
    unittest.main()

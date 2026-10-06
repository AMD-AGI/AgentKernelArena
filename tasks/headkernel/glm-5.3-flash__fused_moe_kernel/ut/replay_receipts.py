"""Durable comparison context; reset and verification receipts are outside timing."""
import functools
import json
import os
from pathlib import Path
import time
from failure_bundle import json_sha, save_cpu_failure


class ReplayReceipts:
    def __init__(self, directory, challenge, source_sha256, *, request=None, provenance=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('native_production_events-' + str(time.time_ns()) + '.jsonl')
        self.path.touch(exist_ok=False)
        self.challenge = challenge
        self.source_sha256 = source_sha256
        self.request = json.loads(json.dumps(request, allow_nan=False))
        self.provenance = json.loads(json.dumps(provenance, allow_nan=False))
        self.failure_attempted = False
        self.emit('comparison_begin', challenge_seed=challenge, source_sha256=source_sha256,
                  enclosing_request=self.request, enclosing_request_sha256=json_sha(self.request))

    def emit(self, event, **fields):
        row = {'event': event, **fields}
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if event.endswith('failure'):
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    def leg(self, case_id, label, reset, verify, *, case=None):
        iteration = -1

        def reset_with_receipt(seed):
            nonlocal iteration
            iteration = seed - self.challenge
            self.emit('replay_preparation', case_id=case_id, leg=label, seed=seed, iteration=iteration)
            return reset(seed)

        @functools.wraps(verify)
        def verify_with_receipt(truth):
            context = {'case_id': case_id, 'leg': label, 'seed': truth['seed'], 'iteration': iteration,
                       'stage': 'eager' if iteration < 0 else 'graph_replay'}
            self.emit('verify_begin', **context)
            try:
                result = verify(truth)
            except BaseException as error:
                # Persist the original failure before potentially large tensor I/O.
                try:
                    self.emit('verify_failure', **context, error_type=type(error).__name__, error=str(error))
                except BaseException as receipt_error:
                    error.add_note('Failure receipt persistence failed: '+str(receipt_error))
                artifact = {'status': 'not_an_assertion'}
                if isinstance(error, AssertionError):
                    artifact = {'status': 'already_attempted'}
                    if not self.failure_attempted:
                        self.failure_attempted = True
                        try:
                            artifact = save_cpu_failure(self.path.parent, self.path.stem, verify, truth, error,
                                context={**context, 'native_comparison_challenge_seed': self.challenge,
                                         'source_sha256': self.source_sha256},
                                request=self.request, provenance=self.provenance, case=case)
                        except BaseException as capture_error:
                            artifact = {'status': 'persistence_error', 'error_type': type(capture_error).__name__,
                                        'error': str(capture_error)}
                            error.add_note('Failure artifact persistence failed: '+str(capture_error))
                try:
                    self.emit('verify_failure_artifact', **context, failure_artifact=artifact)
                except BaseException as receipt_error:
                    error.add_note('Failure receipt persistence failed: '+str(receipt_error))
                raise
            self.emit('verify_pass', **context)
            return result

        return reset_with_receipt, verify_with_receipt

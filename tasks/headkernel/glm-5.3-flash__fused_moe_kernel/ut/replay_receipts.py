"""Durable comparison context; reset and verification receipts are outside timing."""
import functools
import json
import os
from pathlib import Path
import time


class ReplayReceipts:
    def __init__(self, directory, challenge, source_sha256):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ('native_production_events-' + str(time.time_ns()) + '.jsonl')
        self.path.touch(exist_ok=False)
        self.challenge = challenge
        self.emit('comparison_begin', challenge_seed=challenge, source_sha256=source_sha256)

    def emit(self, event, **fields):
        row = {'event': event, **fields}
        with self.path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if event.endswith('failure'):
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    def leg(self, case_id, label, reset, verify):
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
                self.emit('verify_failure', **context, error_type=type(error).__name__, error=str(error))
                raise
            self.emit('verify_pass', **context)
            return result

        return reset_with_receipt, verify_with_receipt

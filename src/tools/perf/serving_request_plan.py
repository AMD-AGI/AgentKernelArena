"""Report the native text client's actual requests before its timed HTTP loop.

The generator owns tokenization and random draws. Nominal ISL is not necessarily
the realized token length after decoding and re-encoding random token IDs.
"""
import hashlib
import json
import os


def encode_plan(requests):
    if not isinstance(requests, list) or not requests or len(requests) > 100000:
        raise ValueError('Invalid native request plan size')
    rows = []
    for item in requests:
        if (not isinstance(item, (list, tuple)) or len(item) != 4
                or not isinstance(item[0], str) or item[3] is not None
                or any(type(value) is not int or value <= 0 for value in item[1:3])):
            raise ValueError('Invalid native text request')
        rows.append(dict(prompt_sha256=hashlib.sha256(item[0].encode()).hexdigest(),
                         input_tokens=item[1], output_tokens=item[2]))
    return json.dumps(dict(version=1, requests=rows), sort_keys=True, separators=(',', ':')).encode()


def emit_request_plan(requests):
    payload = encode_plan(requests)
    descriptor = int(os.environ['AKA_REQUEST_PLAN_FD'])
    # The trusted parent owns this anonymous file, outside the candidate tree.
    # A regular file avoids pipe-capacity deadlocks with large request sets.
    with os.fdopen(os.dup(descriptor), 'wb') as stream:
        stream.write(payload)
    if os.environ.get('AKA_REQUEST_PLAN_MODE') == 'prepare':
        raise SystemExit(0)

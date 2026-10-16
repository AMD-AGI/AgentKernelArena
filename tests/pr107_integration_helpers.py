"""Exact normalization of separately tested PR105 additions for old fingerprints.

Original rows, kernels and benchmark bodies retain their historical hashes.
The complete new dispatcher/control path is exercised by test_pr107_main_integration.
"""
import copy

DISPATCH = ('    if case_index is not None and case_index >= 10000:\n'
            '        from _upstream_controls import run_control\n'
            '        return run_control(case_index - 10000, load_module)\n')


# Additive correctness-only domain controls (protected `_domain_controls.py`).
DOMAIN_DISPATCH = ('    if case_index is not None and case_index >= 20000:\n'
                   '        from _domain_controls import run_control\n'
                   '        return run_control(case_index - 20000, load_module)\n')


def original_runner(source):
    assert source.count(DISPATCH) <= 1
    assert source.count(DOMAIN_DISPATCH) <= 1
    return source.replace(DOMAIN_DISPATCH, '', 1).replace(DISPATCH, '', 1)


def original_manifest(data):
    data = copy.deepcopy(data)
    domain = [r for r in data['cases'] if r['params'].get('case_index', -1) >= 20000]
    if domain:
        assert [r['params']['case_index'] for r in domain] == list(range(20000, 20000+len(domain)))
        assert all(r['checks'] == ['correctness'] for r in domain)
        assert len(data['domain_controls']) == len(domain)
        data['cases'] = [r for r in data['cases'] if r not in domain]
        del data['domain_controls']
    else:
        assert 'domain_controls' not in data
    extra = [r for r in data['cases'] if r['params'].get('case_index', -1) >= 10000]
    if extra:
        assert [r['params']['case_index'] for r in extra] == list(range(10000, 10000+len(extra)))
        assert all(r['checks'] == ['correctness'] for r in extra)
        assert len(data['upstream_controls']) == len(extra)
        data['cases'] = [r for r in data['cases'] if r not in extra]
        del data['upstream_controls']
    else:
        assert 'upstream_controls' not in data
    return data

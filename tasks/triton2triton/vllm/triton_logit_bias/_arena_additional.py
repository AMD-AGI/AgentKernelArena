"""Additional scored branch workloads; original scored rows remain unchanged."""
from _arena_replay import clone, restore, to_device


def install(harness, contract):
    original_correctness = harness.run_correctness
    original_performance = harness.run_performance

    def correctness(*, case_index=None):
        extra_index = None if case_index is None else case_index - contract.CONTROL_INDEX - 1
        if extra_index is not None and 0 <= extra_index < len(contract.SCORED_CASE_IDS):
            try:
                _, args = list(contract.scored_inputs(harness))[extra_index]
                function = getattr(harness.load_module(), contract.FUNCTION)
                function(*to_device(args, 'cuda'))
                return True, None
            except Exception as exc:
                return False, f'{type(exc).__name__}: {exc}'
        result = original_correctness(case_index=case_index)
        if case_index is None and result[0]:
            for index in range(len(contract.SCORED_CASE_IDS)):
                result = correctness(case_index=contract.CONTROL_INDEX + 1 + index)
                if not result[0]:
                    return result
        return result

    def performance():
        records = original_performance()
        module = harness.load_module()
        function = getattr(module, contract.FUNCTION)
        for case_id, cpu_args in contract.scored_inputs(harness):
            try:
                args = to_device(cpu_args, 'cuda')
                mutable = tuple(args[index] for index in contract.MUTABLE)
                saved = clone(mutable)
                def prepare():
                    restore(mutable, saved)
                if hasattr(contract, 'scored_launch'):
                    launch = contract.scored_launch(harness, module, args)
                else:
                    def launch():
                        return function(*args)
                milliseconds, metadata = harness._benchmark_cuda_graph_or_events(
                    launch, warmup=harness.WARMUP_ITERATIONS,
                    repetition=harness.BENCHMARK_ITERATIONS,
                    target_ms=contract.SCORED_TARGET_MS, prepare_fn=prepare)
                records.append({'test_case_id':case_id, 'execution_time_ms':milliseconds,
                                **metadata, 'params':{'coverage':'additional operator branch workload'}})
            except Exception as exc:
                records.append({'test_case_id':case_id, 'execution_time_ms':-1.,
                                'error':f'{type(exc).__name__}: {exc}'})
        return records

    harness.run_correctness = correctness
    harness.run_performance = performance

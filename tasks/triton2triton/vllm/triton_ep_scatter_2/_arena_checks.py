"""Full scatter routing/state checks and exact prepared benchmark replay."""
from contextlib import contextmanager
import inspect


def snapshot(values):
    return tuple(v.clone() for v in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('EP scatter 2 modified a read-only input')


def counts_and_starts(routes, experts):
    import torch
    counts = torch.bincount(routes[routes >= 0].long(),minlength=experts).to(torch.int32)
    aligned = ((counts+127)//128)*128
    starts = aligned.cumsum(0)-aligned
    return counts, starts.to(torch.int32), int(aligned.sum())


def check_outputs(outputs, pristine):
    import torch
    x, routes, starts, before_output, before_index = pristine
    counters, output, index = outputs
    for value, original in zip(outputs, (starts,before_output,before_index)):
        if not isinstance(value,torch.Tensor) or (value.shape != original.shape or
                value.dtype != original.dtype or value.device != original.device):
            raise AssertionError('EP scatter 2 output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('EP scatter 2 output must remain finite')
    counts, _, _ = counts_and_starts(routes, len(starts))
    if not torch.equal(counters,starts+counts):
        raise AssertionError('EP scatter 2 final expert counters mismatch')
    valid = routes >= 0
    if not torch.equal(index[~valid],before_index[~valid]):
        raise AssertionError('EP scatter 2 wrote an inactive route')
    slots = index[valid].long()
    experts = routes[valid].long()
    if not ((slots >= starts[experts]) & (slots < starts[experts]+counts[experts])).all():
        raise AssertionError('EP scatter 2 route is outside its expert region')
    if not ((slots >= 0) & (slots < len(output))).all() or slots.unique().numel() != slots.numel():
        raise AssertionError('EP scatter 2 slots must be unique and in range')
    tokens = torch.arange(x.shape[0],device=x.device)[:,None].expand_as(routes)[valid]
    # Keep the original torch.allclose atol=1e-5 and default rtol=1e-5;
    # extend its source-copy check from 16 tokens to every active assignment.
    torch.testing.assert_close(output[slots],x[tokens],atol=1e-5,rtol=1e-5)
    written = torch.zeros(len(output),device=output.device,dtype=torch.bool)
    written[slots] = True
    if not torch.equal(output[~written],before_output[~written]):
        raise AssertionError('EP scatter 2 modified untouched output padding')


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.ep_scatter_2
        patched.append((module,original))
        diagnosed = False

        def verify(x, routes, counters, output, index):
            inputs = (x,routes)
            pristine = snapshot((x,routes,counters,output,index))
            try:
                result = original(x,routes,counters,output,index)
                unchanged(inputs,pristine[:2])
                check_outputs((counters,output,index),pristine)
                return result
            finally:
                for value,saved in zip(inputs,pristine[:2]):
                    value.copy_(saved)

        def diagnostic(x, routes, row_stride=False):
            import torch
            _, starts, total = counts_and_starts(routes,5 if row_stride else 2)
            step = 2 if row_stride else 1
            output = torch.zeros((total*step,x.shape[1]),device=x.device,dtype=x.dtype)[::step]
            index = torch.full((len(x)*step,routes.shape[1]),-1,device=x.device,dtype=routes.dtype)[::step]
            verify(x,routes,starts,output,index)

        def checked(x, routes, counters, output, index):
            nonlocal diagnosed
            import torch
            result = verify(x,routes,counters,output,index)
            if not diagnosed:
                values = torch.arange(34*515,device=x.device).reshape(34,515)
                dx = ((values%127)/16+.25).to(x.dtype)[::2]
                dr = (torch.arange(34*3,device=x.device).reshape(34,3)%5).to(routes.dtype)[::2]
                dr[::3,1] = -1
                dr[0,:2] = 0  # Duplicate routes still need distinct slots.
                dr[-1,:] = -1
                diagnostic(dx,dr,row_stride=True)
                # Exercise the public 8192-program loop with a final token.
                values = torch.arange(8193,device=x.device)
                diagnostic(((values%127)/16+.25).to(x.dtype)[:,None],
                           (values%2).to(routes.dtype)[:,None])
                diagnosed = True
            return result

        module.ep_scatter_2 = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module,original in reversed(patched):
            module.ep_scatter_2 = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    x, routes = state['recv_x'], state['recv_topk']
    counters = state['expert_start_loc']
    initial_starts = state['initial_expert_start_loc'] if 'initial_expert_start_loc' in state else inspect.getclosurevars(options['prepare_fn']).nonlocals['initial_expert_start_loc']
    output, index = state['output_tensor'], state['output_index']
    buffers = (x,routes,counters,output,index,initial_starts)
    pristine = snapshot(buffers)
    outputs = (counters,output,index)

    def measured():
        fn()
        return outputs

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured,timed_run=timed,**options)
        unchanged((x,routes,initial_starts),(pristine[0],pristine[1],pristine[5]))
        check_outputs(timed.outputs,pristine[:5])
        x.mul_(-.5).add_(.25)
        routes.copy_((pristine[1]+1)%len(counters))
        routes[pristine[1]<0] = pristine[1][pristine[1]<0]
        counts, starts, total = counts_and_starts(routes,len(counters))
        if total != len(output):
            raise AssertionError('Perturbed routes changed allocated capacity')
        initial_starts.copy_(starts)
        output.copy_(pristine[3])
        index.copy_(pristine[4])
        replay_pristine = snapshot((x,routes,initial_starts,output,index))
        counters.fill_(-777)
        for start,count in zip(starts.tolist(),counts.tolist()):
            output[start:start+count].fill_(float('nan'))
        index[routes>=0] = -777
        # The original prepare_fn must still run: it restores counters from
        # the same initial_starts buffer before the exact graph/event replay.
        replayed = timed.rerun()
        unchanged((x,routes,initial_starts),(replay_pristine[0],replay_pristine[1],replay_pristine[2]))
        check_outputs(replayed,replay_pristine)
        return ms, {**metadata,'timed_output_checked':True,
                    'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        for value,saved in zip(buffers,pristine):
            value.copy_(saved)


def install(harness):
    correctness, performance = harness.run_correctness, harness.run_performance

    def checked_correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness(*args, **kwargs)

    def checked_performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness,benchmark,fn,**kwargs)
        try:
            return performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness, harness.run_performance = checked_correctness,checked_performance

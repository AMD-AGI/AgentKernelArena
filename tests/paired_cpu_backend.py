"""Deterministic CPU callbacks for graph lifecycle tests; no GPU claims."""
from contextlib import contextmanager
from copy import deepcopy
import json
import hashlib
import struct
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/headkernel/minimax-m3__decode_score_kernel'
sys.path.insert(0, str(TASK/'ut'))
from evaluation_contract import checked_replays, fingerprint
import paired_reference
from workload_controls import RecordedControls, flat


class Storage:
    def __init__(self, role, events, value=None):
        self.role, self.events, self.value = role, events, value
        self.scrubbed = False

    def data_ptr(self): return id(self)
    def nbytes(self): return 16
    def fill_(self, value):
        self.value = value
        self.scrubbed = True
        self.events.append(self.role+'_clear')


class Tensor:
    def __init__(self, role, events): self.storage = Storage(role, events)
    def untyped_storage(self): return self.storage


def leaves(value):
    if isinstance(value, Tensor): yield value
    elif isinstance(value, dict):
        for item in value.values(): yield from leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value: yield from leaves(item)


class FakeCuda:
    def __init__(self, events):
        self.events, self.capture, self.last_role, self.next_stream = events, None, None, 0
        self.current = self.Stream()

    def Stream(self):
        number = self.next_stream; self.next_stream += 1
        return SimpleNamespace(number=number, wait_stream=lambda other:None, synchronize=lambda:None)

    def current_stream(self): return self.current
    def synchronize(self): self.events.append('sync')

    @contextmanager
    def stream(self, stream):
        before = self.current; self.current = stream
        try: yield
        finally: self.current = before

    @contextmanager
    def graph(self, graph, *, stream):
        self.capture = graph
        with self.stream(stream):
            try: yield
            finally: self.capture = None

    def CUDAGraph(self):
        cuda = self
        class Graph:
            fn = None
            role = None
            def replay(self):
                cuda.last_role = self.role
                cuda.events.append(self.role+'_replay')
                self.fn()
        return Graph()

    def Event(self, **kwargs):
        cuda = self
        return SimpleNamespace(record=lambda:cuda.events.append('event'), synchronize=lambda:None,
            elapsed_time=lambda stop:2.0 if cuda.last_role == 'candidate' else 3.0)


def run_cpu_pair(seed=91, failure=None, *, task=TASK, case=None, signatures=None, source_override=None):
    manifest = json.loads((task/'cases.json').read_text())
    if case is None:
        case = manifest['cases'][0]
        manifest['cases'] = [case]
        manifest['capture']['required_case_ids'] = [case['case_id']]
        manifest['capture']['target_calls'] = manifest['capture']['represented_calls'] = case['occurrences']
    policy = manifest['measurement']
    distribution = RecordedControls(case)
    events = []; cuda = FakeCuda(events)
    inputs = {role:{'value':Tensor(role+'_input',events)} for role in ('candidate','reference')}
    outputs = {role:Tensor(role,events) for role in inputs}
    setup = {role:[] for role in inputs}; selected=[]; signature={}; realized=[]
    def role_of(args): return next(role for role, values in inputs.items() if values is args)
    def restore(args, truth):
        role=role_of(args); events.append(role+'_restore')
        if failure == 'restore' and role == 'reference' and selected: raise ValueError('CPU restore failure')
        args['value'].storage.value=truth
    def invoke(role, args):
        setup[role].append({'stream':cuda.current.number,'capture':cuda.capture is not None})
        def execute():
            outputs[role].storage.value=args['value'].storage.value*2
            outputs[role].storage.scrubbed=False
            if failure=='reference_input' and role=='reference' and selected:
                args['value'].storage.value=-123
        if cuda.capture is not None:
            cuda.capture.fn=execute;cuda.capture.role=role
        else:
            events.append(role+'_setup_replay');execute()
        return outputs[role]
    def snapshot(output):
        events.append(output.storage.role+'_snapshot')
        return output.storage.value
    def immutable(args, truth):
        role=role_of(args);events.append(role+'_immutable')
        if failure == 'reference_setup' and role == 'reference' and not selected:
            raise ValueError('CPU reference setup immutable failure')
        if failure=='candidate_input' and role=='candidate' and selected:
            raise ValueError('CPU candidate immutable failure')
        if args['value'].storage.value != truth: raise ValueError('CPU input mutation')
    def initialize(output):
        events.append(output.storage.role+'_initialize');output.storage.value=None
    def compare(actual, expected, args, tolerance, *, expected_inputs):
        events.append('independent_math_and_compare')
        assert outputs['reference'].storage.scrubbed
        if failure=='compare': raise ValueError('CPU comparison failure')
        assert actual == expected == expected_inputs*2
    def reset(value):
        assert outputs['reference'].storage.scrubbed
        events.append('prepare');inputs['candidate']['value'].storage.value=value
        selected.append(distribution.choose(value))
        signature.clear()
        if signatures is not None:
            signature.update(deepcopy(signatures[len(selected)-1]))
        else:
            addresses={}
            for name in {'seq_lens','req_to_token','slot_ids','topk_idx'} & set(case['tensors']):
                tensor=case['tensors'][name]
                digest=hashlib.sha256(('CPU_PLACEHOLDER:'+name+':'+str(value)).encode()).hexdigest()
                if name=='seq_lens':
                    values=flat(distribution.controls(selected[-1])['inputs.seq_lens'])
                    code={'int32':'i','int64':'q'}[tensor['dtype']]
                    digest=hashlib.sha256(struct.pack('<'+code*len(values),*values)).hexdigest()
                addresses[name]={'dtype':'torch.'+tensor['dtype'],'shape':tensor['shape'],'sha256':digest}
            sizes={}
            for name,alias in case['input_aliases'].items():
                sizes[alias]=max(sizes.get(alias,0),case['original_storage_nbytes'][name])
            signature.update(input_seed=value,variant_id=selected[-1],actual_tensor_controls_sha256=selected[-1],
                             addressing=addresses,private_cpu_storage_bytes=sum(sizes.values()))
        realized.append(deepcopy(signature))
        return value
    api={'leaves':leaves,'raw_storage':lambda t:t.storage,'restore_storages':restore,
         'invoke':lambda fn,args,output_owner:invoke(fn,args),'assert_immutable_inputs':immutable,
         'cpu_clone':snapshot,'initialize_output':initialize,'runtime_abi':lambda args,out:({},{}),
         'observe_case':lambda case,tensors,scalars:case,'compare_native_outputs':compare,
         'checked_replays':checked_replays,'tolerance':.02}
    source=source_override or {'source/flash_with_topk_idx.py':'a'*64}
    identities={role:{'source_sha256':source,'module':'CPU_ONLY_'+role,'leg':role,
                      'gpu_binding':'private_triton_code_object_source_checked_v1'} for role in inputs}
    request={'schema_version':1,'request_id':'CPU_ONLY_'+str(seed),'phase':'performance',
             'manifest_sha256':fingerprint(manifest),'source_sha256':source,'package_sha256':'b'*64,'challenge_seed':seed}
    try:
        with patch.dict(sys.modules, {'torch':SimpleNamespace(cuda=cuda)}):
            row=paired_reference.paired_performance(api,case,manifest,request,
                inputs['candidate'],inputs['reference'],'candidate','reference',7,None,
                identities['candidate'],identities['reference'],reset,current_input_signature=lambda:dict(signature))
        row['workload_control_sampling']=distribution.sampling(seed,selected,policy,row['samples_ms'])
        row['realized_input_signatures']=realized
        return SimpleNamespace(row=row,manifest=manifest,request=request,events=events,setup=setup,
                               outputs=outputs,identities=identities)
    except BaseException:
        assert outputs['reference'].storage.scrubbed
        raise


def run_real_operator_failure(task, fail_at=1, failure_kind="launch_count", *,
                              cleanup_failure=False, reference=True):
    """Exercise the real Operator and adapter with CPU tensors and fake CUDA."""
    import importlib.util
    import torch
    modules = {}
    # Module names are restored after the test so decode and prefill stay isolated.
    with patch.dict(sys.modules):
        for name in ("evaluation_contract", "served_contract", "source_guard", "minimax_data",
                     "minimax_native", "paired_reference", "minimax_paired"):
            spec = importlib.util.spec_from_file_location(name, task/"ut"/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        native, paired, adapter = (modules[name] for name in
                                  ("minimax_native", "paired_reference", "minimax_paired"))
        created, returned, events = [], [], []
        cuda = FakeCuda(events)
        operator = native.Operator.__new__(native.Operator)
        probe = SimpleNamespace(name="cpu_kernel", launches=[])
        operator.probes = [probe]
        expected = {"grid":[1], "constexpr":{}, "compiled":{"num_warps":4, "num_stages":1}}
        operator.expected_launches = {"cpu_kernel":expected} if failure_kind == "launch_contract" else None

        def wrapper(**kwargs):
            index = len(created)+1
            storage = torch.full((96,), 30+index, dtype=torch.uint8)
            output = (storage[8:16], storage[32:48], None)
            created.append(storage)
            returned.append(output)
            events.append("wrapper_return_"+str(index))
            if index != fail_at or failure_kind == "launch_contract":
                probe.launches.append({"grid":[1], "constexpr":{}, "num_warps":
                    8 if index == fail_at and failure_kind == "launch_contract" else 4,
                    "num_stages":1})
            return output

        operator.function = wrapper
        reference_operator = operator if reference else object()
        clear_pointers = []

        def raw_storage(value):
            storage = adapter.raw_storage(value)
            clear_pointers.append(value.untyped_storage().data_ptr())
            events.append("storage_clear")
            if cleanup_failure and len(created) == fail_at:
                class FailingClear:
                    def fill_(self, fill):
                        storage.fill_(fill)
                        raise LookupError("injected cleanup error after storage scrub")
                return FailingClear()
            return storage

        api = {"restore_storages":lambda *args:None,
               "invoke":lambda fn,args,owner:adapter.invoke_with_reference_output_owner(
                   fn,args,owner,reference_operator),
               "cpu_clone":lambda output:output,
               "assert_immutable_inputs":lambda *args:None,
               "runtime_abi":lambda *args:({},{}), "observe_case":lambda *args:None,
               "leaves":adapter.leaves, "raw_storage":raw_storage}
        with patch.object(torch, "cuda", cuda):
            try:
                paired.capture_native_graph(api, {}, operator, {}, None)
            except BaseException as error:
                expected_message = ("wrapper did not execute exactly one declared GPU kernel"
                    if failure_kind == "launch_count" else
                    "actual selected native launch differs from its captured case contract")
                assert type(error) is (RuntimeError if failure_kind == "launch_count" else ValueError)
                assert str(error) == expected_message
                traceback = error.__traceback__
                retained = False
                while traceback:
                    if traceback.tb_frame.f_code is native.Operator.__call__.__code__:
                        retained = traceback.tb_frame.f_locals.get("result") is returned[-1]
                    traceback = traceback.tb_next
                assert retained
                assert len(created) == fail_at
                scrubbed = [torch.equal(storage, torch.full_like(storage, 0xAA)) for storage in created]
                if reference:
                    assert all(scrubbed)
                    failure_events = events[events.index("wrapper_return_"+str(fail_at))+1:]
                    assert failure_events.index("sync") < failure_events.index("storage_clear")
                else:
                    assert not scrubbed[-1]
                if cleanup_failure:
                    assert any("injected cleanup error" in note for note in error.__notes__)
                return {"failure_invocation":fail_at, "failure_kind":failure_kind,
                        "reference_only":reference, "new_storage_each_invocation":True,
                        "all_reference_storage_bytes_scrubbed":all(scrubbed),
                        "exception_type":type(error).__name__, "exception":str(error),
                        "original_attestation_exception_preserved":True,
                        "wrapper_writes_synchronized_before_failure_scrub":reference,
                        "result_still_retained_in_traceback":retained,
                        "cleanup_failure_injected":cleanup_failure, "GPU_actions":False}
            raise AssertionError("Expected the real post-call launch attestation to fail")

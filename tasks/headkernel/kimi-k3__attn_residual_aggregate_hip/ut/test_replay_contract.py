"""Synthetic CPU regressions for protected replay and coalesced oracle snapshots."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import fresh_runner as RUNNER
ROOT=Path(__file__).resolve().parent

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

class Tensor:
    def __init__(self, value, log, name, device='cuda'):
        self.value, self.log, self.name = value, log, name
        self.device = SimpleNamespace(type=device)
    def detach(self): return self
    def untyped_storage(self):
        return SimpleNamespace(tensor=self,nbytes=lambda:8,data_ptr=lambda:id(self))
    def to(self, *, device, copy):
        assert copy is True
        self.log.append(('snapshot', self.name, device))
        return Tensor(self.value, self.log, self.name, device)

class FreshChecks(unittest.TestCase):
    def make(self, kind='valid', mutate_during_reference=False):
        log=[]
        x=Tensor(0,log,'input'); y=Tensor(0,log,'output')
        def refresh(seed): log.append('refresh'); x.value=seed
        def initialize(): log.append('poison'); y.value=-999
        def immutable(after,before):
            self.assertEqual(after['x'].device.type,'cpu')
            self.assertEqual(before['x'].device.type,'cpu')
            if after['x'].value != before['x'].value: raise AssertionError('input modified')
        def replay():
            log.append('replay')
            if kind != 'noop': y.value=x.value*2
            if kind == 'bad_input': x.value=-1
        def reference(before):
            log.append('reference')
            self.assertEqual(before['x'].device.type,'cpu')
            self.assertIn(('snapshot','output','cpu'),log)
            if mutate_during_reference: x.value,y.value=999,999
            return {'y':Tensor(before['x'].value*2,log,'expected')}
        def compare(actual,expected):
            log.append('compare')
            self.assertEqual(actual['y'].device.type,'cpu')
            self.assertEqual(expected['y'].device.type,'cpu')
            if actual['y'].value != expected['y'].value: raise AssertionError('wrong output')
        def event(*,enable_timing):
            self.assertTrue(enable_timing)
            return SimpleNamespace(record=lambda:log.append('event'),
                synchronize=lambda:log.append('event_sync'), elapsed_time=lambda stop:0.25)
        class Raw:
            def set_(self,storage,offset,shape,stride):
                self.tensor=storage.tensor
                return self
            def zero_(self):
                self.tensor.value=0
                log.append('clear_reference')
        torch=SimpleNamespace(uint8='uint8',empty=lambda *args,**kwargs:Raw(),
            is_tensor=lambda value:isinstance(value,Tensor),
            cuda=SimpleNamespace(synchronize=lambda:log.append('sync'),Event=event))
        callbacks=RUNNER.FreshCallbacks(refresh_inputs=refresh,initialize_outputs=initialize,
            snapshot_inputs=lambda:{'x':x},snapshot_outputs=lambda:{'y':y},
            validate_metadata=lambda:log.append('metadata'),assert_immutable=immutable,
            reference=reference,compare=compare,replay=replay,torch_module=torch)
        return callbacks,log,x,y

    def test_snapshot_order_and_cpu_truth_survive_later_gpu_changes(self):
        callbacks,log,x,y=self.make(mutate_during_reference=True)
        truth=callbacks.reset_inputs(7);callbacks.initialize_outputs()
        callbacks.replay();callbacks.verify(truth)
        self.assertEqual(truth.value['x'].value,7)
        self.assertLess(log.index('sync'),log.index(('snapshot','output','cpu')))
        self.assertLess(log.index(('snapshot','output','cpu')),log.index('reference'))
        self.assertLess(max(i for i,v in enumerate(log[:log.index('reference')])
                            if v==('snapshot','input','cpu')),log.index('reference'))
        self.assertEqual(x.value,999)
        self.assertIn('clear_reference',log)
        self.assertLess(log.index('reference'),log.index('clear_reference'))
        self.assertLess(log.index('clear_reference'),log.index('compare'))

    def test_input_corruption_rejected_before_reference(self):
        callbacks,log,_,_=self.make(kind='bad_input')
        truth=callbacks.reset_inputs(7);callbacks.initialize_outputs();callbacks.replay()
        with self.assertRaisesRegex(AssertionError,'input modified'): callbacks.verify(truth)
        self.assertNotIn('reference',log)

    def test_noop_replay_and_stale_tokens_fail(self):
        callbacks,_,_,_=self.make(kind='noop')
        truth=callbacks.reset_inputs(7);callbacks.initialize_outputs();callbacks.replay()
        with self.assertRaisesRegex(AssertionError,'wrong output'): callbacks.verify(truth)
        stale=truth;callbacks.reset_inputs(8)
        with self.assertRaisesRegex(AssertionError,'Stale'):callbacks.verify(stale)

    def test_only_graph_replay_is_inside_event_interval(self):
        callbacks,log,_,_=self.make()
        truth=callbacks.reset_inputs(7);callbacks.initialize_outputs()
        self.assertEqual(callbacks.measure(callbacks.replay),0.25)
        callbacks.verify(truth)
        first=log.index('event')
        self.assertEqual(log[first:first+4],['event','replay','event','event_sync'])
        self.assertLess(log.index('event_sync'),log.index('reference'))

    def test_shared_callback_api_stays_unchanged(self):
        contract=module(ROOT/'evaluation_contract.py','evaluation_contract')
        callbacks,log,_,_=self.make()
        case={'case_id':'cpu-probe'}
        policy={'method':'cuda_graph','warmup_iterations':2,'benchmark_iterations':3}
        row=callbacks.performance_row(case,policy,observe=lambda:case,challenge_seed=11)
        self.assertEqual(row['samples_ms'],[0.25]*3)
        self.assertEqual(log.count('refresh'),5)
        self.assertEqual(log.count('replay'),5)
        self.assertEqual(log.count('reference'),5)

class CoalescedTests(unittest.TestCase):
    def test_owned_outputs_preserve_all_checks_and_100_samples(self):
        callbacks,log,x,y=FreshChecks().make()
        counts={'snapshot':0,'reference':0}
        def snapshot():
            counts['snapshot']+=1
            return RUNNER.cpu_copy({'y':y},callbacks.torch),RUNNER.cpu_copy({'x':x},callbacks.torch)
        previous=callbacks._reference
        def reference(truth):
            counts['reference']+=1
            gpu=previous(truth);cpu=RUNNER.cpu_copy(gpu,callbacks.torch)
            RUNNER.clear_device_reference(gpu,callbacks.torch)
            return RUNNER.OwnedCPUOutputs(cpu)
        callbacks._candidate_snapshot=snapshot;callbacks._reference=reference
        case={'case_id':'synthetic-owned-output'}
        row=callbacks.performance_row(case,{'method':'cuda_graph','warmup_iterations':10,'benchmark_iterations':100},observe=lambda:case,challenge_seed=13)
        self.assertEqual(row['samples_ms'],[0.25]*100)
        self.assertEqual(counts,{'snapshot':110,'reference':110})
        self.assertEqual(log.count('refresh'),110);self.assertEqual(log.count('replay'),110)
        self.assertEqual(log.count('clear_reference'),110)

if __name__=='__main__':unittest.main()

"""Synthetic CPU regressions; never deserialize captured model tensors."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fixture_cache import CpuFixtureCache
from fresh_runner import FreshCallbacks
from runtime_adapter import Prepared


class ByteView:
    def __init__(self,data,start=0,stop=None):self.data=data;self.start=start;self.stop=len(data) if stop is None else stop
    def __getitem__(self,index):return ByteView(self.data,self.start+index.start,self.start+index.stop)
    def copy_(self,source):self.data[self.start:self.stop]=source


class Tensor:
    def __init__(self,value,device='cuda',log=None,name='input'):
        self.value=value;self.device=SimpleNamespace(type=device);self.log=[] if log is None else log;self.name=name
    def detach(self):return self
    def to(self,*,device,copy):
        assert copy;self.log.append(('copy',self.name,self.device.type,device))
        return Tensor(self.value,device,self.log,self.name)
    def copy_(self,source):self.value=source.value
    def untyped_storage(self):return SimpleNamespace(tensor=self,nbytes=lambda:1,data_ptr=lambda:id(self))


class CacheTests(unittest.TestCase):
    def test_one_read_and_hash_with_independent_storage_copies(self):
        fixture=bytes(range(251))*7;digest=hashlib.sha256(fixture).hexdigest()
        fake_torch=SimpleNamespace(uint8='uint8',frombuffer=lambda data,**kwargs:data)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'input.bin';path.write_bytes(fixture);cache=CpuFixtureCache(root)
            first=bytearray(len(fixture));second=bytearray(len(fixture));read=Path.read_bytes;sha=hashlib.sha256
            with patch.object(Path,'read_bytes',autospec=True,side_effect=read) as reads,patch('fixture_cache.hashlib.sha256',wraps=sha) as hashes:
                cache.copy_into('input.bin',digest,len(fixture),ByteView(first),fake_torch)
                first[0]^=255
                cache.copy_into('input.bin',digest,len(fixture),ByteView(second),fake_torch)
                self.assertEqual(reads.call_count,1);self.assertEqual(hashes.call_count,1)
            self.assertEqual(second,fixture);self.assertNotEqual(first,second)
            self.assertIsInstance(cache.read_blob('input.bin',digest,len(fixture)),bytes)
            with self.assertRaises(ValueError):CpuFixtureCache(root).read_blob('input.bin','0'*64,len(fixture))
            with self.assertRaises(ValueError):CpuFixtureCache(root).read_blob('input.bin',digest,len(fixture)+1)

    def test_all_100_samples_keep_fresh_truth_oracles_and_timing_counts(self):
        log=[];counts=dict(refresh=0,initialize=0,replay=0,reference=0,measure=0)
        prepared=object.__new__(Prepared);pristine=Tensor(7,'cpu',log)
        prepared.pristine_groups={'s0':pristine};prepared.groups={'s0':Tensor(7,log=log)}
        output=Tensor(0,log=log,name='output');baseline=Tensor(0,log=log,name='reference_input')
        def refresh(seed):
            counts['refresh']+=1;Prepared.restore_original(prepared,prepared.groups)
            prepared.groups['s0'].value+=seed
        def initialize():counts['initialize']+=1;output.value=-999
        def replay():counts['replay']+=1;log.append('replay');output.value=prepared.groups['s0'].value*2
        def immutable(after,before):self.assertEqual(after['s0'].value,before['s0'].value)
        def reference(truth):
            counts['reference']+=1;log.append('reference')
            self.assertEqual(truth['s0'].device.type,'cpu');self.assertIsNot(truth['s0'],pristine)
            baseline.copy_(truth['s0']);expected=Tensor(baseline.value*2,log=log,name='golden')
            baseline.value=-1000
            self.assertEqual(truth['s0'].value,prepared.groups['s0'].value)
            return expected
        def compare(actual,expected):self.assertEqual(actual.value,expected.value)
        class Raw:
            def set_(self,storage,*args):self.tensor=storage.tensor;return self
            def zero_(self):self.tensor.value=0
        fake_torch=SimpleNamespace(uint8='uint8',empty=lambda *a,**k:Raw(),is_tensor=lambda x:isinstance(x,Tensor),
            cuda=SimpleNamespace(synchronize=lambda:log.append('sync')))
        callbacks=FreshCallbacks(refresh_inputs=refresh,initialize_outputs=initialize,
            snapshot_inputs=lambda:Prepared.snapshot_inputs(prepared),snapshot_outputs=lambda:output,
            validate_metadata=lambda:None,assert_immutable=immutable,reference=reference,compare=compare,
            replay=replay,torch_module=fake_torch)
        def measure(call):
            counts['measure']+=1;before=len(log);call();self.assertEqual(log[before:],['replay']);return 0.25
        callbacks.measure=measure
        result=callbacks.performance_row({'case_id':'synthetic'},dict(method='cuda_graph',warmup_iterations=10,benchmark_iterations=100),
            observe=lambda:{'case_id':'synthetic'},challenge_seed=101)
        self.assertEqual(counts,dict(refresh=110,initialize=110,replay=110,reference=110,measure=100))
        self.assertEqual(result['samples_ms'],[0.25]*100);self.assertEqual(pristine.value,7)
        self.assertEqual(log.count(('copy','input','cuda','cpu')),330)
        self.assertEqual(log.count(('copy','input','cpu','cpu')),110)
        for position,event in enumerate(log):
            if event=='reference':
                self.assertIn(('copy','output','cuda','cpu'),log[max(0,position-5):position])


if __name__=='__main__':unittest.main()

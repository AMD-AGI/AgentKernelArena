"""CPU checks for stage-1 routing, scale layout and sealed fixture contracts."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path=[p for p in sys.path if Path(p or ".").resolve()!=HERE]
import unittest
sys.path.insert(0,str(HERE))
import runtime_adapter as adapter
from evaluation_contract import fingerprint,validate_manifest


class Stage1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'cases.json').read_text())
        cls.external=json.loads((ROOT/'fixtures/EXTERNAL-MANIFEST.json').read_text())

    def test_all_three_structural_frequencies_are_counted_once(self):
        validate_manifest(self.manifest)
        self.assertEqual(len(self.manifest['cases']),3)
        self.assertEqual(sum(c['occurrences'] for c in self.manifest['cases']),777952)
        self.assertEqual([c['per_rank_counts']['0'] for c in self.manifest['cases']],[94208,184,2852])
        for case in self.manifest['cases']:
            histogram=case['work_distribution']['valid_rows_histogram']
            self.assertEqual(sum(count for _,count in histogram),case['occurrences'])
            self.assertEqual(sum(case['per_rank_counts'].values()),case['occurrences'])
        self.assertEqual(self.external['case_manifest_fingerprint'],fingerprint(self.manifest))

    def test_external_inventory_has_exact_data_roles_and_no_embedded_blobs(self):
        assets=self.external['assets']
        self.assertEqual(len(assets),29)
        self.assertEqual(sum(row['bytes'] for row in assets),1666534545)
        self.assertEqual(sum(row['codec']=='served-tensor-fixture-v1' for row in assets),3)
        self.assertEqual(len({row['path'] for row in assets}),len(assets))
        self.assertEqual(len({row['object_key'] for row in assets}),len(assets))
        metadata={row['path']:row['sha256'] for row in assets if row['codec']=='served-tensor-fixture-v1'}
        self.assertEqual(metadata,{c['fixture']['path']:c['fixture']['sha256'] for c in self.manifest['cases']})
        self.assertEqual(sum('kernel_weight' in row['roles'] for row in assets),1)

    def test_scale_layout_is_bijective_and_preserves_padding(self):
        for columns in (16,112):
            offsets={adapter.scale_offset(row,col,columns) for row in range(256) for col in range(columns)}
            self.assertEqual(offsets,set(range(256*columns)))
        live={adapter.scale_offset(row,col,16) for row in range(32) for col in range(12)}
        padded={adapter.scale_offset(row,col,16) for row in range(32) for col in range(12,16)}
        self.assertFalse(live&padded)
        self.assertEqual(len(live),32*12)

    def test_generated_routes_preserve_native_work_and_distinct_topk(self):
        for case in self.manifest['cases']:
            tensors=case['tensors'];m=tensors['inputs.a']['shape'][0]
            tm=case['scalars']['arguments']['tile_m'];capacity=tensors['inputs.sorted_token_ids']['shape'][0]
            expert_slots=tensors['inputs.sorted_expert_ids']['shape'][0]
            values=case['work_distribution']['valid_rows_histogram']
            for rows in (values[0][0],values[len(values)//2][0],values[-1][0]):
                with self.subTest(tokens=m,rows=rows):
                    routes=adapter.make_routes(m,tm,rows,capacity,expert_slots,701)
                    self.assertEqual(routes['num_valid_ids'].tolist(),[rows,m])
                    seen=set();counts=Counter()
                    for row,fused in enumerate(routes['sorted_token_ids'][:rows]):
                        token=fused&0xffffff;slot=fused>>24
                        if token>=m or not 0<=slot<16:continue
                        expert=routes['sorted_expert_ids'][row//tm]
                        self.assertEqual(routes['topk_ids'][token*16+slot],expert)
                        self.assertNotIn((token,expert),seen);seen.add((token,expert));counts[expert]+=1
                    self.assertEqual(len(seen),m*16)
                    self.assertEqual(sum((count+tm-1)//tm for count in counts.values()),rows//tm)
                    self.assertTrue(all(v==-1 for v in routes['sorted_expert_ids'][rows//tm:]))

    def test_refresh_seed_changes_routes_without_inventing_work_values(self):
        case=self.manifest['cases'][0];histogram=case['work_distribution']['valid_rows_histogram']
        observed={rows for rows,_ in histogram}
        self.assertTrue(all(adapter.weighted_work(histogram,random.Random(seed)) in observed for seed in range(100)))
        kwargs=(64,32,histogram[0][0],29680,928)
        first=adapter.make_routes(*kwargs,seed=1);second=adapter.make_routes(*kwargs,seed=2)
        self.assertNotEqual(first['topk_ids'],second['topk_ids'])
        self.assertEqual(first['num_valid_ids'],second['num_valid_ids'])

    def test_invalid_work_counts_are_rejected(self):
        for rows in (1,30000,320):
            with self.subTest(rows=rows),self.assertRaises(ValueError):adapter.make_routes(64,32,rows,29680,928,1)

    def test_regular_fixture_path_and_hash_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ok.bin').write_bytes(b'123')
            digest=adapter.file_sha(root/'ok.bin')
            self.assertEqual(adapter.checked_path(root,'ok.bin',digest),root/'ok.bin')
            with self.assertRaises(ValueError):adapter.checked_path(root,'ok.bin','f'*64)
            (root/'link.bin').symlink_to(root/'ok.bin')
            with self.assertRaises(ValueError):adapter.checked_path(root,'link.bin',digest)


    def test_captured_tensor_attributes_cannot_be_silently_discarded(self):
        record={'tensor_attribute_codec_id':None,'inputs':{'a':{'shape':[64,3584]}},'outputs':{'out':None}}
        adapter.validate_tensor_attributes(record)
        bad=copy.deepcopy(record);bad['inputs']['a']['attributes']={'is_shuffled':True}
        with self.assertRaisesRegex(ValueError,'Unexpected captured tensor attributes'):
            adapter.validate_tensor_attributes(bad)
        bad=copy.deepcopy(record);bad['tensor_attribute_codec_id']='unreviewed-packing'
        with self.assertRaisesRegex(ValueError,'no tensor-attribute codec'):
            adapter.validate_tensor_attributes(bad)

    def test_v4_work_histogram_ranges_and_frequencies(self):
        expected=[(10368,19648,290),(157312,165056,75),(280832,295360,138)]
        for case,(minimum,maximum,bins) in zip(self.manifest['cases'],expected):
            histogram=case['work_distribution']['valid_rows_histogram']
            self.assertEqual((histogram[0][0],histogram[-1][0],len(histogram)),(minimum,maximum,bins))
            self.assertEqual(case['work_distribution']['source_run'],'kimi-actual-v4-no-stack-194550')
            self.assertEqual(sum(count for _,count in histogram),case['occurrences'])
        self.assertEqual(self.manifest['workload_histogram_update']['run_id'],'kimi-actual-v4-no-stack-194550')


    def test_runtime_guard_and_generic_evaluator_share_one_reference(self):
        import yaml
        config=yaml.safe_load((ROOT/'config.yaml').read_text())
        policy=json.loads((ROOT/'ut/source_guard_policy.json').read_text())
        provenance=json.loads((ROOT/'SOURCE-PROVENANCE.json').read_text())
        reference,mapping=adapter.resolve_reference_bindings(config,policy,provenance)
        self.assertEqual(reference,'ut/baseline_src/flydsl')
        self.assertFalse((ROOT/'ut/reference').exists())
        self.assertTrue(all(path.startswith(reference+'/') for path in mapping.values()))
        self.assertEqual(adapter.reference_contract(ROOT)[0],ROOT/reference)
        for name,path in mapping.items():self.assertEqual((ROOT/name).read_bytes(),(ROOT/path).read_bytes())
        broken=copy.deepcopy(config)
        broken['trusted_evaluation']['reference_sources'][next(iter(mapping))]='ut/another_reference.py'
        with self.assertRaisesRegex(ValueError,'must share the same frozen reference'):
            adapter.resolve_reference_bindings(broken,policy,provenance)

    def test_sync_repair_pins_original_image_and_exact_local_change(self):
        import yaml
        config=yaml.safe_load((ROOT/'config.yaml').read_text())
        policy=json.loads((ROOT/'ut/source_guard_policy.json').read_text())
        provenance=json.loads((ROOT/'SOURCE-PROVENANCE.json').read_text())
        _,mapping=adapter.resolve_reference_bindings(config,policy,provenance)
        repair=adapter.validate_sync_repairs(ROOT,provenance,mapping)[0]
        self.assertEqual(repair['image_source_sha256'],'7c43585c51bfee5506515d0efec908611c93ac8d0583addb437296d442a91a91')
        self.assertEqual(repair['projected_sha256_before'],'c8a49e74f9579bf94f3b8bbc98e701a5d4248f602b36d1024cfbd38c858ae436')
        self.assertEqual(repair['projected_sha256_after'],'6755f05f732d603541b2d916a8cd3b2df9ac1c570abdc10eb578e0f939417755')
        broken=copy.deepcopy(provenance);broken['synchronization_repairs'][0]['image_source_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'image/reference source mapping'):
            adapter.validate_sync_repairs(ROOT,broken,mapping)
        broken=copy.deepcopy(provenance);broken['synchronization_repairs'][0]['before_context']+='\n# undeclared change'
        with self.assertRaisesRegex(ValueError,'changes beyond the declared'):
            adapter.validate_sync_repairs(ROOT,broken,mapping)

    def test_modified_reference_cannot_hide_behind_updated_final_hash(self):
        import hashlib,yaml
        config=yaml.safe_load((ROOT/'config.yaml').read_text())
        policy=json.loads((ROOT/'ut/source_guard_policy.json').read_text())
        provenance=json.loads((ROOT/'SOURCE-PROVENANCE.json').read_text())
        _,mapping=adapter.resolve_reference_bindings(config,policy,provenance)
        repair=provenance['synchronization_repairs'][0]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);target=root/repair['reference'];target.parent.mkdir(parents=True)
            changed=(ROOT/repair['reference']).read_text()+'\n_undeclared_host_change = 1\n'
            target.write_text(changed);checksum=hashlib.sha256(changed.encode()).hexdigest()
            repair['projected_sha256_after']=checksum
            next(row for row in provenance['files'] if row['file']==repair['file'])['sha256']=checksum
            with self.assertRaisesRegex(ValueError,'changes beyond the declared'):
                adapter.validate_sync_repairs(root,provenance,mapping)


    def test_fresh_callbacks_make_one_owned_snapshot_per_input_boundary(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from fresh_runner import FreshCallbacks
        log=[];counts=dict(refresh=0,initialize=0,replay=0,reference=0,measure=0)
        class Tensor:
            def __init__(self,value,device='cuda',name='input'):
                self.value=value;self.device=SimpleNamespace(type=device);self.name=name
            def detach(self):return self
            def to(self,*,device,copy):
                self_test.assertTrue(copy);log.append(('copy',self.name,self.device.type,device))
                return Tensor(self.value,device,self.name)
            def untyped_storage(self):return SimpleNamespace(tensor=self,nbytes=lambda:1,data_ptr=lambda:id(self))
        class Raw:
            def set_(self,storage,*args):self.tensor=storage.tensor;return self
            def zero_(self):self.tensor.value=0
        self_test=self
        torch=SimpleNamespace(uint8='uint8',empty=lambda *a,**k:Raw(),is_tensor=lambda x:isinstance(x,Tensor),
            cuda=SimpleNamespace(synchronize=lambda:log.append('sync')))
        prepared=object.__new__(adapter.Prepared);prepared.torch=torch
        prepared.inputs={'a':Tensor(7),'out':None};output=Tensor(0,name='output')
        def refresh(seed):counts['refresh']+=1;prepared.inputs['a'].value=7+seed
        def initialize():counts['initialize']+=1;output.value=-999
        def replay():counts['replay']+=1;log.append('replay');output.value=prepared.inputs['a'].value*2
        def immutable(after,before):
            self.assertEqual(after['a'].device.type,'cpu');self.assertEqual(before['a'].device.type,'cpu')
            self.assertEqual(after['a'].value,before['a'].value)
        def reference(truth):
            counts['reference']+=1;log.append('reference')
            self.assertEqual(truth['a'].device.type,'cpu');self.assertIsNot(truth['a'],prepared.inputs['a'])
            expected=Tensor(truth['a'].value*2,name='golden')
            # GPU storage can change after observations; CPU truth remains owned.
            prepared.inputs['a'].value=-1000
            self.assertNotEqual(truth['a'].value,prepared.inputs['a'].value)
            return expected
        def compare(actual,expected):self.assertEqual(actual.value,expected.value)
        callbacks=FreshCallbacks(refresh_inputs=refresh,initialize_outputs=initialize,
            snapshot_inputs=prepared.snapshot_inputs,snapshot_outputs=lambda:output,
            validate_metadata=lambda:None,assert_immutable=immutable,reference=reference,compare=compare,
            replay=replay,torch_module=torch)
        def measure(call):
            counts['measure']+=1;start=len(log);call();self.assertEqual(log[start:],['replay']);return .25
        callbacks.measure=measure
        with patch.object(adapter,'raw_storage',side_effect=lambda value,torch:value):
            result=callbacks.performance_row({'case_id':'synthetic'},dict(method='cuda_graph',warmup_iterations=10,benchmark_iterations=100),
                observe=lambda:{'case_id':'synthetic'},challenge_seed=101)
        self.assertEqual(counts,dict(refresh=110,initialize=110,replay=110,reference=110,measure=100))
        self.assertEqual(result['samples_ms'],[.25]*100)
        self.assertEqual(log.count(('copy','input','cuda','cpu')),330)
        self.assertEqual(log.count(('copy','input','cpu','cpu')),110)
        for index,value in enumerate(log):
            if value=='reference':self.assertIn(('copy','output','cuda','cpu'),log[max(0,index-5):index])


    def test_fast_source_probe_defers_only_candidate_warmup_failures(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('stage1_fast_probe_test',ROOT/'scripts/check_source_binding.py')
        probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        def compare(self,actual,expected):
            if actual!=expected:raise AssertionError('wrong value')
            return True
        def prepare(case):
            value=object.__new__(adapter.Prepared)
            for actual,expected in ((0,1),(1,1),(0,1)):value.compare(actual,expected)
            value.callbacks=SimpleNamespace(_compare=value.compare)
            return value
        with patch.object(adapter.Prepared,'compare',compare):
            prepared,checks=probe.prepare_for_probe(SimpleNamespace(prepare_case=prepare),{})
            self.assertEqual([row['correct'] for row in checks],[False,True,False])
            self.assertFalse(prepared.captured_parity)
            with self.assertRaisesRegex(AssertionError,'wrong value'):prepared.callbacks._compare(0,1)
            def bad_reference(case):
                value=object.__new__(adapter.Prepared);value.compare(0,1);value.compare(0,1)
            with self.assertRaises(probe.ReferenceCalibrationError):
                probe.prepare_for_probe(SimpleNamespace(prepare_case=bad_reference),{})
            self.assertIs(adapter.Prepared.compare,compare)


if __name__=='__main__':unittest.main()

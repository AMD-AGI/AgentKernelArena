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


if __name__=='__main__':unittest.main()

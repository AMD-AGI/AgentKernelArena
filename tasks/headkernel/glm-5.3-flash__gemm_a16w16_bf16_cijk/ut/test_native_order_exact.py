"""Regressions against113 independently enumerated legal paired outcomes."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from native_precision import exact_pair_orders,NativeOrderProofError

DATA=json.loads((Path(__file__).parent/'data/native_order_seed817302019.json').read_text())
def bf16(bits):return torch.tensor(bits,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)

class Tests(unittest.TestCase):
    def test_all113_legal_outcomes_include24_missed_pointwise_failures(self):
        rows=DATA['outcomes'];self.assertEqual(len(rows),113)
        self.assertEqual(sum(not r['old_search_reachable'] for r in rows),59)
        self.assertEqual(sum(not r['old_search_reachable'] and bool(r['original_pointwise_failed_lanes']) for r in rows),24)
        count=len(rows);parts=bf16(DATA['partials']).reshape(16,1,2).repeat(1,1,count)
        actual=bf16([r['bits'] for r in rows]).reshape(1,2*count);positions=torch.tensor([[0,i] for i in range(count)])
        proof=exact_pair_orders(parts,actual,positions,max_orders=0)
        self.assertEqual(proof['pairs_proved_exactly'],113);self.assertEqual(len(proof['exact_fallback']),1)
        for witness in proof['witnesses']:self.assertEqual(sorted(witness['arrival_order']),list(range(16)))

    def test_jointly_unreachable_pair_retains_observed_and_partial_bits(self):
        parts=bf16(DATA['partials']).reshape(16,1,2);actual=bf16(DATA['invalid_shared_pair_bits']).reshape(1,2)
        with self.assertRaises(NativeOrderProofError) as caught:exact_pair_orders(parts,actual,torch.tensor([[0,0]]),max_orders=0)
        evidence=caught.exception.native_precision_evidence;pair=evidence['failed_pairs'][0]
        self.assertEqual(pair['observed_bf16_bits'],DATA['invalid_shared_pair_bits'])
        self.assertEqual(pair['partial_bf16_bits'],DATA['partials'])
        self.assertEqual((pair['row'],pair['pair_column']),(0,0))

    def test_production_failure_receipt_preserves_pair_evidence(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import production_comparison
        evidence={'schema':'native-ASM-exact-order-failure-v1','failed_pairs':[{'row':27,'pair_column':53,'observed_bf16_bits':DATA['invalid_shared_pair_bits'],'partial_bf16_bits':DATA['partials']}]}
        error=NativeOrderProofError(evidence)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(production_comparison.task,'ROOT',Path(directory)),patch.object(production_comparison,'compare',side_effect=error),patch.dict(production_comparison.CONTEXT,{'case_id':DATA['case_id'],'seed':DATA['seed']},clear=True):
                with self.assertRaises(NativeOrderProofError):production_comparison.main()
                saved=json.loads((Path(directory)/'build/native_production_failure.json').read_text())
            self.assertEqual(saved['native_precision_evidence'],evidence)
            self.assertEqual(saved['seed'],DATA['seed'])

if __name__=='__main__':unittest.main(verbosity=2)

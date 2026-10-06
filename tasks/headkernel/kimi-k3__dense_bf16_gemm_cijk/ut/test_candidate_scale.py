"""CPU regression coverage for low-amplitude constant-output shortcuts."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_precision import candidate_close, candidate_error_metrics


class Tests(unittest.TestCase):
    def test_low_amplitude_zero_and_nonzero_constant_shortcuts_rejected(self):
        expected = torch.tensor([[-0.0155, -0.003, 0.002, 0.0138]], dtype=torch.bfloat16)
        for value in (0.0, 0.001, -0.001, 0.002):
            actual = torch.full_like(expected, value)
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
            with self.assertRaisesRegex(AssertionError, 'Scale-relative'):
                candidate_close(actual, expected)

    def test_binary_rescaling_preserves_normalized_error(self):
        expected = torch.tensor([[1., -2., 4.]], dtype=torch.bfloat16)
        actual = expected + torch.tensor([[0.0078125, 0., 0.]], dtype=torch.bfloat16)
        values = [candidate_error_metrics(actual * 2**k, expected * 2**k)['normalized_l2'] for k in (-10, 0, 10)]
        self.assertEqual(values, [values[0]] * 3)

    def test_exact_zero_reference_requires_exact_output(self):
        expected = torch.zeros((2, 2), dtype=torch.bfloat16)
        self.assertTrue(candidate_close(expected, expected)['scale_relative_pass'])
        with self.assertRaisesRegex(AssertionError, 'Scale-relative'):
            candidate_close(expected + 1e-4, expected)

    def test_pointwise_check_still_rejects_localized_error(self):
        expected = torch.ones((10000,), dtype=torch.bfloat16)
        actual = expected.clone()
        actual[0] = 1.5
        self.assertTrue(candidate_error_metrics(actual, expected)['scale_relative_pass'])
        with self.assertRaises(AssertionError):
            candidate_close(actual, expected)

    def test_nonfinite_values_fail_closed(self):
        expected = torch.ones((2,), dtype=torch.bfloat16)
        for value in (float('nan'), float('inf')):
            with self.assertRaises(AssertionError):
                candidate_error_metrics(torch.full_like(expected, value), expected)

    def test_accuracy_budget_is_bf16_precision_based(self):
        expected = torch.ones((2,), dtype=torch.bfloat16)
        self.assertEqual(candidate_close(expected, expected)['normalized_l2_limit'], 2 / 255)


if __name__ == '__main__':
    unittest.main(verbosity=2)

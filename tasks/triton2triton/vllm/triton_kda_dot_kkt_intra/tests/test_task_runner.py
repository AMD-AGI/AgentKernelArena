from scripts.contract_checks import ContractFailure, check_outputs
import pytest
import torch


def test_correctness_rejects_empty_result_tuple():
    refs = (torch.zeros(1), torch.zeros(1))
    with pytest.raises(ContractFailure, match='Missing or extra output tuple members'):
        check_outputs((), refs, atol=1e-2, rtol=1e-2)
    check_outputs(tuple(x.clone() for x in refs), refs, atol=1e-2, rtol=1e-2)

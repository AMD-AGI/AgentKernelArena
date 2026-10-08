"""MiniMax adapter for the shared paired-reference comparison schema."""
from copy import copy, deepcopy
import hashlib

from evaluation_contract import checked_replays, fingerprint, observe_case, require
from minimax_data import initialize_outputs, snapshot_output, work_controls
from minimax_native import Operator
from paired_reference import paired_performance

GPU_BINDING = "private_triton_code_object_source_checked_v1"


def leaves(value):
    import torch
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)


def raw_storage(value):
    import torch
    size = value.untyped_storage().nbytes()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        value.untyped_storage(), 0, (size,), (1,))


def reference_view(inputs):
    import torch
    result = copy(inputs)
    result.storage = {name: torch.empty_like(value) for name, value in inputs.storage.items()}
    result.tensors = result.views(result.storage)
    result.args = {**result.scalars, **result.tensors}
    return result


def binding(operator, definition, leg):
    return {"source_sha256": {definition["source_file"]: operator.source_hash},
            "module": operator.function.__module__, "leg": leg, "gpu_binding": GPU_BINDING}


def input_signature(inputs, truth, seed):
    # Control hashes are taken from the CPU-owned pre-candidate snapshot.
    # Full floating storage equality is checked on reference restoration; no
    # second generator or reconstructed floating approximation is substituted.
    from paired_input_validation import signature_from_geometry
    args = inputs.reference_args(truth)
    return signature_from_geometry(args, seed, sum(value.numel() for value in truth.values()),
                                   inputs.current_control_variant, work_controls, fingerprint)


def invoke_with_reference_output_owner(operator, args, output_owner, reference_operator):
    """Only the private reference transfers ownership before post-call checks."""
    if operator is reference_operator:
        return operator(args, reference_output_owner=output_owner)
    return operator(args)


def performance(evaluation, manifest, request):
    import torch
    inputs = evaluation.inputs
    reference = reference_view(inputs)
    baseline = Operator(evaluation.root, evaluation.definition, reference=True)
    if "launch_contract" in evaluation.case:
        baseline.select_launch_contract(evaluation.case["launch_contract"])
    identity = binding(evaluation.operator, evaluation.definition, "candidate")
    reference_identity = binding(baseline, evaluation.definition, "reference")
    require(identity["module"] != reference_identity["module"], "candidate/reference modules are not private")
    selected = []
    realized = []
    prepared = {}

    def owner(args):
        if args is inputs.args:
            return inputs
        require(args is reference.args, "unexpected paired input owner")
        return reference

    def restore(args, truth):
        target = owner(args)
        require(set(target.storage) == set(truth), "paired storage set differs")
        for name, value in target.storage.items():
            require(value.numel() == truth[name].numel(), "paired physical storage length differs")
            value.copy_(truth[name])
        target.current_state = inputs.current_state
        target.current_control_variant = inputs.current_control_variant

    def observe(args, output):
        return owner(args).observe_arguments(output)

    def compare(actual, expected, args, tolerance, *, expected_inputs):
        # The protected reference has already been observed on CPU and its
        # complete output storage scrubbed. The independent mathematical oracle
        # runs after both device measurements so it cannot warm only one leg.
        require(args is inputs.args and tolerance == evaluation.definition["tolerance"],
                "paired mathematical comparison contract differs")
        evaluation.check_independent_output(actual, reference.args)
        evaluation.compare_outputs(actual, expected)

    def reset(seed):
        truth = inputs.reset(seed)
        selected.append(inputs.current_control_variant)
        prepared["signature"] = input_signature(inputs, truth, seed)
        realized.append(deepcopy(prepared["signature"]))
        return truth

    api = {"leaves": leaves, "raw_storage": raw_storage,
           "restore_storages": restore,
           "invoke": lambda fn, args, output_owner: invoke_with_reference_output_owner(
               fn, args, output_owner, baseline),
           "assert_immutable_inputs": lambda args, truth: owner(args).assert_immutable(truth),
           "cpu_clone": snapshot_output, "initialize_output": initialize_outputs,
           "runtime_abi": observe, "observe_case": observe_case,
           "compare_native_outputs": compare, "checked_replays": checked_replays,
           "tolerance": evaluation.definition["tolerance"]}
    # Both initial input views come from the same CPU-owned snapshot. Every
    # scheduled reference restore happens later, after candidate observation.
    try:
        row = paired_performance(api, evaluation.case, manifest, request,
            inputs.args, reference.args, evaluation.operator, baseline,
            evaluation.paired_initial_truth, None, identity, reference_identity,
            reset, current_input_signature=lambda: prepared["signature"])
        row["paired_reference"]["setup_input_seed"] = evaluation.paired_setup_seed
        row["realized_input_signatures"] = realized
        row["workload_control_sampling"] = inputs.control_distribution.sampling(
            request["challenge_seed"], selected, manifest["measurement"], row["samples_ms"])
        # Graph capture uses the same verified wrapper calls as ordinary native
        # execution; both independent source bindings must actually engage.
        candidate_proof, reference_proof = evaluation.operator.proof(), baseline.proof()
        require(candidate_proof["engaged_kernels"] == reference_proof["engaged_kernels"]
                == evaluation.definition["kernels"], "paired kernels were not engaged")
        evaluation.paired_specialization = {"case_id": evaluation.case["case_id"],
            "candidate_binding": identity, "reference_binding": reference_identity,
            "invoked_and_synchronized": True, "candidate_launch_evidence": candidate_proof,
            "reference_launch_evidence": reference_proof}
        return row
    finally:
        evaluation.paired_initial_truth = None
        del reference, baseline

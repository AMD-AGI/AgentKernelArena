"""Complete observed support, original plus targeted checks, and sampled timing."""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
import math
import random

from evaluation_contract import canonical, fingerprint, require, validate_report
from minimax_work import structural_work

SCHEMA = "recorded-workload-control-coverage-v2"
SAMPLER = "sha256-domain-seeded-python-randrange-integer-weights-v1"
TARGETING = "endpoints-adjacent-interiors-and-midpoint-pair-v1"


def flat(control):
    require(isinstance(control, dict) and set(control) == {"shape", "runs"}, "invalid exact control encoding")
    size = 1
    for value in control["shape"]:
        require(type(value) is int and value >= 0, "invalid exact control shape")
        size *= value
    values = []
    for value, count in control["runs"]:
        require(type(value) is int and type(count) is int and count > 0, "invalid exact control run")
        values.extend([value] * count)
    require(len(values) == size, "exact control shape/run length differs")
    return values


class RecordedControls:
    def __init__(self, case):
        self.case = case
        self.distribution = case["work_distribution"]
        self.variant_ids = tuple(sorted(self.distribution))
        require(self.variant_ids, "missing recorded control distribution")
        self.distribution_sha256 = fingerprint(self.distribution)
        self.cumulative = []
        total = 0
        self.donors = {}
        for variant in self.variant_ids:
            row = self.distribution[variant]
            require(set(row) == {"tensor_controls", "occurrences"}, "unrecognized control-distribution fields")
            controls = row["tensor_controls"]
            require(fingerprint(controls) == variant, "recorded control hash differs")
            require(type(row["occurrences"]) is int and row["occurrences"] > 0, "invalid control occurrence weight")
            require(structural_work(controls, case["scalars"]) == case["scalars"]["work.variant"],
                    "recorded controls escape the existing ABI/compiler case")
            for control in controls.values():
                flat(control)
            total += row["occurrences"]
            self.cumulative.append(total)
            exact = [i for i, state in enumerate(case["states"]) if state["tensor_controls"] == controls]
            if exact:
                self.donors[variant] = exact[0]
                continue
            # The diagnosed omissions are exact decode sequence lengths inside
            # one retained block class. Other controls must match a real donor;
            # this does not synthesize unobserved causal or sparse work.
            require(case["kind"] in ("decode_score", "sparse_decode"), "uncaptured prefill geometry requires a separate derivation")
            target = flat(controls["inputs.seq_lens"])
            other = {k: v for k, v in controls.items() if k != "inputs.seq_lens"}
            donors = []
            for i, state in enumerate(case["states"]):
                source = state["tensor_controls"]
                if {k: v for k, v in source.items() if k != "inputs.seq_lens"} != other:
                    continue
                lengths = flat(source["inputs.seq_lens"])
                if len(lengths) == len(target) and all(0 <= t <= s for t, s in zip(target, lengths)):
                    donors.append(i)
            require(donors, "no captured paging donor covers the recorded lengths")
            self.donors[variant] = donors[0]
        require(total == case["occurrences"], "control weights do not cover all observed case calls")
        self.total_occurrences = total
        require(all(fingerprint(state["tensor_controls"]) in self.distribution for state in case["states"]),
                "original recorded state is absent from the observed distribution")

    def controls(self, variant):
        require(variant in self.distribution, "unobserved workload-control variant")
        return self.distribution[variant]["tensor_controls"]

    def choose(self, private_seed):
        require(type(private_seed) is int and private_seed >= 0, "control sampler needs a nonnegative integer private seed")
        domain = {"sampler": SAMPLER, "case_id": self.case["case_id"],
                  "distribution_sha256": self.distribution_sha256, "private_seed": private_seed}
        # randrange uses rejection sampling; all integer occurrence intervals
        # receive their exact discrete weight without floating normalization.
        ticket = random.Random(fingerprint(domain)).randrange(self.total_occurrences)
        return self.variant_ids[bisect_right(self.cumulative, ticket)]

    def geometry(self, states, variant):
        controls = self.controls(variant)
        donor = self.donors[variant]
        geometry = dict(states[donor])
        if self.case["states"][donor]["tensor_controls"] != controls:
            lengths = geometry["seq_lens"]
            require(list(lengths.shape) == controls["inputs.seq_lens"]["shape"], "donor length shape differs")
            geometry["seq_lens"] = lengths.new_tensor(flat(controls["inputs.seq_lens"])).reshape(lengths.shape)
        return donor, geometry

    def original_plan(self, policy):
        states = self.case["states"]
        def entry(seed):
            index = seed % len(states)
            return {"seed": seed, "recorded_state_index": index,
                    "variant_id": fingerprint(states[index]["tensor_controls"])}
        return {"seed_checks": [entry(seed) for seed in policy["correctness_seeds"]],
                "negative_controls": {**entry(policy["correctness_seeds"][0]+1000),
                                      "no_op": True, "wrong_output": True}}

    def targeted_variants(self):
        represented = {fingerprint(state["tensor_controls"]) for state in self.case["states"]}
        if set(self.variant_ids) <= represented:
            return []
        # The rank order uses exact numeric controls, never hash order. Check
        # both endpoints, one setting inside each boundary, and both central
        # settings. This covers the partial/full-block transition and interior
        # values without pretending the entire 128-value class was executed.
        def key(variant):
            controls = self.controls(variant)
            return tuple((name, tuple(flat(controls[name]))) for name in sorted(controls))
        ordered = sorted(self.variant_ids, key=key)
        n = len(ordered)
        indices = sorted({0, min(1, n-1), (n-1)//2, n//2, max(0, n-2), n-1})
        return [ordered[i] for i in indices]

    def coverage(self, policy, completed, original):
        require(original == self.original_plan(policy), "original correctness seed/state/control checks changed")
        require(completed == self.targeted_variants(), "targeted correctness missed or repeated a selected setting")
        recorded = {fingerprint(state["tensor_controls"]) for state in self.case["states"]}
        tested = recorded | set(completed)
        return {"schema": SCHEMA, "distribution_sha256": self.distribution_sha256,
                "execution_mode": "original_plus_targeted",
                "represented_variant_count": len(self.variant_ids),
                "complete_recorded_distribution_represented": True,
                "original_checks": original, "targeted_selection": TARGETING,
                "targeted_variant_ids": completed,
                "targeted_seeds_per_variant": list(policy["correctness_seeds"]),
                "targeted_variant_seed_pairs": len(completed)*len(policy["correctness_seeds"]),
                "targeted_negative_controls_per_variant": {"no_op": True, "wrong_output": True},
                "targeted_negative_control_checks": len(completed)*2,
                "correctness_tested_variant_ids": sorted(tested),
                "untested_correctness_variant_ids": sorted(set(self.variant_ids)-tested),
                "all_recorded_values_checked_in_correctness": tested == set(self.variant_ids),
                "exhaustive_per_variant_seed_control_matrix": False,
                "historical_payloads_recovered": False}

    def sampling(self, seed, selected, policy, samples_ms):
        warmup, count = policy["warmup_iterations"], policy["benchmark_iterations"]
        require(warmup == 10 and count == 100, "original timing sample counts changed")
        require(selected == [self.choose(seed+i) for i in range(warmup+count)], "timing control draws differ from private-seed plan")
        require(len(samples_ms) == count and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in samples_ms),
                "mixed-distribution mean requires every raw device sample")
        return {"schema": SCHEMA, "distribution_sha256": self.distribution_sha256,
                "sampler": SAMPLER, "sampling": "occurrence_weighted_with_replacement",
                "private_challenge_seed": seed, "total_occurrences": self.total_occurrences,
                "warmup_variant_ids": selected[:warmup], "measured_variant_ids": selected[warmup:],
                "measured_histogram": dict(sorted(Counter(selected[warmup:]).items())),
                "schedule_fingerprint": fingerprint({"case_id": self.case["case_id"], "seed": seed,
                    "distribution_sha256": self.distribution_sha256, "selected": selected}),
                "represented_variant_count": len(self.variant_ids),
                "measured_variant_count": len(set(selected[warmup:])),
                "untimed_variant_ids": sorted(set(self.variant_ids)-set(selected[warmup:])),
                "aggregation": "arithmetic_mean_from_all_100_raw_device_samples",
                "mean_gpu_cost_ms": math.fsum(samples_ms)/count,
                "timing_guarantee": "sampled_distribution_only", "exhaustive_timing_claim": False}


def validate_scope_reports(manifest, rows, phase, *, challenge_seed=None):
    if phase == "compile":
        return
    by_id = {row["case"]["case_id"]: row for row in rows}
    require(len(by_id) == len(rows) == len(manifest["cases"]), "control evidence case coverage differs")
    policy = manifest["measurement"]
    for case in manifest["cases"]:
        distribution = RecordedControls(case)
        row = by_id[case["case_id"]]
        if phase == "correctness":
            expected = distribution.coverage(policy, distribution.targeted_variants(), distribution.original_plan(policy))
            require(canonical(row.get("workload_control_coverage")) == canonical(expected), "missing original/targeted control correctness evidence")
        else:
            evidence = row.get("workload_control_sampling", {})
            seed = evidence.get("private_challenge_seed")
            require(type(challenge_seed) is int and seed == challenge_seed, "timing control schedule does not bind the trusted private challenge")
            selected = evidence.get("warmup_variant_ids", []) + evidence.get("measured_variant_ids", [])
            expected = distribution.sampling(seed, selected, policy, row["samples_ms"])
            require(canonical(evidence) == canonical(expected), "weighted timing control evidence differs")


def paired_report_costs(reference, candidate, manifest, expected_requests):
    """Trusted postprocessing of full phase reports and separately bound requests.

    Each case estimates mean GPU cost under its conditional recorded histogram.
    Compare those means on the same draw schedule; do not average per-draw
    ratios or reweight the already weighted draws. This is not an E2E claim.
    """
    require(set(expected_requests) == {"reference", "candidate"}, "both trusted requests are required")
    seed = expected_requests["reference"]["challenge_seed"]
    require(seed == expected_requests["candidate"]["challenge_seed"], "paired private challenges differ")
    validated = {}
    for leg, report in (("reference", reference), ("candidate", candidate)):
        request = expected_requests[leg]
        require(request["phase"] == "performance", "paired GPU costs need performance reports")
        measured = validate_report(report, manifest, request)
        validate_scope_reports(manifest, report["cases"], "performance", challenge_seed=seed)
        validated[leg] = {row["test_case_id"]: row["execution_time_ms"] for row in measured}
    right = {row["case"]["case_id"]: row for row in candidate["cases"]}
    rows = []
    for left in reference["cases"]:
        case_id = left["case"]["case_id"]
        a, b = left["workload_control_sampling"], right[case_id]["workload_control_sampling"]
        require(a["schedule_fingerprint"] == b["schedule_fingerprint"], "paired workload schedules differ")
        x, y = validated["reference"][case_id], validated["candidate"][case_id]
        rows.append({"case_id": case_id, "reference_mean_ms": x, "candidate_mean_ms": y,
                     "reference_over_candidate": x/y, "schedule_fingerprint": a["schedule_fingerprint"]})
    return {"metric": "ratio_of_arithmetic_mean_GPU_costs_on_matched_histogram_draws",
            "case_results": rows,
            "existing_arithmetic_mean_case_speedup": math.fsum(r["reference_over_candidate"] for r in rows)/len(rows),
            "raw_sample_recomputed": True, "timing_guarantee": "sampled_distribution_only",
            "untested_values_claimed_tested": False, "end_to_end_gain_claim": False}

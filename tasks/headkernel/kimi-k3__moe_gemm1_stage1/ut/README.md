The protected stage-1 contract is [STAGE1-CONTRACT.md](STAGE1-CONTRACT.md), backed by [the sealed cases](../cases.json) and [the external fixture inventory](../fixtures/EXTERNAL-MANIFEST.json).

`runtime_adapter.py` binds current private candidate/reference FlyDSL packages and validates the native A8W4 ABI. `fresh_runner.py` owns replay, CPU snapshots and timing order. `evaluation_contract.py` is the unchanged canonical portable report helper. `source_guard.py` and `source_guard_policy.json` enforce the permitted stage-1 emission edits.

`baseline_src/flydsl/` is the complete frozen reference closure, including the declared synchronization repair. It is the single reference tree used by the native runtime, source edit boundary and generic evaluator. The original image-source hashes and exact patch mapping remain mandatory checks. Raw captured fixture JSON and storage blobs are materialized beneath the task's `fixtures/` directory through the trusted host evaluator.

CPU contract checks are available through:

```bash
python3 ut/test_runtime_adapter.py
```

They verify routing, physical scale indexing, case frequencies and fixture metadata. They do not replace GPU execution or framework task-validator qualification.

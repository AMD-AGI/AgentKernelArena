The protected stage-1 contract is [STAGE1-CONTRACT.md](STAGE1-CONTRACT.md), backed by [the sealed cases](../cases.json) and [the external fixture inventory](../fixtures/EXTERNAL-MANIFEST.json).

`runtime_adapter.py` binds current private candidate/reference FlyDSL packages and validates the native A8W4 ABI. `fresh_runner.py` owns replay, CPU snapshots and timing order. `evaluation_contract.py` is the unchanged canonical portable report helper. `source_guard.py` and `source_guard_policy.json` enforce the permitted stage-1 emission edits.

`baseline_src/flydsl/` is the complete frozen native reference closure. `reference/source/` holds the frozen counterparts used by the edit boundary. Both directories are required. Raw captured fixture JSON and storage blobs are materialized beneath the task's `fixtures/` directory through the trusted host evaluator.

CPU contract checks are available through:

```bash
python3 ut/test_runtime_adapter.py
```

They verify routing, physical scale indexing, case frequencies and fixture metadata. They do not replace GPU execution or framework task-validator qualification.

# AITER configuration API fixture

`aiter_config_api_fixture.py` contains the pinned upstream `AITER_CONFIG` class
used by CPU tests of the Qwen dispatch table. `api_provenance.json` records the
source URL, revision, source hash, and fixture hash. The tests supply their own
small CSV reader and temporary directories; this fixture is not a runtime task
dependency or a captured experiment report.

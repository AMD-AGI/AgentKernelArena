<!-- Thanks for contributing to AgentKernelArena! Please fill in the sections
     below. See CONTRIBUTING.md for details. -->

## Summary

<!-- What does this PR change and why? Link related issues (e.g. Closes #123). -->

## Type of change

- [ ] Bug fix
- [ ] New agent integration (`agents/`)
- [ ] New / updated task(s) (`tasks/`)
- [ ] Scoring / evaluation logic (`src/score.py`, `src/evaluator.py`, ...)
- [ ] Docs
- [ ] CI / tooling
- [ ] Other:

## Test environment

<!-- Required for changes touching runtime, scoring, or tasks. -->

- GPU model:
- ROCm version:
- Docker image:
- OS:
- Agent(s) and version(s):
- Task category exercised: <!-- hip2hip / torch2hip / triton2triton / instruction2triton / flydsl2flydsl / repository -->

## Verification

<!-- Key commands and a short output summary. -->

- [ ] `ruff check .` passes
- [ ] `make check-perf-helpers` passes (if touching `src/tools/perf/` or task perf stubs)
- [ ] Ran a smoke test: `make docker-run CONFIG=config.yaml`
- [ ] For scoring / evaluation changes: before/after results on at least one task category attached

## Checklist

- [ ] Changes are focused and scoped
- [ ] Updated `config.yaml` / `README.md` / agent registry (`agents/__init__.py`) if adding an agent
- [ ] Docs updated if behavior changed

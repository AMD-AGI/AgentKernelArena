# Makefile for AgentKernelArena — Docker-first workflow
#
# All experiments run inside the pinned ROCm/SGLang container. Docker is the only
# supported path; the legacy host venv / `python main.py` workflow has been removed.
# See src/scripts/docker_benchmark.sh and docs/install/install.md.

SHELL := /bin/bash

.PHONY: help docker-shell docker-check-agents docker-smoke docker-run docker-parallel-run docker-setup-flydsl \
        check-docker-runner check-evaluator check-held-out check-visualization \
        visualization-build visualization-serve visualization-run \
        sync-perf-helpers check-perf-helpers materialize-perf-workspace \
        materialize-perf-task cleanup-works install-cursor-agent vllm

help:
	@echo "AgentKernelArena Experimentation Platform - Makefile Commands"
	@echo "======================================================"
	@echo "Docker-first workflow (the only supported path):"
	@echo "make docker-shell        - Enter the runtime image with repo and agent auth mounted"
	@echo "make docker-check-agents - Verify the first-class host CLI selected by CONFIG"
	@echo "                         Use CONFIG=... for another config; AGENTS=... overrides it"
	@echo "                         AGENTS=all explicitly checks all three first-class CLIs"
	@echo "make docker-smoke        - Verify Docker Python, ROCm tools, imports, and GPU access"
	@echo "make docker-run CONFIG=example_configs/quickstart_claude_mi300.yaml RUN_ARGS=\"--run-suffix test\" - Run an experiment in Docker"
	@echo "make docker-parallel-run CONFIG=example_configs/benchmark_cursor_mi355x.yaml GPU_IDS=0,1 - Run an experiment across one worker container per GPU"
	@echo "                         Default CONFIG is the MI300/MI300X Claude quickstart"
	@echo "                         On other GPUs, pass a matching CONFIG explicitly"
	@echo "                         Images: gfx942->mi30x, gfx950->mi35x; override with AKA_DOCKER_IMAGE=..."
	@echo "make docker-setup-flydsl - Install FlyDSL when absent (for flydsl2flydsl, torch2flydsl, and triton2flydsl)"
	@echo "make check-docker-runner - Check Docker runner syntax and runtime-specific arguments"
	@echo "make check-evaluator     - Run centralized evaluator unit tests"
	@echo "make check-held-out      - Run held-out module unit tests"
	@echo "make visualization-run   - Build and serve the local comparison dashboard"
	@echo "make check-visualization - Run visualization module unit tests"
	@echo ""
	@echo "Maintenance:"
	@echo "make sync-perf-helpers   - Refresh committed perf-helper stubs in task sources"
	@echo "make check-perf-helpers  - Verify task perf-helper stubs and markers are valid"
	@echo "make materialize-perf-workspace WORKSPACE=workspace_x - Inject canonical perf helpers into workspace(s)"
	@echo "make materialize-perf-task TASK=tasks/... OUT=/tmp/aka-task - Copy task(s) and inject canonical perf helpers"
	@echo "make cleanup-works       - Remove workspace_* directories and logs"
	@echo "make install-cursor-agent - Install the Cursor Agent CLI on the host"

DOCKER_RUNNER := src/scripts/docker_benchmark.sh
CONFIG ?= example_configs/quickstart_claude_mi300.yaml
RUN_ARGS ?=
AGENTS ?=
WORKSPACES ?= $(WORKSPACE)
TASKS ?= $(TASK)
OUT ?= /tmp/aka-materialized-tasks
FORCE ?= 0
VISUALIZATION_ARGS ?= --include-workspace-runs
VISUALIZATION_HOST ?= 127.0.0.1
VISUALIZATION_PORT ?= 8080
MATERIALIZE_FORCE_ARG := $(if $(filter 1 true yes,$(FORCE)),--force,)

docker-shell:
	@$(DOCKER_RUNNER) shell

docker-check-agents:
	@AKA_AGENTS="$(AGENTS)" $(DOCKER_RUNNER) check-agents --config_name $(CONFIG)

docker-smoke:
	@$(DOCKER_RUNNER) smoke

docker-run:
	@$(DOCKER_RUNNER) run --config_name $(CONFIG) $(RUN_ARGS)

docker-parallel-run:
	@GPU_IDS="$(GPU_IDS)" $(DOCKER_RUNNER) parallel-run --config_name $(CONFIG) $(RUN_ARGS)

# Install FlyDSL into the container's persistent pip user-base when the selected
# image does not ship it. Needed by all three FlyDSL task types.
docker-setup-flydsl:
	@$(DOCKER_RUNNER) setup-flydsl

check-docker-runner:
	@bash tests/test_docker_benchmark.sh

check-evaluator:
	@python3 -m unittest discover -s tests -p 'test_evaluator_*.py'

check-held-out:
	@python3 -m unittest discover -s tests -p 'test_held_out.py'

check-visualization:
	@python3 -m unittest discover -s tests -p 'test_visualization.py'

visualization-build:
	@python3 -m src.visualization build $(VISUALIZATION_ARGS)

visualization-serve:
	@python3 -m src.visualization serve --host "$(VISUALIZATION_HOST)" --port "$(VISUALIZATION_PORT)"

visualization-run:
	@python3 -m src.visualization run $(VISUALIZATION_ARGS) \
		--host "$(VISUALIZATION_HOST)" --port "$(VISUALIZATION_PORT)"

# Refresh committed perf-helper stubs/markers in task sources. Runtime workspaces
# are materialized from src/tools/perf/ by setup_workspace().
sync-perf-helpers:
	@python3 src/tools/sync_perf_helpers.py

# Verify all task perf-helper stubs/markers are in the expected committed form.
check-perf-helpers:
	@python3 src/tools/sync_perf_helpers.py --check

# Materialize canonical perf helpers into existing copied task workspace(s).
materialize-perf-workspace:
	@test -n "$(WORKSPACES)" || (echo "Usage: make materialize-perf-workspace WORKSPACE=workspace_x"; exit 2)
	@python3 src/tools/materialize_perf_helpers.py workspace $(WORKSPACES)

# Copy one or more task source directories to OUT, then materialize helpers there.
materialize-perf-task:
	@test -n "$(TASKS)" || (echo "Usage: make materialize-perf-task TASK=tasks/... [OUT=/tmp/aka-task] [FORCE=1]"; exit 2)
	@python3 src/tools/materialize_perf_helpers.py task --out "$(OUT)" $(MATERIALIZE_FORCE_ARG) $(TASKS)

cleanup-works:
	@echo "Removing workspace directories and logs..."
	@rm -rf workspace_*
	@rm -rf logs
	@echo "✓ Workspace directories and logs removed"

install-cursor-agent:
	@echo "Installing Cursor agent..."
	@curl https://cursor.com/install -fsSL | bash

# Run the pinned local vLLM endpoint. Agent/provider wiring is integration-specific.
vllm:
	@if ss -ltn | grep ':30001 ' > /dev/null; then \
		echo "vLLM server is already running on port 30001."; \
	else \
		docker run -d \
			--ipc=host \
			--network=host \
			--privileged \
			--cap-add=SYS_ADMIN \
			--cap-add=SYS_PTRACE \
			--device=/dev/kfd \
			--device=/dev/dri \
			--device=/dev/mem \
			--group-add=render \
			--security-opt=seccomp=unconfined \
			rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909 \
			vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
			--served-model-name llamas_team_local_llm \
			--api-key dummy \
			--host 0.0.0.0 \
			--port 30001 \
			--enable-auto-tool-choice \
			--tool-call-parser hermes \
			--trust-remote-code; \
		echo "Configure a compatible agent integration to use the OpenAI-compatible endpoint at port 30001."; \
		echo "vLLM server will be running on port 30001, please wait 3 minutes for it to start..."; \
		echo "You can use docker logs -f container_id to check the server status"; \
	fi

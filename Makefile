# Makefile for AgentKernelArena — Docker-first workflow
#
# All benchmarking runs inside the pinned ROCm/SGLang container. Docker is the only
# supported path; the legacy host venv / `python main.py` workflow has been removed.
# See src/scripts/docker_benchmark.sh and docs/install/install.md.

SHELL := /bin/bash

.PHONY: help docker-shell docker-check-agents docker-smoke docker-run docker-setup-flydsl \
        sync-perf-helpers check-perf-helpers materialize-perf-workspace \
        materialize-perf-task cleanup-works install-cursor-agent vllm

help:
	@echo "AgentKernelArena Evaluation Framework - Makefile Commands"
	@echo "======================================================"
	@echo "Docker-first workflow (the only supported path):"
	@echo "make docker-shell        - Enter the benchmark Docker image with repo and agent auth mounted"
	@echo "make docker-check-agents - Verify Codex, Claude Code, and Cursor Agent login reuse in Docker"
	@echo "make docker-smoke        - Verify Docker Python, ROCm tools, imports, and GPU access"
	@echo "make docker-run CONFIG=config.yaml RUN_ARGS=\"--run-suffix test\" - Run benchmark in Docker"
	@echo "                         Images: gfx942->mi30x, gfx950->mi35x, gfx10*/11*/12*->navi; override with AKA_DOCKER_IMAGE=..."
	@echo "make docker-setup-flydsl - Install/upgrade FlyDSL 0.2.x in the container (needed for flydsl2flydsl tasks)"
	@echo ""
	@echo "Maintenance:"
	@echo "make sync-perf-helpers   - Refresh committed perf-helper stubs in task sources"
	@echo "make check-perf-helpers  - Verify task perf-helper stubs and markers are valid"
	@echo "make materialize-perf-workspace WORKSPACE=workspace_x - Inject canonical perf helpers into workspace(s)"
	@echo "make materialize-perf-task TASK=tasks/... OUT=/tmp/aka-task - Copy task(s) and inject canonical perf helpers"
	@echo "make cleanup-works       - Remove workspace_* directories and logs"
	@echo "make install-cursor-agent- Install the Cursor Agent CLI on the host"

DOCKER_RUNNER := src/scripts/docker_benchmark.sh
CONFIG ?= config.yaml
RUN_ARGS ?=
WORKSPACES ?= $(WORKSPACE)
TASKS ?= $(TASK)
OUT ?= /tmp/aka-materialized-tasks
FORCE ?= 0
MATERIALIZE_FORCE_ARG := $(if $(filter 1 true yes,$(FORCE)),--force,)

docker-shell:
	@$(DOCKER_RUNNER) shell

docker-check-agents:
	@$(DOCKER_RUNNER) check-agents

docker-smoke:
	@$(DOCKER_RUNNER) smoke

docker-run:
	@$(DOCKER_RUNNER) run --config_name $(CONFIG) $(RUN_ARGS)

# Install/upgrade FlyDSL into the container's persistent Python package overlay. Run once
# per machine/image; needed only for flydsl2flydsl tasks.
docker-setup-flydsl:
	@$(DOCKER_RUNNER) setup-flydsl

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

# Run vLLM server with latest ROCm 6.4.1 and vLLM 0.10.1
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
		echo "Don't forget to set local_llm_enabled: true in configs/config.yml"; \
		echo "vLLM server will be running on port 30001, please wait 3 minutes for it to start..."; \
		echo "You can use docker logs -f container_id to check the server status"; \
	fi

# Makefile for KernelBench HIP Kernel Development
# Dynamic ROCm Environment Setup

SHELL := /bin/bash
PYTHON_VERSION := 3.12
VENV_DIR := .venv
REQUIREMENTS := requirements.txt

# Detect ROCm version (supports 7.1, 7.0, 6.4)
ROCM_VERSION := $(shell \
	if [ -d /opt/rocm-7.1.0 ] || [ -d /opt/rocm-7.1 ]; then echo "7.1"; \
	elif [ -d /opt/rocm-7.0.0 ] || [ -d /opt/rocm-7.0 ]; then echo "7.0"; \
	elif [ -d /opt/rocm-6.4.1 ] || [ -d /opt/rocm-6.4.0 ] || [ -d /opt/rocm-6.4 ]; then echo "6.4"; \
	else echo "unknown"; fi)
ROCM_PATH_DETECTED := $(shell \
	if [ -d /opt/rocm-7.1.0 ]; then echo "/opt/rocm-7.1.0"; \
	elif [ -d /opt/rocm-7.1 ]; then echo "/opt/rocm-7.1"; \
	elif [ -d /opt/rocm-7.0.0 ]; then echo "/opt/rocm-7.0.0"; \
	elif [ -d /opt/rocm-7.0 ]; then echo "/opt/rocm-7.0"; \
	elif [ -d /opt/rocm-6.4.1 ]; then echo "/opt/rocm-6.4.1"; \
	elif [ -d /opt/rocm-6.4.0 ]; then echo "/opt/rocm-6.4.0"; \
	elif [ -d /opt/rocm-6.4 ]; then echo "/opt/rocm-6.4"; \
	else echo "/opt/rocm"; fi)

# ROCm environment variables
export ROCM_PATH := $(ROCM_PATH_DETECTED)
export PYTORCH_ROCM_ARCH := gfx90a;gfx942
export CMAKE_PREFIX_PATH := $(ROCM_PATH_DETECTED):$(ROCM_PATH_DETECTED)/hip:/usr/local:/usr
export MAX_JOBS := 8
export HIP_FORCE_DEV_KERNARG := 1
export HSA_NO_SCRATCH_RECLAIM := 1
export HIPCC_COMPILE_FLAGS_APPEND := --offload-arch=gfx942
export AMDGPU_TARGETS := gfx942
export ROCM_ARCH := gfx942

.PHONY: help setup clean

help:
	@echo "AgentKernelArena Evaluation Framework - Makefile Commands"
	@echo "======================================================"
	@echo "make setup  - Complete environment setup (venv + deps)"
	@echo "make clean  - Remove virtual environment"

setup-venv:
	@echo "Detected ROCm version: $(ROCM_VERSION) at $(ROCM_PATH_DETECTED)"
	@if [ "$(ROCM_VERSION)" = "unknown" ]; then \
		echo "ERROR: Could not detect ROCm installation"; \
		exit 1; \
	fi
	@echo "Creating virtual environment with uv..."
	@uv venv $(VENV_DIR) --python $(PYTHON_VERSION)
	@echo "✓ Virtual environment created"
	@echo "Installing PyTorch for ROCm $(ROCM_VERSION)..."
	@source $(VENV_DIR)/bin/activate && \
		uv pip install --upgrade pip setuptools wheel && \
		uv pip install setuptools==75.8.0 && \
		uv pip install setuptools_scm packaging && \
		if [ "$(ROCM_VERSION)" = "7.1" ]; then \
			uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1; \
		elif [ "$(ROCM_VERSION)" = "7.0" ]; then \
			uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0; \
		else \
			uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4; \
		fi
	@echo "✓ PyTorch installed"
	@echo "Installing Python dependencies..."
	@if [ ! -f $(REQUIREMENTS) ]; then \
		echo "Creating requirements.txt..."; \
		echo "# Core ML libraries" > $(REQUIREMENTS); \
		echo "torch" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# Build tools" >> $(REQUIREMENTS); \
		echo "ninja" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# LLM service dependencies" >> $(REQUIREMENTS); \
		echo "pyyaml" >> $(REQUIREMENTS); \
		echo "httpx" >> $(REQUIREMENTS); \
		echo "" >> $(REQUIREMENTS); \
		echo "# Utilities" >> $(REQUIREMENTS); \
		echo "numpy" >> $(REQUIREMENTS); \
	fi
	@source $(VENV_DIR)/bin/activate && uv pip install -r $(REQUIREMENTS)
	@echo "✓ Setup complete! Activate with: source $(VENV_DIR)/bin/activate"

cleanup-venv:
	@echo "Removing virtual environment and build caches..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "build_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

cleanup-works:
	@echo "Removing workspace directories and logs..."
	@rm -rf workspace_*
	@rm -rf logs
	@echo "✓ Workspace directories and logs removed"

install-cursor-agent:
	@echo "Installing Cursor agent..."
	@curl https://cursor.com/install -fsSL | bash


ACTIVATE_VENV_CMD = exec bash -c "source .venv/bin/activate && exec bash"
act:
	$(ACTIVATE_VENV_CMD) 


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
.PHONY: source_venv, install, clean_venv, venv, download_dbs, configure_vscode

USE_NODE ?= false
SRUN_CMD = $(if $(filter true,$(USE_NODE)),echo "Executing in Node..." && srun -p gpu --gres=gpu:2 --pty /bin/bash -c 'make $(MAKECMDGOALS) USE_NODE=false',)

LOG_FILE = make_run.log

node_info:
	@{ \
	echo Hostname: $(shell hostname); \
	echo OS: $(shell uname -s); \
	echo Kernel Version: $(shell uname -r); \
	echo "CPU: $(shell lscpu | grep 'Model name' | awk -F: '{print $$2}' | xargs)"; \
	echo RAM: $(shell free -h | grep 'Mem:' | awk '{print $$2}'); \
	echo Available GPUs: $$(nvidia-smi --query-gpu=name --format=csv,noheader || echo "None"); \
	echo "----------------------------------------"; \
	echo $(SRUN_CMD); \
	if [ "$(USE_NODE)" = "true" ]; then \
		echo "SRUN_CMD is true, finishing execution."; \
		exit 0; \
	fi; \
	} 2>&1 | tee -a $(LOG_FILE); \

source_venv:
	@{ \
	echo "Activating virtual environment..."; \
	. venv/bin/activate; \
	} 2>&1 | tee -a $(LOG_FILE)

venv: clean_venv
	@{ \
	echo "Creating virtual environment..."; \
	python3 -m venv venv; \
	. venv/bin/activate && python -m pip install --upgrade pip; \
	. venv/bin/activate && pip install -r requirements.txt; \
	echo "Initializing .env file..."; \
	echo "PROJECT_PATH=$(shell pwd)" > .env; \
	} 2>&1 | tee -a $(LOG_FILE)

clean_venv: node_info
	@{ \
	if [ "$(USE_NODE)" = "true" ]; then \
		exit 0; \
	fi; \
	echo "Cleaning virtual environment..."; \
	rm -rf venv; \
	rm -rf ~/.cache/; \
	rm -rf .env; \
	} 2>&1 | tee -a $(LOG_FILE)

download_dbs: source_venv
	@{ \
	echo "Downloading databases..."; \
	. venv/bin/activate && python download_db/get_gtsrb_db.py; \
	. venv/bin/activate && python download_db/get_lisa_db.py; \
	} 2>&1 | tee -a $(LOG_FILE)

configure_vscode:
	@{ \
	echo "Configuring VS Code settings..."; \
	mkdir -p .vscode; \
	pwd | sed 's|^|{\"markdown.preview.scrollEditorWithPreview\": false, \"python.defaultInterpreterPath\": \"|' > .vscode/settings.json; \
	echo '\"/bin/python\"}' >> .vscode/settings.json; \
	} 2>&1 | tee -a $(LOG_FILE)

install: node_info clean_venv venv download_dbs configure_vscode
	@{ \
	echo "Installation complete."; \
	} 2>&1 | tee -a $(LOG_FILE)

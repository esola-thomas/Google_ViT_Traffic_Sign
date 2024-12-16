.PHONY: install, clean_venv, venv, download_datasets, configure_vscode, node_info, install_cuda, generate_env_script, process_mapilary

USE_NODE ?= false
SRUN_CMD = $(if $(filter true,$(USE_NODE)),echo "Executing in Node..." && srun -p gpu --gres=gpu:2 --pty /bin/bash -c 'make $(MAKECMDGOALS) USE_NODE=false',)

LOG_FILE = make_run.log

export PROJECT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

CUDA_VERSION=11.4.1
CUDA_INSTALLER=cuda_$(CUDA_VERSION)_470.57.02_linux.run
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/$(CUDA_VERSION)/local_installers/$(CUDA_INSTALLER)
DOWNLOAD_DIR=$(PROJECT_PATH)/downloads
CUDA_INSTALL_DIR=$(DOWNLOAD_DIR)/cuda
MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_URL=https://repo.anaconda.com/miniconda/$(MINICONDA_INSTALLER)
MINICONDA_DIR=$(DOWNLOAD_DIR)/miniconda
CUDNN_VERSION=8.2.4.15
CUDNN_TAR_FILE=cudnn-11.4-linux-x64-v$(CUDNN_VERSION).tgz
CUDNN_URL=https://developer.download.nvidia.com/compute/redist/cudnn/v8.2.4/$(CUDNN_TAR_FILE)

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

venv: clean_venv
	@{ \
	echo "Creating virtual environment..."; \
	python3 -m venv venv; \
	. venv/bin/activate && python -m pip install --upgrade pip; \
	. venv/bin/activate && pip install -r requirements.txt; \
	echo "Initializing .env file..."; \
	echo "PROJECT_PATH=$(shell pwd)" > .env; \
	} 2>&1 | tee -a $(LOG_FILE)

clean_venv:
	@{ \
	if [ "$(USE_NODE)" = "true" ]; then \
		exit 0; \
	fi; \
	make node_info; \
	echo "Cleaning virtual environment..."; \
	rm -rf venv; \
	rm -rf ~/.cache/; \
	rm -rf .env; \
	find -iname "*.pyc" -delete; \
	rm -rf downloads/; \
	rm -rf make_run.log; \
	rm -rf setup_env.sh; \
	} 2>&1 | tee -a $(LOG_FILE); \

download_datasets:
	@{ \
	echo "Downloading databases..."; \
	. venv/bin/activate && python download_datasets/get_dataset_from_yml.py; \
	echo "Downloading Mapillary dataset..."; \
	. venv/bin/activate && python download_datasets/get_mapilary_from_yml.py; \
	} 2>&1 | tee -a $(LOG_FILE)

configure_vscode:
	@{ \
	echo "Configuring VS Code settings..."; \
	mkdir -p .vscode; \
	pwd | sed 's|^|{\"markdown.preview.scrollEditorWithPreview\": false, \"python.defaultInterpreterPath\": \"|; s|$$|/downloads/miniconda/envs/lora_vit/bin/python\"}|' > .vscode/settings.json; \
	} 2>&1 | tee -a $(LOG_FILE)

cuda:
	mkdir -p $(DOWNLOAD_DIR)
	cd $(DOWNLOAD_DIR) && wget $(CUDA_URL)
	sh $(DOWNLOAD_DIR)/$(CUDA_INSTALLER) --silent --toolkit --override --installpath=$(CUDA_INSTALL_DIR)
	export PATH=$(CUDA_INSTALL_DIR)/bin:$$PATH
	export LD_LIBRARY_PATH=$(CUDA_INSTALL_DIR)/lib64:$$LD_LIBRARY_PATH

cudnn:
	cd $(DOWNLOAD_DIR) && wget $(CUDNN_URL)
	tar -xzvf $(DOWNLOAD_DIR)/$(CUDNN_TAR_FILE) -C $(DOWNLOAD_DIR)
	chmod a+r $(CUDA_INSTALL_DIR)/include/cudnn*.h $(CUDA_INSTALL_DIR)/lib64/libcudnn*

miniconda:
	cd $(DOWNLOAD_DIR) && wget $(MINICONDA_URL)
	bash $(DOWNLOAD_DIR)/$(MINICONDA_INSTALLER) -b -p $(MINICONDA_DIR)
	export PATH=$(MINICONDA_DIR)/bin:$$PATH
	conda init bash
	source ~/.bashrc
	conda create -n lora-vit -c conda-forge tensorflow-gpu=2.11.0 python=3.8.10 -y
	. $(MINICONDA_DIR)/bin/activate lora-vit && pip install -r $(PROJECT_PATH)/requirements.txt

verify:
	export PATH=$(CUDA_INSTALL_DIR)/bin:$$PATH && \
	nvcc --version && \
	export PATH=$(MINICONDA_DIR)/bin:$$PATH && \
	source ~/.bashrc && \
	conda activate lora-vit

install: node_info clean_venv cuda cudnn miniconda generate_env_script verify download_datasets
	@{ \
	echo "Installation complete."; \
	} 2>&1 | tee -a $(LOG_FILE)

jupyter_server:
	echo "Starting Jupyter server...";
	conda activate lora-vit && python -m jupyter notebook --ip=10.1.1.52 --port=8888 --no-browser;
	
generate_env_script:
	@{ \
	echo "Generating environment setup script..."; \
	echo "#!/bin/bash" > setup_env.sh; \
	echo "export PROJECT_PATH=$(PROJECT_PATH)" >> setup_env.sh; \
	echo "export PATH=$(PROJECT_PATH)/downloads/miniconda/bin:\$$PATH" >> setup_env.sh; \
	echo "export LD_LIBRARY_PATH=$(PROJECT_PATH)/downloads/miniconda/envs/lora_vit/lib:\$$LD_LIBRARY_PATH" >> setup_env.sh; \
	echo "source $(PROJECT_PATH)/downloads/miniconda/bin/activate lora_vit" >> setup_env.sh; \
	echo "export PATH=$(PROJECT_PATH)/downloads/cuda/bin:\$$PATH" >> setup_env.sh; \
	echo "export LD_LIBRARY_PATH=$(PROJECT_PATH)/downloads/cuda/lib64:\$$LD_LIBRARY_PATH" >> setup_env.sh; \
	chmod +x setup_env.sh; \
	echo "Environment setup script generated as setup_env.sh"; \
	} 2>&1 | tee -a $(LOG_FILE)

process_mapilary:
	@{ \
	echo "Processing Mapillary dataset..."; \
	echo "Make sure the dataset has been downloaded with 'make download_datasets'"; \
	conda activate lora-vit; \
	python process_mapilary.py; \
	} 2>&1 | tee -a $(LOG_FILE)

install_podman:
	@{ \
	echo "Installing Podman..."; \
	if [ -x "$(command -v apt-get)" ]; then \
		apt-get download podman && dpkg -x podman*.deb $(HOME)/podman && rm podman*.deb; \
	elif [ -x "$(command -v yum)" ]; then \
		yumdownloader podman && rpm2cpio podman*.rpm | cpio -idmv && mv usr $(HOME)/podman && rm podman*.rpm; \
	elif [ -x "$(command -v dnf)" ]; then \
		dnf download podman && rpm2cpio podman*.rpm | cpio -idmv && mv usr $(HOME)/podman && rm podman*.rpm; \
	elif [ -x "$(command -v zypper)" ]; then \
		zypper download podman && rpm2cpio podman*.rpm | cpio -idmv && mv usr $(HOME)/podman && rm podman*.rpm; \
	else \
		echo "Package manager not found. Please install Podman manually."; \
		exit 1; \
	fi; \
	export PATH=$(HOME)/podman/bin:$$PATH; \
	echo "Podman installation complete."; \
	} 2>&1 | tee -a $(LOG_FILE)
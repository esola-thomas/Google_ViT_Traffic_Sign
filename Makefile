.PHONY: install, clean_venv, venv, download_datasets, configure_vscode, node_info, install_cuda, generate_env_script

USE_NODE ?= false
SRUN_CMD = $(if $(filter true,$(USE_NODE)),echo "Executing in Node..." && srun -p gpu --gres=gpu:2 --pty /bin/bash -c 'make $(MAKECMDGOALS) USE_NODE=false',)

LOG_FILE = make_run.log

export PROJECT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

CUDA_VERSION=11.4.0
CUDA_INSTALLER=cuda_$(CUDA_VERSION)_470.42.01_linux.run
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/$(CUDA_VERSION)/local_installers/$(CUDA_INSTALLER)

CUDNN_URL=https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/11.x/cudnn-local-repo-rhel9-8.9.7.29-1.0-1.x86_64.rpm/
CUDNN_INSTALLER=cudnn_linux.rpm

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

install_cuda:
	@{ \
	echo "Downloading CUDA Toolkit..."; \
	mkdir -p downloads/cuda; \
	wget $(CUDA_URL) -O downloads/$(CUDA_INSTALLER); \
	echo "Installing CUDA Toolkit..."; \
	sh downloads/$(CUDA_INSTALLER) --silent --toolkit --override --toolkitpath=$(PROJECT_PATH)/downloads/cuda; \
	rm downloads/$(CUDA_INSTALLER); \
	echo "CUDA Toolkit installed."; \
	echo "Downloading cuDNN..."; \
	wget $(CUDNN_URL) -O downloads/$(CUDNN_INSTALLER); \
	echo "Installing cuDNN..."; \
	rpm2cpio downloads/$(CUDNN_INSTALLER) | cpio -idmv -D $(PROJECT_PATH)/downloads/cuda; \
	rm downloads/$(CUDNN_INSTALLER); \
	echo "cuDNN installed."; \
	} 2>&1 | tee -a $(LOG_FILE)

conda_env:
	@{ \
	echo "Downloading Miniconda installer..."; \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
	echo "Installing Miniconda..."; \
	bash miniconda.sh -b -p $(PROJECT_PATH)/downloads/miniconda ; \
	rm miniconda.sh; \
	echo "Initializing Conda..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda init; \
	source ~/.bashrc; \
	echo "Creating Conda environment..."; \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
	echo "Installing Miniconda..."; \
	bash miniconda.sh -b -p $(PROJECT_PATH)/downloads/miniconda ; \
	rm miniconda.sh; \
	echo "Initializing Conda..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda init; \
	source ~/.bashrc; \
	echo "Creating Conda environment..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda create -n lora_vit python=3.9 -y; \
	echo "Activating Conda environment..."; \
	source $(PROJECT_PATH)/downloads/miniconda/bin/activate lora_vit; \
	echo "Installing CUDA Toolkit..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda install -c conda-forge cudatoolkit=$(CUDA_VERSION) -y; \
	echo "Installing cuDNN..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda install -c conda-forge cudnn -y; \
	echo "Installing Python dependencies..."; \
	$(PROJECT_PATH)/downloads/miniconda/bin/conda run -n lora_vit pip install -r $(PROJECT_PATH)/requirements.txt; \
	echo "Conda environment setup complete."; \
	} 2>&1 | tee -a $(LOG_FILE)

install: node_info clean_venv conda_env venv download_datasets configure_vscode generate_env_script
	@{ \
	echo "Installation complete."; \
	} 2>&1 | tee -a $(LOG_FILE)

jupyter_server:
	echo "Starting Jupyter server...";
	srun -p gpu --gres=gpu:1 --pty /bin/bash -c 'source venv/bin/activate && python -m jupyter notebook --ip=10.1.1.52 --port=8888 --no-browser';
	
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
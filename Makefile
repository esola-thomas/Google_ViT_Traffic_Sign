.PHONY: install, clean_venv, venv, download_datasets, configure_vscode, node_info, install_cuda

USE_NODE ?= false
SRUN_CMD = $(if $(filter true,$(USE_NODE)),echo "Executing in Node..." && srun -p gpu --gres=gpu:2 --pty /bin/bash -c 'make $(MAKECMDGOALS) USE_NODE=false',)

LOG_FILE = make_run.log

export PROJECT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

CUDA_VERSION=12.6.3
CUDA_INSTALLER=cuda_$(CUDA_VERSION)_560.35.05_linux.run
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/$(CUDA_VERSION)/local_installers/$(CUDA_INSTALLER)

CUDNN_URL=https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-rhel9-9.5.1-1.0-1.x86_64.rpm
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

# install_cuda:
# 	mkdir -p downloads
# 	cd downloads

# 	@echo "Downloading CUDA Toolkit..."
# 	if [ ! -f $(CUDA_INSTALLER) ]; then \
# 		wget $(CUDA_URL) -O $(CUDA_INSTALLER); \
# 	fi

# 	if [ ! -f $(CUDNN_INSTALLER) ]; then \
# 		wget $(CUDNN_URL) -O $(CUDNN_INSTALLER); \
# 	fi

# 	@{ \
# 	echo "Installing CUDA Toolkit..."; \
# 	mkdir -p downloads; \
# 	sh $(CUDA_INSTALLER) --silent --toolkit --override --toolkitpath="$(PROJECT_PATH)/downloads"; \
# 	echo "Cleaning up..."; \
# 	rm $(CUDA_INSTALLER); \
# 	echo "CUDA Toolkit installed at $(CUDA_PATH)"; \
# 	echo "Add the following lines to your ~/.bashrc or ~/.bash_profile to set the environment variables:"; \
# 	echo "export PATH=$(PROJECT_PATH)/downloads/bin:\$$PATH"; \
# 	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:\$$LD_LIBRARY_PATH"; \
# 	echo "Updating ~/.bashrc with environment variables..."; \
# 	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:\$$LD_LIBRARY_PATH" >> ~/.bashrc; \
# 	echo "Sourcing ~/.bashrc to apply changes..."; \
# 	source ~/.bashrc; \
# 	echo "Installing CUDNN..."; \
# 	rpm2cpio $(CUDNN_INSTALLER) | cpio -idmv -D $(PROJECT_PATH)/downloads; \
# 	} 2>&1 | tee -a $(LOG_FILE)
	
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

install: node_info clean_venv conda_env venv download_datasets configure_vscode
	@{ \
	echo "Installation complete."; \
	} 2>&1 | tee -a $(LOG_FILE)

jupyter_server:
	echo "Starting Jupyter server...";
	srun -p gpu --gres=gpu:1 --pty /bin/bash -c 'source venv/bin/activate && python -m jupyter notebook --ip=10.1.1.52 --port=8888 --no-browser';

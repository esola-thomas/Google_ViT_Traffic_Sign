# Traffic Sign Detection with Google ViT and LoRA

This repository contains work to apply LoRA (Low-Rank Adaptation) to the Google Vision Transformer (ViT) to enhance its performance on traffic sign detection. The next step will be to enhance its performance in avoiding false classification when phantom attacks or other sorts of attacks are present, trying to get the ViT to say a traffic sign is in view when it is a fake one.

## Project Overview

1. **Traffic Sign Detection with LoRA**: Applying LoRA to Google ViT to improve traffic sign detection accuracy.
2. **Robustness Against Attacks**: Enhancing ViT's robustness against phantom attacks and other adversarial attacks to prevent false classifications.

## Make Commands
Define the available make commands that can be run in this section.

- `make install`: Install the necessary dependencies and create the python venv. To execute on a node with GPU access, set `USE_NODE=true`. If not set, the command will execute on the head node.
- `make clean_venv`: Clean the virtual environment for a fresh start. To execute on a node with GPU access, set `USE_NODE=true`. If not set, the command will execute on the head node.


```sh
git clone <>
cd <>
make install
```

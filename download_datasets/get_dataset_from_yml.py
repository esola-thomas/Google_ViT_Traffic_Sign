import kagglehub
from dotenv import load_dotenv, set_key
import os
import yaml

# Load existing .env file
env_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=env_path)

# Load datasets information from YAML file
project_path = os.getenv('PROJECT_PATH')

with open(os.path.join(project_path, 'download_datasets/datasets.yml'), 'r') as file:
    datasets = yaml.safe_load(file)['datasets']

# Iterate over datasets and download them
for dataset in datasets:
    path = kagglehub.dataset_download(f"{dataset['owner']}/{dataset['name']}")
    print(f"Path to {dataset['description']} files:", path)
    set_key(env_path, dataset['env_variable'], path)
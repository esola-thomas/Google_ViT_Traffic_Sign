import kagglehub
from dotenv import load_dotenv, set_key
import os

# Download latest version
path = kagglehub.dataset_download("mbornoe/lisa-traffic-light-dataset")

print("Path to dataset files:", path)
# Load existing .env file
env_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=env_path)

# Add the path to the .env file
set_key(env_path, 'LISA_LIGHT_DB_PATH', path)
import kagglehub
from dotenv import load_dotenv, set_key
import os

# Download latest version
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)
# Load existing .env file
env_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path=env_path)

# Add the path to the .env file
set_key(env_path, 'GTSRB_DB_PATH', path)
import yaml
import requests
import os
import zipfile
from tqdm import tqdm

# Load the YAML file
with open('download_datasets/mapilary.yml', 'r') as file:
    data = yaml.safe_load(file)

# Create a directory to store the downloaded files
os.makedirs('downloads', exist_ok=True)

# Function to download a file with a progress bar
def download_file(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(file_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

# Function to unzip a file with a progress bar
def unzip_file(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit='file') as t:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                t.update(1)

# Download each file
for item in tqdm(data['files'], desc="Overall Progress", unit="file"):
    file_name = item['name']
    url = item['url']
    file_path = os.path.join('downloads', file_name)
    
    # Download the file
    download_file(url, file_path)
    print(f'Downloaded {file_name}')

    # Unzip the file if it is a zip file
    if file_name.endswith('.zip'):
        unzip_file(file_path, 'downloads')
        print(f'Unzipped {file_name}')
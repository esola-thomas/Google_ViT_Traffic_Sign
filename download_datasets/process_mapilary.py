import os
import yaml
import json
from PIL import Image
from tqdm import tqdm

PROJECT_PATH = os.getenv('PROJECT_PATH')
IMAGES_DIR = os.path.join(PROJECT_PATH, 'downloads/mtsd_v2_fully_annotated/images')
ANNOTATIONS_DIR = os.path.join(PROJECT_PATH, 'downloads/mtsd_v2_fully_annotated/annotations')
PROCESSED_DIR = os.path.join(PROJECT_PATH, 'downloads/mtsd_v2_fully_annotated/processed')

# Load categories to process
with open(os.path.join(PROJECT_PATH, 'download_datasets/process_mapilary.yml')) as f:
    categories_to_process = yaml.safe_load(f)

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_image(image_id):
    image_path = os.path.join(IMAGES_DIR, f'{image_id}.jpg')
    annotation_path = os.path.join(ANNOTATIONS_DIR, f'{image_id}.json')

    if not os.path.exists(annotation_path):
        return

    with open(annotation_path) as f:
        annotation = json.load(f)

    image = Image.open(image_path)

    for obj in annotation['objects']:
        if obj['label'] in categories_to_process:
            bbox = obj['bbox']
            cropped_image = image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
            resized_image = cropped_image.resize((224, 224))

            sanitized_label = obj['label'].replace('--', '_')
            category_dir = os.path.join(PROCESSED_DIR, sanitized_label)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir, exist_ok=True)
            output_path = os.path.join(category_dir, f'{image_id}_{obj["key"]}.png')
            resized_image.save(output_path)

def main():
    print(IMAGES_DIR)
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    total_images = len(image_files)
    
    print(f'Total images to process: {total_images}')
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_id = os.path.splitext(image_file)[0]
        process_image(image_id)

if __name__ == '__main__':
    main()
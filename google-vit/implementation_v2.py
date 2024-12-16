import os
from PIL import Image
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from datasets import load_dataset, Dataset, concatenate_datasets, Features, ClassLabel, Image as DatasetsImage
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import re

# 1. Load and Modify the Pre-trained ViT Model
model_name = 'google/vit-base-patch16-224-in21k'
config = ViTConfig.from_pretrained(model_name)
original_num_labels = config.num_labels
new_num_labels = original_num_labels + 1

config.num_labels = new_num_labels

model = ViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

if model.classifier.weight.size(0) == new_num_labels:
    print(f"Classifier layer successfully updated to {new_num_labels} classes.")
else:
    raise ValueError("Classifier layer was not updated correctly.")

# 2. Integrate LoRA with FEATURE_EXTRACTION Task Type and Correct Target Modules
target_modules=[
        f"vit.encoder.layer.{i}.attention.attention.query" for i in range(12)
    ] + [
        f"vit.encoder.layer.{i}.attention.attention.key" for i in range(12)
    ] + [
        f"vit.encoder.layer.{i}.attention.attention.value" for i in range(12)
    ] + [
        f"vit.encoder.layer.{i}.attention.output.dense" for i in range(12)
    ] + [
        f"vit.encoder.layer.{i}.intermediate.dense" for i in range(12)
    ] + [
        f"vit.encoder.layer.{i}.output.dense" for i in range(12)
    ]

existing_modules = set()
for name, module in model.named_modules():
    existing_modules.add(name)

for tm in target_modules:
    pattern = tm.replace("*", ".*")
    if any(re.fullmatch(pattern, name) for name in existing_modules):
        print(f"Module pattern '{tm}' found in the model.")
    else:
        print(f"Module pattern '{tm}' NOT found in the model.")

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print("LoRA integrated into the model.")

# 3. Prepare the Dataset
# login(token="your_hf_token")  # Uncomment and set your token if needed

original_dataset = load_dataset('imagenet-1k', split='train[:1%]')  # Adjust as necessary
class_names = original_dataset.features['label'].names
print(f"Original number of classes: {len(class_names)}")

# Add new class
class_names.append('stop_sign')

features = Features({
    'image': DatasetsImage(),
    'label': ClassLabel(names=class_names)
})

# Load new category images
new_category_images = []
new_category_label = len(class_names) - 1

new_category_path = '/home/ernestos/ws/downloads/mtsd_v2_fully_annotated/processed/regulatory_stop_g1'  # Replace with your path

for img_name in os.listdir(new_category_path):
    img_path = os.path.join(new_category_path, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
        new_category_images.append({'image': image, 'label': new_category_label})
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

new_dataset = Dataset.from_dict({
    'image': [item['image'] for item in new_category_images],
    'label': [item['label'] for item in new_category_images]
}, features=features)

combined_dataset = concatenate_datasets([original_dataset, new_dataset])
combined_dataset = combined_dataset.shuffle(seed=42)

print(f"Combined dataset size: {len(combined_dataset)}")
print(f"Classes after addition: {combined_dataset.features['label'].names}")

# 4. Data Preprocessing
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

def preprocess(examples):
    encoding = feature_extractor(
        images=examples['image'],
        return_tensors='pt',
        size=224
    )
    return {
        'pixel_values': encoding['pixel_values'].squeeze(),
        'labels': examples['label']
    }

combined_dataset = combined_dataset.map(preprocess, batched=True, remove_columns=['image'])
combined_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

print(combined_dataset[0])

# 5. Create DataLoaders
train_test_split = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

# 6. Define Training Parameters and Optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 7. Define Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 8. Training Loop with Evaluation
epochs = 3

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_dataloader, leave=True, desc=f'Epoch {epoch + 1}/{epochs}')
    epoch_loss = 0
    for batch in loop:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_epoch_loss:.4f}")

    eval_accuracy = evaluate(model, eval_dataloader, device)
    print(f"Epoch {epoch + 1} - Evaluation Accuracy: {eval_accuracy:.4f}")

# 9. Save the Fine-Tuned Model
save_directory = 'vit-lora-finetuned'
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
model.config.save_pretrained(save_directory)

print(f"Model and LoRA adapters saved to '{save_directory}'")

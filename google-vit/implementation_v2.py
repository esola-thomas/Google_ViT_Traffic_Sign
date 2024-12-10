from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "google/vit-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

from datasets import load_dataset

dataset_path = "/home/ernestos/ws/downloads/mtsd_v2_fully_annotated/processed/regulatory_stop_g1"
dataset = load_dataset("imagefolder", data_dir=dataset_path)

# Split the dataset into train and validation sets
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

from torchvision.transforms import Compose

# Create transforms for preprocessing images
transforms = Compose([
    lambda img: image_processor(img, return_tensors="pt")['pixel_values'][0]
])

# Apply transforms to the dataset
def preprocess_data(example):
    example['pixel_values'] = transforms(example['image'])
    return example

train_dataset = train_dataset.map(preprocess_data, batched=False)
val_dataset = val_dataset.map(preprocess_data, batched=False)

from peft import get_peft_model, LoraConfig, TaskType

from peft import TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

# Prepare dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):  # Adjust the number of epochs as needed
    model.train()
    for batch in train_loader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(pixel_values=inputs, labels=labels)

# Save the model
model.save_pretrained("./vit_with_stop_sign")

# Evaluate the model
# Use the validation dataset to compute accuracy or other metrics

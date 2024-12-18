import os
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTConfig,
    ViTImageProcessor
)
from datasets import (
    load_dataset,
    Dataset,
    concatenate_datasets,
    Features,
    ClassLabel,
    Image as DatasetsImage
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import types

# -----------------------------
# 1. Load and Modify the Pre-trained ViT Model
# -----------------------------

model_name = 'google/vit-base-patch16-224'  # Ensure using the correct model variant
config = ViTConfig.from_pretrained(model_name)
original_num_labels = config.num_labels
print(f"Original number of labels in the model config: {original_num_labels}")

# Assuming you're fine-tuning on ImageNet-1k with 1000 classes and adding 1 new class
desired_num_labels = 1001  # 1000 original + 1 new
config.num_labels = desired_num_labels

# Load the pre-trained ViT model with the updated configuration
model = ViTForImageClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True  # Allows loading models with different classifier sizes
)

# Verify and update the classifier layer
if model.classifier.weight.size(0) == desired_num_labels:
    print(f"Classifier layer successfully updated to {desired_num_labels} classes.")
    # Initialize the new class's weights to zero to start with neutral predictions
    with torch.no_grad():
        nn.init.zeros_(model.classifier.weight[-1])
else:
    raise ValueError("Classifier layer was not updated correctly.")

# Set the main input name to "pixel_values" as expected by ViT models
model.config.main_input_name = "pixel_values"

# -----------------------------
# 2. Integrate LoRA with Standard Configuration
# -----------------------------

# Define target modules within the ViT encoder layers where LoRA will be applied
target_modules = []
for i in range(config.num_hidden_layers):
    target_modules.extend([
        f"vit.encoder.layer.{i}.attention.attention.query",
        f"vit.encoder.layer.{i}.attention.attention.key",
        f"vit.encoder.layer.{i}.attention.attention.value",
        f"vit.encoder.layer.{i}.attention.output.dense",
        f"vit.encoder.layer.{i}.intermediate.dense",
        f"vit.encoder.layer.{i}.output.dense",
    ])

# Verify that all target modules exist within the model
existing_modules = set()
for name, module in model.named_modules():
    existing_modules.add(name)

for tm in target_modules:
    if tm in existing_modules:
        print(f"Module '{tm}' found in the model.")
    else:
        print(f"Module '{tm}' NOT found in the model. Check the module naming.")

# Define LoRA configuration with standard initialization
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,  # Using FEATURE_EXTRACTION as IMAGE_CLASSIFICATION is unavailable
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    init_lora_weights="gaussian"  # Standard initialization: "gaussian" or "kaiming"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

print("LoRA integrated into the model.")

# -----------------------------
# 3. Override the Forward Method to Prevent 'input_ids'
# -----------------------------

def custom_forward(self, **kwargs):
    """
    Custom forward method to ensure only 'pixel_values' and 'labels' are passed to the base model.
    Removes 'input_ids' if present to prevent TypeError.
    """
    # Remove 'input_ids' if present
    kwargs.pop('input_ids', None)
    # Ensure 'pixel_values' is present
    if 'pixel_values' not in kwargs:
        raise ValueError("Missing 'pixel_values' in inputs.")
    return self.base_model(**kwargs)

# Bind the custom forward method to the model instance
model.forward = types.MethodType(custom_forward, model)

# -----------------------------
# 4. Prepare the Dataset
# -----------------------------

# Uncomment and set your Hugging Face token if authentication is required
# from huggingface_hub import login
# login(token="your_hf_token")  # Replace with your actual token

# Load the original ImageNet dataset (Using a subset for demonstration)
original_dataset = load_dataset('imagenet-1k', split='train[:1%]')  # Adjust the split as needed
class_names = original_dataset.features['label'].names
print(f"Original number of classes: {len(class_names)}")  # Should print 1000

# Add the new class name
class_names.append('stop_sign')

# Define new features with the updated ClassLabel
new_features = Features({
    'image': DatasetsImage(),
    'label': ClassLabel(names=class_names)
})

# Load new category images
new_category_images = []
new_category_label = len(class_names) - 1  # Index of the new class (1000)

# Replace with your actual path to the new category images
new_category_path = 'mapilary_stopsign/regulatory_stop_g1'

# Iterate through the directory and load images
for img_name in os.listdir(new_category_path):
    img_path = os.path.join(new_category_path, img_name)
    if os.path.isfile(img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            new_category_images.append({'image': image, 'label': new_category_label})
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Create a Dataset from the new category images
new_dataset = Dataset.from_dict({
    'image': [item['image'] for item in new_category_images],
    'label': [item['label'] for item in new_category_images]
}).cast(new_features)

# Cast the original_dataset to the new ClassLabel
original_dataset = original_dataset.cast(new_features)

# Verify labels in original_dataset are within [0, 999]
max_original_label = max(original_dataset['label'])
assert max_original_label < 1000, f"Original dataset labels exceed 999. Max label: {max_original_label}"

# Verify labels in new_dataset are exactly 1000
all_new_labels = set(new_dataset['label'])
assert all_new_labels == {1000}, f"New dataset labels are not set to 1000. Found labels: {all_new_labels}"

# Combine the datasets
combined_dataset = concatenate_datasets([original_dataset, new_dataset])

# Shuffle the combined dataset to ensure a good mix of classes
combined_dataset = combined_dataset.shuffle(seed=42)

print(f"Combined dataset size: {len(combined_dataset)}")
print(f"Classes after addition: {combined_dataset.features['label'].names}")

# -----------------------------
# 5. Data Preprocessing
# -----------------------------

# Initialize the feature extractor with the pre-trained model's configuration
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Ensure standard ImageNet normalization
feature_extractor.image_mean = [0.485, 0.456, 0.406]
feature_extractor.image_std = [0.229, 0.224, 0.225]

def preprocess(examples):
    """
    Preprocesses a batch of examples by converting images to RGB and applying the feature extractor.
    """
    # Ensure all images are in RGB format
    images = [img.convert('RGB') for img in examples['image']]
    
    # Apply feature extractor to obtain pixel_values
    encoding = feature_extractor(
        images=images,
        return_tensors='pt'
    )
    
    return {
        'pixel_values': encoding['pixel_values'],
        'labels': examples['label']
    }

# Apply the preprocessing to the combined dataset
combined_dataset = combined_dataset.map(
    preprocess,
    batched=True,
    remove_columns=['image']
)

# Set the dataset format to PyTorch tensors for efficient loading
combined_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

print("Dataset sample:")
print(combined_dataset[0])

# -----------------------------
# 6. Create DataLoaders
# -----------------------------

# Split the dataset into training and evaluation sets
train_test_split = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Initialize DataLoaders for training and evaluation
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# -----------------------------
# 7. Define Training Parameters and Optimizer
# -----------------------------

# Determine the device to run the training on (GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Move the model to the selected device
model.to(device)

# Define the optimizer using standard AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Using PyTorch's AdamW to avoid FutureWarning

# -----------------------------
# 8. Define Evaluation Function
# -----------------------------

def evaluate(model, dataloader, device):
    """
    Evaluates the model on the provided dataloader and returns the accuracy.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Get predictions by selecting the class with the highest logit
            preds = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# -----------------------------
# 9. Training Loop with Evaluation
# -----------------------------

epochs = 3  # Define the number of training epochs

# Verify model's num_labels
print("Model config num_labels:", model.config.num_labels)  # Should print 1001

# Verify classifier layer shape
print("Classifier layer shape:", model.classifier.weight.shape)  # Should print torch.Size([1001, 768])

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    loop = tqdm(train_dataloader, leave=True, desc=f'Epoch {epoch + 1}/{epochs}')
    epoch_loss = 0
    for batch in loop:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Training Loss: {avg_epoch_loss:.4f}")

    # Evaluation after each epoch
    eval_accuracy = evaluate(model, eval_dataloader, device)
    print(f"Epoch {epoch + 1} - Evaluation Accuracy: {eval_accuracy:.4f}")

# -----------------------------
# 10. Save the Fine-Tuned Model
# -----------------------------

save_directory = 'vit-lora-finetuned'
os.makedirs(save_directory, exist_ok=True)

# Save the PEFT model (including LoRA adapters)
model.save_pretrained(save_directory)
model.config.save_pretrained(save_directory)

print(f"Model and LoRA adapters saved to '{save_directory}'")

# -----------------------------
# 11. Optional: Merge LoRA Weights into the Base Model for Efficient Inference
# -----------------------------

# If you wish to merge the LoRA adapters into the base model to eliminate the need for separate adapter loading during inference:
model.merge_and_unload()
print("LoRA adapters merged into the base model.")

# Save the merged model separately if desired
merged_save_directory = 'lora-google-vit-stop-sign'
os.makedirs(merged_save_directory, exist_ok=True)
model.save_pretrained(merged_save_directory)
model.config.save_pretrained(merged_save_directory)
print(f"Merged model saved to '{merged_save_directory}'")

#%%
# Import Modules/Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib
from matplotlib import pyplot as plt
import os
import glob
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
#%%
# Import Libraries and Setup GPU Device
# Confirm GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#%%
# Data Preparation and Transformations
# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.RandomHorizontalFlip(p=0.5),  # Apply random horizontal flip
    transforms.RandomRotation(degrees=15),  # Apply random rotation within 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, etc.
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])
#%%
# Custom Dataset Class
# Create a custom dataset class to handle images without subfolders
class CatDogDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = 0 if 'cat' in os.path.basename(img_path) else 1
        if self.transform:
            image = self.transform(image)
        return image, label

# Load all image paths from the 'dataset/train' directory
image_paths = glob.glob('dataset/train/*.jpg')

# Check if the dataset is loaded correctly
if len(image_paths) == 0:
    raise ValueError("No images found in the dataset/train directory. Please check the path and ensure images are available.")
else:
    print(f"Number of images found: {len(image_paths)}")

# Create dataset
full_dataset = CatDogDataset(image_paths, transform=transform)

# Split the dataset into 70% training and 30% testing
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
#%%
# Demonstrate Some Transformed Samples
# Display a few samples after transformations to verify the augmentation and normalization
def imshow(img, title=None):
    if img.min() < 0 or img.max() > 1:
        img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.savefig('training_validation_loss.png')
print('Loss plot saved as training_validation_loss.png')

# Get some random training images
sample_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
dataiter = iter(sample_loader)
images, labels = next(dataiter)

# Show 6 random images: original, transformed, and normalized
fig, axes = plt.subplots(3, 6, figsize=(18, 9))

# Ensure consistent image paths and transformations
for i in range(6):
    img_path = image_paths[i]
    
    # Load the original image
    original_img = Image.open(img_path).convert('RGB')
    axes[0, i].imshow(original_img)
    axes[0, i].set_title(f"Original: {'cat' if 'cat' in img_path else 'dog'}")
    axes[0, i].axis('off')
    
    # Apply the same transformation used for training
    if transform:
        transformed_img_tensor = transform(original_img)
    
    # Transformed image (after augmentation)
    transformed_img = transformed_img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5  # Reverse normalization for visualization
    transformed_img = np.clip(transformed_img, 0, 1)  # Clip values to [0, 1] range
    axes[1, i].imshow(transformed_img)
    axes[1, i].set_title(f"Transformed: {'cat' if 'cat' in img_path else 'dog'}")
    axes[1, i].axis('off')
    
    # Normalized image
    normalized_img = transformed_img_tensor.permute(1, 2, 0).numpy()
    normalized_img = np.clip(normalized_img, 0, 1)  # Clip values to [0, 1] range
    axes[2, i].imshow(normalized_img)
    axes[2, i].set_title(f"Normalized: {'cat' if 'cat' in img_path else 'dog'}")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
#%%
# Initialize Pre-trained Model
# Load a pre-trained VGG16 model and modify it for binary classification
model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # Freeze the feature extraction layers

# Modify the classifier to output 2 classes (cat and dog)
model.classifier[6] = nn.Linear(4096, 2)

# Move model to GPU if available
model = model.to(device)
#%%
# Loss and Optimizer Setup
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#%%
# Training Loop
# Training loop with model checkpointing to prevent data loss in case of interruption

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, checkpoint_path='models/checkpoint.pth'):
    model.train()
    train_losses, val_losses = [], []

    # Load checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            # Calculate average training loss
            train_losses.append(running_loss / len(train_loader))
            # Validate the model
            val_loss = validate_model(model, criterion, val_loader)
            val_losses.append(val_loss)

            # Print epoch details
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

            # Save checkpoint after each epoch
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)

    return train_losses, val_losses
#%%
# Validation Loop
# Function to validate the model on the validation set

def validate_model(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)
#%%
# Training the Model
# Train the model and save progress periodically
train_losses, val_losses = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10)
#%%
# Plotting Loss Curves
# Load checkpoint if available
checkpoint_path = 'models/checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)  # Removed 'weights_only=True'
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
else:
    train_losses = []
    val_losses = []

# Check the length of losses and their contents
print(f"Length of train_losses: {len(train_losses)}")
print(f"Length of val_losses: {len(val_losses)}")
print(f"train_losses: {train_losses}")
print(f"val_losses: {val_losses}")

# Plot training and validation loss curves to visualize model performance
if len(train_losses) > 0 and len(val_losses) > 0:
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('figures/training_validation_loss.png')
    print('Loss plot saved as training_validation_loss.png')
else:
    print("No loss data to plot.")
#%%
# Save the Final Model
# Save the final trained model
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/classifier_model.pth')
#%%
# Test the Model on Sample Images
# Randomly select 5-6 images from the validation set to evaluate the model's performance

def test_model_on_validation_samples(model, val_dataset, num_samples=5):
    model.eval()
    sample_indices = random.sample(range(len(val_dataset)), num_samples)
    for idx in tqdm(sample_indices, desc="Testing on validation samples"):
        image, label = val_dataset[idx]
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = 'cat' if predicted.item() == 0 else 'dog'
            true_label = 'cat' if label == 0 else 'dog'
            print(f"True Label: {true_label}, Predicted: {predicted_label}")
            imshow(image.cpu().squeeze(), title=f"True: {true_label}, Predicted: {predicted_label}")

# Test the model on some sample validation images
test_model_on_validation_samples(model, test_dataset, num_samples=5)
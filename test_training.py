"""Test training loop with 1 epoch"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from medmnist import PathMNIST
import numpy as np
from models.simple_cnn import SimpleCNN

print("=" * 50)
print("TESTING TRAINING LOOP (1 epoch only)")
print("=" * 50)

# Load data with correct format
print("\n1. Loading data...")
train_dataset = PathMNIST(split='train', download=True)

# FIXED: Convert to correct channel order (C, H, W) instead of (H, W, C)
print("   Converting to PyTorch format (Channels, Height, Width)...")
images_list = []
labels_list = []

for i, (img, label) in enumerate(train_dataset):
    # Original shape: (28, 28, 3) -> Change to (3, 28, 28)
    img = np.transpose(img, (2, 0, 1))
    images_list.append(img)
    labels_list.append(label[0])  # FIXED: Extract scalar from label array
    
    # Print first image shape as verification
    if i == 0:
        print(f"   First image shape after conversion: {img.shape}")

# Convert to tensors
images = torch.tensor(np.array(images_list)).float() / 255.0
labels = torch.tensor(np.array(labels_list)).long()  # FIXED: No squeeze needed

# Create dataset and subset
dataset = torch.utils.data.TensorDataset(images, labels)
small_dataset = Subset(dataset, range(500))
train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)
print(f"   Using {len(small_dataset)} samples")
print(f"   Image shape: {small_dataset[0][0].shape}")  # Should be (3, 28, 28)
print(f"   Label shape: {small_dataset[0][1].shape}")  # Should be scalar

# Setup
print("\n2. Setting up model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"   Device: {device}")

# Train 1 epoch
print("\n3. Training 1 epoch...")
model.train()
running_loss = 0.0
for batch_idx, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)  # FIXED: No squeeze needed - labels are already 1D
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    
    if batch_idx % 5 == 0:
        print(f"   Batch {batch_idx}, Loss: {loss.item():.4f}")

avg_loss = running_loss / len(train_loader)
print(f"\n✅ Training complete!")
print(f"   Average loss: {avg_loss:.4f}")

# Save test model
torch.save(model.state_dict(), 'test_model.pth')
print("✅ Model saved to test_model.pth")

# Clean up
import os
os.remove('test_model.pth')
print("✅ Test model cleaned up")
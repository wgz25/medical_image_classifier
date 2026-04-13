"""Test evaluation on test set"""
import torch
from torch.utils.data import DataLoader, Subset
from medmnist import PathMNIST
import numpy as np
from models.simple_cnn import SimpleCNN
from sklearn.metrics import accuracy_score

print("=" * 50)
print("TESTING EVALUATION")
print("=" * 50)

# Load test data with correct format
print("\n1. Loading test data...")
test_dataset = PathMNIST(split='test', download=True)

# Convert to correct channel order (C, H, W) instead of (H, W, C)
print("   Converting to PyTorch format (Channels, Height, Width)...")
images_list = []
labels_list = []

for img, label in test_dataset:
    # Original shape: (28, 28, 3) -> Change to (3, 28, 28)
    img = np.transpose(img, (2, 0, 1))
    images_list.append(img)
    labels_list.append(label[0])  # Extract scalar from label array

# Convert to tensors
images = torch.tensor(np.array(images_list)).float() / 255.0
labels = torch.tensor(np.array(labels_list)).long()

# Create dataset and subset (use first 200 for quick test)
dataset = torch.utils.data.TensorDataset(images, labels)
small_dataset = Subset(dataset, range(200))
test_loader = DataLoader(small_dataset, batch_size=32)
print(f"   Using {len(small_dataset)} test samples")
print(f"   Image shape: {small_dataset[0][0].shape}")  # Should be (3, 28, 28)

# Create model
print("\n2. Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=9).to(device)
model.eval()

# Evaluate
print("\n3. Running evaluation...")
predictions = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        predictions.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, predictions)
print(f"\n Evaluation complete!")
print(f"   Test accuracy: {accuracy * 100:.2f}%")
print(f"   Random chance would be: ~11.1%")
if accuracy > 0.15:
    print(f"  Your model is learning!")
else:
    print(f" Model needs more training (expected for random weights)")
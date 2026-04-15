import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN
from utils.data_utils import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F


print("=" * 60)
print("CLINICAL IMAGE CLASSIFIER - TRAINING")
print("=" * 60)

# Load data with correct format
print("\n Loading data...")
train_dataset, val_dataset, _ = load_data()

# extract labels from training set for class weight calc
train_labels = []
for _, label in train_dataset:
    train_labels.append(label)
train_labels = np.array(train_labels)

# calculate class weights to handle imbalance
print("\n Calculating class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes = np.unique(train_labels),
    y = train_labels
)
class_weights = torch.tensor(class_weights, dtype = torch.float32)
print(f" Class weights: {class_weights.numpy()}")

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"   Training samples: {len(train_dataset)}")
print(f"   Validation samples: {len(val_dataset)}")
print(f"   Batch size: {batch_size}")

# Setup model
print("\n Setting up model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
# add weight decay for regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

print(f"   Device: {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
num_epochs = 5
train_losses = []
val_accuracies = []
best_val_accuracy = 0

print(f"\n Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    scheduler.step(accuracy)

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
        print(f'   Best model saved! ({accuracy:.2f}%)')
    
    print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Accuracy = {accuracy:.2f}%')

# Save model
torch.save(model.state_dict(), 'checkpoints/model.pth')
print(f"\n Model saved to checkpoints/model.pth")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
print(" Training curves saved to training_curves.png")

print("\n Training complete!")
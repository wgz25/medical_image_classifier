import medmnist
from medmnist import PathMNIST
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("EXPLORING MEDICAL IMAGES")
print("=" * 50)

print("\n1. Loading PathMNIST dataset...")
train_dataset = PathMNIST(split='train', download=True)

print(f"\n2. Dataset Info:")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Image shape: {train_dataset[0][0].shape}")  # (28, 28, 3)
print(f"   Number of classes: {len(train_dataset.info['label'])}")
print(f"   Classes: {train_dataset.info['label']}")

# Fix channel order for visualization
print("\n3. Converting to PyTorch format (C, H, W)...")
img, label = train_dataset[0]
img_fixed = np.transpose(img, (2, 0, 1))  # (3, 28, 28)
print(f"   Original shape: {img.shape}")
print(f"   Converted shape: {img_fixed.shape}")

# Show sample images (using original format for matplotlib)
print("\n4. Displaying sample images...")
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    # Matplotlib expects (H, W, C) for RGB
    ax.imshow(img)  # Original (28, 28, 3) works for display
    ax.set_title(f'Class: {label[0]}')  # Extract scalar from label array
    ax.axis('off')

plt.suptitle('PathMNIST Dataset Examples (Original Format)')
plt.tight_layout()
plt.savefig('data_exploration.png')
print("   Sample images saved to data_exploration.png")

print("\n Data exploration complete!")
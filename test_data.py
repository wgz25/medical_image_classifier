"""Test data loading with MedMNIST"""
from medmnist import PathMNIST
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("TESTING DATA LOADING")
print("=" * 50)

print("\n1. Loading PathMNIST dataset...")
train_dataset = PathMNIST(split='train', download=True)

print(f"\n2. Dataset Info:")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Number of classes: {len(train_dataset.info['label'])}")
print(f"   Class labels: {train_dataset.info['label']}")

print(f"\n3. Sample Image:")
sample_image, sample_label = train_dataset[0]
print(f"   Image type: {type(sample_image).__name__}")
print(f"   Image size (PIL): {sample_image.size}")
print(f"   Label: {sample_label[0]}")

# Convert to numpy
img_np = np.array(sample_image)
print(f"   Numpy shape: {img_np.shape}")

# Show image
plt.figure(figsize=(4, 4))
plt.imshow(img_np)
plt.title(f"Sample Image - Class: {sample_label[0]}")
plt.axis('off')
plt.savefig('test_sample.png')
print("\n4. Sample image saved to test_sample.png")

print("\n Data loading test passed!")
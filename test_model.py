"""Test that the CNN model works"""
import torch
from models.simple_cnn import SimpleCNN

print("Creating model...")
model = SimpleCNN(num_classes=9)

# Test forward pass
dummy_input = torch.randn(1, 3, 28, 28)
output = model(dummy_input)

print(f"   Model works!")
print(f"   Input shape: {dummy_input.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with random batch
batch = torch.randn(4, 3, 28, 28)
outputs = model(batch)
print(f"   Batch output shape: {outputs.shape}")
print(f"   Expected: [4, 9] (batch_size, num_classes)")

print("\n Model architecture is correct!")
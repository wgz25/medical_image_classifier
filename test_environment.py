# test environment 
"""Test that everything is installed correctly"""
import sys
import torch
import torchvision
import medmnist
import matplotlib
import numpy
import sklearn
import tqdm

print("=" * 50)
print("all imports working!")
print("=" * 50)
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"MedMNIST: {medmnist.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print("=" * 50)

# Quick tensor test
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor test: {x}")
print("Environment is ready!")
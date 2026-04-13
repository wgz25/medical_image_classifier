import torch
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN
from utils.data_utils import load_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("CLINICAL IMAGE CLASSIFIER - EVALUATION")
print("=" * 60)

# Load data
print("\n Loading test data...")
_, _, test_dataset = load_data()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"   Test samples: {len(test_dataset)}")

# Load model
print("\n Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=9).to(device)
model.load_state_dict(torch.load('checkpoints/model.pth', map_location=device))
model.eval()
print(f"   Device: {device}")

# Evaluate
print("\n Running evaluation...")
predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        predictions.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, predictions)
print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\n Classification Report:")
print(classification_report(all_labels, predictions, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print(" Confusion matrix saved to confusion_matrix.png")

print("\n Evaluation complete!")
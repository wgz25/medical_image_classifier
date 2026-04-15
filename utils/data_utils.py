import torch
import numpy as np
from medmnist import PathMNIST

def load_data():
    """Load PathMNIST dataset with correct channel order for PyTorch"""
    
    # load training, validation, and test datasets. download automatically if not cached
    print("Loading PathMNIST datasets...")
    train_dataset = PathMNIST(split='train', download=True)
    val_dataset = PathMNIST(split='val', download=True)
    test_dataset = PathMNIST(split='test', download=True)
    
    def convert_to_correct_format(dataset, dataset_name=""):
        """Convert MedMNIST format to PyTorch format (C, H, W)"""
        print(f"   Converting {dataset_name} dataset...")
        images_list = []
        labels_list = []
        
        # convert for different order, medmnist default is (H, W, C)
        for img, label in dataset:
            # img shape is (28, 28, 3) - convert to (3, 28, 28)
            img = np.transpose(img, (2, 0, 1))
            images_list.append(img)
            labels_list.append(label[0])  # Extract scalar from label array
        
        # normalize input from (0, 255) (rgb values) to (0, 1). .float = float32 for standard, .long = float64, required for cross-entropy loss
        images = torch.tensor(np.array(images_list)).float() / 255.0
        labels = torch.tensor(np.array(labels_list)).long()
        
        print(f"      Images shape: {images.shape}")
        print(f"      Labels shape: {labels.shape}")
        
        # return a TensorDataset, wraps all tensors into one set
        return torch.utils.data.TensorDataset(images, labels)
    
    # update and return 
    train_data = convert_to_correct_format(train_dataset, "training")
    val_data = convert_to_correct_format(val_dataset, "validation")
    test_data = convert_to_correct_format(test_dataset, "test")
    
    return train_data, val_data, test_data

# quick test
if __name__ == "__main__":
    train, val, test = load_data()
    print(f"\n Data loading works!")
    print(f"   Training sample shape: {train[0][0].shape}")
    print(f"   Training label: {train[0][1]}")
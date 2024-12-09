import pandas as pd
import torch

def load_data(skeletons_path, labels_path):
    # Load skeletons data
    # Load skeletons data
    skeletons = torch.load(skeletons_path)
    
    # Load labels data
    labels = torch.load(labels_path)
    
    return skeletons, labels

# Example usage
skeletons_path = 'dataset/data_skeletons.pt'
labels_path = 'dataset/data_labels.pt'
skeletons, labels = load_data(skeletons_path, labels_path)

print(skeletons)
print(labels)
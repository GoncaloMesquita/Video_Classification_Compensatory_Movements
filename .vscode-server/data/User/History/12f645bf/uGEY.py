import pandas as pd

def load_data(skeletons_path, labels_path):
    # Load skeletons data
    skeletons = pd.read_csv(skeletons_path)
    
    # Load labels data
    labels = pd.read_csv(labels_path)
    
    return skeletons, labels

# Example usage
skeletons_path = '/data/skeletons.csv'
labels_path = '/data/labels.csv'
skeletons, labels = load_data(skeletons_path, labels_path)

print(skeletons.head())
print(labels.head())
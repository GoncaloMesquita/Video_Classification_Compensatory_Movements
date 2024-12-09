import pandas as pd
import torch

def load_data(skeletons_path, labels_path):
    # Load skeletons data
    # Load skeletons data
    skeletons = torch.load(skeletons_path)
    
    # Load labels data
    labels = torch.load(labels_path)
    
    return skeletons, labels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def analyze_skeleton_data_3d(skeleton_data, labels):
    """
    Analyze 3D skeleton data and labels, providing statistics on variance, missing values, 
    and class distribution. Handles skeleton data in the format [examples, frames, joints].
    
    Parameters:
    - skeleton_data: A 3D NumPy array or tensor where each example is [frames, joints].
    - labels: A list or array of labels corresponding to each example.
    
    Returns:
    - Dictionary with statistical information.
    """
    analysis_report = {}

    # Convert the 3D skeleton data into a 2D structure for easier analysis.
    # Flatten the data: Each row corresponds to a single frame, and columns represent joints.
    num_examples, num_frames, num_joints = skeleton_data.shape
    flattened_skeleton_data = skeleton_data.reshape(num_examples * num_frames, num_joints)

    # Convert the flattened data to a pandas DataFrame for analysis
    skeleton_df = pd.DataFrame(flattened_skeleton_data)

    # 1. Check for missing values (NaNs)
    missing_values = skeleton_df.isnull().sum().sum()
    analysis_report['Missing Values'] = missing_values
    
    # 2. Variance of skeleton data
    variance_per_joint = skeleton_df.var()
    low_variance_joints = variance_per_joint[variance_per_joint < 1e-5].index.tolist()
    analysis_report['Low Variance Joints'] = low_variance_joints
    analysis_report['Overall Variance'] = variance_per_joint.describe()
    
    # Histogram of 100 examples in the skeleton data
    sample_data = skeleton_data[:100].reshape(100 * num_frames, num_joints)
    plt.figure(figsize=(12, 8))
    plt.hist(sample_data.flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Joint Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of 100 Examples in Skeleton Data')
    plt.show()

    # Dataset statistics
    dataset_stats = {
        'Mean': skeleton_df.mean().mean(),
        'Variance': skeleton_df.var().mean(),
        'Max': skeleton_df.max().max(),
        'Min': skeleton_df.min().min(),
        '25th Percentile': skeleton_df.quantile(0.25).mean(),
        '50th Percentile (Median)': skeleton_df.median().mean(),
        '75th Percentile': skeleton_df.quantile(0.75).mean()
    }
    analysis_report['Dataset Stats'] = dataset_stats
    # 3. Check class imbalance in labels
    label_counts = Counter(labels)
    analysis_report['Label Distribution'] = label_counts
    total_samples = sum(label_counts.values())
    label_ratios = {label: count / total_samples for label, count in label_counts.items()}
    analysis_report['Label Ratios'] = label_ratios

    # Plotting class distribution for better visualization
    plt.figure(figsize=(8, 6))
    plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution of Labels')
    plt.show()

    # 4. Outlier detection based on standard deviation (z-score method)
    outliers = (np.abs(skeleton_df - skeleton_df.mean()) > 3 * skeleton_df.std()).sum().sum()
    analysis_report['Outliers'] = outliers

    # 5. Basic Statistics: Mean, Standard Deviation
    basic_stats = skeleton_df.describe()
    analysis_report['Basic Stats'] = basic_stats

    print(analysis_report)
    return analysis_report



# Example usage
skeletons_path = 'dataset/data_skeletons.pt'
labels_path = 'dataset/data_labels.pt'
skeletons, labels = load_data(skeletons_path, labels_path)

print(skeletons)
print(labels)
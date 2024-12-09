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
    plt.savefig('histogram.png')
    plt.show()

    # Dataset statistics
    dataset_stats = {
        'Mean': skeleton_df.mean().mean(),
        'Variance': skeleton_df.var().mean(),
        'Max': skeleton_df.max().max(),
        'Min': skeleton_df.min().min(),
        '25th Percentile': skeleton_df.quantile(0.25).mean(),
        '50th Percentile (Median)': skeleton_df.median().mean(),
        '75th Percentile': skeleton_df.quantile(0.75).mean(),
        'Zeros Percentage': (skeleton_df == 0).sum().sum() / skeleton_df.size * 100
    }
    analysis_report['Dataset Stats'] = dataset_stats
    # 3. Check class imbalance in labels
    # Flatten the multi-labels to count occurrences of each label combination
    label_counts = Counter(map(tuple, labels))
    ones_per_column = (skeleton_df == 1).sum()
    analysis_report['Ones Per Column'] = ones_per_column


    # Plotting class distribution for better visualization


    # 4. Outlier detection based on standard deviation (z-score method)
    outliers = (np.abs(skeleton_df - skeleton_df.mean()) > 3 * skeleton_df.std()).sum().sum()
    analysis_report['Outliers'] = outliers

    # 5. Basic Statistics: Mean, Standard Deviation

    return analysis_report



# Example usage
skeletons_path = 'dataset/data_skeletons.pt'
labels_path = 'dataset/data_labels.pt'
skeletons, labels = load_data(skeletons_path, labels_path)

max_length = max(len(row) for row in skeletons)
padded_rows = []
padded_rows = np.zeros((len(skeletons), max_length, 99))

for idx, row in enumerate(skeletons):
    # flattened_row = np.ravel(row)  
    row_np = np.array(row)
    row_pad = np.pad(row_np, ((0, max_length-row_np.shape[0]),(0, 0) ), mode='constant')

    padded_rows[idx] = row_pad

analyze_skeleton_data_3d(padded_rows, labels)
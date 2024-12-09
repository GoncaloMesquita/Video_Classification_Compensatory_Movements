import pandas as pd
import glob
import argparse
import os
import numpy as np

label_files = "dataset/E*_labels.csv"
keypoint_files = "dataset/E*_mp_world_landmarks.csv"

# Find the matching files
labels_files = glob.glob(label_files)
keypoints_files = glob.glob(keypoint_files)

df = []
labels = []

dict = {}

for file1, file2 in zip(labels_files, keypoints_files):
    label = pd.read_csv(file1)
    key = pd.read_csv(file2)
    
    # Iterate over each row in the label DataFrame
    for m in label["pid"].unique():  # Assuming "pid" indicates unique patient IDs
        
        # Filter rows for the current patient
        patient_data = label[label["pid"] == m]
        
        for index, row in patient_data.iterrows():
            data = []
            if "affected" in patient_data.columns:
                n = row["affected"] 
        
                for i in range(row["frame_init"], row["frame_end"]):
                    
                    data.append(key[(key["pid"] == m) & (key["affected"] == n) & (key["frame"] == i) ].iloc[:, 2:].values.flatten().tolist())
                    
            else:
                
                for i in range(row["frame_init"], row["frame_end"]):
                    
                    data.append(key[(key["pid"] == m) & (key["frame"] == i)].iloc[:, 2:].values.flatten().tolist())
                    
            df.append(data)
            labels.append(row[["comp", "comp_sh", "comp_sh_ele", "comp_sh_abd", "comp_tr", "comp_hd"]].tolist())
            
    dict[m] = df 
    

empty_indices = [i for i, x in enumerate(df) if not x]
df = [x for x in df if x]
labels = [x for i, x in enumerate(labels) if i not in empty_indices]

max_length = max(len(row) for row in df)
padded_rows = []
padded_rows = np.zeros((len(df), max_length, 166))

for idx, row in enumerate(df):
    # flattened_row = np.ravel(row)  
    row_np = np.array(row)
    row_pad = np.pad(row_np, ((0, max_length-row_np.shape[0]),(0, 0) ), mode='constant')

    padded_rows[idx] = row_pad


labels = np.array(labels)

np.save("dataset/dataset.npy", padded_rows)
np.save("dataset/labels.npy", labels) 
print()

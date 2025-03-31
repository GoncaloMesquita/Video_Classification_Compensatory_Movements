import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from torch.utils.data import Dataset, DataLoader
import json
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix, roc_curve
import pickle
from sklearn.metrics import RocCurveDisplay, auc
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd


class EarlyStopping:
    
    def __init__(self, patience, model_name, learning_rate, batch_size, output_dir, verbose=True, delta=0, optuna=False):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.output = output_dir
        self.model = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optuna = optuna
        self.check_point = None

    def __call__(self, val_loss, model, fold, epoch):

        self.fold = fold
        self.epoch = epoch
        

        if self.best_score is None:
            self.best_score = val_loss
            self.check_point = self.save_checkpoint(model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.check_point = self.save_checkpoint(model)
            self.counter = 0
          
        return self.check_point

    def save_checkpoint(self, model):
        if self.verbose and not self.optuna:
            # torch.save(model.state_dict(), f"{self.output}/{self.model}_{self.fold}_best.pth")
            torch.save(model.state_dict(), f"{self.output}/{self.model}_{self.fold}_best.pth")
        print("Validation loss improved. Saving the model...", flush=True)
        return f"{self.output}/{self.model}_{self.fold}_best.pth"


def metrics(targets, predictions, mode, output_dir, model_name):
    # Calculate metrics
    # accuracy = accuracy_score(targets, predictions)
    precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
    recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    
    # precision_sample = precision_score(targets, predictions, average='samples', zero_division=0)
    # recall_sample = recall_score(targets, predictions, average='samples', zero_division=0)
    # f1_sample = f1_score(targets, predictions, average='samples', zero_division=0)
    
    accuracy_per_label = np.mean(targets == predictions, axis=0)
    precision_per_label = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_label = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_label = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Store metrics in a dictionary
    metrics_dict = {
        'mode': mode,
        # 'accuracy_micro': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        # 'precision_sample': precision_sample,
        # 'recall_sample': recall_sample,
        # 'f1_sample': f1_sample,
        'accuracy_per_label': accuracy_per_label.tolist(),
        'precision_per_label': precision_per_label.tolist(),
        'recall_per_label': recall_per_label.tolist(),
        'f1_per_label': f1_per_label.tolist()
    }
    
    # if mode == 'test':
    # Print metrics
    # print(f" Accuracy: {accuracy}, Precision micro: {precision_micro:.3f}, Recall micro: {recall_micro:.3f}, F1 micro: {f1_micro:.3f}", flush=True)
    # print(f"Precision sample: {precision_sample:.3f}, Recall sample: {recall_sample:.3f}, F1 sample: {f1_sample:.3f}", flush=True)
    # print(f"Accuracy per label: {accuracy_per_label}, Precision per label: {precision_per_label}, Recall per label: {recall_per_label}, F1 per label: {f1_per_label}", flush=True)
        
    # Save metrics to a JSON file
    
    data = [
        # round(accuracy, 3),
        round(precision_micro, 3),
        round(recall_micro, 3),
        round(f1_micro, 3),
        # round(precision_sample, 3),
        # round(recall_sample, 3),
        # round(f1_sample, 3),
        np.round(accuracy_per_label, 3).tolist(),
        np.round(precision_per_label, 3).tolist(),
        np.round(recall_per_label, 3).tolist(),
        np.round(f1_per_label, 3).tolist()
    ]
    return data


def metrics_evaluate(data, output, model_name):
    
    data = np.array(data, dtype=object)
    avg_accuracy = np.mean([d[0] for d in data])
    avg_precision_micro = np.mean([d[1] for d in data])
    avg_recall_micro = np.mean([d[2] for d in data])
    avg_f1_micro = np.mean([d[3] for d in data])
    avg_precision_sample = np.mean([d[4] for d in data])
    avg_recall_sample = np.mean([d[5] for d in data])
    avg_f1_sample = np.mean([d[6] for d in data])
    avg_accuracy_per_label = np.mean([d[7] for d in data], axis=0)
    avg_precision_per_label = np.mean([d[8] for d in data], axis=0)
    avg_recall_per_label = np.mean([d[9] for d in data], axis=0)
    avg_f1_per_label = np.mean([d[10] for d in data], axis=0)
    
    metrics_dict = {
    'model_name': model_name,
    'accuracy': f"{avg_accuracy:.4f} +/- {np.std([d[0] for d in data]):.4f}",
    'precision_micro': f"{avg_precision_micro:.4f} +/- {np.std([d[1] for d in data]):.4f}",
    'recall_micro': f"{avg_recall_micro:.4f} +/- {np.std([d[2] for d in data]):.4f}",
    'f1_micro': f"{avg_f1_micro:.4f} +/- {np.std([d[3] for d in data]):.4f}",
    
    'precision_sample': f"{avg_precision_sample:.4f} +/- {np.std([d[4] for d in data]):.4f}",
    'recall_sample': f"{avg_recall_sample:.4f} +/- {np.std([d[5] for d in data]):.4f}",
    'f1_sample': f"{avg_f1_sample:.4f} +/- {np.std([d[6] for d in data]):.4f}",

    'accuracy_per_label': [f"{avg:.4f} +/- {std:.4f}" for avg, std in zip(avg_accuracy_per_label, np.std([d[7] for d in data], axis=0))],
    'precision_per_label': [f"{avg:.4f} +/- {std:.4f}" for avg, std in zip(avg_precision_per_label, np.std([d[8] for d in data], axis=0))],
    'recall_per_label': [f"{avg:.4f} +/- {std:.4f}" for avg, std in zip(avg_recall_per_label, np.std([d[9] for d in data], axis=0))],
    'f1_per_label': [f"{avg:.4f} +/- {std:.4f}" for avg, std in zip(avg_f1_per_label, np.std([d[10] for d in data], axis=0))]
    }

        # Save metrics to a JSON file
    output_filepath = os.path.join(output, f'{model_name}_avg_metrics.json')
    with open(output_filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    return


def plotting_loss(t_loss, v_loss, fold, epochs, model, output_dir, batch_size, learning_rate):
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs + 1), t_loss, label='Training Loss', marker='o')
    plt.plot(range(0, epochs + 1), v_loss, label='Validation Loss', marker='o')
    plt.title(f'Training and Validation Loss. Fold: {fold}, Epochs: {epochs}, Batch Size: {batch_size}.jpg')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_filename = f'Losses-PS:{model}_{batch_size}_{fold}_{learning_rate}.jpg'
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    return


def center_crop_square(image, target_size=224):
    width, height = image.shape[1], image.shape[0]
    new_side = min(width, height)
    left = (width - new_side) / 2
    top = (height - new_side) / 2
    right = (width + new_side) / 2
    bottom = (height + new_side) / 2
    image = image[int(top):int(bottom), int(left):int(right)]
    return image


class CustomDataset(Dataset):
    
    def __init__(self, data, targets, model_name, type_train, data2=None, target_size=256, max_frames=None):
        self.data = data
        self.targets = targets
        self.data2 = data2
        self.max_len = max(len(row) for row in data)
        self.lengths = torch.tensor([len(seq) for seq in data], dtype=torch.long)
        self.model_name = model_name
        self.target_size = target_size
        self.max_frames = max_frames  # Limit on frames to load per video, if specified
        self.type_train = type_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Primary data (assume it's already preprocessed)
        if self.type_train:
            sample_data = self.data[idx]
            sample_target = self.targets[idx]
            return sample_data, sample_target, 0, 0
            
        sample_data = self.data[idx]
        sample_target = self.targets[idx]

        # Process secondary data (data2) if provided
        if self.data2 is not None:
            # Assume each entry in data2 is a folder path containing frames
            sample_data2 = self.data2[idx]
            # sample_data2 = self.load_and_preprocess_frames(folder_path)
        else:
            sample_data2 = None

        return sample_data, sample_target, self.model_name, sample_data2


def create_dataloader(x, y, batch_size, shuffle, model_name, seq_size, trainII, x2):
    
    if trainII:
        dataset = CustomDataset(x, y, model_name, trainII, x2) 
        dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    else:
        dataset = CustomDataset(x, y, model_name, trainII, x2) 
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, seq_size=seq_size), shuffle=shuffle)
    
    return  dataloader


def collate_fn(batch, seq_size):
    """
    batch: list of tuples:
       (sequence, label, model_name, sequence2)
    sequence: can be a list/array of shape [num_frames, ...]
    label: your label (float or int)
    model_name: string or something
    sequence2: optional second input
    """
    sequences, labels, model_names, sequences2 = zip(*batch)
    
    # Just pick the first model_name for the batch (assuming they're the same)
    model_name = model_names[0]
    
    # If you have a special max_length for certain models:
    if model_name == 'SkateFormer':
        max_length = seq_size
    elif model_name in ['moment', 'moment+dino']:
        max_length = seq_size
    else:
        # If none of the above, just find the largest
        max_length = max(len(seq) for seq in sequences)

    # Pad the first input
    padded_ske = []
    for seq in sequences:
        seq = torch.tensor(seq)  # shape: [num_frames, ...]
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            # Example: replicate the *last* frame for padding
            pad_frame = seq[-1].unsqueeze(0).repeat(pad_length, *[1]*(seq.ndim-1))
            padded_seq = torch.cat((seq, pad_frame), dim=0)
        else:
            # If the sequence is longer, truncate
            padded_seq = seq[:max_length]
        padded_ske.append(padded_seq)
    padded_sequences = torch.stack(padded_ske, dim=0)  
    # shape => [batch_size, max_length, frame_dim...]

    # Convert labels to a tensor
    # (Assuming each label is just a scalar or a 1D array)
    labels_tensor = torch.tensor(labels)

    # If sequences2 is all None, return None
    if sequences2[0] is None:
        return padded_sequences, labels_tensor, torch.tensor([len(seq) for seq in sequences]), None

    # Otherwise, do the same padding logic for sequences2
    padded_ske2 = []
    for seq2 in sequences2:
        seq2 = torch.tensor(seq2)
        if len(seq2) < max_length:
            pad_length = max_length - len(seq2)
            pad_frame = seq2[-1].unsqueeze(0).repeat(pad_length, *[1]*(seq2.ndim-1))
            padded_seq2 = torch.cat((seq2, pad_frame), dim=0)
        else:
            padded_seq2 = seq2[:max_length]
        padded_ske2.append(padded_seq2)
    padded_sequences2 = torch.stack(padded_ske2, dim=0)
    
    return padded_sequences, labels_tensor, torch.tensor([len(seq) for seq in sequences]), padded_sequences2


    
def initial_distance(skeletons, size, n_dim):
    features = []
    
    # Iterate through each film
    for sequence in skeletons:
        
        initial_position = np.array(sequence[0])
        
        initial_position = initial_position.reshape(-1, n_dim)
        
        feature_sequence = []
        
        for frame in sequence:
            current_position = np.array(frame)
            current_position = current_position.reshape(-1, n_dim)  # Reshape each frame similarly
            
            distance = current_position - initial_position
            
            feature_sequence.append(distance.reshape(size))
        
        # Append the sequence of features for the current film
        features.append(feature_sequence)
    
    return features


def moving_average_filter(sequence, window_size=5):

    person = []
    
    for seq in sequence:
        
        filtered_sequence = []
        
        for i in range(len(seq)):
            
            start_idx = max(0, i - window_size + 1)
            window = seq[start_idx:i + 1] 
            window_average = np.mean(window, axis=0)
            filtered_sequence.append(window_average)
            
        person.append(filtered_sequence)
        
    return person


def remove_none_from_nested_list(nested_list):
    return [
        [
            [
                frame for frame in subject if frame is not None
            ] for subject in trial if subject is not None
        ] for trial in nested_list if trial is not None
    ]

def normalize_skeleton(ske, n_dim):

    normalized_skeletons = {}
    
    for person_id, videos in ske.items():
        normalized_videos = []
        
        for video in videos:
            normalized_frames = []
            video_pose = np.array(video).reshape(len(video), 17, 3)[:,:,0:2].tolist()
            # Get reference distance from first frame for consistent scaling
            first_frame = np.array(video)[:,0:2].reshape(-1, n_dim)
            
            # Get shoulder coordinates (ignoring confidence)
            left_shoulder = first_frame[5][:2]  # Use only x,y from left shoulder
            right_shoulder = first_frame[6][:2]  # Use only x,y from right shoulder
            
            # Calculate reference distance (shoulder width)
            reference_dist = np.linalg.norm(left_shoulder - right_shoulder)
            
            for frame in video_pose:
                # Convert to numpy array
                frame_arr = np.array(frame).reshape(-1, n_dim)
                
                # Get hip coordinates (ignoring confidence)
                left_hip = frame_arr[11][:2].copy()  # Use only x,y
                right_hip = frame_arr[12][:2].copy()  # Use only x,y
                
                # Calculate the midpoint between hips
                mid_hip = (left_hip + right_hip) / 2.0
                
                # Create array to hold normalized coordinates
                normalized_frame = frame_arr.copy()
                
                # Center all joints by subtracting mid_hip (only for x,y coordinates)
                for j in range(frame_arr.shape[0]):
                    normalized_frame[j, 0] = frame_arr[j, 0] - mid_hip[0]  # Center x
                    normalized_frame[j, 1] = frame_arr[j, 1] - mid_hip[1]  # Center y
                    # Keep confidence value (index 2) unchanged
                
                # Scale based on reference shoulder distance if it's valid
                if reference_dist > 0:
                    for j in range(frame_arr.shape[0]):
                        normalized_frame[j, 0] = normalized_frame[j, 0] / reference_dist  # Scale x
                        normalized_frame[j, 1] = normalized_frame[j, 1] / reference_dist  # Scale y
                        # Keep confidence value unchanged
                
                # Add to normalized frames
                normalized_frames.append(normalized_frame.tolist())
            
            normalized_videos.append(normalized_frames)
        
        normalized_skeletons[person_id] = normalized_videos
    
    return normalized_skeletons


def load_data(data_dir_labels, data_dir_skeletons, num_patients, input_size, mode, n_dim):
    
    ske = torch.load(data_dir_skeletons, weights_only=False)
    labels = torch.load(data_dir_labels, weights_only=False)
    true_data = None
    
    if data_dir_labels.rsplit('/', 1)[1][0:5]  == 'MMAct': 
        ske = {
            key: remove_none_from_nested_list(value) for key, value in ske.items()
        }
        
        if mode != 'pseudo-label' and mode != 'visualization':
            for key in labels.keys():
                # For MMAct dataset, flatten nested lists to get the first level of labels
                labels[key] = [outer_list[0]  for outer_list in labels[key]]
                labels[key] = [[1-label for label in sublist] for sublist  in labels[key]]
        else:
            for key in labels.keys():
                labels[key] = [[1 - label for  label in sublist] for sublist in labels[key]]
    
    if mode == 'pseudo-label' and data_dir_labels.rsplit('/', 1)[1][0:7]  == 'Toronto':
        true_data = labels
    
    if data_dir_labels.rsplit('/', 1)[1][0:7]  == 'Toronto':
        label_video = {}
        for key in labels.keys():
            label_video[key] = []
            for label in labels[key]:
                if any(l == 0 for l in label):
                    label_video[key].append([0])
                else:
                    label_video[key].append([1])
                    
        labels = label_video
        

    append_indices = []
    for key in ske.keys():
        if isinstance(ske[key], list):  
            indices_to_keep = [i for i, sublist in enumerate(ske[key]) if sublist != []]

            ske[key] = [ske[key][i] for i in indices_to_keep]
            labels[key] = [labels[key][i] for i in indices_to_keep]
            append_indices.append(indices_to_keep)
            
    cross_val_data = []
    if data_dir_labels.rsplit('/', 1)[1][0:5]  == 'MMAct': 
        ske = normalize_skeleton(ske, n_dim)
    ske = {key: moving_average_filter(value) for key, value in ske.items()}
    ske = {key: initial_distance(value, input_size, n_dim) for key, value in ske.items()}


    for i in range(1, num_patients+1):  # Assuming persons are indexed from 1 to 18
        # Test set for the current person
        test_ske = ske[i]
        test_labels = labels[i]
        if data_dir_labels.rsplit('/', 1)[1][0:4]  == 'SERE':
        
            test_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in test_labels]

        # Train set 1(excluding test set)
        train_ske = [ske[j] for j in range(1, num_patients) if j != i]
        train_labels = [labels[j] for j in range(1, num_patients) if j != i]

        # Flatten the train data
        train_ske = [item for sublist in train_ske for item in sublist]
        train_labels = [item for sublist in train_labels for item in sublist]
        if data_dir_labels.rsplit('/', 1)[1][0:4]  == 'SERE':
            train_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in train_labels]

        # Append the split data (train, test) to the cross-validation list
        cross_val_data.append((train_ske, train_labels, test_ske, test_labels))

        
    return cross_val_data, append_indices, true_data


def load_data_video(data_dir_labels, data_trial, index_keep, num_patients, resize_dim=(256, 256)):
    

    patient_trials = load('dataset/patient_trials.joblib')
    
    for participant, indices in zip(patient_trials.keys(), index_keep):
        patient_trials[participant] = [patient_trials[participant][i] for i in indices]
            
    cross_val_data = []
    
    for i in range(1, num_patients):  
        
        test_ske = patient_trials[f'P{i}']
        train_ske = [patient_trials[f'P{j}'] for j in range(1, num_patients) if j != i]
        train_ske = [item for sublist in train_ske for item in sublist]

        cross_val_data.append((train_ske, test_ske))
        
    return cross_val_data
    
    
def plot_auc_curves(targets, predictions, fold, n_splits, mean_fpr, mode, first_label, dataset_name):
    
    if dataset_name == 'SERE':
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    elif dataset_name == 'Toronto' or first_label==True:
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = ['standing', 'crouching', 'walking',
                       'running', 'checking_time', 'waving_hand', 
                       'using_phone', 'talking_on_phone', 'kicking', 
                       'pointing', 'throwing', 'jumping', 
                       'exiting', 'entering', 'setting_down', 
                       'talking', 'opening', 'closing',
                       'carrying', 'loitering', 'transferring_object',
                       'looking_around', 'pushing', 'pulling',
                       'picking_up', 'fall', 'sitting_down', 
                       'using_pc', 'drinking', 'pocket_out', 
                       'pocket_in', 'sitting', 'using_phone_desk', 
                       'talking_on_phone_desk', 'standing_up', 
                       'carrying_light', 'carrying_heavy', 'Carrying_light']

    if mode == 'validation' or mode == 'train':
        
        aucs = []
        for i, label in enumerate(label_names):
            
            if len(np.unique(targets[:, i])) < 2:
                continue
                
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
            aucs.append(auc(fpr, tpr))

        print(f'AUCs: {aucs}')
        return
        
    if mode == 'testing': 
        
        tprs, fprs, aucs, folds = [], [], [], []
        viz = {}
        m = 0
        n = 0
        for i, label in enumerate(label_names):
            
            if len(np.unique(targets[:, i])) < 2:
                tprs.append(np.nan)
                aucs.append(np.nan)  
                folds.append(np.nan)
                continue
                
            fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])   
            m = m+1
            if m == 3:
                m = 0
                n = n+1
                
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            auc1 = auc(fpr, tpr)
            aucs.append(auc1)
            folds.append(fold)
        
        
        data = [tprs, fprs, aucs, folds]
            
    print(f"AUCs: {aucs}")  
    return data


def plot_auc_test(data, output, model_name, mean_fpr, first_label, dataset_name):


    # 1) Pick label names depending on arguments
    if first_label or dataset_name == 'Toronto':
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = [
            'standing', 'crouching', 'walking', 'running', 'checking_time',
            'waving_hand', 'using_phone', 'talking_on_phone', 'kicking',
            'pointing', 'throwing', 'jumping', 'exiting', 'entering',
            'setting_down', 'talking', 'opening', 'closing', 'carrying',
            'loitering', 'transferring_object', 'looking_around', 'pushing',
            'pulling', 'picking_up', 'fall', 'sitting_down', 'using_pc',
            'drinking', 'pocket_out', 'pocket_in', 'sitting',
            'using_phone_desk', 'talking_on_phone_desk',
            'standing_up', 'carrying_light', 'carrying_heavy', 'Carrying_light'
        ]
    else:
        label_names = [
            'General compensation', 'Shoulder Compensation', 'Shoulder Elevation',
            'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation'
        ]
    
    label_names = [
            'standing', 'crouching', 'walking', 'running', 'checking_time',
            'waving_hand', 'using_phone', 'talking_on_phone', 'kicking',
            'pointing', 'throwing', 'jumping', 'exiting', 'entering',
            'setting_down', 'talking', 'opening', 'closing', 'carrying',
            'loitering', 'transferring_object', 'looking_around', 'pushing',
            'pulling', 'picking_up', 'fall', 'sitting_down', 'using_pc',
            'drinking', 'pocket_out', 'pocket_in', 'sitting',
            'using_phone_desk', 'talking_on_phone_desk',
            'standing_up', 'carrying_light', 'carrying_heavy', 'Carrying_light'
        ]


    # Convert data to np.array for easier indexing
    data = np.array(data, dtype=object)
    tprs, fprs, aucs, folds = (np.array(data[:, i]) for i in range(4))

    # 2) Prepare arrays to hold per-label results
    mean_tprs = []
    mean_aucs = []
    std_aucs = []
    valid_aucs = []
    valid_tprs = []
    valid_folds = []

    # 3) Compute mean TPR, AUC for each label
    for i, label in enumerate(label_names):
        valid_tpr = [tpr[i] for tpr in tprs if not np.isnan(tpr[i]).any()]
        valid_auc = [ac[i] for ac in aucs if not np.isnan(ac[i])]
        valid_fold = [fd[i] for fd in folds if not np.isnan(fd[i])]

        if valid_tpr:
            mean_tpr = np.mean(valid_tpr, axis=0)
            mean_tpr[-1] = 1.0
        else:
            mean_tpr = np.full_like(mean_fpr, np.nan)

        mean_auc_val = auc(mean_fpr, mean_tpr) if valid_tpr else np.nan
        std_auc_val = np.std(valid_auc) if valid_auc else np.nan

        valid_tprs.append(valid_tpr)
        valid_aucs.append(valid_auc)
        valid_folds.append(valid_fold)
        mean_tprs.append(mean_tpr)
        mean_aucs.append(mean_auc_val)
        std_aucs.append(std_auc_val)

    # 4) If label_names > 6, break them into chunks of 6
    def chunk_labels(seq, size=6):
        for pos in range(0, len(seq), size):
            yield pos, seq[pos:pos+size]

    # We'll create separate figures for each chunk of up to 6 labels
    for chunk_idx, label_chunk in chunk_labels(label_names, 6):
        # Slice out the relevant precomputed arrays
        chunk_tprs = mean_tprs[chunk_idx:chunk_idx+len(label_chunk)]
        chunk_mean_aucs = mean_aucs[chunk_idx:chunk_idx+len(label_chunk)]
        chunk_std_aucs = std_aucs[chunk_idx:chunk_idx+len(label_chunk)]
        chunk_valid_tprs = valid_tprs[chunk_idx:chunk_idx+len(label_chunk)]
        chunk_valid_aucs = valid_aucs[chunk_idx:chunk_idx+len(label_chunk)]
        chunk_valid_folds = valid_folds[chunk_idx:chunk_idx+len(label_chunk)]

        # --------------------------------------------------
        # Plot 1: Normal scale TPR vs FPR
        # --------------------------------------------------
        fig, ax3 = plt.subplots(3, 2, figsize=(20, 16))
        m = 0
        n = 0
        for i, label in enumerate(label_chunk):
            ax3[m, n].plot(mean_fpr, chunk_tprs[i],
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)"
                      % (chunk_mean_aucs[i], chunk_std_aucs[i]))
            colors = plt.cm.get_cmap('tab20', len(chunk_valid_tprs[i]))
            for j, tpr in enumerate(chunk_valid_tprs[i]):
                ax3[m, n].plot(mean_fpr, tpr, color=colors(j / len(chunk_valid_tprs[i])),
                               alpha=0.4,
                               label=f'Fold {chunk_valid_folds[i][j] + 1} ROC (AUC = {chunk_valid_aucs[i][j]:.2f})')

            # For standard deviation shading, we need the standard dev of tprs across folds
            # if you want that logic, you can compute it similarly to how we do it for mean_tpr:
            # (You might be mixing the concept of std of TPR with std of AUC here, so be mindful.)
            # For demonstration, we won't re-calc tpr std across folds here unless you have that data.

            ax3[m, n].set_xlabel('False Positives')
            ax3[m, n].set_ylabel('True Positives')
            ax3[m, n].set_title(f'{model_name} ROC for {label}: TPR vs FPR')
            ax3[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left')

            m += 1
            if m == 3:
                m = 0
                n += 1

        plt.tight_layout()
        # If multiple chunks, add an index in the filename
        plt.savefig(f'{output}/auc_curves_normal_scale_chunk{chunk_idx}.png')
        plt.close()

        # --------------------------------------------------
        # Plot 2: Log scale TPR vs FPR
        # --------------------------------------------------
        fig, ax1 = plt.subplots(3, 2, figsize=(20, 16))
        m, n = 0, 0
        for i, label in enumerate(label_chunk):
            ax1[m, n].plot(mean_fpr, chunk_tprs[i], label="Mean")
            colors = plt.cm.get_cmap('tab20', len(chunk_valid_tprs[i]))
            for j, tpr in enumerate(chunk_valid_tprs[i]):
                ax1[m, n].plot(mean_fpr, tpr, color=colors(j / len(chunk_valid_tprs[i])),
                               alpha=0.4,
                               label=f'Fold {chunk_valid_folds[i][j] + 1} ROC (AUC = {chunk_valid_aucs[i][j]:.2f})')

            ax1[m, n].set_xscale('log')
            ax1[m, n].set_xlabel('False Positives')
            ax1[m, n].set_ylabel('True Positives')
            ax1[m, n].set_title(f'{model_name} ROC (Log Scale) for {label}')
            ax1[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left')

            m += 1
            if m == 3:
                m = 0
                n += 1

        plt.tight_layout()
        plt.savefig(f'{output}/auc_curves_log_chunk{chunk_idx}.png')
        plt.close()

        # --------------------------------------------------
        # Plot 3: TN vs FN
        # --------------------------------------------------
        fig, ax2 = plt.subplots(3, 2, figsize=(20, 16))
        m, n = 0, 0
        for i, label in enumerate(label_chunk):
            # mean_fnr = 1 - chunk_tprs[i]
            # mean_tnr = 1 - mean_fpr
            # For each fold TPR, you'd do something similar if needed
            # We'll do a simple version:
            mean_fnr = 1.0 - chunk_tprs[i]
            mean_tnr = 1.0 - mean_fpr

            ax2[m, n].plot(mean_fnr, mean_tnr, label="Mean")
            colors = plt.cm.get_cmap('tab20', len(chunk_valid_tprs[i]))
            for j, tpr in enumerate(chunk_valid_tprs[i]):
                fnr = 1.0 - tpr
                ax2[m, n].plot(fnr, mean_tnr, color=colors(j / len(chunk_valid_tprs[i])),
                               alpha=0.4, label=f'Fold {chunk_valid_folds[i][j] + 1}')

            ax2[m, n].set_xscale('log')
            ax2[m, n].set_xlabel('False Negatives')
            ax2[m, n].set_ylabel('True Negatives')
            ax2[m, n].set_title(f'{model_name} TN vs FN for {label}')
            ax2[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left')

            m += 1
            if m == 3:
                m = 0
                n += 1

        plt.tight_layout()
        plt.savefig(f'{output}/TN_vs_FN_chunk{chunk_idx}.png')
        plt.close()

        # Save mean AUCs to a table
    table_data = {
        'Label': label_names,
        'Mean AUC': mean_aucs,
        'Std AUC': std_aucs
    }
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(table_data)
    
    # Save the DataFrame to a CSV file
    output_filepath = os.path.join(output, f'{model_name}_mean_aucs.csv')
    df.to_csv(output_filepath, index=False)
    
    print(f"Mean AUCs saved to {output_filepath}")

    return


def load_pseudo_label(data_dir_skeletons, data_true_dir, data_pseudo_dir, first_label, true_labels, model_name, num_patients, input_size):
    
    ske = torch.load(data_dir_skeletons, weights_only=False)
    
    data_true = torch.load(data_true_dir, weights_only=False)

    num_files = len([name for name in os.listdir(data_pseudo_dir.rsplit('/', 1)[0]) if name.rsplit('_', 1)[0] == 'pseudo_labels' ])  
    
    data_pseudo = {}
    for i in range(1, num_files + 1):
        if i not in data_pseudo:  
            data_pseudo[i] = []
            
        with open(f"{data_pseudo_dir.rsplit('.', 1)[0][:-1]}{i}.pkl", 'rb') as f:
            data = pickle.load(f)
            data_pseudo[i].append(data)

    for key in data_pseudo.keys():
        if isinstance(data_pseudo[key], list) and len(data_pseudo[key]) > 0 and isinstance(data_pseudo[key][0], list):
            data_pseudo[key] = data_pseudo[key][0]
            
    for key in ske.keys():
        if isinstance(ske[key], list):  
            indices_to_keep = [i for i, sublist in enumerate(ske[key]) if sublist != []]
            ske[key] = [ske[key][i] for i in indices_to_keep]
            data_true[key] = [data_true[key][i] for i in indices_to_keep]
    
        
    if data_dir_skeletons.rsplit('/', 1)[1][0:5]  == 'MMAct': 
        ske = {
            key: remove_none_from_nested_list(value) for key, value in ske.items()
        }
        # for key in data_true.keys():
        #     data_true[key] = [[[1 - item for item in label] if isinstance(label, list) else 1 - label for label in sublist] for sublist in data_true[key]] 
        for key in data_true.keys():
            data_true[key] = [[[1 - value for value in frame] for frame in trial] for trial in data_true[key]]
            
    for key in ske.keys():
        if isinstance(ske[key], list):  
            indices_to_keep = [i for i, sublist in enumerate(ske[key]) if sublist != []]
            ske[key] = [ske[key][i] for i in indices_to_keep]
            data_true[key] = [data_true[key][i] for i in indices_to_keep]
        
    if data_dir_skeletons.rsplit('/', 1)[1][0:5]  == 'MMAct': 
            
        for key in data_pseudo.keys():
            if isinstance(data_pseudo[key], list) and len(data_pseudo[key]) == 1:
                data_pseudo[key] = data_pseudo[key][0]  # Remove the singleton dimension
            else:
                data_pseudo[key] = data_pseudo[key]  
        # Process MMAct data for pseudo-labeling with elimination of mismatched frames
        data_pseudo, data_true, ske = filter_matching_frames(data_pseudo, data_true,ske)
        
    if data_pseudo_dir.rsplit('/', 1)[1][0:5]  == 'MMAct': 
        ske = normalize_skeleton(ske, n_dim)     
    ske = {key: moving_average_filter(value) for key, value in ske.items()}
    ske = {key: initial_distance(value, input_size,) for key, value in ske.items()}    

                        
    if len(data_pseudo_dir.rsplit('/', 2)[1].rsplit('_',3)) == 3:
        nan_indices = {}
        for key in data_pseudo.keys():
            nan_indices[key] = [[i for i, frame in enumerate(frames) if np.isnan(frame).any()] for frames in data_pseudo[key]]
            data_pseudo[key] = [
            [frame for i, frame in enumerate(frames) if i not in nan_indices[key][seq_idx]]
            for seq_idx, frames in enumerate(data_pseudo[key])]
            data_true[key] = [
            [frame for i, frame in enumerate(frames) if i not in nan_indices[key][seq_idx]]
            for seq_idx, frames in enumerate(data_true[key])]
            ske[key] = [
            [frame for i, frame in enumerate(frames) if i not in nan_indices[key][seq_idx]]
            for seq_idx, frames in enumerate(ske[key])]
    
    pseudo_data_set_info(data_true, data_pseudo, data_pseudo_dir, num_patients, first_label)
    cross_val_data = []
    
    for i in range(1, num_patients):
        
        # Test set for the current person
        test_ske = torch.tensor( [frame for trial in ske[i] for frame in trial])
        test_labels = torch.tensor([frame for trial in data_true[i] for frame in trial])
        
        # test_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in test_labels]

        # Train set (excluding test set)
        if true_labels:
            train_ske = torch.tensor([frame for j in range(1, num_patients) if j != i for trial in ske[j] for frame in trial])
            train_labels = torch.tensor([frame for j in range(1, num_patients) if j != i for trial in data_true[j] for frame in trial])
        else:
            train_ske = torch.tensor([frame for j in range(1, num_patients) if j != i for trial in ske[j] for frame in trial])
            train_labels = torch.tensor([frame for j in range(1, num_patients) if j != i for trial in data_pseudo[j] for frame in trial])

        # train_labels = train_labels.unsqueeze(-1)
        # test_labels = test_labels.unsqueeze(-1)
        
        # Append the split data (train, test) to the cross-validation list
        cross_val_data.append((train_ske, train_labels, test_ske, test_labels))
        
    return cross_val_data


def pseudo_data_set_info(dt, dp, save_dir, num_patients, first_label):

    # print(len(dp),len(dp[1]), len(dp[1][0]), len(dp[1][0][0]), len(dp[1][0][0][0]) )

    data_true_list = torch.tensor([frame for i in range(1, num_patients) for trial in dt[i] for frame in trial])
    data_pseudo_list = torch.tensor([label for i in range(1, num_patients) for trial in dp[i] for frame in trial for label in frame])
    
    if first_label: 
        data_true_flat = data_true_list.view(-1)
        data_pseudo_flat = data_pseudo_list.view(-1)
    else:
        data_true_flat = data_true_list.view(-1)
        data_pseudo_flat = data_pseudo_list.view(-1)

    # 1. Accuracy
    accuracy = accuracy_score(data_true_flat, data_pseudo_flat)
    print("Accuracy:", accuracy)

    # 2. Precision, Recall, and F1 Score (per label)
    precision = precision_score(data_true_flat, data_pseudo_flat, average='weighted', zero_division=0)
    recall = recall_score(data_true_flat, data_pseudo_flat, average='weighted', zero_division=0)
    f1 = f1_score(data_true_flat, data_pseudo_flat, average='weighted', zero_division=0)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # 3. Confusion Matrix
    conf_matrix = confusion_matrix(data_true_flat, data_pseudo_flat)
    print("Confusion Matrix:\n", conf_matrix)

    # 4. Hamming Loss
    hamming = hamming_loss(data_true_flat, data_pseudo_flat)
    print("Hamming Loss:", hamming)

    # 5. Mismatch Count
    mismatch_count = (data_true_flat != data_pseudo_flat).sum().item()
    print("Mismatch Count:", mismatch_count)
    # Save metrics to a JSON file
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'hamming_loss': hamming,
        'mismatch_count': mismatch_count
    }
    
    pseudo_labels_filepath = os.path.join(os.path.dirname(save_dir), 'info_labels.pkl')
    with open(pseudo_labels_filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    return


def filter_matching_frames(data_pseudo, data_true, ske):

    
    for key in data_pseudo.keys():
        
        # For each trial in this participant's data
        for trial_idx, (pseudo_trial, true_trial, skeleton) in enumerate(zip(data_pseudo[key], data_true[key], ske[key])):
            # Convert to numpy arrays if not already
            pseudo_frames = np.array(pseudo_trial)
            true_frames = np.array(true_trial)
            skeleton_frames = np.array(skeleton)
            
            if pseudo_frames.shape[0] !=  true_frames.shape[0]:
                frames_delete = np.abs(pseudo_frames.shape[0] - true_frames.shape[0])
                if pseudo_frames.shape[0] > true_frames.shape[0]:
                    pseudo_frames = pseudo_frames[:-frames_delete]
                    data_pseudo[key][trial_idx] = pseudo_frames.tolist()
                    ske[key][trial_idx] = skeleton_frames[:-frames_delete].tolist()
                    
                else:
                    true_frames = true_frames[:-frames_delete]
                    data_true[key][trial_idx] = true_frames.tolist()
                        
    return data_pseudo, data_true, ske
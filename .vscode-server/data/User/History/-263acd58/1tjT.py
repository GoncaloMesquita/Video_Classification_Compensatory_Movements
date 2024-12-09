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
from torch.utils.data.distributed import DistributedSampler
import gc
import cv2
from torchvision import transforms
from joblib import load



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
    accuracy = accuracy_score(targets, predictions)
    precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
    recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    
    precision_sample = precision_score(targets, predictions, average='samples', zero_division=0)
    recall_sample = recall_score(targets, predictions, average='samples', zero_division=0)
    f1_sample = f1_score(targets, predictions, average='samples', zero_division=0)
    
    accuracy_per_label = np.mean(targets == predictions, axis=0)
    precision_per_label = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_label = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_label = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Store metrics in a dictionary
    metrics_dict = {
        'mode': mode,
        'accuracy_micro': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_sample': precision_sample,
        'recall_sample': recall_sample,
        'f1_sample': f1_sample,
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
        round(accuracy, 3),
        round(precision_micro, 3),
        round(recall_micro, 3),
        round(f1_micro, 3),
        round(precision_sample, 3),
        round(recall_sample, 3),
        round(f1_sample, 3),
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
        self.lengths = torch.tensor([len(seq) for seq in data])
        self.model_name = model_name
        self.target_size = target_size
        self.max_frames = max_frames  # Limit on frames to load per video, if specified
        self.type_train = type_train

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])

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


def create_dataloader(x, y, batch_size, shuffle, model_name, trainII, x2):
    
    if trainII:
        dataset = CustomDataset(x, y, model_name, trainII, x2) 
        train_sampler = DistributedSampler(dataset, num_replicas=3, rank=[0,1,2])
        dataloader = DataLoader(dataset,batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    else:
        dataset = CustomDataset(x, y, model_name, trainII, x2) 
        train_sampler = DistributedSampler(dataset, num_replicas=3, rank=[0,1,2])
        dataloader = DataLoader(dataset,batch_size=batch_size, collate_fn=collate_fn, sampler = train_sampler, num_workers=2, pin_memory=True)
    
    return  dataloader, train_sampler


def collate_fn(batch):
    
    sequences, labels, model_name, sequences2 = zip(*batch)  # Unzip the sequences and labels
    lengths = torch.tensor([len(seq) for seq in sequences])  # Get lengths of each sequence

    if model_name[0] == 'SkateFormer':
        max_length = 769
        
    elif model_name[0] == 'moment' or model_name[0] == 'moment+dino':

        max_length = 512
        sequences, labels, sequences2 = zip(*[(seq, label, sequence2) for seq, label, sequence2 in zip(sequences, labels, sequences2) if len(seq) <= max_length])
        
    else:
        max_length = max(len(seq) for seq in sequences)
        
    padded_ske = []
    for seq in sequences:
        seq = torch.tensor(np.array(seq))
        pad_length = max_length - len(seq)
        if pad_length > 0:
            pad_frames = seq[0].unsqueeze(0).repeat(pad_length, *[1 for _ in seq[0].shape])  
            # Concatenate seq and the padding frames
            padded_seq = torch.cat((seq, pad_frames), dim=0)
        else:
            padded_seq = seq
        padded_ske.append(padded_seq)
    padded_sequences = torch.stack(padded_ske)
    
    if sequences2[0] is None:
        return padded_sequences, torch.tensor(labels), lengths, None
    
    else:
        padded_ske = []
        for seq in sequences2:
            seq = torch.tensor(np.array(seq))
            pad_length = max_length - len(seq)
            if pad_length > 0:
                pad_frames = seq[0].unsqueeze(0).repeat(pad_length, *[1 for _ in seq[0].shape])  
                # Concatenate seq and the padding frames
                padded_seq = torch.cat((seq, pad_frames), dim=0)
            else:
                padded_seq = seq
            padded_ske.append(padded_seq)
        padded_sequences2 = torch.stack(padded_ske)
    
    return padded_sequences, torch.tensor(labels), lengths, padded_sequences2

    
def initial_distance(skeletons):
    features = []
    
    # Iterate through each film
    for sequence in skeletons:
        
        initial_position = np.array(sequence[0])
        
        initial_position = initial_position.reshape(-1, 3)
        
        feature_sequence = []
        
        for frame in sequence:
            current_position = np.array(frame)
            current_position = current_position.reshape(-1, 3)  # Reshape each frame similarly
            
            distance = current_position - initial_position
            
            feature_sequence.append(distance.reshape(99))
        
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


def load_data(data_dir_labels, data_dir_skeletons):
    
    ske = torch.load(data_dir_skeletons, weights_only=False)
    labels = torch.load(data_dir_labels, weights_only=False)
    
    append_indices = []
    for key in ske.keys():
        if isinstance(ske[key], list):  
            indices_to_keep = [i for i, sublist in enumerate(ske[key]) if sublist != []]

            ske[key] = [ske[key][i] for i in indices_to_keep]
            labels[key] = [labels[key][i] for i in indices_to_keep]
            append_indices.append(indices_to_keep)
            
    cross_val_data = []

    ske = {key: moving_average_filter(value) for key, value in ske.items()}
    ske = {key: initial_distance(value) for key, value in ske.items()}

    for i in range(1, 19):  # Assuming persons are indexed from 1 to 18
        # Test set for the current person
        test_ske = ske[i]
        test_labels = labels[i]
        test_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in test_labels]

        # Train set (excluding test set)
        train_ske = [ske[j] for j in range(1, 19) if j != i]
        train_labels = [labels[j] for j in range(1, 19) if j != i]

        # Flatten the train data
        train_ske = [item for sublist in train_ske for item in sublist]
        train_labels = [item for sublist in train_labels for item in sublist]
        train_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in train_labels]

        # Append the split data (train, test) to the cross-validation list
        cross_val_data.append((train_ske, train_labels, test_ske, test_labels))

        
    return cross_val_data, append_indices


def load_data_video(data_dir_labels, data_trial, index_keep,  resize_dim=(256, 256)):
    

    patient_trials = load('dataset/patient_trials.joblib')
    
    for participant, indices in zip(patient_trials.keys(), index_keep):
        patient_trials[participant] = [patient_trials[participant][i] for i in indices]
            
    cross_val_data = []
    
    for i in range(1, 19):  
        
        test_ske = patient_trials[f'P{i}']
        train_ske = [patient_trials[f'P{j}'] for j in range(1, 19) if j != i]
        train_ske = [item for sublist in train_ske for item in sublist]

        cross_val_data.append((train_ske, test_ske))
        
    return cross_val_data
    
    
def plot_auc_curves(targets, predictions, fold, n_splits, mean_fpr, mode, first_label):
    
    label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    if first_label:
        label_names = ['General compensation']

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


def plot_auc_test(data, output, model_name, mean_fpr, first_label):
    
    if first_label:
        label_names = ['General compensation']
    else:
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation','Exaggerated Shoulder Abduction', 'Trunk Compensation','Head Compensation' ]
        
    data = np.array(data, dtype=object)
    tprs, fprs, aucs, folds = np.array(data[:, 0]), np.array(data[:, 1]), np.array(data[:, 2]), np.array(data[:, 3])

    mean_tprs = []
    mean_aucs = []
    std_aucs = []
    valid_aucs = []
    valid_tprs = []
    valid_folds = []
    
    for i, label in enumerate(label_names):
        
        valid_tpr = [tpr[i] for tpr in tprs if not np.isnan(tpr[i]).any()]
        valid_auc = [auc[i] for auc in aucs if not np.isnan(auc[i])]
        valid_fold = [fold[i] for fold in folds if not np.isnan(fold[i])]

        
        mean_tpr = np.mean(valid_tpr, axis=0) if valid_tpr else np.full_like(mean_fpr, np.nan)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr) if valid_tpr else np.nan
        
        std_auc = np.std(valid_auc) if valid_auc else np.nan
        
        valid_tprs.append(valid_tpr)
        valid_aucs.append(valid_auc)
        valid_folds.append(valid_fold)
        
        mean_tprs.append(mean_tpr)
        mean_aucs.append(mean_auc)
        std_aucs.append(std_auc)

    m=0
    n=0
    fig, ax3 = plt.subplots(3, 2, figsize=(20, 16))

    for i, label in enumerate(label_names):

        ax3[m, n].plot(mean_fpr, mean_tprs[i], label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_aucs[i], std_aucs[i]))
        # colors = plt.cm.viridis(np.linspace(0, 4, len(valid_tprs[i])))
        colors = plt.cm.get_cmap('tab20', len(valid_tprs[i]))
        for j, tpr in enumerate(valid_tprs[i]):

            ax3[m, n].plot(mean_fpr, tpr, color=colors(j / len(valid_tprs[i])), alpha=0.4, label=f'Fold {valid_folds[i][j] + 1} ROC (AUC = {valid_aucs[i][j]:.2f})')
            
        std_tpr = np.std(mean_tprs[i], axis=0)
        tprs_upper = np.minimum(mean_tprs[i] + std_tpr, 1)
        tprs_lower = np.maximum(mean_tprs[i] - std_tpr, 0)
        ax3[m, n].fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
        ax3[m, n].set_xlabel('False Positives')
        ax3[m, n].set_ylabel('True Positives')
        ax3[m, n].set_title(f'{model_name} ROC Curve for {label}: TPR vs FPR')
        ax3[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left')
        m = m+1
        if m == 3:
            m = 0
            n = n+1
            
    plt.tight_layout()
    plt.savefig(f'{output}/auc_curves_normal_scale.png')
    plt.close()
    
    fig, ax1 = plt.subplots(3, 2, figsize=(20, 16))
    m,n = 0 ,0 
    for i, label in enumerate(label_names):
        ax1[m,n].plot(mean_fpr, mean_tprs[i], label="Mean" )
        # colors = plt.cm.viridis(np.linspace(0, 4, len(valid_tprs[i])))
        colors = plt.cm.get_cmap('tab20', len(valid_tprs[i]))
        for j, tpr in enumerate(valid_tprs[i]):

            ax1[m, n].plot(mean_fpr, tpr, color=colors(j / len(valid_tprs[i])), alpha=0.4, label=f'Fold {valid_folds[i][j] + 1} ROC (AUC = {valid_aucs[i][j]:.2f})')
        tprs_upper = np.minimum(mean_tprs[i] + std_tpr, 1)
        tprs_lower = np.maximum(mean_tprs[i] - std_tpr, 0)
        ax1[m,n].fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
        ax1[m,n].set_xscale('log')
        ax1[m,n].set_xlabel('False Positives')
        ax1[m,n].set_ylabel('True Positives')
        ax1[m,n].set_title(f'{model_name} ROC Curve for {label}: TPR vs FPR')
        ax1[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left') 
        m = m+1
        if m == 3:
            m = 0
            n = n+1
            
    plt.tight_layout()
    plt.savefig(f'{output}/auc_curves_log.png')
    plt.close()

    # Plot 3: True Negatives (normal scale) vs False Negatives (log scale)
    fig, ax2 = plt.subplots(3, 2, figsize=(20, 16))
    m, n = 0, 0
    for i, label in enumerate(label_names):
        mean_fnr = 1 - mean_tprs[i]
        mean_tnr = 1 - mean_fpr
        ax2[m, n].plot(mean_fnr, mean_tnr, label="Mean")
        colors = plt.cm.get_cmap('tab20', len(valid_tprs[i]))
        # colors = plt.cm.viridis(np.linspace(0, 4, len(valid_tprs[i])))
        for j, tpr in enumerate(valid_tprs[i]):
            fnr = 1 - tpr
            ax2[m, n].plot(fnr, mean_tnr, color=colors(j / len(valid_tprs[i])), alpha=0.4, label=f'Fold {valid_folds[i][j] + 1}')
            
        std_fnr = np.std(mean_fnr, axis=0)
        fnr_upper = np.minimum(mean_fnr + std_fnr, 1)
        fnr_lower = np.maximum(mean_fnr - std_fnr, 0)
        ax2[m, n].fill_between(mean_fnr, fnr_lower, fnr_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
        ax2[m, n].set_xscale('log')
        ax2[m, n].set_xlabel('False Negatives')
        ax2[m, n].set_ylabel('True Negatives')
        ax2[m, n].set_title(f'{model_name} ROC Curve for {label}: TN vs FN')
        ax2[m, n].legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1), loc='upper left')

        
        m = m + 1
        if m == 3:
            m = 0
            n = n + 1

    plt.tight_layout()
    plt.savefig(f'{output}/TN_vs_FN.png')
    plt.close()
    
    return


def load_pseudo_label(data_dir_skeletons, data_true_dir, data_pseudo_dir, first_label, true_labels):
    
    ske = torch.load(data_dir_skeletons, weights_only=False)
    data_true = torch.load(data_true_dir, weights_only=False)
    with open(data_pseudo_dir, 'rb') as f:
        data_pseudo = pickle.load(f)
        
    for key in data_pseudo.keys():
        if isinstance(data_pseudo[key], list) and len(data_pseudo[key]) > 0 and isinstance(data_pseudo[key][0], list):
            data_pseudo[key] = data_pseudo[key][0]
    
    for key in ske.keys():
        if isinstance(ske[key], list):  
            indices_to_keep = [i for i, sublist in enumerate(ske[key]) if sublist != []]
            ske[key] = [ske[key][i] for i in indices_to_keep]
    
    ske = {key: moving_average_filter(value) for key, value in ske.items()}
    ske = {key: initial_distance(value) for key, value in ske.items()}
    
    for i in range(1, 19):
        for trial_idx, trial in enumerate(data_true[i]):
            for frame_idx, frame in enumerate(trial):
                if len(frame) == 12:
                    if frame[:6] == frame[6:]:
                        data_true[i][trial_idx][frame_idx] = frame[:6]
    
    pseudo_data_set_info(data_true, data_pseudo, data_pseudo_dir)
    cross_val_data = []
    
    for i in range(1, 19):
        
        # Test set for the current person
        test_ske = torch.tensor( [frame for trial in ske[i] for frame in trial])
        test_labels = torch.tensor([frame for trial in data_true[i] for frame in trial])
        
        # test_labels = [[1 - label if idx == 0 else label for idx, label in enumerate(sublist)] for sublist in test_labels]

        # Train set (excluding test set)
        if true_labels:
            train_ske = torch.tensor([frame for j in range(1, 19) if j != i for trial in ske[j] for frame in trial])
            train_labels = torch.tensor([frame for j in range(1, 19) if j != i for trial in data_true[j] for frame in trial])
        else:
            train_ske = torch.tensor([frame for j in range(1, 19) if j != i for trial in ske[j] for frame in trial])
            train_labels = torch.tensor([frame for j in range(1, 19) if j != i for trial in data_pseudo[j] for frame in trial])

        if first_label:
            if true_labels:
                train_labels = train_labels[:, 0].unsqueeze(-1)
            else:
                train_labels = train_labels.unsqueeze(-1)
            test_labels = test_labels[:, 0].unsqueeze(-1)
            
        # Append the split data (train, test) to the cross-validation list
        cross_val_data.append((train_ske, train_labels, test_ske, test_labels))
        
    return cross_val_data


def pseudo_data_set_info(dt, dp, save_dir):

    data_true_list = torch.tensor([frame for i in range(1, 19) for trial in dt[i] for frame in trial])

    data_pseudo_list = torch.tensor([frame for i in range(1, 19) for trial in dp[i] for frame in trial])
        
    data_true_flat = data_true_list[:,0].view(-1)
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
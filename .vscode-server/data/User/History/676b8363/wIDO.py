import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class EarlyStopping:
    
    def __init__(self, patience, model_name, learning_rate, batch_size, output_dir, verbose=True, delta=0):
        
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

    def __call__(self, val_loss, model, fold, epoch):

        self.fold = fold
        self.epoch = epoch

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.verbose:
            print("Validation loss improved. Saving the model...")
        torch.save(model.state_dict(), f"{self.output}/{self.model}_{self.fold}_best.pth")
        return f"{self.output}/{self.model}_{self.fold}_best.pth"


def metrics(targets, predictions, mode, output_dir, model_name):
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='samples', zero_division=0)
    recall = recall_score(targets, predictions, average='samples', zero_division=0)
    f1 = f1_score(targets, predictions, average='samples', zero_division=0)
    
    accuracy_per_label = np.mean(targets == predictions, axis=0)
    precision_per_label = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_label = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_label = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Store metrics in a dictionary
    metrics_dict = {
        'mode': mode,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy_per_label': accuracy_per_label.tolist(),
        'precision_per_label': precision_per_label.tolist(),
        'recall_per_label': recall_per_label.tolist(),
        'f1_per_label': f1_per_label.tolist()
    }
    
    # Print metrics
    print(f"{mode}")
    print(f" Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Accuracy per label: {accuracy_per_label}, Precision per label: {precision_per_label}, Recall per label: {recall_per_label}, F1 per label: {f1_per_label}")
    
    # Save metrics to a JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_metrics_{mode}.json")
    with open(output_file, 'w') as f:
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

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):

        # self.data = torch.tensor(data, dtype=torch.float32) if not torch.is_tensor(data) else data
        # self.targets = torch.tensor(targets, dtype=torch.float32) if not torch.is_tensor(targets) else targets
        self.transform = transform
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_target = self.targets[idx]

        if self.transform:
            sample_data = self.transform(sample_data)
            
        sample_data_tensor = torch.tensor(sample_data, dtype=torch.float32)  # Adjust dtype as needed
        sample_target_tensor = torch.tensor(sample_target, dtype=torch.float32) 

        return sample_data_tensor, sample_target_tensor

def create_dataloader(x, y, batch_size):

    dataset = CustomDataset(x, y) 
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return  dataloader

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unzip the sequences and labels
    lengths = torch.tensor([len(seq) for seq in sequences])  # Get lengths of each sequence

    # Pad the sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True)


    return packed_sequences, lengths, torch.stack(labels)
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_labels, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)  
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Fully connected layer for binary multi-label classification
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, num_labels)  # Second hidden layer
        
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(packed_input, (h0, c0))
        
        out, _ = pad_packed_sequence(out, batch_first=True)

        
        
        lengths = lengths.to(x.device)
        mask = torch.arange(out.size(1), device=x.device).expand(batch_size, out.size(1)) < lengths.unsqueeze(1)
        out = out * mask.unsqueeze(-1)  
        
        out_sum = out.sum(dim=1)
        out_count = lengths.float()  
        out_mean = out_sum / out_count.unsqueeze(1)  
        
        # Retrieve the last valid output for each sequence
        # idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1).to(x.device)
        # out_mean = out.gather(1, idx).squeeze(1)

        # Layer normalization
        out_mean = self.layer_norm(out_mean)
        
        # Fully connected layers
        out = self.fc1(out_mean)
        out = self.relu(out)
        
        out = self.fc2(out)

        return out



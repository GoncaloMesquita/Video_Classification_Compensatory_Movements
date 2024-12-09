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
                            batch_first=True, dropout=dropout)
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)  
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Fully connected layer for binary multi-label classification
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, num_labels)  # Output layer
        
    # def forward(self, x, lengths):
        
    #     batch_size = x.size(0)
        
    #     packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

    #     # Initialize hidden and cell state
    #     h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
    #     c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
    #     # Forward propagate through LSTM
    #     out, hstate = self.lstm(packed_input, (h0, c0))
        
    #     out, _ = pad_packed_sequence(out, batch_first=True)
        
    #     # Retrieve the last valid output for each sequence
    #     idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1).to(x.device)
    #     out = out.gather(1, idx).squeeze(1)
        
    #     # Take the last output
    #     # out = out[:, -1, :]
        
    #     # Apply batch normalization
    #     out = self.batch_norm(out)
    #     out = self.fc1(out)
    #     out = self.batch_norm1(out) 
    #     out = self.relu(out)
         
        
    #     out = self.fc2(out)
    #     out = self.batch_norm2(out)  
    #     out = self.relu(out)
        
        
    #     out = self.fc3(out)
        
    #     return out  # The output will be passed directly to BCEWithLogitsLoss
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # Pack padded sequence for LSTM
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(packed_input, (h0, c0))
        
        # Unpack the sequence
        out, _ = pad_packed_sequence(out, batch_first=True)

        # Prepare to compute the mean over non-padded outputs
        lengths.to(x.device)
        mask = torch.arange(out.size(1), device=x.device).expand(batch_size, out.size(1)) < lengths.unsqueeze(1)
        out = out * mask.unsqueeze(-1)  # Apply mask to out

        # Calculate the mean, avoiding padded values
        out_sum = out.sum(dim=1)
        out_count = lengths.float()  # Convert lengths to float for averaging
        out_mean = out_sum / out_count.unsqueeze(1)  # Divide by the number of valid time steps

        # Layer normalization
        out_mean = self.layer_norm(out_mean)
        
        # Fully connected layers
        out = self.fc1(out_mean)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        
        # Apply sigmoid activation for multi-label classification
        
        return out



import torch
import torch.nn as nn

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
        
        # Fully connected layer for binary multi-label classification
        self.fc = nn.Linear(hidden_size, num_labels)
        
    def forward(self, x):
        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, hstate = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        
        # Apply batch normalization
        out = self.batch_norm(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out  # The output will be passed directly to BCEWithLogitsLoss

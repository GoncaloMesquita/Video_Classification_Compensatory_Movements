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
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size // 2, )  # Output layer
        
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


        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch_norm1(out)  # Apply batch normalization after first layer
        out = self.dropout(out)  # Apply dropout after first layer
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.batch_norm2(out)  # Apply batch normalization after second layer
        out = self.dropout(out)  # Apply dropout after second layer
        
        out = self.fc3(out)
        out = self.batch_norm3(out)  # Apply batch normalization after third layer
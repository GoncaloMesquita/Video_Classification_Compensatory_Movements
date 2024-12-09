import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_labels, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(packed_input, (h0, c0))
        
        out, _ = pad_packed_sequence(out, batch_first=True)

        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1).to(x.device)
        out = out.gather(1, idx).squeeze(1)
        
        out = self.dropout(out)
        
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        
        return out



from models.LSTM import LSTM
import torch
# from act_model import AcTModel

def create_model(model_name, input_size, hidden_size, num_layers, num_labels, dropout, checkpoint, mode):
    # Import the necessary modules
    
    if model_name == 'LSTM':
        if mode == 'train':
            model = LSTM(input_size, hidden_size, num_layers, num_labels, dropout)
            
        if mode == 'test':
            model = LSTM(input_size, hidden_size, num_layers, num_labels, dropout)
            model.load_state_dict(torch.load(checkpoint), strict=True)
            
        
    # elif model_name == 'AcT':
    #     model = AcTModel()
    
    return model
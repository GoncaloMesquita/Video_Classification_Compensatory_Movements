import torch
import torch.nn as nn
# from momentfm import MOMENTPipeline
from models.Moment import MOMENTPipeline


class Moment_II(nn.Module):
    def __init__(self, num_classes, dropout, input_size, embedding_dim=512): 
        super(Moment_II, self).__init__()
        
        # Load the pre-trained MOMENT model for embedding extraction
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification',
                'n_channels': input_size, # number of input channels
                'seq_length': 512,
                'num_class': num_classes,
                'dropout': dropout,
            },
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
        
        self.moment_model.init() 
        
    def forward(self, x):
        
        x = x.permute(0, 2, 1)
        output = self.moment_model(x)
        
        return output.logits
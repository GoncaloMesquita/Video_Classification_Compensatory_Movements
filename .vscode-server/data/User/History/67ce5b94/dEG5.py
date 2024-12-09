import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
class Moment(nn.Module):
    def __init__(self, num_classes, embedding_dim=128): 
        super(Moment, self).__init__()
        
        # Load the pre-trained MOMENT model for embedding extraction
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 99, # number of input channels
                'freeze_encoder': True, # Freeze the patch embedding layer
                'freeze_embedder': True, # Freeze the transformer encoder
                'freeze_head': False, # The linear forecasting head must be trained
                'enable_gradient_checkpointing': False,
                'reduction': 'mean',
            },
            # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
        
        self.moment_model.init() 
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        with torch.no_grad():  # Freezing MOMENT weights if you don't want to fine-tune
            embeddings = self.moment_model(x)
        
        output = self.classifier(embeddings.embeddings)
        return output
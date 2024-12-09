import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
class Moment(nn.Module):
    def __init__(self, num_classes, embedding_dim=1024): 
        super(Moment, self).__init__()
        
        # Load the pre-trained MOMENT model for embedding extraction
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={"task_name": "embedding"}
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
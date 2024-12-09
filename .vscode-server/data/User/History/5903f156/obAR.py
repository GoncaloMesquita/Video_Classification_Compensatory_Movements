import torch
import torch.nn as nn
from momentfm import MOMENTPipeline

class MomentMLPModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=768): 
        super(MomentMLPModel, self).__init__()
        
        # Load the pre-trained MOMENT model for embedding extraction
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={"task_name": "embedding"}
        )
        self.moment_model.init()  # Initialize the pre-trained model
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        
        with torch.no_grad():  # Freezing MOMENT weights if you don't want to fine-tune
            embeddings = self.moment_model(x)
        
        output = self.classifier(embeddings)
        return output
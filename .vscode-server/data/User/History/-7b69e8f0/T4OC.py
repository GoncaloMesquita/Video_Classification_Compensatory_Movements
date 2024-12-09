import torch
import torch.nn as nn
from momentfm import MOMENTPipeline
from transformers import AutoModel, AutoImageProcessor


class MomentDino(nn.Module):
    def __init__(self, num_classes, dropout): 
        super(MomentDino, self).__init__()
        
        embedding_moment = 2048
        embedding_dino = 2048   
        self.dropout = dropout
        self.attention_weights = []
        
        self.moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small", 
            model_kwargs={
                'task_name': 'classification',
                'n_channels': 99, # number of input channels
                'seq_length': 512,
                'num_class': num_classes,
                'dropout': dropout,
            },
            )
        
        self.moment_model.init() 
        
        self.dino_processor  = AutoModel.from_pretrained("facebook/dino-v2-vitb14", output_attentions=True)
        self.dinov2 = AutoImageProcessor.from_pretrained(self.dino_processor)
        
        self.ll = nn.Linear(384, 256)
        
        self.norm1 = nn.LayerNorm(embedding_moment + embedding_dino )
        self.dropout = nn.Dropout(dropout)
        
        self.fc_fusion = nn.Linear(embedding_moment + embedding_dino, 256)
        self.norm2 = nn.LayerNorm(256)
        self.final_fc = nn.Linear(256, num_classes)
    
    
    def forward(self, x1, x2):
        
        frame_embeddings = []
        x = x.permute(0, 2, 1)
        
        output = self.moment_model(x1)
        
        for frame in x2:
            frame_input = self.dino_processor(images=frame, return_tensors="pt").pixel_values.to(frame.device)
            dino_output, attention_weight = self.dinov2(frame_input)
            frame_embeddings.append(dino_output.last_hidden_state.mean(dim=1))
            self.attention_weights.append(attention_weight)
        
        dino_embedding = torch.stack(frame_embeddings, dim=0)
        dino_embedding = self.ll(dino_embedding)
        combined_embedding = torch.cat((output.embeddings, dino_embedding), dim=1)
        
        # Apply dropout, normalization, and non-linear function
        combined_embedding = self.dropout(combined_embedding)
        combined_embedding = self.norm1(combined_embedding)
        combined_embedding = nn.ReLU()(combined_embedding)
        
        out = self.fc_fusion(combined_embedding)
        
        # Apply dropout, normalization, and non-linear function
        out = self.dropout(out)
        out = self.norm2(out)
        out = nn.ReLU()(out)
        
        out = self.final_fc(out)
        
        return out